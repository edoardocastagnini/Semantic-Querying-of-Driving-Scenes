import time
import numpy as np
from collections import defaultdict

from config.settings import (
    QUERIES, YOLO_CONF, YOLO_IOU, IMG_SIZE, TRACKER, ROAD_CLASSES,
    CLIP_INTERVAL, DRAW_ALL, DRAW_COUNTERS, VISUAL_PERSIST_FRAMES,
    ALLOW_MASK_REUSE_WHEN_MISSING, OUTPUT_VIDEO,
    TRAFFIC_SIGN_CONF, TRAFFIC_SIGN_IOU, TRAFFIC_SIGN_IMG_SIZE,
    DRAW_TRAFFIC_SIGNS, TRAFFIC_SIGN_SMOOTHING_ENABLED,
    TRAFFIC_SIGN_ROUTE_MODE,
)
from models.loader import load_all_models
from core.clip_utils import (
    build_text_features, build_negative_text_features, build_query_type_cache,
    encode_crop, compute_scores,
)
from core.geometry import box_center, extract_semantic_crop
from core.crossing import update_crossing_state
from core.tracking import create_track_state, maybe_link_to_recent_lost, cleanup_tracks
from core.semantics import update_semantics
from core.traffic_signs import (
    selected_traffic_sign_class_ids, traffic_sign_label,
    make_traffic_sign_detection, update_traffic_sign_tracks,
    build_traffic_sign_query_routes, build_clip_traffic_sign_query_routes,
    traffic_sign_queries_for_class,
)
from visualization.drawing import (
    draw_visual, draw_counters, color_from_id, draw_traffic_sign_detection,
)
from io_utils.video_io import setup_capture, setup_writer
from io_utils.output_writer import build_summary, save_summary


# ── MODELS ───────────────────────────────────────────────────────

models = load_all_models()
device        = models["device"]
yolo          = models["yolo"]
traffic_sign_yolo = models["traffic_sign_yolo"]
clip_model    = models["clip_model"]
clip_preprocess = models["clip_preprocess"]
tokenizer     = models["tokenizer"]

if TRAFFIC_SIGN_ROUTE_MODE == "clip":
    traffic_sign_query_routes, traffic_sign_route_diagnostics = build_clip_traffic_sign_query_routes(
        QUERIES, clip_model, tokenizer, device
    )
elif TRAFFIC_SIGN_ROUTE_MODE == "aliases":
    traffic_sign_query_routes = build_traffic_sign_query_routes(QUERIES)
    traffic_sign_route_diagnostics = {}
else:
    raise ValueError(f"Unknown TRAFFIC_SIGN_ROUTE_MODE: {TRAFFIC_SIGN_ROUTE_MODE}")
main_clip_queries = [q for q in QUERIES if q not in traffic_sign_query_routes]

text_feature_cache = build_text_features(clip_model, tokenizer, device, queries=main_clip_queries)
negative_text_feature_cache = build_negative_text_features(clip_model, tokenizer, device)
query_type_cache, _ = build_query_type_cache(clip_model, tokenizer, device)

print("[INFO] Query semantic types:")
for q in QUERIES:
    info = query_type_cache[q]
    route = "traffic-sign detector" if q in traffic_sign_query_routes else "main YOLO + CLIP"
    print(f"  - {q}: route={route}, human_like={info['is_human_like']} (score={info['human_like_score']:.3f})")

use_traffic_sign_detector = traffic_sign_yolo is not None and bool(traffic_sign_query_routes)
traffic_sign_class_ids = (
    selected_traffic_sign_class_ids(traffic_sign_query_routes)
    if use_traffic_sign_detector else []
)
if traffic_sign_yolo is not None:
    selected_msg = "disabled by queries" if not use_traffic_sign_detector else traffic_sign_class_ids
    print(f"[INFO] Traffic sign classes: {selected_msg}")
    print("[INFO] Traffic sign query routes:")
    for query in QUERIES:
        class_ids = traffic_sign_query_routes.get(query, [])
        if not class_ids:
            print(f"  - {query}: main detector")
            continue
        diag = traffic_sign_route_diagnostics.get(query, {})
        reason = diag.get("route", TRAFFIC_SIGN_ROUTE_MODE)
        best = diag.get("best_class_label", "-")
        score = diag.get("best_class_score", 0.0)
        margin = diag.get("class_margin", 0.0)
        print(
            f"  - {query}: classes={class_ids} reason={reason} "
            f"best={best} score={score:.3f} margin={margin:.3f}"
        )


# ── VIDEO ───────────────────────────────────────────────────────────────

cap, video_info = setup_capture()
writer = setup_writer(video_info)
width  = video_info["width"]
height = video_info["height"]
total_frames = video_info["total_frames"]


# ── RUNTIME GLOBAL STATE ─────────────────────────────────────────────────────

tracks               = {}          # {track_id: state_dict}
counters             = defaultdict(int)
traffic_sign_frame_hits = defaultdict(int)
traffic_sign_tracks = {"_next_id": 1}
count_registry       = defaultdict(list)
recently_lost_tracks = []
frame_idx            = 0
start_time           = time.time()


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

print("[INFO] Starting processing...")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_idx += 1
    raw_frame = frame.copy()

    results = yolo.track(
        source=frame,
        persist=True,
        tracker=TRACKER,
        classes=ROAD_CLASSES,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        device=device,
        verbose=False,
        half=(device == "cuda"),
        imgsz=IMG_SIZE,
    )

    result   = results[0]
    boxes    = result.boxes
    masks    = result.masks
    seen_ids = set()

    if boxes is not None and len(boxes) > 0:
        xyxy_list  = boxes.xyxy.cpu().numpy()
        cls_list   = boxes.cls.int().cpu().tolist()
        conf_list  = boxes.conf.cpu().tolist()
        track_ids  = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(cls_list)

        mask_polygons = None
        if masks is not None and hasattr(masks, "xy") and masks.xy is not None:
            mask_polygons = masks.xy

        for det_idx, (xyxy, cls_id, det_conf, track_id) in enumerate(
            zip(xyxy_list, cls_list, conf_list, track_ids)
        ):
            if track_id < 0:
                continue

            seen_ids.add(track_id)
            polygon_xy = None
            if mask_polygons is not None and det_idx < len(mask_polygons):
                polygon_xy = mask_polygons[det_idx]

            # Create state if new track, or get existing state for track_id
            state = tracks.get(track_id)
            new_track = False
            if state is None:
                state = create_track_state()
                state["track_id"] = track_id
                tracks[track_id] = state
                new_track = True

            # Update state with detection info
            state["last_seen"]      = frame_idx
            state["frames_seen"]   += 1
            state["last_box"]       = [int(v) for v in xyxy]
            state["detector_label"] = str(yolo.names[int(cls_id)])
            state["detector_conf"]  = float(det_conf)
            state["conf_history"].append(float(det_conf))
            state["visual_hold_until"] = frame_idx + VISUAL_PERSIST_FRAMES

            if polygon_xy is not None:
                poly_list = np.asarray(polygon_xy).tolist()
                state["last_mask_xy"]       = poly_list
                state["last_good_mask_xy"]  = poly_list
                state["last_good_mask_frame"] = frame_idx
            else:
                state["last_mask_xy"] = None

            if new_track:
                maybe_link_to_recent_lost(
                    track_id, state, width, height, frame_idx, recently_lost_tracks
                )

            # Crossing
            cx, cy = box_center(xyxy)
            state["center_history"].append((cx, cy))
            update_crossing_state(state, frame.shape)

            # CLIP
            clip_polygon = polygon_xy
            if clip_polygon is None and ALLOW_MASK_REUSE_WHEN_MISSING:
                if (state["last_good_mask_xy"] is not None and
                        frame_idx - state["last_good_mask_frame"] <= VISUAL_PERSIST_FRAMES):
                    clip_polygon = state["last_good_mask_xy"]

            if text_feature_cache and (frame_idx % CLIP_INTERVAL == 0 or state["clip_updates"] == 0):
                crop = extract_semantic_crop(frame, xyxy, polygon_xy=clip_polygon)
                if crop is not None:
                    image_feature = encode_crop(crop, clip_model, clip_preprocess, device)
                    raw_scores = compute_scores(image_feature, text_feature_cache)
                    negative_scores = compute_scores(image_feature, negative_text_feature_cache)
                    update_semantics(
                        state, raw_scores, width, height, frame_idx,
                        count_registry, counters, negative_scores=negative_scores,
                    )

            # Label and color
            if state["matched_queries"]:
                best_query = max(state["matched_queries"],
                                 key=lambda q: state["ema_scores"].get(q, -999))
                best_score = state["ema_scores"][best_query]
                is_human_like = query_type_cache[best_query]["is_human_like"]

                if is_human_like and state["motion_state"] == "crossing":
                    label = f"crossing {best_query} {best_score:.2f}"
                    color = (0, 0, 255)
                else:
                    label = f"{best_query} {best_score:.2f}"
                    color = (0, 220, 0)

            elif DRAW_ALL:
                label = f"{state['detector_label']} {state['detector_conf']:.2f}"
                color = color_from_id(track_id)
            else:
                label = None
                color = None

            if label is not None:
                draw_poly = polygon_xy
                if draw_poly is None and ALLOW_MASK_REUSE_WHEN_MISSING:
                    if (state["last_good_mask_xy"] is not None and
                            frame_idx - state["last_good_mask_frame"] <= VISUAL_PERSIST_FRAMES):
                        draw_poly = state["last_good_mask_xy"]
                frame = draw_visual(frame, xyxy, draw_poly, label, color)

    # Visual persistence for tracks not seen in current frame
    if VISUAL_PERSIST_FRAMES > 0:
        for tid, state in tracks.items():
            if tid in seen_ids or frame_idx > state.get("visual_hold_until", -1):
                continue
            box = state.get("last_box")
            if box is None:
                continue

            if state["matched_queries"]:
                best_query = max(state["matched_queries"],
                                 key=lambda q: state["ema_scores"].get(q, -999))
                best_score = state["ema_scores"].get(best_query, -1.0)
                is_human_like = query_type_cache[best_query]["is_human_like"]

                if is_human_like and state["motion_state"] == "crossing":
                    label = f"crossing {best_query} {best_score:.2f}"
                    color = (0, 0, 180)
                else:
                    label = f"{best_query} {best_score:.2f}"
                    color = (0, 150, 0)
            elif DRAW_ALL:
                label = "hold"
                color = (120, 120, 120)
            else:
                continue

            draw_poly = None
            if ALLOW_MASK_REUSE_WHEN_MISSING:
                if (state["last_good_mask_xy"] is not None and
                        frame_idx - state["last_good_mask_frame"] <= VISUAL_PERSIST_FRAMES):
                    draw_poly = state["last_good_mask_xy"]

            frame = draw_visual(frame, box, draw_poly, label, color)

    # Parallel traffic-sign detector: independent from the main YOLO+CLIP tracking pipeline.
    if use_traffic_sign_detector:
        sign_detections = []
        sign_results = traffic_sign_yolo.predict(
            source=raw_frame,
            classes=traffic_sign_class_ids,
            conf=TRAFFIC_SIGN_CONF,
            iou=TRAFFIC_SIGN_IOU,
            device=device,
            verbose=False,
            half=(device == "cuda"),
            imgsz=TRAFFIC_SIGN_IMG_SIZE,
        )
        sign_result = sign_results[0]
        sign_boxes = sign_result.boxes
        if sign_boxes is not None and len(sign_boxes) > 0:
            sign_xyxy_list = sign_boxes.xyxy.cpu().numpy()
            sign_cls_list = sign_boxes.cls.int().cpu().tolist()
            sign_conf_list = sign_boxes.conf.cpu().tolist()

            for xyxy, cls_id, det_conf in zip(sign_xyxy_list, sign_cls_list, sign_conf_list):
                class_id = int(cls_id)
                raw_label = traffic_sign_label(class_id, traffic_sign_yolo.names)
                matched_queries = traffic_sign_queries_for_class(
                    class_id, traffic_sign_query_routes, QUERIES
                )
                label = matched_queries[0] if matched_queries else raw_label
                traffic_sign_frame_hits[raw_label] += 1
                if TRAFFIC_SIGN_SMOOTHING_ENABLED:
                    sign_detections.append(
                        make_traffic_sign_detection(
                            xyxy, class_id, float(det_conf), label,
                            matched_queries=matched_queries,
                        )
                    )
                elif DRAW_TRAFFIC_SIGNS:
                    draw_traffic_sign_detection(frame, xyxy, label, float(det_conf))

        if TRAFFIC_SIGN_SMOOTHING_ENABLED and DRAW_TRAFFIC_SIGNS:
            visible_signs = update_traffic_sign_tracks(
                sign_detections, traffic_sign_tracks, frame_idx, width, height
            )
            for sign in visible_signs:
                for query in sign.get("matched_queries", []):
                    if query not in sign["counted_queries"]:
                        counters[query] += 1
                        sign["counted_queries"].add(query)
                draw_traffic_sign_detection(
                    frame, sign["box"], sign["label"], float(sign["conf"])
                )

    if DRAW_COUNTERS:
        draw_counters(frame, counters)

    writer.write(frame)
    cleanup_tracks(frame_idx, tracks, recently_lost_tracks)

    if frame_idx % 100 == 0 or frame_idx == total_frames:
        elapsed  = time.time() - start_time
        fps_proc = frame_idx / max(elapsed, 1e-6)
        print(f"[INFO] {frame_idx}/{total_frames} frames | avg FPS {fps_proc:.2f}")


# ── FINALIZATION ────────────────────────────────────────────────────────────

cap.release()
writer.release()

elapsed  = time.time() - start_time
fps_proc = frame_idx / max(elapsed, 1e-6)

summary = build_summary(
    device, query_type_cache, frame_idx, fps_proc, counters, tracks,
    traffic_sign_frame_hits,
    query_routing={
        "mode": TRAFFIC_SIGN_ROUTE_MODE,
        "traffic_sign_detector_enabled": use_traffic_sign_detector,
        "main_yolo_clip_queries": main_clip_queries,
        "traffic_sign_queries": traffic_sign_query_routes,
        "traffic_sign_route_diagnostics": traffic_sign_route_diagnostics,
    },
)
save_summary(summary)

print("[INFO] Done.")
print(f"[INFO] Video  -> {OUTPUT_VIDEO}")
