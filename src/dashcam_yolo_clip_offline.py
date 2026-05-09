#!/usr/bin/env python3
"""
========================================================
Offline Dashcam Semantic Search + Crossing Humans
YOLO + ByteTrack + CLIP
========================================================

Pipeline:
--------------------------------------------------------
1. YOLO detects and tracks road objects
2. CLIP semantically filters tracked crops for arbitrary queries
3. CLIP also checks whether a query is semantically human-like
4. If a matched query is human-like, a temporal crossing detector
   can relabel it as:
      crossing <query>
5. Final statistics are exported to JSON
"""

# ======================================================
# Imports
# ======================================================

import os
import json
import time
from collections import defaultdict, deque

import cv2
import torch
from PIL import Image

import open_clip
from ultralytics import YOLO


# ======================================================
# CONFIG
# ======================================================

INPUT_VIDEO  = "videos/input.mp4"
OUTPUT_VIDEO = "outputs/output_crossing_open_vocab.mp4"
OUTPUT_JSON  = "outputs/output_crossing_open_vocab.json"

QUERIES = [
    "pedestrian",
    "vehicle",
    "traffic light",
    "bus",
    "bicycle",
]

YOLO_MODEL = "yolo26s.pt"
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
TRACKER = "bytetrack.yaml"

# COCO:
# 0 person
# 1 bicycle
# 2 car
# 3 motorcycle
# 4 airplane
# 5 bus
# 6 train
# 7 truck
# 8 boat
# 9 traffic light
# 10 fire hydrant
# 11 stop sign
# 12 parking meter
# 13 bench
# 14 bird
# 15 cat
# 16 dog

#these are the base classes we want to track with yolo, then we will use CLIP to filter them semantically for our queries of interest, allowing for open-vocabulary on those classes (so for example using synonyms)
ROAD_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

YOLO_CONF = 0.25
YOLO_IOU  = 0.45

CLIP_SIM_THRESHOLD = 0.26
CLIP_INTERVAL = 3
EMA_KEEP = 0.80
MIN_HITS = 2
MAX_TRACK_AGE = 90
CROP_PAD = 0.05
MIN_CROP_AREA = 1600

DRAW_ALL = False
DRAW_COUNTERS = True
BOX_THICKNESS = 2

# ----------------------------
# Crossing logic for pedestrians
# ----------------------------

HISTORY_LEN = 30
MIN_TRACK_FRAMES = 5

CENTER_BAND_X = 0.26
ROAD_Y_MIN = 0.22
ROAD_Y_MAX = 0.92

CROSSING_SCORE_ON = 3.0
CROSSING_KEEP_FRAMES = 8

SCORE_IN_ROAD = 1.0
SCORE_LATERAL_MOVE = 0.8
SCORE_TOWARD_CAMERA = 0.6
SCORE_OUTSIDE_ROAD = -0.8

STATE_SWITCH_MIN_FRAMES = 3

MIN_LATERAL_DISP = 0.015
MIN_VERTICAL_PROGRESS = 0.010

HUMAN_QUERY_PROTOTYPES = [
    "person",
    "pedestrian",
    "human",
    "a person walking",
    "a pedestrian in a road scene",
    "a human in a dashcam image",
]

HUMAN_QUERY_THRESHOLD = 0.85


# ======================================================
# Device
# ======================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[INFO] Device: {DEVICE}")


# ======================================================
# HELPERS
# ======================================================

def build_prompt_ensemble(query):
    return [
        f"a photo of a {query}",
        f"a road scene containing a {query}",
        f"a cropped object that is a {query}",
        f"a dashcam image of a {query}",
    ]


def encode_text_list(texts):
    with torch.no_grad():
        tokens = tokenizer(texts).to(DEVICE)
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def encode_text_mean(texts):
    feats = encode_text_list(texts)
    feat = feats.mean(dim=0, keepdim=True)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def build_text_features():
    cache = {}
    with torch.no_grad():
        for query in QUERIES:
            prompts = build_prompt_ensemble(query)
            text_feature = encode_text_mean(prompts)
            cache[query] = text_feature
    return cache


def build_query_type_cache():
    human_proto = encode_text_mean(HUMAN_QUERY_PROTOTYPES)
    query_type_cache = {}

    for query in QUERIES:
        q_feat = encode_text_mean(build_prompt_ensemble(query))
        sim = float((q_feat @ human_proto.T).item())
        query_type_cache[query] = {
            "human_like_score": sim,
            "is_human_like": sim >= HUMAN_QUERY_THRESHOLD,
        }

    return query_type_cache, human_proto


def extract_crop(frame, xyxy):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in xyxy]
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 1 or bh <= 1:
        return None

    px = int(bw * CROP_PAD)
    py = int(bh * CROP_PAD)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    if crop.shape[0] * crop.shape[1] < MIN_CROP_AREA:
        return None

    return crop


def encode_crop(crop):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    image_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_feature = clip_model.encode_image(image_tensor)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

    return image_feature


def compute_scores(image_feature):
    scores = {}
    for query, text_feature in text_feature_cache.items():
        sim = float((image_feature @ text_feature.T).item())
        scores[query] = sim
    return scores


def create_track_state():
    return {
        "last_seen": -1,
        "frames_seen": 0,
        "last_box": None,

        "detector_label": "",
        "detector_conf": 0.0,

        "ema_scores": {},
        "raw_scores": {},

        "positive_hits": defaultdict(int),

        "matched_queries": [],
        "counted_queries": set(),

        "best_query": None,
        "best_score": -1.0,

        "clip_updates": 0,

        # temporal motion
        "center_history": deque(maxlen=HISTORY_LEN),
        "motion_state": "normal",
        "candidate_motion_state": "normal",
        "candidate_motion_count": 0,
        "crossing_score": 0.0,
        "crossing_hold": 0,
    }


def update_semantics(state, raw_scores):
    matched = []

    for query, raw_score in raw_scores.items():
        prev = state["ema_scores"].get(query, None)
        ema = raw_score if prev is None else EMA_KEEP * prev + (1 - EMA_KEEP) * raw_score

        state["ema_scores"][query] = ema
        state["raw_scores"][query] = raw_score

        if ema >= CLIP_SIM_THRESHOLD:
            state["positive_hits"][query] += 1
        else:
            state["positive_hits"][query] = max(0, state["positive_hits"][query] - 1)

        if state["positive_hits"][query] >= MIN_HITS:
            matched.append(query)

            if query not in state["counted_queries"]:
                counters[query] += 1
                state["counted_queries"].add(query)

    # best query must be recomputed every update from current EMA scores
    if len(state["ema_scores"]) > 0:
        best_query = max(state["ema_scores"], key=lambda q: state["ema_scores"][q])
        state["best_query"] = best_query
        state["best_score"] = state["ema_scores"][best_query]

    state["matched_queries"] = matched
    state["clip_updates"] += 1


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def in_road_zone(cx_norm, cy_norm):
    return (
        cy_norm >= ROAD_Y_MIN
        and cy_norm <= ROAD_Y_MAX
        and abs(cx_norm - 0.5) <= CENTER_BAND_X
    )


def update_crossing_state(state, frame_shape):
    if len(state["center_history"]) < MIN_TRACK_FRAMES:
        return

    h, w = frame_shape[:2]
    pts = list(state["center_history"])

    x0, y0 = pts[0]
    x1, y1 = pts[-1]

    dx = (x1 - x0) / max(w, 1)
    dy = (y1 - y0) / max(h, 1)

    cxn = x1 / max(w, 1)
    cyn = y1 / max(h, 1)

    score = 0.0

    if in_road_zone(cxn, cyn):
        score += SCORE_IN_ROAD

        if abs(dx) >= MIN_LATERAL_DISP:
            score += SCORE_LATERAL_MOVE

        if dy >= MIN_VERTICAL_PROGRESS:
            score += SCORE_TOWARD_CAMERA

        if abs(dx) >= MIN_LATERAL_DISP and dy >= 0:
            score += 0.4
    else:
        score += SCORE_OUTSIDE_ROAD

    state["crossing_score"] = 0.85 * state["crossing_score"] + score

    if state["crossing_score"] >= CROSSING_SCORE_ON:
        state["crossing_hold"] = CROSSING_KEEP_FRAMES
    elif state["crossing_hold"] > 0:
        state["crossing_hold"] -= 1

    proposed = "crossing" if (state["crossing_hold"] > 0 or state["crossing_score"] >= CROSSING_SCORE_ON) else "normal"

    if proposed == state["candidate_motion_state"]:
        state["candidate_motion_count"] += 1
    else:
        state["candidate_motion_state"] = proposed
        state["candidate_motion_count"] = 1

    if state["motion_state"] == "normal":
        state["motion_state"] = proposed
    elif state["candidate_motion_count"] >= STATE_SWITCH_MIN_FRAMES:
        state["motion_state"] = proposed


def draw_box(frame, xyxy, label, color):
    x1, y1, x2, y2 = [int(v) for v in xyxy]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        BOX_THICKNESS
    )

    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2
    )

    top = max(0, y1 - th - 10)

    cv2.rectangle(
        frame,
        (x1, top),
        (x1 + tw + 10, y1),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def draw_counters(frame):
    h, w = frame.shape[:2]

    panel_h = 40 + 26 * len(QUERIES)

    x1 = w - 360
    y1 = 10

    cv2.rectangle(
        frame,
        (x1, y1),
        (w - 10, y1 + panel_h),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "Counters",
        (x1 + 10, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    yy = y1 + 55

    for q in QUERIES:
        cv2.putText(
            frame,
            f"{q}: {counters[q]}",
            (x1 + 10, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2
        )
        yy += 26


def cleanup_tracks(frame_idx):
    old_ids = []

    for tid, state in tracks.items():
        if frame_idx - state["last_seen"] > MAX_TRACK_AGE:
            old_ids.append(tid)

    for tid in old_ids:
        del tracks[tid]


# ======================================================
# LOAD MODELS
# ======================================================

print("[INFO] Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("[INFO] Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL,
    pretrained=CLIP_PRETRAINED,
    device=DEVICE,
)

clip_model.eval()
tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

text_feature_cache = build_text_features()
query_type_cache, human_query_prototype = build_query_type_cache()

print("[INFO] Query semantic types:")
for q in QUERIES:
    info = query_type_cache[q]
    print(
        f"  - {q}: human_like={info['is_human_like']} "
        f"(score={info['human_like_score']:.3f})"
    )

print("[INFO] Models loaded")


# ======================================================
# VIDEO SETUP
# ======================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


# ======================================================
# GLOBAL STATE
# ======================================================

tracks   = {}
counters = defaultdict(int)

frame_idx = 0
start_time = time.time()


# ======================================================
# MAIN LOOP
# ======================================================

print("[INFO] Starting processing...")

while True:
    ok, frame = cap.read()

    if not ok:
        break

    frame_idx += 1

    results = yolo.track(
        source=frame,
        persist=True,
        tracker=TRACKER,
        classes=ROAD_CLASSES,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        device=DEVICE,
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        xyxy_list = boxes.xyxy.cpu().numpy()
        cls_list  = boxes.cls.int().cpu().tolist()
        conf_list = boxes.conf.cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [-1] * len(cls_list)
        )

        for xyxy, cls_id, det_conf, track_id in zip(
            xyxy_list,
            cls_list,
            conf_list,
            track_ids
        ):
            if track_id < 0:
                continue

            state = tracks.get(track_id)
            if state is None:
                state = create_track_state()
                tracks[track_id] = state

            state["last_seen"] = frame_idx
            state["frames_seen"] += 1
            state["last_box"] = [int(v) for v in xyxy]
            state["detector_label"] = str(yolo.names[int(cls_id)])
            state["detector_conf"] = float(det_conf)

            # temporal motion history for crossing logic
            cx, cy = box_center(xyxy)
            state["center_history"].append((cx, cy))
            update_crossing_state(state, frame.shape)

            # CLIP semantic filtering
            should_run_clip = (
                frame_idx % CLIP_INTERVAL == 0
                or state["clip_updates"] == 0
            )

            if should_run_clip:
                crop = extract_crop(frame, xyxy)

                if crop is not None:
                    image_feature = encode_crop(crop)
                    raw_scores = compute_scores(image_feature)
                    update_semantics(state, raw_scores)

            # draw results
            if state["matched_queries"]:
                best_query = max(
                    state["matched_queries"],
                    key=lambda q: state["ema_scores"].get(q, -999)
                )

                best_score = state["ema_scores"][best_query]
                is_human_like = query_type_cache[best_query]["is_human_like"]

                if is_human_like and state["motion_state"] == "crossing":
                    label = f"ID {track_id} | crossing {best_query} | {best_score:.2f}"
                    color = (0, 0, 255)
                else:
                    label = f"ID {track_id} | {best_query} | {best_score:.2f}"
                    color = (0, 220, 0)

                draw_box(frame, xyxy, label, color)

            elif DRAW_ALL:
                label = (
                    f"ID {track_id} | "
                    f"{state['detector_label']} | "
                    f"{state['detector_conf']:.2f}"
                )

                draw_box(
                    frame,
                    xyxy,
                    label,
                    (180, 180, 180)
                )

    if DRAW_COUNTERS:
        draw_counters(frame)

    writer.write(frame)
    cleanup_tracks(frame_idx)

    if frame_idx % 100 == 0 or frame_idx == total_frames:
        elapsed = time.time() - start_time
        fps_proc = frame_idx / max(elapsed, 1e-6)

        print(
            f"[INFO] "
            f"{frame_idx}/{total_frames} frames "
            f"| avg FPS {fps_proc:.2f}"
        )

# ======================================================
# FINALIZE
# ======================================================

cap.release()
writer.release()

elapsed = time.time() - start_time
fps_proc = frame_idx / max(elapsed, 1e-6)

summary = {
    "input_video": INPUT_VIDEO,
    "output_video": OUTPUT_VIDEO,
    "device": DEVICE,
    "queries": QUERIES,
    "query_type_cache": {
        q: {
            "human_like_score": float(query_type_cache[q]["human_like_score"]),
            "is_human_like": bool(query_type_cache[q]["is_human_like"]),
        }
        for q in QUERIES
    },
    "tracker": TRACKER,
    "frames_processed": frame_idx,
    "processing_fps": fps_proc,
    "counters": {
        q: int(counters[q])
        for q in QUERIES
    },
    "tracks": {},
}

for track_id, state in tracks.items():
    summary["tracks"][str(track_id)] = {
        "frames_seen": int(state["frames_seen"]),
        "detector_label": state["detector_label"],
        "matched_queries": list(state["matched_queries"]),
        "best_query": state["best_query"],
        "best_score": float(state["best_score"]),
        "motion_state": state["motion_state"],
        "crossing_score": float(state["crossing_score"]),
        "ema_scores": {
            k: float(v)
            for k, v in state["ema_scores"].items()
        },
    }

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("[INFO] Done.")
print(f"[INFO] Video saved to: {OUTPUT_VIDEO}")
print(f"[INFO] JSON saved to: {OUTPUT_JSON}")