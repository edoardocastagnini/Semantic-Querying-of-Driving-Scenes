import os
import json

from config.settings import (
    INPUT_VIDEO, OUTPUT_VIDEO, OUTPUT_JSON, QUERIES,
    VIS_MODE, MASK_ALPHA, USE_MASK_FOR_CLIP_CROP,
    ALLOW_MASK_REUSE_WHEN_MISSING, VISUAL_PERSIST_FRAMES, YOLO_MODEL,
    TRACKER, MAX_TRACK_AGE, KEEP_LOST_TRACKS_FOR,
    REID_MAX_CENTER_DIST, REID_MIN_IOU,
    MIN_FRAMES_TO_COUNT, MIN_MEAN_DET_CONF_FOR_COUNT,
    MIN_LABEL_DOMINANCE_FOR_COUNT, COUNT_COOLDOWN_FRAMES, COUNT_COOLDOWN_DIST,
    MIN_LATERAL_DISP_BBOX, MIN_VERTICAL_PROGRESS_BBOX,
    MIN_ABS_LATERAL_TRAVEL_BBOX, MIN_IN_ROAD_FRAMES,
    CROSSING_SCORE_ON, HISTORY_LEN,
)


def build_summary(device: str, query_type_cache: dict,
                  frame_idx: int, fps_proc: float,
                  counters: dict, tracks: dict) -> dict:
    """Builds a comprehensive summary dictionary of the processing run, including configuration, counters, and track states, ready to be serialized to JSON."""
    summary = {
        "input_video": INPUT_VIDEO,
        "output_video": OUTPUT_VIDEO,
        "device": device,
        "queries": QUERIES,
        "query_type_cache": {
            q: {
                "human_like_score": float(query_type_cache[q]["human_like_score"]),
                "is_human_like": bool(query_type_cache[q]["is_human_like"]),
            }
            for q in QUERIES
        },
        "count_mode": "stable_track_id_only_no_line",
        "crossing_motion_config": {
            "mode": "mean_velocity + road-entry + persistence + bbox_normalized",
            "MIN_LATERAL_DISP_BBOX": MIN_LATERAL_DISP_BBOX,
            "MIN_VERTICAL_PROGRESS_BBOX": MIN_VERTICAL_PROGRESS_BBOX,
            "MIN_ABS_LATERAL_TRAVEL_BBOX": MIN_ABS_LATERAL_TRAVEL_BBOX,
            "MIN_IN_ROAD_FRAMES": MIN_IN_ROAD_FRAMES,
            "CROSSING_SCORE_ON": CROSSING_SCORE_ON,
            "HISTORY_LEN": HISTORY_LEN,
        },
        "segmentation_config": {
            "vis_mode": VIS_MODE,
            "mask_alpha": MASK_ALPHA,
            "use_mask_for_clip_crop": USE_MASK_FOR_CLIP_CROP,
            "allow_mask_reuse_when_missing": ALLOW_MASK_REUSE_WHEN_MISSING,
            "visual_persist_frames": VISUAL_PERSIST_FRAMES,
            "model_path": YOLO_MODEL,
        },
        "tracking_config": {
            "tracker": TRACKER,
            "max_track_age": MAX_TRACK_AGE,
            "keep_lost_tracks_for": KEEP_LOST_TRACKS_FOR,
            "reid_max_center_dist": REID_MAX_CENTER_DIST,
            "reid_min_iou": REID_MIN_IOU,
        },
        "counting_config": {
            "min_frames_to_count": MIN_FRAMES_TO_COUNT,
            "min_mean_det_conf_for_count": MIN_MEAN_DET_CONF_FOR_COUNT,
            "min_label_dominance_for_count": MIN_LABEL_DOMINANCE_FOR_COUNT,
            "count_cooldown_frames": COUNT_COOLDOWN_FRAMES,
            "count_cooldown_dist": COUNT_COOLDOWN_DIST,
        },
        "frames_processed": frame_idx,
        "processing_fps": fps_proc,
        "counters": {q: int(counters[q]) for q in QUERIES},
        "tracks": {},
    }

    for track_id, state in tracks.items():
        summary["tracks"][str(track_id)] = {
            "reid_root_id": state.get("reid_root_id"),
            "frames_seen": int(state["frames_seen"]),
            "detector_label": state["detector_label"],
            "matched_queries": list(state["matched_queries"]),
            "best_query": state["best_query"],
            "best_score": float(state["best_score"]),
            "motion_state": state["motion_state"],
            "crossing_score": float(state["crossing_score"]),
            "counted_once": bool(state["counted_once"]),
            "ema_scores": {k: float(v) for k, v in state["ema_scores"].items()},
            "last_box": state["last_box"],
            "last_mask_xy": state["last_mask_xy"],
        }

    return summary


def save_summary(summary: dict) -> None:
    """Serializes and saves the summary to disk as JSON."""
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON   -> {OUTPUT_JSON}")
