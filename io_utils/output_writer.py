#!/usr/bin/env python3
"""
io_utils/output_writer.py
================
Handles writing to output.
"""

import os
import json

from config.settings import (
    INPUT_VIDEO, OUTPUT_VIDEO, OUTPUT_JSON, QUERIES,
    CLIP_SIM_THRESHOLD, CLIP_MARGIN_THRESHOLD, CLIP_NEGATIVE_MARGIN_THRESHOLD,
    CLIP_NEGATIVE_QUERIES, CLIP_MULTI_VIEW_ENABLED, CLIP_VIEW_WEIGHTS,
    MASK_CROP_PAD, CONTEXT_CROP_PAD,
    CLIP_CONTEXT_BLUR_KERNEL, CLIP_CONTEXT_NEUTRAL_COLOR,
    CLIP_CONTEXT_OCCLUDER_MODE, CLIP_TEMPORAL_AGGREGATION_ENABLED,
    CLIP_TEMPORAL_WINDOW, CLIP_TEMPORAL_AGGREGATION,
    VIS_MODE, MASK_ALPHA, USE_MASK_FOR_CLIP_CROP,
    ALLOW_MASK_REUSE_WHEN_MISSING, VISUAL_PERSIST_FRAMES, YOLO_MODEL,
    TRAFFIC_SIGN_MODEL, TRAFFIC_SIGN_CLASSES, TRAFFIC_SIGN_CONF,
    TRAFFIC_SIGN_IOU, TRAFFIC_SIGN_IMG_SIZE, DRAW_TRAFFIC_SIGNS,
    TRAFFIC_SIGN_SMOOTHING_ENABLED, TRAFFIC_SIGN_BOX_EMA_KEEP,
    TRAFFIC_SIGN_MIN_HITS_TO_DRAW, TRAFFIC_SIGN_VISUAL_PERSIST_FRAMES,
    TRAFFIC_SIGN_MAX_TRACK_AGE, TRAFFIC_SIGN_MATCH_IOU,
    TRAFFIC_SIGN_MATCH_CENTER_DIST, TRAFFIC_SIGN_ROUTE_MODE,
    TRAFFIC_SIGN_ROUTE_SIM_THRESHOLD, TRAFFIC_SIGN_ROUTE_MARGIN_THRESHOLD,
    TRAFFIC_SIGN_ROUTE_AMBIGUOUS_MARGIN, TRAFFIC_SIGN_ROUTE_TOP_K,
    TRAFFIC_SIGN_ROUTE_SIGNLIKE_THRESHOLD, TRAFFIC_SIGN_ROUTE_GROUP_THRESHOLD,
    TRAFFIC_SIGN_ROUTE_GROUP_MARGIN, TRAFFIC_SIGN_ROUTE_NEGATIVE_MARGIN,
    TRAFFIC_SIGN_ROUTE_NEGATIVE_QUERIES,
    TRACKER, MAX_TRACK_AGE, KEEP_LOST_TRACKS_FOR,
    REID_MAX_CENTER_DIST, REID_MIN_IOU,
    MIN_FRAMES_TO_COUNT, MIN_MEAN_DET_CONF_FOR_COUNT,
    MIN_LABEL_DOMINANCE_FOR_COUNT, COUNT_COOLDOWN_FRAMES, COUNT_COOLDOWN_DIST,
    MIN_LATERAL_DISP_BBOX, MIN_VERTICAL_PROGRESS_BBOX,
    MIN_ABS_LATERAL_TRAVEL_BBOX, MIN_IN_ROAD_FRAMES,
    CROSSING_SCORE_ON, HISTORY_LEN,
)


def _float_nested_scores(scores: dict) -> dict:
    return {
        str(k): (
            _float_nested_scores(v) if isinstance(v, dict) else float(v)
        )
        for k, v in scores.items()
    }


def build_summary(device: str, query_type_cache: dict,
                  frame_idx: int, fps_proc: float,
                  counters: dict, tracks: dict,
                  traffic_sign_frame_hits: dict | None = None,
                  query_routing: dict | None = None) -> dict:
    """
    Builds a comprehensive summary dictionary of the processing run, including configuration, counters, and track states, ready to be serialized to JSON.
    
    Parameters:
        device (str): selected device ("cuda" | "mps" | "cpu")
        query_type_cache (dict): query cache
        frame_idx (int): frame index
        fps_proc (float): frames per second metric
        counters (dict): counters of detected objects
        tracks (dict): dictionary with trackss
        traffic_sign_frame_hits (dict): traffic sign frame hit counts
        query_routing (dict): query routings

    Returns:
        dict: summary of all detections and system configuration
    """
    
    summary = {
        "input_video": INPUT_VIDEO,
        "output_video": OUTPUT_VIDEO,
        "device": device,
        "queries": QUERIES,
        "query_routing": query_routing or {},
        "query_type_cache": {
            q: {
                "human_like_score": float(query_type_cache[q]["human_like_score"]),
                "is_human_like": bool(query_type_cache[q]["is_human_like"]),
            }
            for q in QUERIES
        },
        "clip_filter_config": {
            "sim_threshold": CLIP_SIM_THRESHOLD,
            "positive_margin_threshold": CLIP_MARGIN_THRESHOLD,
            "negative_margin_threshold": CLIP_NEGATIVE_MARGIN_THRESHOLD,
            "negative_queries": CLIP_NEGATIVE_QUERIES,
            "multi_view_enabled": CLIP_MULTI_VIEW_ENABLED,
            "view_weights": CLIP_VIEW_WEIGHTS,
            "mask_crop_pad": MASK_CROP_PAD,
            "context_crop_pad": CONTEXT_CROP_PAD,
            "context_occluder_mode": CLIP_CONTEXT_OCCLUDER_MODE,
            "context_blur_kernel": CLIP_CONTEXT_BLUR_KERNEL,
            "context_neutral_color": CLIP_CONTEXT_NEUTRAL_COLOR,
            "temporal_aggregation_enabled": CLIP_TEMPORAL_AGGREGATION_ENABLED,
            "temporal_window": CLIP_TEMPORAL_WINDOW,
            "temporal_aggregation": CLIP_TEMPORAL_AGGREGATION,
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
        "traffic_sign_detection_config": {
            "model_path": TRAFFIC_SIGN_MODEL,
            "selected_classes": TRAFFIC_SIGN_CLASSES,
            "conf": TRAFFIC_SIGN_CONF,
            "iou": TRAFFIC_SIGN_IOU,
            "img_size": TRAFFIC_SIGN_IMG_SIZE,
            "draw": DRAW_TRAFFIC_SIGNS,
            "smoothing_enabled": TRAFFIC_SIGN_SMOOTHING_ENABLED,
            "box_ema_keep": TRAFFIC_SIGN_BOX_EMA_KEEP,
            "min_hits_to_draw": TRAFFIC_SIGN_MIN_HITS_TO_DRAW,
            "visual_persist_frames": TRAFFIC_SIGN_VISUAL_PERSIST_FRAMES,
            "max_track_age": TRAFFIC_SIGN_MAX_TRACK_AGE,
            "match_iou": TRAFFIC_SIGN_MATCH_IOU,
            "match_center_dist": TRAFFIC_SIGN_MATCH_CENTER_DIST,
            "route_mode": TRAFFIC_SIGN_ROUTE_MODE,
            "route_sim_threshold": TRAFFIC_SIGN_ROUTE_SIM_THRESHOLD,
            "route_margin_threshold": TRAFFIC_SIGN_ROUTE_MARGIN_THRESHOLD,
            "route_ambiguous_margin": TRAFFIC_SIGN_ROUTE_AMBIGUOUS_MARGIN,
            "route_top_k": TRAFFIC_SIGN_ROUTE_TOP_K,
            "route_signlike_threshold": TRAFFIC_SIGN_ROUTE_SIGNLIKE_THRESHOLD,
            "route_group_threshold": TRAFFIC_SIGN_ROUTE_GROUP_THRESHOLD,
            "route_group_margin": TRAFFIC_SIGN_ROUTE_GROUP_MARGIN,
            "route_negative_margin": TRAFFIC_SIGN_ROUTE_NEGATIVE_MARGIN,
            "route_negative_queries": TRAFFIC_SIGN_ROUTE_NEGATIVE_QUERIES,
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
        "traffic_sign_frame_hits": {
            str(k): int(v) for k, v in (traffic_sign_frame_hits or {}).items()
        },
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
            "last_clip_views_used": list(state.get("last_clip_views_used", [])),
            "last_view_scores": _float_nested_scores(state.get("last_view_scores", {})),
            "last_combined_scores": {
                k: float(v) for k, v in state.get("last_combined_scores", {}).items()
            },
            "last_temporal_scores": {
                k: float(v) for k, v in state.get("last_temporal_scores", {}).items()
            },
            "last_temporal_window_size": int(state.get("last_temporal_window_size", 0)),
            "last_box": state["last_box"],
            "last_mask_xy": state["last_mask_xy"],
        }

    return summary


def save_summary(summary: dict) -> None:
    """
    Serializes and saves the summary to disk as JSON.
    
    Parameters:
        summary (dict): logs to be saved
    """
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON   -> {OUTPUT_JSON}")
