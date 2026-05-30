#!/usr/bin/env python3
"""
core/tracking.py
================
Handles object tracking.
"""

from collections import defaultdict, deque

from config.settings import (
    MAX_TRACK_AGE, KEEP_LOST_TRACKS_FOR,
    REID_MAX_CENTER_DIST, REID_MIN_IOU, REID_REQUIRE_SAME_QUERY,
    HISTORY_LEN, CONF_HISTORY_LEN, QUERY_HISTORY_LEN, CLIP_TEMPORAL_WINDOW,
)
from core.geometry import center_distance_norm, box_iou


def create_track_state() -> dict:
    """
    Initializes a new track state dictionary with default values.
    
    Returns: 
        dict: new track state
    """
    
    return {
        "track_id": None,
        "reid_root_id": None,
        "last_seen": -1,
        "frames_seen": 0,
        "last_box": None,
        "last_mask_xy": None,
        "last_good_mask_xy": None,
        "last_good_mask_frame": -1,
        "detector_label": "",
        "detector_conf": 0.0,
        "ema_scores": {},
        "raw_scores": {},
        "last_view_scores": {},
        "last_negative_view_scores": {},
        "last_combined_scores": {},
        "last_clip_views_used": [],
        "pending_clip_scores": deque(maxlen=CLIP_TEMPORAL_WINDOW),
        "pending_negative_clip_scores": deque(maxlen=CLIP_TEMPORAL_WINDOW),
        "last_temporal_scores": {},
        "last_temporal_negative_scores": {},
        "last_temporal_window_size": 0,
        "positive_hits": defaultdict(int),
        "matched_queries": [],
        "counted_queries": set(),
        "counted_once": False,
        "best_query": None,
        "best_score": -1.0,
        "best_query_history": deque(maxlen=QUERY_HISTORY_LEN),
        "clip_updates": 0,
        "center_history": deque(maxlen=HISTORY_LEN),
        "road_history": deque(maxlen=HISTORY_LEN),
        "in_road_frames": 0,
        "entered_road": False,
        "motion_state": "normal",
        "candidate_motion_state": "normal",
        "candidate_motion_count": 0,
        "crossing_score": 0.0,
        "crossing_hold": 0,
        "conf_history": deque(maxlen=CONF_HISTORY_LEN),
        "visual_hold_until": -1,
    }


def maybe_link_to_recent_lost(track_id: int, state: dict,
                               frame_w: int, frame_h: int,
                               current_frame_idx: int,
                               recently_lost_tracks: list) -> None:
    """
    Tries to link a newly created track to a recently lost track for re-identification. Compares the new track's last box
    with recently lost tracks using a combination of normalized center distance and IoU, and optionally requiring the same best query. 
    If a good match is found, it transfers the reid_root_id and counting history to the new track state. Modifies `state` in-place.
    
    Parameters:
        track_id (int): id of the track
        state (dict): current state
        frame_w (int): frame width
        frame_h (int): frame height
        current_frame_idx (int): current frame index
        recently_lost_tracks (list): list of recently lost tracks
    """

    best_idx = None
    best_score = -999.0

    for idx, lost in enumerate(recently_lost_tracks):
        gap = current_frame_idx - lost["last_seen"]
        if gap < 0 or gap > KEEP_LOST_TRACKS_FOR:
            continue
        dist = center_distance_norm(state["last_box"], lost["last_box"], frame_w, frame_h)
        iou = box_iou(state["last_box"], lost["last_box"])
        if dist > REID_MAX_CENTER_DIST and iou < REID_MIN_IOU:
            continue
        if REID_REQUIRE_SAME_QUERY:
            if state.get("best_query") is not None and lost.get("best_query") is not None:
                if state["best_query"] != lost["best_query"]:
                    continue
        score = 1.5 * iou - 1.0 * dist - 0.002 * gap
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is not None:
        lost = recently_lost_tracks.pop(best_idx)
        state["reid_root_id"] = lost.get("reid_root_id", lost.get("track_id", track_id))
        state["counted_once"] = state["counted_once"] or lost.get("counted_once", False)
        state["counted_queries"] = set(state["counted_queries"]) | set(lost.get("counted_queries", []))
    else:
        state["reid_root_id"] = track_id


def cleanup_tracks(current_frame_idx: int, tracks: dict,
                   recently_lost_tracks: list) -> None:
    """
    Removes stale tracks that haven't been seen for more than MAX_TRACK_AGE frames. 
    Before removing, it saves relevant info in recently_lost_tracks for potential re-identification. 
    Also cleans up recently_lost_tracks to only keep entries within the KEEP_LOST_TRACKS_FOR window. 
    Modifies `tracks` and `recently_lost_tracks` in-place.

    Parameters:
        current_frame_idx (int): current frame index
        tracks (dict):
        recently_lost_tracks (list):
    """

    stale = [tid for tid, s in tracks.items()
             if current_frame_idx - s["last_seen"] > MAX_TRACK_AGE]

    for tid in stale:
        s = tracks[tid]
        recently_lost_tracks.append({
            "track_id": tid,
            "reid_root_id": s.get("reid_root_id", tid),
            "last_seen": s["last_seen"],
            "last_box": s["last_box"],
            "best_query": s["best_query"],
            "counted_once": s["counted_once"],
            "counted_queries": list(s["counted_queries"]),
        })
        del tracks[tid]

    recently_lost_tracks[:] = [
        x for x in recently_lost_tracks
        if current_frame_idx - x["last_seen"] <= KEEP_LOST_TRACKS_FOR
    ]
