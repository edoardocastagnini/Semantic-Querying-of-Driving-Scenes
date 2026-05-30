#!/usr/bin/env python3
"""
core/semantics.py
================
Handles object detection and matching.
"""

import numpy as np
from collections import defaultdict

from config.settings import (
    CLIP_SIM_THRESHOLD, CLIP_MARGIN_THRESHOLD, CLIP_NEGATIVE_MARGIN_THRESHOLD,
    EMA_KEEP, MIN_HITS,
    MIN_FRAMES_TO_COUNT, MIN_MEAN_DET_CONF_FOR_COUNT,
    MIN_LABEL_DOMINANCE_FOR_COUNT, CONF_HISTORY_LEN,
    COUNT_COOLDOWN_FRAMES, COUNT_COOLDOWN_DIST,
)
from core.geometry import normalized_center


def is_valid_match(scores: dict, negative_scores: dict | None = None) -> tuple[bool, str | None, float]:
    """
    Checks if the best score is above CLIP_SIM_THRESHOLD and sufficiently better than the second best.
    Otherwise returns a tuple (False, None, -1.0)
    
    Parameters:
        scores (dict): CLIP output scores
        negative_scores (dict | None): CLIP negative scores

    Returns:
        boolean, str, float : (is_valid, best_query, best_score)
    """

    if not scores:
        return False, None, -1.0
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_query, best_score = sorted_scores[0]
    if best_score < CLIP_SIM_THRESHOLD:
        return False, best_query, best_score
    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
        if best_score - second_score < CLIP_MARGIN_THRESHOLD:
            return False, best_query, best_score
    if negative_scores:
        best_negative = max(negative_scores.values())
        if best_score - best_negative < CLIP_NEGATIVE_MARGIN_THRESHOLD:
            return False, best_query, best_score
    return True, best_query, best_score


def dominant_query_fraction(hist: list) -> tuple[str | None, float]:
    """
    Returns the most frequent query in the history and its fraction.
    
    Parameters:
        hist (list): score history
    
    Returns:
        (str, float): key of the most frequent query and its score
    """

    valid = [q for q in hist if q is not None]
    if not valid:
        return None, 0.0
    best = max(set(valid), key=valid.count)
    return best, valid.count(best) / len(valid)


def is_mature_for_count(state: dict) -> bool:
    """
    Determines whether a track has enough values for processing.

    Parameters:
        state (dict): track

    Returns:
        bool: True if the track has enough history, confidence and label consistency to be counted
    """

    if state["frames_seen"] < MIN_FRAMES_TO_COUNT:
        return False
    if len(state["conf_history"]) < min(10, CONF_HISTORY_LEN):
        return False
    if float(np.mean(state["conf_history"])) < MIN_MEAN_DET_CONF_FOR_COUNT:
        return False
    best, frac = dominant_query_fraction(state["best_query_history"])
    if best is None or frac < MIN_LABEL_DOMINANCE_FOR_COUNT:
        return False
    return True


def try_count(query: str, cx_norm: float, cy_norm: float,
              current_frame: int, count_registry: dict,
              counters: dict, anti_double_key: int = None) -> bool:
    """
    Tries to count a track for the given query. Checks the cooldown registry to prevent double counting.

    Parameters:
        query (str): input query
        cx_norm (float): normalized center x coordinate
        cy_norm (float): normalized center y coordinate
        current_frame (int): frame index
        count_registry (dict): cooldown registry for double counting prevention
        counters (dict): dict with counters of detected objects
        anti_double_key (int): anti double key for double counting prevention 

    Returns:
        True if the count was registered, False if otherwise.
    """

    count_registry[query] = [
        (f, x, y, k) for f, x, y, k in count_registry[query]
        if current_frame - f <= COUNT_COOLDOWN_FRAMES
    ]
    for _, x, y, k in count_registry[query]:
        dist = ((cx_norm - x) ** 2 + (cy_norm - y) ** 2) ** 0.5
        if dist < COUNT_COOLDOWN_DIST:
            return False
        if anti_double_key is not None and k is not None and anti_double_key == k:
            return False
    counters[query] += 1
    count_registry[query].append((current_frame, cx_norm, cy_norm, anti_double_key))
    return True


def maybe_count_track(state: dict, frame_w: int, frame_h: int,
                      current_frame_idx: int, count_registry: dict,
                      counters: dict) -> None:
    """
    Counts the track if it has a valid matched query and is mature enough, using try_count to check cooldowns.
    
    Parameters:
        state (dict):
        frame_w (int): frame width
        frame_h (int): frame height
        current_frame_idx (int): index of current frame
        count_registry (dict): cooldown registry for double counting prevention
        counters (dict): dict with counters of detected objects

    """

    if state["counted_once"] or not state["matched_queries"]:
        return
    if not is_mature_for_count(state):
        return
    query = max(state["matched_queries"], key=lambda q: state["ema_scores"].get(q, -999))
    box = state["last_box"]
    if box is None:
        return
    cx_norm, cy_norm = normalized_center(box, frame_w, frame_h)
    anti_double_key = state.get("reid_root_id", state.get("track_id"))
    if query not in state["counted_queries"]:
        if try_count(query, cx_norm, cy_norm, current_frame_idx,
                     count_registry, counters, anti_double_key=anti_double_key):
            state["counted_queries"].add(query)
            state["counted_once"] = True


def update_semantics(state: dict, raw_scores: dict, frame_w: int, frame_h: int,
                     current_frame_idx: int, count_registry: dict,
                     counters: dict, negative_scores: dict | None = None) -> None:
    """
    Update the semantic state of a track based on new CLIP raw scores. Applies exponential moving average to smooth scores,
    updates positive hit counts, and determines matched queries. Also tries to count the track if it becomes mature enough.
    
    Parameters:
        state (dict): sematic state
        raw_scores (dict): CLIP scores
        frame_w (int): frame width 
        frame_h (int): frame height
        current_frame_idx (int): index of current frame 
        count_registry (dict): for double count prevention
        counters (dict): positive hit count
        negative_scores (dict | None): 
    """

    matched = []
    is_valid, _, _ = is_valid_match(raw_scores, negative_scores=negative_scores)

    for query, raw_score in raw_scores.items():
        prev = state["ema_scores"].get(query, None)
        ema = raw_score if prev is None else EMA_KEEP * prev + (1 - EMA_KEEP) * raw_score
        state["ema_scores"][query] = ema
        state["raw_scores"][query] = raw_score

        if is_valid and ema >= CLIP_SIM_THRESHOLD:
            state["positive_hits"][query] += 1
        else:
            state["positive_hits"][query] = max(0, state["positive_hits"][query] - 1)

        if state["positive_hits"][query] >= MIN_HITS:
            matched.append(query)

    if state["ema_scores"]:
        best_query = max(state["ema_scores"], key=lambda q: state["ema_scores"][q])
        state["best_query"] = best_query
        state["best_score"] = state["ema_scores"][best_query]
        state["best_query_history"].append(best_query)

    state["matched_queries"] = matched
    state["clip_updates"] += 1

    maybe_count_track(state, frame_w, frame_h, current_frame_idx,
                      count_registry, counters)
