#!/usr/bin/env python3
"""
core/traffic_signs.py
================
For handling traffic sign detection.
"""

from config.settings import (
    TRAFFIC_SIGN_CLASS_NAMES, TRAFFIC_SIGN_CLASSES,
    TRAFFIC_SIGN_BOX_EMA_KEEP, TRAFFIC_SIGN_MIN_HITS_TO_DRAW,
    TRAFFIC_SIGN_VISUAL_PERSIST_FRAMES, TRAFFIC_SIGN_MAX_TRACK_AGE,
    TRAFFIC_SIGN_MATCH_IOU, TRAFFIC_SIGN_MATCH_CENTER_DIST,
    TRAFFIC_SIGN_ROUTE_SIM_THRESHOLD, TRAFFIC_SIGN_ROUTE_MARGIN_THRESHOLD,
    TRAFFIC_SIGN_ROUTE_AMBIGUOUS_MARGIN, TRAFFIC_SIGN_ROUTE_TOP_K,
    TRAFFIC_SIGN_ROUTE_SIGNLIKE_THRESHOLD, TRAFFIC_SIGN_ROUTE_GROUP_THRESHOLD,
    TRAFFIC_SIGN_ROUTE_GROUP_MARGIN, TRAFFIC_SIGN_ROUTE_NEGATIVE_MARGIN,
    TRAFFIC_SIGN_ROUTE_NEGATIVE_QUERIES,
)
from core.clip_utils import build_prompt_ensemble, encode_text_mean
from core.geometry import box_iou, center_distance_norm


SPEED_LIMIT_CLASS_IDS = list(range(2, 13))
IGNORED_TRAFFIC_SIGN_ROUTE_CLASS_IDS = {14}
ALL_TRAFFIC_SIGN_CLASS_IDS = sorted(
    class_id for class_id in TRAFFIC_SIGN_CLASS_NAMES
    if class_id not in IGNORED_TRAFFIC_SIGN_ROUTE_CLASS_IDS
)

TRAFFIC_SIGN_QUERY_ALIASES = {
    "traffic sign": ALL_TRAFFIC_SIGN_CLASS_IDS,
    "traffic signs": ALL_TRAFFIC_SIGN_CLASS_IDS,
    "road sign": ALL_TRAFFIC_SIGN_CLASS_IDS,
    "road signs": ALL_TRAFFIC_SIGN_CLASS_IDS,
    "street sign": ALL_TRAFFIC_SIGN_CLASS_IDS,
    "street signs": ALL_TRAFFIC_SIGN_CLASS_IDS,
    "sign": ALL_TRAFFIC_SIGN_CLASS_IDS,
    "signs": ALL_TRAFFIC_SIGN_CLASS_IDS,
    "speed limit sign": SPEED_LIMIT_CLASS_IDS,
    "speed limit signs": SPEED_LIMIT_CLASS_IDS,
    "speed limit": SPEED_LIMIT_CLASS_IDS,
    "traffic light": [0, 1],
    "traffic lights": [0, 1],
    "green light": [0],
    "green traffic light": [0],
    "red light": [1],
    "red traffic light": [1],
    "stop": [13],
    "stop sign": [13],
    "crosswalk": [15],
    "crosswalk sign": [15],
    "pedestrians crossing": [16],
    "pedestrian crossing": [16],
    "pedestrian crossing sign": [16],
    "one way": [17],
    "one way sign": [17],
    "roundabout": [18],
    "roundabout sign": [18],
    "no entry": [19],
    "no entry sign": [19],
    "no parking": [20],
    "no parking sign": [20],
    "no stopping": [21],
    "no stopping sign": [21],
    "yield": [22],
    "yield sign": [22],
    "priority road": [23],
    "priority road sign": [23],
    "no turn": [24],
    "no turn sign": [24],
    "parking": [25],
    "parking sign": [25],
}

TRAFFIC_SIGN_ROUTE_GROUPS = {
    "traffic sign": {
        "class_ids": ALL_TRAFFIC_SIGN_CLASS_IDS,
        "prompts": [
            "traffic sign",
            "road sign",
            "street sign",
            "roadside traffic sign",
        ],
    },
    "speed limit sign": {
        "class_ids": SPEED_LIMIT_CLASS_IDS,
        "prompts": [
            "speed limit sign",
            "speed limit traffic sign",
            "road speed limit sign",
        ],
    },
    "traffic light": {
        "class_ids": [0, 1],
        "prompts": [
            "traffic light",
            "red or green traffic light",
            "traffic signal light",
        ],
    },
}


def normalize_query(query: str) -> str:
    """
    Normalizes the user query. Replaces symbols ("-", "_", "/") with whitespaces.

    Parameters:
        query (str): not-normalized user query

    Returns:
        str: normalized query
    """

    normalized = str(query).strip().lower()
    for char in "-_/":
        normalized = normalized.replace(char, " ")
    return " ".join(normalized.split())


def traffic_sign_classes_for_query(query: str) -> list[int] | None:
    normalized = normalize_query(query)
    if normalized in TRAFFIC_SIGN_QUERY_ALIASES:
        return list(TRAFFIC_SIGN_QUERY_ALIASES[normalized])

    name_to_id = {
        normalize_query(name): class_id
        for class_id, name in TRAFFIC_SIGN_CLASS_NAMES.items()
    }
    if normalized in name_to_id:
        return [name_to_id[normalized]]

    if normalized.startswith("speed limit "):
        suffix = normalized.replace("speed limit ", "", 1).replace(" sign", "").strip()
        if suffix.isdigit():
            class_name = f"speed limit {suffix}"
            if class_name in name_to_id:
                return [name_to_id[class_name]]

    return None


def build_traffic_sign_query_routes(queries: list[str]) -> dict[str, list[int]]:
    """
    Builds traffic sign query routing.

    Parameters:
        queries (list): list of queries
    """

    routes = {}
    for query in queries:
        class_ids = traffic_sign_classes_for_query(query)
        if class_ids:
            filtered = sorted(
                class_id for class_id in set(class_ids)
                if class_id not in IGNORED_TRAFFIC_SIGN_ROUTE_CLASS_IDS
            )
            if filtered:
                routes[query] = filtered
    return routes


def traffic_sign_text_label(label: str) -> str:
    label = str(label).strip()
    replacements = {
        "Green Light": "green traffic light",
        "Red Light": "red traffic light",
        "Stop": "stop sign",
        "Parking": "parking sign",
        "No parking": "no parking sign",
        "No stopping": "no stopping sign",
        "No turn": "no turn sign",
        "Pedestrians crossing": "pedestrian crossing sign",
        "Yield": "yield sign",
        "One way": "one way sign",
        "Roundabout": "roundabout sign",
        "No entry": "no entry sign",
        "Priority road": "priority road sign",
        "crosswalk": "crosswalk sign",
    }
    if label in replacements:
        return replacements[label]
    if label.startswith("Speed Limit"):
        return f"{label.lower()} traffic sign"
    return label.lower()


def _score_query_against_features(query: str, features: dict, clip_model,
                                  tokenizer, device: str) -> list[tuple[int | str, float]]:
    query_feature = encode_text_mean(build_prompt_ensemble(query), clip_model, tokenizer, device)
    scores = [
        (key, float((query_feature @ feature.T).item()))
        for key, feature in features.items()
    ]
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _build_traffic_sign_class_features(clip_model, tokenizer, device: str) -> dict[int, object]:
    return {
        class_id: encode_text_mean(
            build_prompt_ensemble(traffic_sign_text_label(label)),
            clip_model,
            tokenizer,
            device,
        )
        for class_id, label in TRAFFIC_SIGN_CLASS_NAMES.items()
        if class_id not in IGNORED_TRAFFIC_SIGN_ROUTE_CLASS_IDS
    }


def _build_traffic_sign_group_features(clip_model, tokenizer, device: str) -> dict[str, object]:
    return {
        group_name: encode_text_mean(group["prompts"], clip_model, tokenizer, device)
        for group_name, group in TRAFFIC_SIGN_ROUTE_GROUPS.items()
    }


def _build_route_negative_features(clip_model, tokenizer, device: str) -> dict[str, object]:
    return {
        query: encode_text_mean(build_prompt_ensemble(query), clip_model, tokenizer, device)
        for query in TRAFFIC_SIGN_ROUTE_NEGATIVE_QUERIES
    }


def build_clip_traffic_sign_query_routes(queries: list[str], clip_model,
                                         tokenizer, device: str) -> tuple[dict[str, list[int]], dict]:
    """
    Uses CLIP text embeddings to decide which user queries should be handled by
    the secondary traffic-sign detector and which detector classes they need.
    """
    class_features = _build_traffic_sign_class_features(clip_model, tokenizer, device)
    group_features = _build_traffic_sign_group_features(clip_model, tokenizer, device)
    negative_features = _build_route_negative_features(clip_model, tokenizer, device)
    routes = {}
    diagnostics = {}

    for query in queries:
        class_scores = _score_query_against_features(
            query, class_features, clip_model, tokenizer, device
        )
        group_scores = _score_query_against_features(
            query, group_features, clip_model, tokenizer, device
        )
        negative_scores = _score_query_against_features(
            query, negative_features, clip_model, tokenizer, device
        )

        best_class_id, best_class_score = class_scores[0]
        second_class_score = class_scores[1][1] if len(class_scores) > 1 else -1.0
        class_margin = best_class_score - second_class_score
        best_group_name, best_group_score = group_scores[0]
        best_negative_name, best_negative_score = negative_scores[0]
        class_vs_negative_margin = best_class_score - best_negative_score
        group_vs_negative_margin = best_group_score - best_negative_score

        route_class_ids = None
        route_reason = "main_yolo_clip"

        group_is_confident = (
            best_group_score >= TRAFFIC_SIGN_ROUTE_GROUP_THRESHOLD and
            best_group_score >= best_class_score - TRAFFIC_SIGN_ROUTE_GROUP_MARGIN and
            group_vs_negative_margin >= TRAFFIC_SIGN_ROUTE_NEGATIVE_MARGIN
        )
        if group_is_confident:
            route_class_ids = list(TRAFFIC_SIGN_ROUTE_GROUPS[best_group_name]["class_ids"])
            route_reason = f"clip_group:{best_group_name}"
        elif (best_class_score >= TRAFFIC_SIGN_ROUTE_SIM_THRESHOLD and
              best_group_score >= TRAFFIC_SIGN_ROUTE_SIGNLIKE_THRESHOLD and
              class_vs_negative_margin >= TRAFFIC_SIGN_ROUTE_NEGATIVE_MARGIN):
            if class_margin >= TRAFFIC_SIGN_ROUTE_MARGIN_THRESHOLD:
                route_class_ids = [int(best_class_id)]
                route_reason = "clip_specific"
            else:
                max_drop = TRAFFIC_SIGN_ROUTE_AMBIGUOUS_MARGIN
                route_class_ids = [
                    int(class_id)
                    for class_id, score in class_scores[:TRAFFIC_SIGN_ROUTE_TOP_K]
                    if best_class_score - score <= max_drop
                ]
                route_reason = "clip_ambiguous_topk"

        if route_class_ids:
            routes[query] = sorted(set(route_class_ids))

        diagnostics[query] = {
            "route": route_reason,
            "class_ids": sorted(set(route_class_ids or [])),
            "best_class_id": int(best_class_id),
            "best_class_label": TRAFFIC_SIGN_CLASS_NAMES[int(best_class_id)],
            "best_class_score": best_class_score,
            "second_class_score": second_class_score,
            "class_margin": class_margin,
            "best_group": best_group_name,
            "best_group_score": best_group_score,
            "signlike_threshold": TRAFFIC_SIGN_ROUTE_SIGNLIKE_THRESHOLD,
            "best_negative": best_negative_name,
            "best_negative_score": best_negative_score,
            "class_vs_negative_margin": class_vs_negative_margin,
            "group_vs_negative_margin": group_vs_negative_margin,
            "negative_margin_threshold": TRAFFIC_SIGN_ROUTE_NEGATIVE_MARGIN,
            "top_classes": [
                {
                    "class_id": int(class_id),
                    "label": TRAFFIC_SIGN_CLASS_NAMES[int(class_id)],
                    "score": score,
                }
                for class_id, score in class_scores[:TRAFFIC_SIGN_ROUTE_TOP_K]
            ],
        }

    return routes, diagnostics


def traffic_sign_queries_for_class(class_id: int,
                                   query_routes: dict[str, list[int]],
                                   query_order: list[str]) -> list[str]:
    return [
        query for query in query_order
        if query in query_routes and class_id in query_routes[query]
    ]


def selected_traffic_sign_class_ids(query_routes: dict[str, list[int]] | None = None) -> list[int] | None:
    """
    Converts TRAFFIC_SIGN_CLASSES into YOLO class ids.
    Returns None when the list is empty, meaning all traffic-sign classes.
    """
    if query_routes:
        selected = sorted({class_id for ids in query_routes.values() for class_id in ids})
        return selected or None

    if not TRAFFIC_SIGN_CLASSES:
        return None

    name_to_id = {name.lower(): class_id for class_id, name in TRAFFIC_SIGN_CLASS_NAMES.items()}
    selected = []
    for item in TRAFFIC_SIGN_CLASSES:
        if isinstance(item, int):
            if item not in TRAFFIC_SIGN_CLASS_NAMES:
                raise ValueError(f"Unknown traffic sign class id: {item}")
            selected.append(item)
            continue

        key = str(item).strip().lower()
        if key not in name_to_id:
            valid = ", ".join(TRAFFIC_SIGN_CLASS_NAMES.values())
            raise ValueError(f"Unknown traffic sign class name: {item}. Valid names: {valid}")
        selected.append(name_to_id[key])

    return sorted(set(selected))


def traffic_sign_label(class_id: int, model_names=None) -> str:
    if class_id in TRAFFIC_SIGN_CLASS_NAMES:
        return TRAFFIC_SIGN_CLASS_NAMES[class_id]
    if model_names is not None and class_id in model_names:
        return str(model_names[class_id])
    return f"class {class_id}"


def make_traffic_sign_detection(xyxy, class_id: int, conf: float, label: str,
                                matched_queries: list[str] | None = None) -> dict:
    return {
        "box": [float(v) for v in xyxy],
        "class_id": int(class_id),
        "conf": float(conf),
        "label": label,
        "matched_queries": list(matched_queries or []),
    }


def update_traffic_sign_tracks(detections: list[dict], tracks: dict,
                               frame_idx: int, frame_w: int, frame_h: int) -> list[dict]:
    """
    Associates traffic-sign detections across frames and returns stable visual tracks.
    Matching is class-aware and uses IoU plus normalized center distance so small signs
    can still be linked when the box jitters by a few pixels.
    """
    matched_track_ids = set()

    for det in detections:
        best_tid = None
        best_score = -1e9

        for tid, track in tracks.items():
            if tid == "_next_id":
                continue
            if tid in matched_track_ids:
                continue
            if track["class_id"] != det["class_id"]:
                continue

            iou = box_iou(track["box"], det["box"])
            dist = center_distance_norm(track["box"], det["box"], frame_w, frame_h)
            if iou < TRAFFIC_SIGN_MATCH_IOU and dist > TRAFFIC_SIGN_MATCH_CENTER_DIST:
                continue

            score = (2.0 * iou) - dist
            if score > best_score:
                best_tid = tid
                best_score = score

        if best_tid is None:
            next_id = tracks.get("_next_id", 1)
            best_tid = next_id
            tracks["_next_id"] = next_id + 1
            tracks[best_tid] = {
                "id": best_tid,
                "box": det["box"],
                "class_id": det["class_id"],
                "label": det["label"],
                "matched_queries": det.get("matched_queries", []),
                "counted_queries": set(),
                "conf": det["conf"],
                "hits": 0,
                "last_seen": frame_idx,
            }

        track = tracks[best_tid]
        keep = TRAFFIC_SIGN_BOX_EMA_KEEP if track["hits"] > 0 else 0.0
        track["box"] = [
            keep * old + (1.0 - keep) * new
            for old, new in zip(track["box"], det["box"])
        ]
        track["conf"] = keep * track["conf"] + (1.0 - keep) * det["conf"]
        track["label"] = det["label"]
        track["class_id"] = det["class_id"]
        track["matched_queries"] = det.get("matched_queries", [])
        track["hits"] += 1
        track["last_seen"] = frame_idx
        matched_track_ids.add(best_tid)

    stale_ids = [
        tid for tid, track in tracks.items()
        if tid != "_next_id" and frame_idx - track["last_seen"] > TRAFFIC_SIGN_MAX_TRACK_AGE
    ]
    for tid in stale_ids:
        del tracks[tid]

    visible = []
    for tid, track in tracks.items():
        if tid == "_next_id":
            continue
        if track["hits"] < TRAFFIC_SIGN_MIN_HITS_TO_DRAW:
            continue
        if frame_idx - track["last_seen"] > TRAFFIC_SIGN_VISUAL_PERSIST_FRAMES:
            continue
        visible.append(track)

    return visible
