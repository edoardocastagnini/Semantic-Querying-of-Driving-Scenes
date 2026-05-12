import numpy as np

from config.settings import (
    MIN_TRACK_FRAMES, CENTER_BAND_X, ROAD_Y_MIN, ROAD_Y_MAX,
    CROSSING_SCORE_ON, CROSSING_KEEP_FRAMES,
    SCORE_IN_ROAD, SCORE_LATERAL_MOVE, SCORE_TOWARD_CAMERA, SCORE_OUTSIDE_ROAD,
    STATE_SWITCH_MIN_FRAMES,
    MIN_LATERAL_DISP_BBOX, MIN_VERTICAL_PROGRESS_BBOX,
    MIN_ABS_LATERAL_TRAVEL_BBOX, MIN_IN_ROAD_FRAMES,
)


def in_road_zone(cx_norm: float, cy_norm: float) -> bool:
    """True if the normalized center is in the central zone of the road."""
    return ROAD_Y_MIN <= cy_norm <= ROAD_Y_MAX and abs(cx_norm - 0.5) <= CENTER_BAND_X


def compute_mean_velocity(pts: list) -> tuple[float, float]:
    """Calculate the mean velocity (vx, vy) from the position history."""
    if len(pts) < 2:
        return 0.0, 0.0
    vx_list = [pts[i][0] - pts[i - 1][0] for i in range(1, len(pts))]
    vy_list = [pts[i][1] - pts[i - 1][1] for i in range(1, len(pts))]
    return float(np.mean(vx_list)), float(np.mean(vy_list))


def update_crossing_state(state: dict, frame_shape: tuple) -> None:
    """
    Update the crossing state of a track based on its movement and position history.
    """
    if len(state["center_history"]) < MIN_TRACK_FRAMES:
        return

    h, w = frame_shape[:2]
    pts = list(state["center_history"])
    mean_vx, mean_vy = compute_mean_velocity(pts)

    x_last, y_last = pts[-1]
    cxn = x_last / max(w, 1)
    cyn = y_last / max(h, 1)

    inside = in_road_zone(cxn, cyn)
    state["road_history"].append(1 if inside else 0)
    state["in_road_frames"] = sum(state["road_history"])

    if len(state["road_history"]) >= 2 and state["road_history"][-2] == 0 and state["road_history"][-1] == 1:
        state["entered_road"] = True

    box = state["last_box"]
    if box is None:
        return

    box_w = max(box[2] - box[0], 1)
    box_h = max(box[3] - box[1], 1)
    dx_bbox = mean_vx / box_w
    dy_bbox = mean_vy / box_h

    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    total_dx_bbox = abs(x1 - x0) / box_w
    total_dy_bbox = (y1 - y0) / box_h

    score = 0.0
    if inside:
        score += SCORE_IN_ROAD
        if state["in_road_frames"] >= MIN_IN_ROAD_FRAMES:
            score += 0.6
        if abs(dx_bbox) >= MIN_LATERAL_DISP_BBOX:
            score += SCORE_LATERAL_MOVE
        if total_dx_bbox >= MIN_ABS_LATERAL_TRAVEL_BBOX:
            score += 0.8
        if dy_bbox >= MIN_VERTICAL_PROGRESS_BBOX:
            score += SCORE_TOWARD_CAMERA
        if state.get("entered_road", False):
            score += 0.5
        if abs(dx_bbox) >= MIN_LATERAL_DISP_BBOX and total_dy_bbox >= 0:
            score += 0.3
    else:
        score += SCORE_OUTSIDE_ROAD

    state["crossing_score"] = 0.88 * state["crossing_score"] + score

    if state["crossing_score"] >= CROSSING_SCORE_ON:
        state["crossing_hold"] = CROSSING_KEEP_FRAMES
    elif state["crossing_hold"] > 0:
        state["crossing_hold"] -= 1

    proposed = "crossing" if (
        state["crossing_hold"] > 0 or state["crossing_score"] >= CROSSING_SCORE_ON
    ) else "normal"

    if proposed == state["candidate_motion_state"]:
        state["candidate_motion_count"] += 1
    else:
        state["candidate_motion_state"] = proposed
        state["candidate_motion_count"] = 1

    if state["motion_state"] == "normal":
        state["motion_state"] = proposed
    elif state["candidate_motion_count"] >= STATE_SWITCH_MIN_FRAMES:
        state["motion_state"] = proposed
