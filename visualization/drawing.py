import cv2
import numpy as np

from config.settings import (
    BOX_THICKNESS, MASK_ALPHA, MASK_BORDER_THICKNESS,
    DRAW_MASK_LABELS, VIS_MODE, QUERIES,
)
from core.geometry import polygon_to_int_xy


# Palette colori per track diversi
_PALETTE = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72, 249, 10),   (146, 204, 23),  (61, 219, 134),
    (26, 147, 52),   (0, 212, 187),   (44, 153, 168),  (0, 194, 255),
    (52, 69, 147),   (100, 115, 255), (0, 24, 236),    (132, 56, 255),
    (82, 0, 133),    (203, 56, 255),  (255, 149, 200), (255, 55, 199),
]


def color_from_id(idx) -> tuple[int, int, int]:
    if idx is None or idx < 0:
        return (180, 180, 180)
    return _PALETTE[int(idx) % len(_PALETTE)]


def draw_box(frame, xyxy, label: str, color: tuple) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def draw_polygon_mask(frame, polygon_xy, color: tuple,
                      alpha: float = MASK_ALPHA,
                      border_thickness: int = MASK_BORDER_THICKNESS):
    pts = polygon_to_int_xy(polygon_xy)
    if pts is None:
        return frame
    overlay = frame.copy()
    pts_cv = pts.reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts_cv], color)
    cv2.polylines(overlay, [pts_cv], True, color, border_thickness)
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)


def draw_mask_label(frame, polygon_xy, label: str, color: tuple) -> None:
    pts = polygon_to_int_xy(polygon_xy)
    if pts is None or not label:
        return
    x = int(np.min(pts[:, 0]))
    y = int(np.min(pts[:, 1]))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(0, y - th - 10)
    cv2.rectangle(frame, (x, top), (x + tw + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def draw_visual(frame, xyxy, polygon_xy, label: str, color: tuple):
    if VIS_MODE in ["mask", "both"] and polygon_xy is not None:
        frame = draw_polygon_mask(frame, polygon_xy, color=color,
                                  alpha=MASK_ALPHA, border_thickness=MASK_BORDER_THICKNESS)
        if VIS_MODE == "mask" and DRAW_MASK_LABELS:
            draw_mask_label(frame, polygon_xy, label, color)
    if VIS_MODE in ["box", "both"]:
        draw_box(frame, xyxy, label, color)
    return frame


def draw_counters(frame, counters: dict) -> None:
    h, w = frame.shape[:2]
    panel_h = 60 + 26 * len(QUERIES)
    x1, y1 = w - 360, 10
    cv2.rectangle(frame, (x1, y1), (w - 10, y1 + panel_h), (30, 30, 30), -1)
    cv2.putText(frame, "Counters", (x1 + 10, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    yy = y1 + 55
    for q in QUERIES:
        cv2.putText(frame, f"{q}: {counters[q]}", (x1 + 10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        yy += 26
