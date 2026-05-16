import cv2
import numpy as np

from config.settings import (
    CROP_PAD, MASK_CROP_PAD, CONTEXT_CROP_PAD, CONTEXT_OCCLUDER_PAD,
    MIN_CROP_AREA, USE_MASK_FOR_CLIP_CROP,
    CLIP_CONTEXT_BLUR_KERNEL, CLIP_CONTEXT_NEUTRAL_COLOR,
    CLIP_CONTEXT_OCCLUDER_MODE,
)


def polygon_to_int_xy(polygon_xy) -> np.ndarray | None:
    """Converts a polygon list of [x,y] to a numpy array of shape (N,2) with integer coordinates"""
    if polygon_xy is None:
        return None
    arr = np.asarray(polygon_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
        return None
    return np.round(arr).astype(np.int32)


def mask_bbox_from_polygon(polygon_xy) -> list | None:
    """Returns the bounding box [x1,y1,x2,y2] of the polygon, or None if invalid."""
    pts = polygon_to_int_xy(polygon_xy)
    if pts is None:
        return None
    return [
        int(np.min(pts[:, 0])),
        int(np.min(pts[:, 1])),
        int(np.max(pts[:, 0])),
        int(np.max(pts[:, 1])),
    ]


def box_center(xyxy) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def normalized_center(xyxy, frame_w: int, frame_h: int) -> tuple[float, float]:
    cx, cy = box_center(xyxy)
    return cx / max(frame_w, 1), cy / max(frame_h, 1)


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def center_distance_norm(box_a, box_b, frame_w: int, frame_h: int) -> float:
    if box_a is None or box_b is None:
        return 999.0
    cxa, cya = normalized_center(box_a, frame_w, frame_h)
    cxb, cyb = normalized_center(box_b, frame_w, frame_h)
    return float(np.hypot(cxa - cxb, cya - cyb))


def extract_crop(frame, xyxy):
    """Extract a crop from the frame with padding. Returns None if too small."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    px, py = int(bw * CROP_PAD), int(bh * CROP_PAD)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(w, x2 + px); y2 = min(h, y2 + py)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] * crop.shape[1] < MIN_CROP_AREA:
        return None
    return crop


def padded_box_bounds(frame, xyxy, pad: float = CROP_PAD):
    """Returns padded box bounds [x1,y1,x2,y2] clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(w, x2 + px); y2 = min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def extract_mask_crop(frame, polygon_xy):
    """Extract a masked crop using the segmentation polygon."""
    pts = polygon_to_int_xy(polygon_xy)
    if pts is None:
        return None
    h, w = frame.shape[:2]
    bbox = mask_bbox_from_polygon(pts)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * MASK_CROP_PAD), int(bh * MASK_CROP_PAD)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(w, x2 + px); y2 = min(h, y2 + py)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] * crop.shape[1] < MIN_CROP_AREA:
        return None
    shifted = pts.copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shifted.reshape((-1, 1, 2))], 255)
    masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
    if masked_crop.size == 0 or masked_crop.shape[0] * masked_crop.shape[1] < MIN_CROP_AREA:
        return None
    return masked_crop


def _valid_blur_kernel(value: int, max_size: int) -> int:
    kernel = max(3, int(value))
    if kernel % 2 == 0:
        kernel += 1
    max_kernel = max(3, max_size if max_size % 2 == 1 else max_size - 1)
    return min(kernel, max_kernel)


def extract_clean_box_context_crop(frame, xyxy, all_polygons=None, det_idx=None):
    """
    Extracts a padded box crop and neutralizes all non-target segmentation masks
    inside the crop. Background remains visible.
    """
    bounds = padded_box_bounds(frame, xyxy, pad=CONTEXT_CROP_PAD)
    if bounds is None:
        return None
    occluder_bounds = padded_box_bounds(frame, xyxy, pad=CONTEXT_OCCLUDER_PAD)

    x1, y1, x2, y2 = bounds
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0 or crop.shape[0] * crop.shape[1] < MIN_CROP_AREA:
        return None

    if all_polygons is None or det_idx is None:
        return crop

    if occluder_bounds is None:
        return crop

    ox1, oy1, ox2, oy2 = occluder_bounds
    ox1 = max(0, ox1 - x1); oy1 = max(0, oy1 - y1)
    ox2 = min(crop.shape[1], ox2 - x1); oy2 = min(crop.shape[0], oy2 - y1)
    if ox2 <= ox1 or oy2 <= oy1:
        return crop

    occluder_region = np.zeros(crop.shape[:2], dtype=np.uint8)
    occluder_region[oy1:oy2, ox1:ox2] = 255
    blurred_crop = None

    for other_idx, polygon_xy in enumerate(all_polygons):
        if other_idx == det_idx:
            continue

        pts = polygon_to_int_xy(polygon_xy)
        if pts is None:
            continue

        shifted = pts.copy()
        shifted[:, 0] -= x1
        shifted[:, 1] -= y1

        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shifted.reshape((-1, 1, 2))], 255)
        mask = cv2.bitwise_and(mask, occluder_region)
        if not np.any(mask):
            continue

        if CLIP_CONTEXT_OCCLUDER_MODE == "blur":
            if blurred_crop is None:
                max_kernel_size = min(crop.shape[:2])
                kernel = _valid_blur_kernel(CLIP_CONTEXT_BLUR_KERNEL, max_kernel_size)
                blurred_crop = cv2.GaussianBlur(crop, (kernel, kernel), 0)
            crop[mask > 0] = blurred_crop[mask > 0]
        else:
            crop[mask > 0] = CLIP_CONTEXT_NEUTRAL_COLOR

    return crop


def extract_clip_views(frame, xyxy, polygon_xy=None, all_polygons=None, det_idx=None):
    """
    Returns CLIP views for one detection:
    - mask: segmented target object only
    - clean_box_context: padded box with non-target masks neutralized
    """
    views = {}

    if USE_MASK_FOR_CLIP_CROP and polygon_xy is not None:
        mask_crop = extract_mask_crop(frame, polygon_xy)
        if mask_crop is not None:
            views["mask"] = mask_crop

    context_crop = extract_clean_box_context_crop(
        frame, xyxy, all_polygons=all_polygons, det_idx=det_idx
    )
    if context_crop is not None:
        views["clean_box_context"] = context_crop

    return views


def extract_semantic_crop(frame, xyxy, polygon_xy=None):
    """
    Extract the best available crop: masked crop if USE_MASK_FOR_CLIP_CROP,
    otherwise bounding box crop.
    """
    if USE_MASK_FOR_CLIP_CROP and polygon_xy is not None:
        mask_crop = extract_mask_crop(frame, polygon_xy)
        if mask_crop is not None:
            return mask_crop
    return extract_crop(frame, xyxy)
