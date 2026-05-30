#!/usr/bin/env python3
"""
core/geometry.py
================
Handles processing crops, polygons and bounding boxes.
"""

import cv2
import numpy as np

from config.settings import (
    CROP_PAD, MASK_CROP_PAD, CONTEXT_CROP_PAD,
    MIN_CROP_AREA, USE_MASK_FOR_CLIP_CROP,
    CLIP_CONTEXT_BLUR_KERNEL, CLIP_CONTEXT_NEUTRAL_COLOR,
    CLIP_CONTEXT_OCCLUDER_MODE,
)


def polygon_to_int_xy(polygon_xy: list) -> np.ndarray | None:
    """
    Converts a polygon list of [x,y] to a numpy array of shape (N,2) with integer coordinates.
    
    Parameters:
        polygon_xy (list): input polygon.

    Returns:
        np.ndarray | None: converted polygon or None
    """

    if polygon_xy is None:
        return None
    arr = np.asarray(polygon_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
        return None
    return np.round(arr).astype(np.int32)


def mask_bbox_from_polygon(polygon_xy: list) -> list | None:
    """
    Returns the bounding box [x1,y1,x2,y2] of the polygon, or None if invalid.
    
    Parameters:
        polygon_xy (list): input polygon

    Returns:
        list: list of bounding box coordinates
    """
    
    pts = polygon_to_int_xy(polygon_xy)
    if pts is None:
        return None
    return [
        int(np.min(pts[:, 0])),
        int(np.min(pts[:, 1])),
        int(np.max(pts[:, 0])),
        int(np.max(pts[:, 1])),
    ]


def box_center(xyxy: list) -> tuple[float, float]:
    """
    Computes the center x and y coordinate of a given box.

    Parameters:
        xyxy (list): box coordinates
    
    Returns:
        tuple[float, float]: center x, center y
    """

    x1, y1, x2, y2 = xyxy
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def normalized_center(xyxy, frame_w: int, frame_h: int) -> tuple[float, float]:
    """
    Normalizes the center coordinates of the box with respect to the size of the original image.

    Parameters:
        xyxy (list): coordinates of the box
        frame_w (int): image width
        frame_h (int): image height 

    Returns:
        tuple[float, float]: normalized center x coordinate, normalized y coordinate
    """

    cx, cy = box_center(xyxy)
    return cx / max(frame_w, 1), cy / max(frame_h, 1)


def box_iou(a, b) -> float:
    """
    
    Parameters:

    
    Returns:
        float: 
    """

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


def center_distance_norm(box_a: list, box_b: list, frame_w: int, frame_h: int) -> float:
    """
    Calculates the distance between two centers of boxes.

    Parameters:
        box_a (list): xyxy coord of the first box 
        box_b (): xyxy coord of the second box
        frame_w (int): image width 
        frame_h (int): image height

    Returns:
        float: distance between normalized centers or 999.0 if at least one of the boxes is None
    """
    
    if box_a is None or box_b is None:
        return 999.0
    cxa, cya = normalized_center(box_a, frame_w, frame_h)
    cxb, cyb = normalized_center(box_b, frame_w, frame_h)
    return float(np.hypot(cxa - cxb, cya - cyb))


def extract_crop(frame: np.ndarray, xyxy: list) -> np.ndarray:
    """
    Extract a crop from the frame with padding. Returns None if too small.
    
    Parameters:
        frame (np.ndarray): input frame
        xyxy (list): coordinates of the crop

    Returns:
        np.ndarray: image crop or None
    """

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


def padded_box_bounds(frame, xyxy, pad: float = CROP_PAD) -> list:
    """
    Returns padded box bounds [x1,y1,x2,y2] clipped to the frame.
    
    Parameters:
        frame (np.ndarray): image frame
        xyxy (list): coordinates of the crop
        pad (float): crop padding

    Returns:
        list: list of padded box bounds
    """

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


def extract_mask_crop(frame: np.ndarray, polygon_xy: list) -> np.ndarray:
    """
    Extract a masked crop using the segmentation polygon.

    Parameters:
        frame (np.ndarray): input frame
        polygon_xy (list): coordinates of the polygon
    
    Returns:
        np.ndarray: cropped mask
    """

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
    """
    Checks whether the blur kernels validity.

    Parameters:
        value (int): size of the kernel
        max_size (int): maximal kernel size

    Returns:
        int: blur kernel size
    """
    
    kernel = max(3, int(value))
    if kernel % 2 == 0:
        kernel += 1
    max_kernel = max(3, max_size if max_size % 2 == 1 else max_size - 1)
    return min(kernel, max_kernel)


def extract_clean_box_context_crop(frame: np.ndarray, xyxy: list, all_polygons: list = None, det_idx: int = None) -> np.ndarray:
    """
    Extracts a padded box crop and neutralizes all non-target segmentation masks
    inside the whole padded context crop. Background remains visible.

    Parameters:
        frame (np.ndarray): input frame
        xyxy (list): list of crop coordinates
        allPolygons (list | None): list of all polygons
        det_idx (int | None): detection index
        
    Returns:
        np.ndarray: padded crop box
    """

    bounds = padded_box_bounds(frame, xyxy, pad=CONTEXT_CROP_PAD)
    if bounds is None:
        return None

    x1, y1, x2, y2 = bounds
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0 or crop.shape[0] * crop.shape[1] < MIN_CROP_AREA:
        return None

    if all_polygons is None or det_idx is None:
        return crop

    occluder_region = np.full(crop.shape[:2], 255, dtype=np.uint8)
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


def extract_clip_views(frame: np.ndarray, xyxy: list, polygon_xy: list = None, 
                       all_polygons: list = None, det_idx: int = None) -> dict:
    """
    Returns CLIP views for one detection:
    - mask: segmented target object only
    - clean_box_context: padded box with non-target masks neutralized

    Parameters:
        frame (np.ndarray): input frame
        xyxy (list): cropbox coordinates
        polygon_xy (list): polygon coordinates
        all_polygons (list): list of all polygons
        det_idx (int): detection index

    Returns:
        dict: CLIP detection views
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


def extract_semantic_crop(frame: np.ndarray, xyxy: list, polygon_xy: list = None) -> np.ndarray:
    """
    Extract the best available crop: masked crop if USE_MASK_FOR_CLIP_CROP,
    otherwise bounding box crop.

    Parameters:
        frame (np.ndarray): input frame
        xyxy (list): bounding box coordinates
        polygon_xy (list): masked crop coordinates

    Returns:
        np.ndarray: image crop
    """

    if USE_MASK_FOR_CLIP_CROP and polygon_xy is not None:
        mask_crop = extract_mask_crop(frame, polygon_xy)
        if mask_crop is not None:
            return mask_crop
    return extract_crop(frame, xyxy)
