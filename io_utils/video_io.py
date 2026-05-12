import os
import cv2

from config.settings import INPUT_VIDEO, OUTPUT_VIDEO


def setup_capture() -> tuple[cv2.VideoCapture, dict]:
    """
    Opens the input video and retrieves its properties. Returns a tuple of (VideoCapture object, video_info dict).
    Raises an error if the video cannot be opened.
    """
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0

    video_info = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }
    return cap, video_info


def setup_writer(video_info: dict) -> cv2.VideoWriter:
    """Creates and returns the VideoWriter for the output video."""
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_info["fps"],
        (video_info["width"], video_info["height"]),
    )
    return writer
