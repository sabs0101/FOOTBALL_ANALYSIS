"""
Video Input/Output and processing utilities.
"""

from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple
import cv2
import numpy as np


def get_video_properties(video_path: str) -> Dict[str, Any]:
    """
    Extract key metadata from a video file.

    Args:
        video_path: Path to video file.

    Returns:
        Dict containing fps, frame_count, width, height, and duration_seconds.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path.resolve()}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {path.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Guard against invalid FPS or frame count
    if fps <= 0:
        fps = 25.0
    duration = frame_count / fps if fps > 0 else 0.0

    return {
        "path": str(path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


class VideoReader:
    """
    Iterator and context manager to read frames sequentially from video.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path.resolve()}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video stream: {self.video_path.resolve()}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_idx = self.current_frame
            self.current_frame += 1
            yield frame_idx, frame

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    """
    Video writer utility to save processed frames into standard MP4 format.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.width = width
        self.height = height
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, fps, (width, height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to initialize VideoWriter at: {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def write(self, frame: np.ndarray):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
