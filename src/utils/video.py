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


import queue
import threading


class VideoWriter:
    """
    High-performance Asynchronous Video writer utility that encodes broadcast-ready H.264 MP4
    in a dedicated background thread with zero-blocking egress (PyAV + OpenCV fallback).
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "h264",
        queue_size: int = 64,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.width = width
        self.height = height
        self.use_av = False
        self.av_container = None
        self.av_stream = None
        self.cv_writer = None

        try:
            import av
            self.av = av
            self.av_container = av.open(str(self.output_path), mode="w")
            self.av_stream = self.av_container.add_stream("h264", rate=int(round(fps)))
            self.av_stream.width = width
            self.av_stream.height = height
            self.av_stream.pix_fmt = "yuv420p"
            self.av_stream.options = {"crf": "22", "preset": "ultrafast", "tune": "zerolatency"}
            self.use_av = True
        except Exception:
            fourcc_code = "avc1" if codec == "h264" else codec
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            self.cv_writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
            if not self.cv_writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.cv_writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))

        # Asynchronous frame writing thread
        self._write_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._write_thread = threading.Thread(target=self._writer_loop, name=f"Writer-{self.output_path.stem}", daemon=True)
        self._write_thread.start()

    def _writer_loop(self):
        """Dedicated background loop for non-blocking disk video encoding."""
        while True:
            frame = self._write_queue.get()
            if frame is None:
                break
            try:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                if self.use_av and self.av_container is not None:
                    av_frame = self.av.VideoFrame.from_ndarray(frame, format="bgr24")
                    for packet in self.av_stream.encode(av_frame):
                        self.av_container.mux(packet)
                elif self.cv_writer is not None:
                    self.cv_writer.write(frame)
            except Exception:
                pass
            finally:
                self._write_queue.task_done()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def write(self, frame: np.ndarray):
        """Enqueue frame for asynchronous background encoding."""
        if frame is None:
            return
        try:
            self._write_queue.put(frame, timeout=0.2)
        except queue.Full:
            pass

    def release(self):
        """Flush remaining frames and finalize video file."""
        if hasattr(self, "_write_queue") and self._write_queue is not None:
            try:
                self._write_queue.put(None, timeout=0.5)
                if hasattr(self, "_write_thread") and self._write_thread is not None:
                    self._write_thread.join(timeout=10.0)
            except Exception:
                pass

        if self.use_av and self.av_container is not None:
            try:
                for packet in self.av_stream.encode():
                    self.av_container.mux(packet)
                self.av_container.close()
            except Exception:
                pass
            self.av_container = None
        elif self.cv_writer is not None:
            self.cv_writer.release()
            self.cv_writer = None
