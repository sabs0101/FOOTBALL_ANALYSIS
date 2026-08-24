"""
YOLO-based Player and Ball Detection Module for Football Analytics (Polished & Hardened).
Supports multi-class confidence tuning (high recall for small football, precision for players),
FP16 CUDA acceleration, and robust bounding box sanitization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO

from ..utils.config import get_device


@dataclass
class DetectionResult:
    """
    Container for object detections in a single frame.

    Attributes:
        xyxy: Array of shape (N, 4) with bounding box coordinates [x1, y1, x2, y2].
        confidences: Array of shape (N,) with confidence scores [0.0, 1.0].
        class_ids: Array of shape (N,) with integer class IDs (0=Person, 32=Sports Ball).
        class_names: List of string class labels ('person', 'sports_ball').
        tracker_ids: Array of shape (N,) with persistent tracking IDs (-1 for untracked/ball).
        trails: Dict mapping track ID to historical foot position points.
        frame_idx: Index of the processed frame.
    """
    xyxy: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    class_names: List[str]
    tracker_ids: Optional[np.ndarray] = None
    trails: Optional[Dict[int, Any]] = None
    frame_idx: int = 0

    @property
    def num_detections(self) -> int:
        return len(self.xyxy)

    def get_players(self) -> "DetectionResult":
        """Filter detections to only person/players (class_id = 0)."""
        mask = self.class_ids == 0
        tids = self.tracker_ids[mask] if self.tracker_ids is not None else None
        return DetectionResult(
            xyxy=self.xyxy[mask],
            confidences=self.confidences[mask],
            class_ids=self.class_ids[mask],
            class_names=[self.class_names[i] for i, m in enumerate(mask) if m],
            tracker_ids=tids,
            trails=self.trails,
            frame_idx=self.frame_idx,
        )

    def get_ball(self) -> "DetectionResult":
        """Filter detections to only sports ball (class_id = 32)."""
        mask = self.class_ids == 32
        tids = self.tracker_ids[mask] if self.tracker_ids is not None else None
        return DetectionResult(
            xyxy=self.xyxy[mask],
            confidences=self.confidences[mask],
            class_ids=self.class_ids[mask],
            class_names=[self.class_names[i] for i, m in enumerate(mask) if m],
            tracker_ids=tids,
            trails=self.trails,
            frame_idx=self.frame_idx,
        )

    def get_foot_positions(self) -> np.ndarray:
        """Calculate bottom-center ground contact points for player boxes."""
        if len(self.xyxy) == 0:
            return np.empty((0, 2), dtype=np.float32)
        feet_x = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        feet_y = self.xyxy[:, 3]
        return np.column_stack((feet_x, feet_y)).astype(np.float32)


class PlayerDetector:
    """
    YOLOv8/11 detector optimized for broadcast football footage.
    Implements dual confidence thresholds to catch small fast-moving footballs
    while preserving high precision on players.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        model_dir: str = "models",
        conf_threshold: float = 0.18,
        conf_ball: float = 0.12,
        iou_threshold: float = 0.50,
        imgsz: int = 1280,
        device: str = "auto",
        half: bool = True,
        filter_classes: Optional[List[int]] = None,
    ):
        self.conf_threshold = conf_threshold
        self.conf_ball = conf_ball
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.filter_classes = filter_classes or [0, 32]

        self.torch_device = get_device(device)
        self.device = str(self.torch_device)
        self.half = half and (self.torch_device.type == "cuda")

        model_path = Path(model_name)
        if not model_path.exists():
            model_path = Path(model_dir) / model_name

        print(f"[Detector] Initializing YOLO model: {model_name} on {self.device} (FP16: {self.half}, imgsz: {self.imgsz})")
        self.model = YOLO(str(model_path) if model_path.exists() else model_name)

        if self.torch_device.type == "cuda":
            self.model.to(self.device)

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """
        Run inference on video frame and extract sanitized bounding boxes.
        """
        h, w, _ = frame.shape

        # Use the lower threshold to catch small balls
        min_conf = min(self.conf_threshold, self.conf_ball)

        results = self.model.predict(
            source=frame,
            conf=min_conf,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.filter_classes,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else np.empty((0, 4), dtype=np.float32)
        confs = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else np.empty((0,), dtype=np.float32)
        classes = result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) > 0 else np.empty((0,), dtype=int)

        # Apply dual confidence thresholding: conf_threshold for person (0), conf_ball for sports_ball (32)
        keep = []
        for i in range(len(boxes)):
            cid = classes[i]
            conf = confs[i]
            if cid == 32 and conf >= self.conf_ball:
                keep.append(i)
            elif cid == 0 and conf >= self.conf_threshold:
                keep.append(i)

        if keep:
            boxes = boxes[keep]
            confs = confs[keep]
            classes = classes[keep]
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            classes = np.empty((0,), dtype=int)

        # Sanitize coordinate boundaries within image bounds
        if len(boxes) > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)

        names = self.model.names
        class_names = [names[c] if c in names else f"class_{c}" for c in classes]

        return DetectionResult(
            xyxy=boxes,
            confidences=confs,
            class_ids=classes,
            class_names=class_names,
            frame_idx=frame_idx,
        )
