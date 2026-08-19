"""
Player and Football Object Detection Module using YOLO.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class DetectionResult:
    """
    Structured container for object detection outputs on a single frame.

    Attributes:
        xyxy: Bounding box coordinates [x1, y1, x2, y2] of shape (N, 4).
        confidences: Detection confidence scores of shape (N,).
        class_ids: Class IDs of shape (N,).
        class_names: List of human-readable class names.
        frame_idx: Index of the processed frame.
    """
    xyxy: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    class_names: List[str]
    frame_idx: int = 0

    @property
    def num_detections(self) -> int:
        return len(self.xyxy)

    def filter_by_class(self, target_class_id: int) -> "DetectionResult":
        """Filter detections to a specific class (e.g. 0 for player)."""
        mask = self.class_ids == target_class_id
        return DetectionResult(
            xyxy=self.xyxy[mask],
            confidences=self.confidences[mask],
            class_ids=self.class_ids[mask],
            class_names=[self.class_names[i] for i, m in enumerate(mask) if m],
            frame_idx=self.frame_idx,
        )

    def get_players(self) -> "DetectionResult":
        """Convenience method to get only player detections (class 0)."""
        return self.filter_by_class(0)

    def get_ball(self) -> "DetectionResult":
        """Convenience method to get only sports ball detections (class 32)."""
        return self.filter_by_class(32)


class PlayerDetector:
    """
    YOLO-based detector for football players, referees, and the ball.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        model_dir: str = "models",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50,
        device: Union[str, torch.device] = "auto",
        half: bool = True,
        filter_classes: Optional[List[int]] = None,
    ):
        """
        Initialize the PlayerDetector.

        Args:
            model_name: Name of the YOLO model checkpoint (e.g. yolov8m.pt).
            model_dir: Directory where model weights are stored.
            conf_threshold: Confidence threshold for bounding boxes.
            iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
            device: 'auto', 'cuda', 'cpu', or device index.
            half: Use FP16 half-precision inference if running on CUDA.
            filter_classes: List of class IDs to retain (e.g. [0, 32]). Defaults to [0, 32].
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.filter_classes = filter_classes if filter_classes is not None else [0, 32]
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Resolve device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, torch.device):
            self.device = str(device)
        elif str(device).isdigit():
            self.device = f"cuda:{device}"
        elif device == "cuda":
            self.device = "cuda:0"
        else:
            self.device = str(device)

        self.half = half and ("cuda" in self.device)

        # Model path
        model_path = self.model_dir / model_name
        print(f"[Detector] Initializing YOLO model: {model_name} on {self.device} (FP16: {self.half})")
        self.model = YOLO(str(model_path) if model_path.exists() else model_name)
        self.model.to(self.device)

        # Class dictionary
        self.names = self.model.names

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """
        Run inference on a single image frame.

        Args:
            frame: BGR image from OpenCV (H, W, 3).
            frame_idx: Index of the current video frame.

        Returns:
            DetectionResult dataclass.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.filter_classes,
            device=self.device,
            verbose=False,
        )

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return DetectionResult(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                frame_idx=frame_idx,
            )

        xyxy = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
        confidences = boxes_obj.conf.cpu().numpy().astype(np.float32)
        class_ids = boxes_obj.cls.cpu().numpy().astype(np.int32)
        class_names = [self.names.get(int(cid), f"class_{cid}") for cid in class_ids]

        return DetectionResult(
            xyxy=xyxy,
            confidences=confidences,
            class_ids=class_ids,
            class_names=class_names,
            frame_idx=frame_idx,
        )

    def detect_batch(
        self, frames: List[np.ndarray], start_idx: int = 0
    ) -> List[DetectionResult]:
        """
        Run inference on a batch of image frames for maximum GPU throughput.

        Args:
            frames: List of BGR images (H, W, 3).
            start_idx: Starting frame index.

        Returns:
            List of DetectionResult dataclasses.
        """
        if not frames:
            return []

        results = self.model.predict(
            source=frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.filter_classes,
            device=self.device,
            verbose=False,
        )

        batch_results = []
        for i, res in enumerate(results):
            current_frame_idx = start_idx + i
            boxes_obj = res.boxes
            if boxes_obj is None or len(boxes_obj) == 0:
                batch_results.append(
                    DetectionResult(
                        xyxy=np.empty((0, 4), dtype=np.float32),
                        confidences=np.empty((0,), dtype=np.float32),
                        class_ids=np.empty((0,), dtype=np.int32),
                        class_names=[],
                        frame_idx=current_frame_idx,
                    )
                )
                continue

            xyxy = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
            confidences = boxes_obj.conf.cpu().numpy().astype(np.float32)
            class_ids = boxes_obj.cls.cpu().numpy().astype(np.int32)
            class_names = [self.names.get(int(cid), f"class_{cid}") for cid in class_ids]

            batch_results.append(
                DetectionResult(
                    xyxy=xyxy,
                    confidences=confidences,
                    class_ids=class_ids,
                    class_names=class_names,
                    frame_idx=current_frame_idx,
                )
            )

        return batch_results
