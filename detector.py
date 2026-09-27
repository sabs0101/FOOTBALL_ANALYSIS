"""
Player and Football Object Detection Module using YOLO (Accelerated).
Supports TensorRT engines, PyTorch CUDA FP16, and unified detection results.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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
        track_ids: Optional persistent tracking IDs.
        frame_idx: Index of the processed frame.
    """
    xyxy: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    class_names: List[str]
    track_ids: Optional[np.ndarray] = None
    tracker_ids: Optional[np.ndarray] = None
    trails: Optional[Dict[int, Any]] = None
    frame_idx: int = 0

    def __post_init__(self):
        if self.track_ids is None and self.tracker_ids is not None:
            self.track_ids = self.tracker_ids
        elif self.tracker_ids is None and self.track_ids is not None:
            self.tracker_ids = self.track_ids

    def __len__(self) -> int:
        return len(self.xyxy)

    @property
    def num_detections(self) -> int:
        return len(self.xyxy)

    def filter_by_class(self, target_class_id: int) -> "DetectionResult":
        """Filter detections to a specific class (e.g. 0 for player)."""
        mask = self.class_ids == target_class_id
        tids = self.track_ids[mask] if self.track_ids is not None else None
        return DetectionResult(
            xyxy=self.xyxy[mask],
            confidences=self.confidences[mask],
            class_ids=self.class_ids[mask],
            class_names=[self.class_names[i] for i, m in enumerate(mask) if m],
            track_ids=tids,
            tracker_ids=tids,
            trails=self.trails,
            frame_idx=self.frame_idx,
        )

    def get_players(self) -> "DetectionResult":
        """Convenience method to get only player detections (class 0)."""
        return self.filter_by_class(0)

    def get_ball(self) -> "DetectionResult":
        """Convenience method to get only sports ball detections (class 32)."""
        return self.filter_by_class(32)

    def get_referees(self) -> "DetectionResult":
        """Convenience method to get only referee detections (class 1)."""
        return self.filter_by_class(1)

    def get_foot_positions(self) -> np.ndarray:
        """Calculate bottom-center ground contact points for player boxes."""
        if len(self.xyxy) == 0:
            return np.empty((0, 2), dtype=np.float32)
        feet_x = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        feet_y = self.xyxy[:, 3]
        return np.column_stack((feet_x, feet_y)).astype(np.float32)


class PlayerDetector:
    """
    YOLO-based detector for football players, referees, and the ball with TensorRT & CUDA FP16 support.
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
        enable_tracking: bool = False,
        tracker_type: str = "bytetrack.yaml",
        prefer_engine: bool = True,
    ):
        """
        Initialize the PlayerDetector.

        Args:
            model_name: Name of the YOLO model checkpoint (.pt, .engine, .onnx).
            model_dir: Directory where model weights are stored.
            conf_threshold: Confidence threshold for bounding boxes.
            iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
            device: 'auto', 'cuda', 'cpu', or device index.
            half: Use FP16 half-precision inference if running on CUDA.
            filter_classes: List of class IDs to retain (e.g. [0, 32]). Defaults to [0, 32].
            enable_tracking: Enable Ultralytics built-in ByteTrack.
            tracker_type: Tracker configuration yaml.
            prefer_engine: Auto-load .engine file if available on disk.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.filter_classes = filter_classes if filter_classes is not None else [0, 32]
        self.enable_tracking = enable_tracking
        self.tracker_type = tracker_type
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

        # Model path resolution
        model_path = Path(model_name)
        if not model_path.exists():
            model_path = self.model_dir / model_name

        if (
            prefer_engine
            and "cuda" in self.device
            and model_path.suffix == ".pt"
        ):
            engine_path = model_path.with_suffix(".engine")
            if engine_path.exists():
                model_path = engine_path

        self.backend = (
            "TensorRT" if model_path.suffix == ".engine"
            else "PyTorch-CUDA-FP16" if self.half
            else "PyTorch-CPU"
        )
        print(f"[Detector] Initializing YOLO model: {model_path.name} on {self.device} (Backend: {self.backend}, FP16: {self.half})")
        
        try:
            self.model = YOLO(str(model_path) if model_path.exists() else model_name)
            if "cuda" in self.device and not str(model_path).endswith(".engine"):
                self.model.to(self.device)
        except Exception as e:
            if str(model_path).endswith(".engine"):
                print(f"[Detector] WARNING: TensorRT engine load failed ({e}), falling back to .pt")
                fallback_pt = model_path.with_suffix(".pt")
                self.model = YOLO(str(fallback_pt) if fallback_pt.exists() else model_name)
                if "cuda" in self.device:
                    self.model.to(self.device)
            else:
                raise

        # Class dictionary
        self.names = self.model.names

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """
        Run inference on a single image frame.
        """
        inference_args = {
            "source": frame,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "classes": self.filter_classes,
            "device": self.device,
            "verbose": False,
        }

        if self.enable_tracking:
            results = self.model.track(**inference_args, persist=True, tracker=self.tracker_type)
        else:
            results = self.model.predict(**inference_args)

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
        
        track_ids = None
        if self.enable_tracking and boxes_obj.id is not None:
            track_ids = boxes_obj.id.cpu().numpy().astype(np.int32)

        return DetectionResult(
            xyxy=xyxy,
            confidences=confidences,
            class_ids=class_ids,
            class_names=class_names,
            track_ids=track_ids,
            tracker_ids=track_ids,
            frame_idx=frame_idx,
        )

    def detect_batch(
        self, frames: List[np.ndarray], start_idx: int = 0
    ) -> List[DetectionResult]:
        """
        Run inference on a batch of image frames for maximum GPU throughput.
        """
        if not frames:
            return []

        inference_args = {
            "source": frames,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "classes": self.filter_classes,
            "device": self.device,
            "verbose": False,
        }

        if self.enable_tracking:
            results = self.model.track(**inference_args, persist=True, tracker=self.tracker_type)
        else:
            results = self.model.predict(**inference_args)

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
            
            track_ids = None
            if self.enable_tracking and boxes_obj.id is not None:
                track_ids = boxes_obj.id.cpu().numpy().astype(np.int32)

            batch_results.append(
                DetectionResult(
                    xyxy=xyxy,
                    confidences=confidences,
                    class_ids=class_ids,
                    class_names=class_names,
                    track_ids=track_ids,
                    tracker_ids=track_ids,
                    frame_idx=current_frame_idx,
                )
            )

        return batch_results
