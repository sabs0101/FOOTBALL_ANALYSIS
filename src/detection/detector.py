"""
High-Performance Player and Ball Detection Module for Football Analytics.
Supports TensorRT (.engine), ONNX Runtime, and PyTorch CUDA FP16 acceleration with
dual-confidence thresholding, zero-copy batching, and graceful fallbacks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
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

    def __len__(self) -> int:
        return len(self.xyxy)

    @property
    def num_detections(self) -> int:
        return len(self.xyxy)

    @property
    def track_ids(self) -> Optional[np.ndarray]:
        """Alias for tracker_ids to ensure cross-module compatibility."""
        return self.tracker_ids

    @track_ids.setter
    def track_ids(self, value: Optional[np.ndarray]):
        self.tracker_ids = value

    def filter_by_class(self, target_class_id: int) -> "DetectionResult":
        """Filter detections to a specific class (e.g. 0 for player, 32 for ball)."""
        mask = self.class_ids == target_class_id
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

    def get_players(self) -> "DetectionResult":
        """Filter detections to only person/players (class_id = 0)."""
        return self.filter_by_class(0)

    def get_ball(self) -> "DetectionResult":
        """Filter detections to only sports ball (class_id = 32)."""
        return self.filter_by_class(32)

    def get_referees(self) -> "DetectionResult":
        """Filter detections to only referee detections (class_id = 1 if present)."""
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
    High-Performance YOLO Detector optimized for broadcast football footage.
    Supports TensorRT FP16 .engine execution, PyTorch CUDA FP16, and multi-class confidence tuning.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        model_dir: str = "models",
        conf_threshold: float = 0.30,
        conf_ball: float = 0.18,
        iou_threshold: float = 0.50,
        imgsz: int = 1280,
        device: str = "auto",
        half: bool = True,
        filter_classes: Optional[List[int]] = None,
        prefer_engine: bool = True,
        warmup_iterations: int = 2,
    ):
        """
        Initialize the PlayerDetector.

        Args:
            model_name: Name or path to YOLO checkpoint (.pt, .engine, or .onnx).
            model_dir: Directory where models are stored.
            conf_threshold: Confidence threshold for players/general classes.
            conf_ball: Confidence threshold specifically for the small fast-moving ball.
            iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
            imgsz: Input inference resolution.
            device: 'auto', 'cuda', 'cpu', or device ID.
            half: Use FP16 half-precision inference if running on CUDA.
            filter_classes: List of class IDs to detect ([0, 32] for player and ball).
            prefer_engine: Auto-load .engine file if available on disk.
            warmup_iterations: Number of warmup frames to eliminate first-frame latency spikes.
        """
        self.conf_threshold = conf_threshold
        self.conf_ball = conf_ball
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.filter_classes = filter_classes or [0, 32]

        self.torch_device = get_device(device)
        self.device = str(self.torch_device)
        self.half = half and (self.torch_device.type == "cuda")

        # Resolve model path and check for optimized engine
        self.model_path, self.backend = self._resolve_model(
            model_name=model_name,
            model_dir=model_dir,
            prefer_engine=prefer_engine,
        )

        print(
            f"[Detector] Initialized YOLO detector: '{self.model_path.name}' "
            f"[Backend: {self.backend}, Device: {self.device}, FP16: {self.half}, imgsz: {self.imgsz}]"
        )

        # Enable cuDNN autotuning for NVIDIA GPUs
        if self.torch_device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Load Ultralytics YOLO model
        try:
            self.model = YOLO(str(self.model_path))
            if self.torch_device.type == "cuda" and not str(self.model_path).endswith(".engine"):
                self.model.to(self.device)
                if self.half and hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                    self.model.model.half()
        except Exception as e:
            # Fallback if engine fails to load
            if self.backend == "TensorRT":
                print(f"[Detector] WARNING: Failed to initialize TensorRT engine ({e}). Falling back to PyTorch .pt model.")
                fallback_pt = self.model_path.with_suffix(".pt")
                if fallback_pt.exists():
                    self.model_path = fallback_pt
                    self.backend = "PyTorch-CUDA-FP16" if self.half else "PyTorch"
                    self.model = YOLO(str(self.model_path))
                    if self.torch_device.type == "cuda":
                        self.model.to(self.device)
                        if self.half and hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                            self.model.model.half()
                else:
                    raise
            else:
                raise

        # Pre-warm GPU kernels
        if warmup_iterations > 0 and self.torch_device.type == "cuda":
            self._warmup(iterations=warmup_iterations)

    def _resolve_model(
        self,
        model_name: str,
        model_dir: str,
        prefer_engine: bool,
    ) -> Tuple[Path, str]:
        """Resolve model file path and determine acceleration backend."""
        model_path = Path(model_name)
        if not model_path.exists():
            model_path = Path(model_dir) / model_name

        # If user passed a .pt model and we are on CUDA with prefer_engine, check for .engine
        if (
            prefer_engine
            and self.torch_device.type == "cuda"
            and model_path.suffix == ".pt"
        ):
            engine_path = model_path.with_suffix(".engine")
            if engine_path.exists():
                return engine_path, "TensorRT"

        # Check explicit suffixes
        if model_path.suffix == ".engine":
            return model_path, "TensorRT"
        elif model_path.suffix == ".onnx":
            return model_path, "ONNX-Runtime"
        else:
            backend = "PyTorch-CUDA-FP16" if self.half else "PyTorch-CUDA-FP32" if self.torch_device.type == "cuda" else "PyTorch-CPU"
            return model_path, backend

    def _warmup(self, iterations: int = 2) -> None:
        """Run dummy inference to warm up GPU caches and TensorRT context."""
        try:
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            with torch.inference_mode():
                for _ in range(iterations):
                    _ = self.model.predict(
                        source=dummy,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def get_backend_info(self) -> Dict[str, Any]:
        """Return metadata about current inference engine."""
        return {
            "model_path": str(self.model_path),
            "backend": self.backend,
            "device": self.device,
            "half": self.half,
            "imgsz": self.imgsz,
            "is_cuda": self.torch_device.type == "cuda",
        }

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """
        Run inference on video frame and extract sanitized bounding boxes with dual thresholding.
        """
        h, w, _ = frame.shape
        min_conf = min(self.conf_threshold, self.conf_ball)

        with torch.inference_mode():
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
            elif cid not in (0, 32) and conf >= self.conf_threshold:
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

    def detect_batch(
        self, frames: List[np.ndarray], start_idx: int = 0
    ) -> List[DetectionResult]:
        """
        Run inference on a batch of image frames for maximum GPU throughput.
        """
        if not frames:
            return []

        min_conf = min(self.conf_threshold, self.conf_ball)
        with torch.inference_mode():
            results = self.model.predict(
                source=frames,
                conf=min_conf,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                classes=self.filter_classes,
                verbose=False,
            )

        batch_results = []
        names = self.model.names

        for i, res in enumerate(results):
            current_frame_idx = start_idx + i
            h, w, _ = frames[i].shape

            boxes_obj = res.boxes
            if boxes_obj is None or len(boxes_obj) == 0:
                batch_results.append(
                    DetectionResult(
                        xyxy=np.empty((0, 4), dtype=np.float32),
                        confidences=np.empty((0,), dtype=np.float32),
                        class_ids=np.empty((0,), dtype=int),
                        class_names=[],
                        frame_idx=current_frame_idx,
                    )
                )
                continue

            boxes = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes_obj.conf.cpu().numpy().astype(np.float32)
            classes = boxes_obj.cls.cpu().numpy().astype(int)

            keep = []
            for k in range(len(boxes)):
                cid = classes[k]
                conf = confs[k]
                if cid == 32 and conf >= self.conf_ball:
                    keep.append(k)
                elif cid == 0 and conf >= self.conf_threshold:
                    keep.append(k)
                elif cid not in (0, 32) and conf >= self.conf_threshold:
                    keep.append(k)

            if keep:
                boxes = boxes[keep]
                confs = confs[keep]
                classes = classes[keep]
                boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                confs = np.empty((0,), dtype=np.float32)
                classes = np.empty((0,), dtype=int)

            class_names = [names[c] if c in names else f"class_{c}" for c in classes]

            batch_results.append(
                DetectionResult(
                    xyxy=boxes,
                    confidences=confs,
                    class_ids=classes,
                    class_names=class_names,
                    frame_idx=current_frame_idx,
                )
            )

        return batch_results
