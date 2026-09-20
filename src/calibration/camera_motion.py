"""
Camera Movement & Zoom Compensation Module for Football Broadcast Analytics (Milestone 9).
Performs Global Motion Estimation (GME) via Lucas-Kanade sparse optical flow with foreground
player masking, RANSAC affine transformation fitting, Pan-Tilt-Zoom (PTZ) velocity decomposition,
and Camera Motion Compensation (CMC) for multi-object tracking.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from ..detection.detector import DetectionResult


@dataclass
class CameraMotionResult:
    """
    Telemetry container for inter-frame camera motion, pan-tilt-zoom kinematics,
    and coordinate transformation matrices.
    """
    frame_idx: int
    dx_pixels: float = 0.0           # Horizontal camera displacement (px/frame)
    dy_pixels: float = 0.0           # Vertical camera displacement (px/frame)
    zoom_factor: float = 1.0         # Scale change ratio s (1.0 = steady, >1 = zoom in, <1 = zoom out)
    rotation_deg: float = 0.0        # Camera roll/rotation in degrees
    pan_direction: str = "STATIC"    # "PAN RIGHT", "PAN LEFT", "STATIC"
    tilt_direction: str = "STATIC"   # "TILT UP", "TILT DOWN", "STATIC"
    zoom_state: str = "STEADY"       # "ZOOM IN", "ZOOM OUT", "STEADY"
    transform_matrix: np.ndarray = None  # 2x3 Affine matrix mapping (x_{t-1}, y_{t-1}) -> (x_t, y_t)
    confidence: float = 1.0          # Inlier ratio from RANSAC
    is_valid: bool = True


class CameraMotionCompensator:
    """
    Estimates global camera ego-motion (pan, tilt, zoom) between consecutive video frames
    using sparse Lucas-Kanade optical flow on background pitch features, excluding moving foreground players.
    """

    def __init__(
        self,
        max_features: int = 400,
        quality_level: float = 0.015,
        min_distance: int = 16,
        block_size: int = 7,
        ransac_threshold: float = 2.5,
        smoothing_alpha: float = 0.60,
        pan_threshold_px: float = 1.5,
        tilt_threshold_px: float = 1.2,
        zoom_threshold: float = 0.006,
    ):
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.ransac_threshold = ransac_threshold
        self.smoothing_alpha = smoothing_alpha
        self.pan_threshold_px = pan_threshold_px
        self.tilt_threshold_px = tilt_threshold_px
        self.zoom_threshold = zoom_threshold

        # LK Optical Flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        # State cache
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_features: Optional[np.ndarray] = None
        self.smoothed_dx: float = 0.0
        self.smoothed_dy: float = 0.0
        self.smoothed_zoom: float = 1.0

    def _create_background_mask(
        self,
        gray_shape: Tuple[int, int],
        detections: Optional[DetectionResult] = None,
        margin: int = 15,
    ) -> np.ndarray:
        """
        Create a binary mask where foreground moving players and referees are zeroed out,
        ensuring optical flow only tracks static background pitch markings.
        """
        h, w = gray_shape
        mask = np.full((h, w), 255, dtype=np.uint8)

        # Mask top broadcast scoreboard & HUD area (top 8% of frame)
        mask[0:int(h * 0.08), :] = 0

        # Mask out player bounding boxes with padding
        if detections is not None and len(detections.xyxy) > 0:
            for box in detections.xyxy:
                x1 = max(0, int(box[0]) - margin)
                y1 = max(0, int(box[1]) - margin)
                x2 = min(w, int(box[2]) + margin)
                y2 = min(h, int(box[3]) + margin)
                mask[y1:y2, x1:x2] = 0

        return mask

    def _detect_features(self, gray: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """Detect good features on the background pitch."""
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            mask=mask,
        )
        return pts

    def estimate_motion(
        self,
        frame: np.ndarray,
        detections: Optional[DetectionResult] = None,
        frame_idx: int = 0,
    ) -> CameraMotionResult:
        """
        Estimate inter-frame camera pan, tilt, zoom, and affine transform matrix from frame t-1 to t.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = curr_gray.shape
        bg_mask = self._create_background_mask((h, w), detections)

        # Initial frame handling
        if self.prev_gray is None or self.prev_features is None or len(self.prev_features) < 10:
            self.prev_gray = curr_gray.copy()
            self.prev_features = self._detect_features(curr_gray, bg_mask)
            identity_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            return CameraMotionResult(
                frame_idx=frame_idx,
                transform_matrix=identity_affine,
                is_valid=True,
            )

        # 1. Lucas-Kanade Pyramidal Optical Flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.prev_features,
            None,
            **self.lk_params,
        )

        # Filter valid flow points
        good_prev = []
        good_curr = []
        if curr_pts is not None and status is not None:
            valid = (status.flatten() == 1)
            good_prev = self.prev_features[valid]
            good_curr = curr_pts[valid]

        # 2. RANSAC Affine Model Fitting
        affine_mat = None
        inlier_ratio = 0.0

        if len(good_prev) >= 8:
            affine_mat, inliers = cv2.estimateAffinePartial2D(
                good_prev,
                good_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=1000,
            )
            if inliers is not None:
                inlier_ratio = float(np.sum(inliers)) / float(len(inliers))

        if affine_mat is None or not np.isfinite(affine_mat).all():
            affine_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            is_valid = False
            raw_dx, raw_dy, raw_scale, raw_rot = 0.0, 0.0, 1.0, 0.0
        else:
            is_valid = True
            # Decompose affine matrix:
            # M = [[s * cos(theta), -s * sin(theta), dx],
            #      [s * sin(theta),  s * cos(theta), dy]]
            raw_dx = float(affine_mat[0, 2])
            raw_dy = float(affine_mat[1, 2])
            raw_scale = float(np.hypot(affine_mat[0, 0], affine_mat[1, 0]))
            raw_rot = float(np.degrees(np.arctan2(affine_mat[1, 0], affine_mat[0, 0])))

        # 3. Temporal Smoothing (EMA)
        self.smoothed_dx = self.smoothing_alpha * raw_dx + (1.0 - self.smoothing_alpha) * self.smoothed_dx
        self.smoothed_dy = self.smoothing_alpha * raw_dy + (1.0 - self.smoothing_alpha) * self.smoothed_dy
        self.smoothed_zoom = self.smoothing_alpha * raw_scale + (1.0 - self.smoothing_alpha) * self.smoothed_zoom

        # 4. Motion Classification (Physical PTZ Directions)
        # Note: If image moves left (dx < 0), camera panned right. If image moves right (dx > 0), camera panned left.
        if raw_dx < -self.pan_threshold_px:
            pan_dir = "PAN RIGHT"
        elif raw_dx > self.pan_threshold_px:
            pan_dir = "PAN LEFT"
        else:
            pan_dir = "STATIC"

        # If image moves up (dy < 0), camera tilted down. If image moves down (dy > 0), camera tilted up.
        if raw_dy < -self.tilt_threshold_px:
            tilt_dir = "TILT DOWN"
        elif raw_dy > self.tilt_threshold_px:
            tilt_dir = "TILT UP"
        else:
            tilt_dir = "STATIC"

        if raw_scale > (1.0 + self.zoom_threshold):
            zoom_state = "ZOOM IN"
        elif raw_scale < (1.0 - self.zoom_threshold):
            zoom_state = "ZOOM OUT"
        else:
            zoom_state = "STEADY"

        # Update cache for next frame
        self.prev_gray = curr_gray.copy()
        # Re-detect background features if count is running low
        if len(good_curr) < 60:
            self.prev_features = self._detect_features(curr_gray, bg_mask)
        else:
            self.prev_features = good_curr.reshape(-1, 1, 2)

        return CameraMotionResult(
            frame_idx=frame_idx,
            dx_pixels=round(raw_dx, 2),
            dy_pixels=round(raw_dy, 2),
            zoom_factor=round(raw_scale, 4),
            rotation_deg=round(raw_rot, 2),
            pan_direction=pan_dir,
            tilt_direction=tilt_dir,
            zoom_state=zoom_state,
            transform_matrix=affine_mat.astype(np.float32),
            confidence=round(inlier_ratio, 2),
            is_valid=is_valid,
        )

    @staticmethod
    def warp_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply affine camera motion transformation to an array of 2D points (N, 2).
        Maps points from frame t-1 coordinate system into current frame t coordinate system.
        """
        if len(points) == 0 or transform_matrix is None:
            return points

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.transform(pts, transform_matrix[:2, :])
        return warped.reshape(-1, 2)

    @staticmethod
    def warp_boxes(boxes: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply affine camera motion transformation to bounding boxes (N, 4) [x1, y1, x2, y2].
        """
        if len(boxes) == 0 or transform_matrix is None:
            return boxes

        warped_boxes = np.empty_like(boxes, dtype=np.float32)
        for i, box in enumerate(boxes):
            corners = np.array([
                [box[0], box[1]],
                [box[2], box[1]],
                [box[2], box[3]],
                [box[0], box[3]],
            ], dtype=np.float32).reshape(-1, 1, 2)

            warped_corners = cv2.transform(corners, transform_matrix[:2, :]).reshape(-1, 2)
            warped_boxes[i] = np.array([
                np.min(warped_corners[:, 0]),
                np.min(warped_corners[:, 1]),
                np.max(warped_corners[:, 0]),
                np.max(warped_corners[:, 1]),
            ], dtype=np.float32)

        return warped_boxes
