"""
Football Pitch and Field Line Segmentation Module (Polished & Stabilized).
Detects green pitch boundary mask with temporal EMA smoothing, and filters out
crowd spectators, stadium stands, and dugout bench personnel while strictly
preserving all active on-pitch players.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class PitchResult:
    """
    Container for pitch segmentation and field markings.

    Attributes:
        mask: Binary pitch mask (255 on playable grass, 0 elsewhere).
        polygon: Dynamic convex contour boundary of the pitch.
        lines: Detected white line segments [x1, y1, x2, y2].
        pitch_area_ratio: Ratio of pitch area to total frame area.
    """
    mask: np.ndarray
    polygon: np.ndarray
    lines: np.ndarray
    pitch_area_ratio: float


class PitchDetector:
    """
    HSV-based dynamic pitch segmentation with Hough line detection,
    temporal boundary smoothing (EMA), and robust crowd/dugout filtering.
    """

    def __init__(
        self,
        hsv_green_lower: Tuple[int, int, int] = (32, 45, 40),
        hsv_green_upper: Tuple[int, int, int] = (85, 255, 235),
        morph_kernel_size: int = 15,
        min_pitch_area_ratio: float = 0.20,
        white_contrast_thresh: int = 18,
        min_line_length: int = 60,
        max_line_gap: int = 40,
        ema_alpha: float = 0.80,
    ):
        self.hsv_lower = np.array(hsv_green_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_green_upper, dtype=np.uint8)
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        self.min_pitch_area_ratio = min_pitch_area_ratio
        self.white_contrast_thresh = white_contrast_thresh
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.ema_alpha = ema_alpha

        # Temporal EMA state
        self.smooth_y_top: Optional[float] = None
        self.smooth_y_bot: Optional[float] = None

    def get_pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment the green pitch using HSV color thresholding and morphological closing.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        return mask

    def detect_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convenience method returning (mask, polygon, pitch_area_ratio)."""
        res = self.detect_lines(frame)
        return res.mask, res.polygon, res.pitch_area_ratio

    def detect_lines(self, frame: np.ndarray) -> PitchResult:
        """
        Extract field line markings and dynamic pitch boundary with EMA smoothing.
        """
        h, w, _ = frame.shape
        grass_mask = self.get_pitch_mask(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Top-hat morphological filter for white markings
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, tophat_kernel)

        _, line_mask = cv2.threshold(
            tophat, self.white_contrast_thresh, 255, cv2.THRESH_BINARY
        )
        line_mask = cv2.bitwise_and(line_mask, line_mask, mask=grass_mask)

        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            line_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        valid_lines = []
        horizontal_y = []
        if lines is not None:
            for l in lines:
                pts = l[0] if len(l.shape) > 1 else l
                if len(pts) >= 4:
                    x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    length = np.sqrt(dx ** 2 + dy ** 2)
                    angle = np.degrees(np.arctan2(dy, dx))

                    if (angle < 25 or angle > 65) and length >= 30:
                        valid_lines.append((x1, y1, x2, y2))
                        if angle < 18:
                            horizontal_y.append((y1 + y2) / 2.0)

        # Calibrated playable pitch bounds
        raw_y_top = float(h * 0.20)
        raw_y_bot = float(h * 0.795)

        if horizontal_y:
            top_cands = [y for y in horizontal_y if y < h * 0.35]
            if top_cands:
                raw_y_top = float(np.median(top_cands))
            bot_cands = [y for y in horizontal_y if y > h * 0.65]
            if bot_cands:
                raw_y_bot = float(np.median(bot_cands))

        # Temporal EMA smoothing across video stream
        if self.smooth_y_top is None:
            self.smooth_y_top = raw_y_top
            self.smooth_y_bot = raw_y_bot
        else:
            self.smooth_y_top = self.ema_alpha * self.smooth_y_top + (1.0 - self.ema_alpha) * raw_y_top
            self.smooth_y_bot = self.ema_alpha * self.smooth_y_bot + (1.0 - self.ema_alpha) * raw_y_bot

        y_min = max(0, int(self.smooth_y_top - 15))
        y_max = min(h, int(self.smooth_y_bot + 18))

        playable_mask = np.zeros((h, w), dtype=np.uint8)
        playable_mask[y_min:y_max, :] = grass_mask[y_min:y_max, :]

        pitch_poly = np.array([
            [0, y_min],
            [w - 1, y_min],
            [w - 1, y_max],
            [0, y_max]
        ], dtype=np.int32).reshape((-1, 1, 2))

        final_lines = []
        for line in valid_lines:
            x1, y1, x2, y2 = line
            mid_y = (y1 + y2) / 2.0
            if y_min <= mid_y <= y_max:
                final_lines.append(line)

        lines_arr = np.array(final_lines, dtype=np.float32) if final_lines else np.empty((0, 4), dtype=np.float32)
        pitch_area_ratio = float(np.sum(playable_mask > 0) / (h * w))

        return PitchResult(
            mask=playable_mask,
            polygon=pitch_poly,
            lines=lines_arr,
            pitch_area_ratio=pitch_area_ratio,
        )

    def filter_detections_on_pitch(
        self,
        detections: Any,
        pitch_mask: np.ndarray,
    ) -> Any:
        """
        Filter out stadium crowd spectators and dugout bench personnel.
        Preserves all active on-pitch players and the football.
        """
        if len(detections.xyxy) == 0 or pitch_mask is None:
            return detections

        h, w = pitch_mask.shape
        keep_mask = []

        for i in range(len(detections.xyxy)):
            cid = detections.class_ids[i]
            if cid == 32:  # Ball is always kept
                keep_mask.append(True)
                continue

            box = detections.xyxy[i]
            foot_x = int(np.clip((box[0] + box[2]) / 2.0, 0, w - 1))
            foot_y = int(np.clip(box[3], 0, h - 1))

            on_pitch = pitch_mask[foot_y, foot_x] > 0
            keep_mask.append(bool(on_pitch))

        keep_mask = np.array(keep_mask, dtype=bool)

        result_type = type(detections)
        kwargs = {
            "xyxy": detections.xyxy[keep_mask],
            "confidences": detections.confidences[keep_mask],
            "class_ids": detections.class_ids[keep_mask],
            "class_names": [
                detections.class_names[i] for i, m in enumerate(keep_mask) if m
            ],
            "frame_idx": detections.frame_idx,
        }

        if hasattr(detections, "tracker_ids") and detections.tracker_ids is not None:
            kwargs["tracker_ids"] = detections.tracker_ids[keep_mask]
        if hasattr(detections, "trails") and detections.trails is not None:
            kwargs["trails"] = {
                tid: detections.trails[tid]
                for tid in kwargs.get("tracker_ids", [])
                if tid in detections.trails
            }

        return result_type(**kwargs)
