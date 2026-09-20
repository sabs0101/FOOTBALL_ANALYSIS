"""
Camera Cut Detection Module for Football Broadcast Video (Milestone 10).
Detects abrupt camera cuts, shot boundaries, and replay transitions using
multi-cue HSV color histogram dissimilarity, structural edge change ratio,
and optical flow motion discontinuity.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class CutDetectionResult:
    """
    Container for frame-by-frame camera cut detection metrics.

    Attributes:
        is_cut: True if a shot boundary / camera cut occurred at this frame.
        cut_type: Category ('HARD_CUT', 'NONE').
        hist_distance: Bhattacharyya distance between consecutive frame HSV histograms [0.0 - 1.0].
        edge_diff: Structural edge change ratio [0.0 - 1.0].
        combined_score: Weighted composite dissimilarity score [0.0 - 1.0].
        frame_idx: Index of current video frame.
        cut_count: Cumulative number of camera cuts detected so far.
    """
    is_cut: bool
    cut_type: str
    hist_distance: float
    edge_diff: float
    combined_score: float
    frame_idx: int
    cut_count: int


class CameraCutDetector:
    """
    Real-Time Camera Shot Boundary & Cut Detector for Football Broadcasts.
    Analyzes color distributions and structural edges across consecutive frames
    with temporal debouncing to identify camera angle switches and replay cuts.
    """

    def __init__(
        self,
        hist_threshold: float = 0.50,
        edge_threshold: float = 0.45,
        combined_threshold: float = 0.52,
        min_cut_interval: int = 15,
        downsample_size: Tuple[int, int] = (320, 180),
        h_bins: int = 30,
        s_bins: int = 32,
    ):
        self.hist_threshold = hist_threshold
        self.edge_threshold = edge_threshold
        self.combined_threshold = combined_threshold
        self.min_cut_interval = min_cut_interval
        self.downsample_size = downsample_size
        self.h_bins = h_bins
        self.s_bins = s_bins

        # Previous frame cache
        self.prev_hist: Optional[np.ndarray] = None
        self.prev_edges: Optional[np.ndarray] = None
        self.last_cut_frame: int = -self.min_cut_interval - 1
        self.cut_count: int = 0

    def _compute_hsv_hist(self, frame_small: np.ndarray) -> np.ndarray:
        """
        Compute normalized 2D Hue-Saturation color histogram on downsampled frame.
        """
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [self.h_bins, self.s_bins],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _compute_edges(self, frame_small: np.ndarray) -> np.ndarray:
        """
        Compute binary Canny edge map on downsampled grayscale frame.
        """
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 50, 150)
        return (edges > 0).astype(np.uint8)

    def detect_cut(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        flow_inlier_ratio: Optional[float] = None,
    ) -> CutDetectionResult:
        """
        Analyze the current frame against the previous frame to determine if a camera cut occurred.

        Args:
            frame: Input BGR image (H, W, 3).
            frame_idx: Integer index of the current frame.
            flow_inlier_ratio: Optional optical flow inlier ratio from GME (0.0 to 1.0).

        Returns:
            CutDetectionResult with dissimilarity scores and cut flag.
        """
        # Downsample for ultra-fast real-time processing
        small = cv2.resize(frame, self.downsample_size, interpolation=cv2.INTER_AREA)

        curr_hist = self._compute_hsv_hist(small)
        curr_edges = self._compute_edges(small)

        if self.prev_hist is None or self.prev_edges is None:
            self.prev_hist = curr_hist
            self.prev_edges = curr_edges
            return CutDetectionResult(
                is_cut=False,
                cut_type="NONE",
                hist_distance=0.0,
                edge_diff=0.0,
                combined_score=0.0,
                frame_idx=frame_idx,
                cut_count=self.cut_count,
            )

        # 1. Color Histogram Bhattacharyya Distance (0.0 = identical, 1.0 = completely disjoint)
        hist_dist = cv2.compareHist(self.prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        hist_dist = float(np.clip(hist_dist, 0.0, 1.0))

        # 2. Structural Edge Difference Ratio (Normalized to edge volume)
        edge_xor = np.logical_xor(self.prev_edges, curr_edges)
        total_edge_pixels = int(np.sum(self.prev_edges) + np.sum(curr_edges))
        if total_edge_pixels > 0:
            edge_diff = float(np.sum(edge_xor) / total_edge_pixels)
        else:
            edge_diff = 0.0
        edge_diff = float(np.clip(edge_diff, 0.0, 1.0))

        # 3. Optical Flow Discontinuity Bonus (if provided and optical flow completely collapsed)
        flow_penalty = 0.0
        if flow_inlier_ratio is not None and flow_inlier_ratio < 0.10:
            flow_penalty = 0.15

        # Combined weighted score
        combined_score = 0.60 * hist_dist + 0.40 * edge_diff + flow_penalty
        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        # Check cut conditions with debounce interval
        is_cut = False
        frames_since_last_cut = frame_idx - self.last_cut_frame

        if frames_since_last_cut >= self.min_cut_interval:
            if (
                combined_score >= self.combined_threshold
                or (hist_dist >= self.hist_threshold and edge_diff >= self.edge_threshold)
            ):
                is_cut = True
                self.cut_count += 1
                self.last_cut_frame = frame_idx

        # Update cache
        self.prev_hist = curr_hist
        self.prev_edges = curr_edges

        return CutDetectionResult(
            is_cut=is_cut,
            cut_type="HARD_CUT" if is_cut else "NONE",
            hist_distance=hist_dist,
            edge_diff=edge_diff,
            combined_score=combined_score,
            frame_idx=frame_idx,
            cut_count=self.cut_count,
        )

    def reset(self):
        """Reset internal history buffers."""
        self.prev_hist = None
        self.prev_edges = None
        self.last_cut_frame = -self.min_cut_interval - 1
        self.cut_count = 0
