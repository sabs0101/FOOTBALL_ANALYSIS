"""
Homography Estimation and Coordinate Transformation Module.
Computes 3x3 Homography Matrix (H) mapping 2D camera pixels to FIFA 2D pitch meters.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from .template import PitchTemplate, PITCH_KEYPOINTS_FIFA, PITCH_LENGTH_M, PITCH_WIDTH_M


@dataclass
class HomographyResult:
    """
    Container for computed Homography Matrix and geometric projection metrics.

    Attributes:
        H: 3x3 Homography matrix mapping (u, v) image pixels -> (X, Y) pitch meters.
        H_inv: 3x3 Inverse homography matrix mapping (X, Y) pitch meters -> (u, v) image pixels.
        reprojection_error: Mean reprojection error in pixels.
        src_points: Image coordinate correspondences (K, 2).
        dst_points: Canonical FIFA pitch coordinate correspondences (K, 2).
        is_valid: Boolean indicating whether H is non-singular and well-conditioned.
    """
    H: np.ndarray
    H_inv: np.ndarray
    reprojection_error: float = 0.0
    src_points: Optional[np.ndarray] = None
    dst_points: Optional[np.ndarray] = None
    is_valid: bool = True


class PitchHomography:
    """
    Calculates projective transformations (Homography) between broadcast camera views
    and standard 2D FIFA football pitch coordinate space.
    """

    def __init__(self, template: Optional[PitchTemplate] = None):
        self.template = template or PitchTemplate()

    def compute_homography(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        ransac_thresh: float = 5.0,
    ) -> HomographyResult:
        """
        Compute optimal 3x3 Homography matrix from 4+ non-collinear point pairs.
        """
        src = np.asarray(src_points, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 1, 2)

        if len(src) < 4:
            raise ValueError(f"At least 4 point correspondences required, got {len(src)}")

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)

        if H is None or np.isnan(H).any() or np.linalg.cond(H) > 1e12:
            return HomographyResult(
                H=np.eye(3, dtype=np.float32),
                H_inv=np.eye(3, dtype=np.float32),
                reprojection_error=999.0,
                is_valid=False,
            )

        H_inv = np.linalg.inv(H)

        projected = cv2.perspectiveTransform(src, H)
        errors = np.linalg.norm(projected - dst, axis=2)
        mean_err = float(np.mean(errors))

        return HomographyResult(
            H=H.astype(np.float32),
            H_inv=H_inv.astype(np.float32),
            reprojection_error=mean_err,
            src_points=src.reshape(-1, 2),
            dst_points=dst.reshape(-1, 2),
            is_valid=True,
        )

    def image_to_pitch(self, points_image: np.ndarray, H: Any) -> np.ndarray:
        """
        Transform camera image coordinates (u, v) into real-world pitch coordinates (X, Y) in meters.
        """
        if len(points_image) == 0 or H is None:
            return np.empty((0, 2), dtype=np.float32)

        mat = H.H if hasattr(H, "H") else H
        pts = np.asarray(points_image, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, mat)
        return transformed.reshape(-1, 2)

    def pitch_to_image(self, points_pitch: np.ndarray, H_inv: Any) -> np.ndarray:
        """
        Project 2D pitch coordinates (X, Y) in meters back into the camera image space (u, v).
        """
        if len(points_pitch) == 0 or H_inv is None:
            return np.empty((0, 2), dtype=np.float32)

        mat = H_inv.H_inv if hasattr(H_inv, "H_inv") else (H_inv.H if hasattr(H_inv, "H") else H_inv)
        pts = np.asarray(points_pitch, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, mat)
        return transformed.reshape(-1, 2)

    def estimate_broadcast_homography(
        self,
        frame_shape: Tuple[int, int],
        top_touchline_y: float = 245.0,
        bottom_touchline_y: float = 745.0,
        halfway_x: float = 960.0,
        center_y: float = 500.0,
        horiz_radius_px: float = 280.0,
        vert_radius_px: float = 130.0,
    ) -> HomographyResult:
        """
        Compute broadcast homography using 7 non-collinear keypoint correspondences:
        - Halfway top & bottom touchline intersections
        - Center spot
        - Center circle top & bottom extremes
        - Center circle left & right extremes
        """
        h, w = frame_shape[:2]

        src_points = np.array([
            [halfway_x - 30.0, top_touchline_y],                     # Halfway top
            [halfway_x + 10.0, bottom_touchline_y],                  # Halfway bottom
            [halfway_x, center_y],                                  # Center spot
            [halfway_x, center_y - vert_radius_px],                 # Center circle top
            [halfway_x, center_y + vert_radius_px],                 # Center circle bottom
            [halfway_x - horiz_radius_px, center_y],                # Center circle left
            [halfway_x + horiz_radius_px, center_y],                # Center circle right
        ], dtype=np.float32)

        dst_points = np.array([
            [52.5, 68.0],                                           # Halfway top
            [52.5, 0.0],                                            # Halfway bottom
            [52.5, 34.0],                                           # Center spot
            [52.5, 34.0 + 9.15],                                    # Center circle top
            [52.5, 34.0 - 9.15],                                    # Center circle bottom
            [52.5 - 9.15, 34.0],                                    # Center circle left
            [52.5 + 9.15, 34.0],                                    # Center circle right
        ], dtype=np.float32)

        return self.compute_homography(src_points, dst_points)
