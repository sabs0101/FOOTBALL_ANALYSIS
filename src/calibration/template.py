"""
Standard FIFA Football Pitch Model and Geometric Keypoints Template.
Defines canonical 2D real-world coordinates in meters (105m x 68m).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np


# Standard FIFA Pitch Dimensions in Meters
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
CENTER_CIRCLE_RADIUS_M = 9.15
PENALTY_AREA_LENGTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
GOAL_AREA_LENGTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32
PENALTY_SPOT_DIST_M = 11.0


# Canonical FIFA Keypoints (X in [0, 105] meters, Y in [0, 68] meters)
PITCH_KEYPOINTS_FIFA: Dict[str, Tuple[float, float]] = {
    # 4 Pitch Corners
    "corner_bottom_left": (0.0, 0.0),
    "corner_bottom_right": (PITCH_LENGTH_M, 0.0),
    "corner_top_right": (PITCH_LENGTH_M, PITCH_WIDTH_M),
    "corner_top_left": (0.0, PITCH_WIDTH_M),

    # Halfway Line & Center Circle
    "halfway_bottom": (PITCH_LENGTH_M / 2.0, 0.0),
    "halfway_top": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M),
    "center_spot": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0),
    "center_circle_bottom": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0 - CENTER_CIRCLE_RADIUS_M),
    "center_circle_top": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0 + CENTER_CIRCLE_RADIUS_M),
    "center_circle_left": (PITCH_LENGTH_M / 2.0 - CENTER_CIRCLE_RADIUS_M, PITCH_WIDTH_M / 2.0),
    "center_circle_right": (PITCH_LENGTH_M / 2.0 + CENTER_CIRCLE_RADIUS_M, PITCH_WIDTH_M / 2.0),

    # Left Penalty Area (18-yard box)
    "left_penalty_bottom_left": (0.0, (PITCH_WIDTH_M - PENALTY_AREA_WIDTH_M) / 2.0),
    "left_penalty_bottom_right": (PENALTY_AREA_LENGTH_M, (PITCH_WIDTH_M - PENALTY_AREA_WIDTH_M) / 2.0),
    "left_penalty_top_right": (PENALTY_AREA_LENGTH_M, (PITCH_WIDTH_M + PENALTY_AREA_WIDTH_M) / 2.0),
    "left_penalty_top_left": (0.0, (PITCH_WIDTH_M + PENALTY_AREA_WIDTH_M) / 2.0),
    "left_penalty_spot": (PENALTY_SPOT_DIST_M, PITCH_WIDTH_M / 2.0),

    # Right Penalty Area (18-yard box)
    "right_penalty_bottom_left": (PITCH_LENGTH_M - PENALTY_AREA_LENGTH_M, (PITCH_WIDTH_M - PENALTY_AREA_WIDTH_M) / 2.0),
    "right_penalty_bottom_right": (PITCH_LENGTH_M, (PITCH_WIDTH_M - PENALTY_AREA_WIDTH_M) / 2.0),
    "right_penalty_top_right": (PITCH_LENGTH_M, (PITCH_WIDTH_M + PENALTY_AREA_WIDTH_M) / 2.0),
    "right_penalty_top_left": (PITCH_LENGTH_M - PENALTY_AREA_LENGTH_M, (PITCH_WIDTH_M + PENALTY_AREA_WIDTH_M) / 2.0),
    "right_penalty_spot": (PITCH_LENGTH_M - PENALTY_SPOT_DIST_M, PITCH_WIDTH_M / 2.0),
}


class PitchTemplate:
    """
    Standardized 2D Pitch Model with meter-to-pixel coordinate transformations
    and tactical top-down pitch canvas generation.
    """

    def __init__(
        self,
        length_meters: float = PITCH_LENGTH_M,
        width_meters: float = PITCH_WIDTH_M,
    ):
        self.length_m = length_meters
        self.width_m = width_meters
        self.keypoints = PITCH_KEYPOINTS_FIFA

    def draw_2d_pitch(
        self,
        canvas_width: int = 420,
        canvas_height: int = 272,
        bg_color: Tuple[int, int, int] = (24, 32, 24),     # Dark forest green in BGR
        line_color: Tuple[int, int, int] = (220, 240, 220), # Crisp white/lime lines
        padding: int = 16,
    ) -> np.ndarray:
        """
        Render a beautiful 2D tactical top-down pitch graphic.

        Args:
            canvas_width: Width of canvas in pixels.
            canvas_height: Height of canvas in pixels.
            bg_color: Background field color (BGR).
            line_color: Line marking color (BGR).
            padding: Margin in pixels around the pitch boundary.

        Returns:
            Image of 2D tactical pitch (canvas_height, canvas_width, 3).
        """
        img = np.full((canvas_height, canvas_width, 3), bg_color, dtype=np.uint8)

        field_w = canvas_width - 2 * padding
        field_h = canvas_height - 2 * padding

        sx = field_w / self.length_m
        sy = field_h / self.width_m

        def m2p(x_m: float, y_m: float) -> Tuple[int, int]:
            px = int(padding + x_m * sx)
            py = int(padding + (self.width_m - y_m) * sy) # Flip Y so 0 is bottom
            return (px, py)

        # 1. Outer Boundary
        p_bl = m2p(0.0, 0.0)
        p_tr = m2p(self.length_m, self.width_m)
        cv2.rectangle(img, p_bl, p_tr, line_color, 1, cv2.LINE_AA)

        # 2. Halfway Line
        p_half_bot = m2p(self.length_m / 2.0, 0.0)
        p_half_top = m2p(self.length_m / 2.0, self.width_m)
        cv2.line(img, p_half_bot, p_half_top, line_color, 1, cv2.LINE_AA)

        # 3. Center Circle & Center Spot
        p_center = m2p(self.length_m / 2.0, self.width_m / 2.0)
        r_px = int(CENTER_CIRCLE_RADIUS_M * sx)
        cv2.circle(img, p_center, r_px, line_color, 1, cv2.LINE_AA)
        cv2.circle(img, p_center, 3, line_color, -1, cv2.LINE_AA)

        # 4. Left Penalty Box (18-yard)
        p_lp_bl = m2p(0.0, (self.width_m - PENALTY_AREA_WIDTH_M) / 2.0)
        p_lp_tr = m2p(PENALTY_AREA_LENGTH_M, (self.width_m + PENALTY_AREA_WIDTH_M) / 2.0)
        cv2.rectangle(img, p_lp_bl, p_lp_tr, line_color, 1, cv2.LINE_AA)

        # 5. Right Penalty Box (18-yard)
        p_rp_bl = m2p(self.length_m - PENALTY_AREA_LENGTH_M, (self.width_m - PENALTY_AREA_WIDTH_M) / 2.0)
        p_rp_tr = m2p(self.length_m, (self.width_m + PENALTY_AREA_WIDTH_M) / 2.0)
        cv2.rectangle(img, p_rp_bl, p_rp_tr, line_color, 1, cv2.LINE_AA)

        # 6. Penalty Spots
        p_l_spot = m2p(PENALTY_SPOT_DIST_M, self.width_m / 2.0)
        p_r_spot = m2p(self.length_m - PENALTY_SPOT_DIST_M, self.width_m / 2.0)
        cv2.circle(img, p_l_spot, 2, line_color, -1, cv2.LINE_AA)
        cv2.circle(img, p_r_spot, 2, line_color, -1, cv2.LINE_AA)

        return img
