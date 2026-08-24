"""
2D Tactical Pitch Radar Minimap Module for Football Analytics (Milestone 7).
Renders real-world metric player positions (X, Y), Team Convex Hulls (Compactness),
and Voronoi Pitch Space Control dominance onto a top-down tactical pitch display.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..calibration.template import PitchTemplate, PITCH_LENGTH_M, PITCH_WIDTH_M
from .annotator import get_track_color


class TacticalRadar:
    """
    Renders top-down 2D tactical radar minimap with real-world metric player/ball positions,
    Team Convex Hulls, and Spatial Control dominance telemetry.
    """

    def __init__(
        self,
        radar_width: int = 380,
        radar_height: int = 245,
        padding: int = 16,
        player_dot_radius: int = 6,
        ball_dot_radius: int = 5,
        bg_color: Tuple[int, int, int] = (14, 22, 16),      # Deep tactical dark pitch
        line_color: Tuple[int, int, int] = (200, 240, 200), # Crisp white/lime pitch markings
        border_color: Tuple[int, int, int] = (0, 215, 255), # Cyan accent border
    ):
        self.radar_width = radar_width
        self.radar_height = radar_height
        self.padding = padding
        self.player_dot_radius = player_dot_radius
        self.ball_dot_radius = ball_dot_radius
        self.bg_color = bg_color
        self.line_color = line_color
        self.border_color = border_color
        self.template = PitchTemplate()

        self.base_pitch = self.template.draw_2d_pitch(
            canvas_width=self.radar_width,
            canvas_height=self.radar_height,
            bg_color=self.bg_color,
            line_color=self.line_color,
            padding=self.padding,
        )

    def _meters_to_radar_pixels(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert real-world meters [0..105, 0..68] to radar pixel coordinates."""
        field_w = self.radar_width - 2 * self.padding
        field_h = self.radar_height - 2 * self.padding

        sx = field_w / PITCH_LENGTH_M
        sy = field_h / PITCH_WIDTH_M

        cx = np.clip(x_m, 0.0, PITCH_LENGTH_M)
        cy = np.clip(y_m, 0.0, PITCH_WIDTH_M)

        px = int(self.padding + cx * sx)
        py = int(self.padding + (PITCH_WIDTH_M - cy) * sy)
        return (px, py)

    def draw_convex_hull(
        self,
        radar_img: np.ndarray,
        hull_vertices_m: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.22,
    ):
        """Draw semi-transparent filled team convex hull polygon on the radar canvas."""
        if len(hull_vertices_m) < 3:
            return

        pts_px = [self._meters_to_radar_pixels(xm, ym) for xm, ym in hull_vertices_m]
        poly = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))

        overlay = radar_img.copy()
        cv2.fillPoly(overlay, [poly], color, cv2.LINE_AA)
        cv2.polylines(radar_img, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, radar_img, 1.0 - alpha, 0, radar_img)

    def render_radar(
        self,
        player_positions_m: np.ndarray,
        player_track_ids: Optional[np.ndarray] = None,
        ball_position_m: Optional[Tuple[float, float]] = None,
        team_colors: Optional[List[Tuple[int, int, int]]] = None,
        tactical_spatial_result: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Draw active players, team convex hulls, ball, and territory dominance on the 2D pitch canvas.
        """
        radar_img = self.base_pitch.copy()

        # 1. Draw Team Convex Hulls (Tactical Compactness)
        if tactical_spatial_result is not None:
            if tactical_spatial_result.team_a_hull is not None:
                self.draw_convex_hull(radar_img, tactical_spatial_result.team_a_hull, (240, 240, 240), alpha=0.20)
            if tactical_spatial_result.team_b_hull is not None:
                self.draw_convex_hull(radar_img, tactical_spatial_result.team_b_hull, (30, 220, 70), alpha=0.20)

        # 2. Draw Player Dots
        num_players = len(player_positions_m)
        for i in range(num_players):
            xm, ym = player_positions_m[i]
            if not (-5.0 <= xm <= PITCH_LENGTH_M + 5.0 and -5.0 <= ym <= PITCH_WIDTH_M + 5.0):
                continue

            px, py = self._meters_to_radar_pixels(xm, ym)
            tid = int(player_track_ids[i]) if (player_track_ids is not None and i < len(player_track_ids)) else i

            if team_colors is not None and i < len(team_colors):
                color = team_colors[i]
            else:
                color = get_track_color(tid)

            cv2.circle(radar_img, (px, py), self.player_dot_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(radar_img, (px, py), self.player_dot_radius, color, -1, cv2.LINE_AA)
            cv2.circle(radar_img, (px, py), self.player_dot_radius, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Football Dot
        if ball_position_m is not None:
            bxm, bym = ball_position_m
            if -5.0 <= bxm <= PITCH_LENGTH_M + 5.0 and -5.0 <= bym <= PITCH_WIDTH_M + 5.0:
                bpx, bpy = self._meters_to_radar_pixels(bxm, bym)
                cv2.circle(radar_img, (bpx, bpy), self.ball_dot_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(radar_img, (bpx, bpy), self.ball_dot_radius, (0, 60, 255), -1, cv2.LINE_AA)
                cv2.circle(radar_img, (bpx, bpy), self.ball_dot_radius, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Outer accent border
        cv2.rectangle(radar_img, (0, 0), (self.radar_width - 1, self.radar_height - 1), self.border_color, 2, cv2.LINE_AA)

        # 5. Top title badge
        cv2.putText(
            radar_img,
            "TACTICAL 2D RADAR (FIFA 105m x 68m)",
            (12, 14),
            cv2.FONT_HERSHEY_DUPLEX,
            0.30,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

        # 6. Bottom Territory Dominance Bar
        if tactical_spatial_result is not None:
            pct_a = tactical_spatial_result.team_a_control_pct
            pct_b = tactical_spatial_result.team_b_control_pct
            control_str = f"SPACE: A {pct_a:.0f}% | B {pct_b:.0f}%"
            cv2.putText(
                radar_img,
                control_str,
                (self.radar_width - 145, self.radar_height - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.28,
                (200, 240, 200),
                1,
                cv2.LINE_AA,
            )

        return radar_img

    def overlay_on_frame(
        self,
        frame: np.ndarray,
        radar_img: np.ndarray,
        position: str = "bottom_right",
        margin: int = 18,
        alpha: float = 0.95,
    ) -> np.ndarray:
        """
        Overlay the 2D tactical radar minimap onto the main broadcast video frame.
        """
        out_frame = frame.copy()
        fh, fw, _ = out_frame.shape
        rh, rw, _ = radar_img.shape

        if position == "bottom_right":
            x1 = fw - rw - margin
            y1 = fh - rh - margin
        elif position == "bottom_left":
            x1 = margin
            y1 = fh - rh - margin
        elif position == "top_right":
            x1 = fw - rw - margin
            y1 = margin + 50
        else:
            x1 = margin
            y1 = margin + 50

        x2 = x1 + rw
        y2 = y1 + rh

        sub_frame = out_frame[y1:y2, x1:x2]
        blended = cv2.addWeighted(radar_img, alpha, sub_frame, 1.0 - alpha, 0)
        out_frame[y1:y2, x1:x2] = blended

        return out_frame
