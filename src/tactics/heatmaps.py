"""
Positional Density Heatmap Module for Football Tactical Analysis (Polished & Hardened).
Generates 2D Gaussian Kernel Density Estimation (KDE) heatmaps with 99th-percentile normalization.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..calibration.template import PitchTemplate, PITCH_LENGTH_M, PITCH_WIDTH_M


class HeatmapGenerator:
    """
    Accumulates real-world pitch coordinates (X, Y) over time and renders
    high-resolution 2D positional density heatmaps for players and teams.
    """

    def __init__(
        self,
        grid_width: int = 1050,
        grid_height: int = 680,
        gaussian_sigma: int = 35,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.sigma = gaussian_sigma
        self.template = PitchTemplate()

        # Positional history buffers
        self.player_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.team_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    def add_positions(
        self,
        track_ids: np.ndarray,
        positions_m: np.ndarray,
        team_ids: Optional[np.ndarray] = None,
    ):
        """
        Accumulate current frame player coordinates into historical buffers.
        """
        num_items = len(track_ids)
        for i in range(num_items):
            tid = int(track_ids[i])
            if tid < 0:
                continue

            xm = float(positions_m[i, 0])
            ym = float(positions_m[i, 1])

            # Must be within pitch boundaries
            if -2.0 <= xm <= PITCH_LENGTH_M + 2.0 and -2.0 <= ym <= PITCH_WIDTH_M + 2.0:
                self.player_positions[tid].append((xm, ym))

                if team_ids is not None and i < len(team_ids):
                    team = int(team_ids[i])
                    if team in [0, 3]:  # Team A Outfield + GK
                        self.team_positions[0].append((xm, ym))
                    elif team in [1, 4]:  # Team B Outfield + GK
                        self.team_positions[1].append((xm, ym))

    def _positions_to_density_grid(self, positions: List[Tuple[float, float]]) -> np.ndarray:
        """
        Convert (X_m, Y_m) coordinates into a 2D Gaussian smoothed density grid with 99th-percentile clipping.
        """
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        if not positions:
            return grid

        sx = self.grid_width / PITCH_LENGTH_M
        sy = self.grid_height / PITCH_WIDTH_M

        for xm, ym in positions:
            px = int(np.clip(xm * sx, 0, self.grid_width - 1))
            py = int(np.clip((PITCH_WIDTH_M - ym) * sy, 0, self.grid_height - 1))
            grid[py, px] += 1.0

        # Gaussian smoothing
        ksize = int(self.sigma * 4) | 1
        smoothed = cv2.GaussianBlur(grid, (ksize, ksize), self.sigma)

        # 99th-percentile normalization to prevent standing hotspots from over-saturating
        non_zero = smoothed[smoothed > 0]
        if len(non_zero) > 0:
            p99 = np.percentile(non_zero, 98.5)
            clipped = np.clip(smoothed / max(1e-5, p99), 0.0, 1.0)
            normalized = (clipped * 255.0).astype(np.uint8)
        else:
            normalized = np.zeros_like(grid, dtype=np.uint8)

        return normalized

    def render_heatmap(
        self,
        positions: List[Tuple[float, float]],
        title: str = "POSITIONAL HEATMAP",
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.60,
    ) -> np.ndarray:
        """
        Render density heatmap overlaid on a FIFA pitch canvas.
        """
        base_pitch = self.template.draw_2d_pitch(
            canvas_width=self.grid_width,
            canvas_height=self.grid_height,
            bg_color=(20, 28, 20),
            line_color=(220, 240, 220),
            padding=20,
        )

        density_grid = self._positions_to_density_grid(positions)
        if np.max(density_grid) == 0:
            return base_pitch

        # Apply Colormap
        color_heatmap = cv2.applyColorMap(density_grid, colormap)

        # Mask out low density regions (keep green pitch visible where player didn't go)
        mask = (density_grid > 15).astype(np.uint8)
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        blended = base_pitch.copy()
        np.copyto(
            blended,
            cv2.addWeighted(color_heatmap, alpha, base_pitch, 1.0 - alpha, 0),
            where=mask_3ch.astype(bool),
        )

        # Overlay title badge
        cv2.rectangle(blended, (18, 12), (380, 42), (15, 20, 25), -1)
        cv2.rectangle(blended, (18, 12), (380, 42), (0, 215, 255), 1)
        cv2.putText(
            blended,
            title,
            (26, 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

        return blended

    def generate_player_heatmap(self, track_id: int) -> np.ndarray:
        """Generate 2D heatmap for a specific player track ID."""
        pts = self.player_positions.get(track_id, [])
        return self.render_heatmap(pts, title=f"PLAYER #{track_id:02d} POSITIONAL HEATMAP")

    def generate_team_heatmap(self, team_id: int, team_name: str = "Team") -> np.ndarray:
        """Generate 2D heatmap for an entire team."""
        pts = self.team_positions.get(team_id, [])
        return self.render_heatmap(pts, title=f"{team_name.upper()} TACTICAL HEATMAP")

    def export_all_heatmaps(self, output_dir: str):
        """Save team and top player heatmaps to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        img_a = self.generate_team_heatmap(0, "Team A (Gladbach)")
        cv2.imwrite(str(out_path / "heatmap_team_a.png"), img_a)

        img_b = self.generate_team_heatmap(1, "Team B (Wolfsburg)")
        cv2.imwrite(str(out_path / "heatmap_team_b.png"), img_b)

        # Export Top 4 Most Active Players
        sorted_players = sorted(
            self.player_positions.keys(),
            key=lambda k: len(self.player_positions[k]),
            reverse=True,
        )[:4]

        for tid in sorted_players:
            img_p = self.generate_player_heatmap(tid)
            cv2.imwrite(str(out_path / f"heatmap_player_{tid:02d}.png"), img_p)

        print(f"[Heatmaps] Exported team and player heatmaps to: {out_path.resolve()}")
