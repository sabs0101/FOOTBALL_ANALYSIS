"""
Spatial Pitch Dominance and Tactical Team Metrics Module (Polished & Hardened).
Computes Team Convex Hulls (Compactness in m²), Centroids, and Voronoi Pitch Space Control.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.spatial import ConvexHull

from ..calibration.template import PITCH_LENGTH_M, PITCH_WIDTH_M


@dataclass
class TacticalSpatialResult:
    """
    Tactical metrics computed for a single match frame.

    Attributes:
        team_a_control_pct: Percentage of pitch space controlled by Team A [0..100].
        team_b_control_pct: Percentage of pitch space controlled by Team B [0..100].
        team_a_hull: (K, 2) 2D polygon vertices of Team A's tactical shape in meters.
        team_b_hull: (M, 2) 2D polygon vertices of Team B's tactical shape in meters.
        team_a_area_m2: Tactical compactness area of Team A in square meters (m²).
        team_b_area_m2: Tactical compactness area of Team B in square meters (m²).
        team_a_centroid_m: Center of mass (X, Y) of Team A in meters.
        team_b_centroid_m: Center of mass (X, Y) of Team B in meters.
        voronoi_grid: 2D array (68, 105) mapping each square meter to controlling team ID (0=Team A, 1=Team B).
    """
    team_a_control_pct: float = 50.0
    team_b_control_pct: float = 50.0
    team_a_hull: Optional[np.ndarray] = None
    team_b_hull: Optional[np.ndarray] = None
    team_a_area_m2: float = 0.0
    team_b_area_m2: float = 0.0
    team_a_centroid_m: Tuple[float, float] = (52.5, 34.0)
    team_b_centroid_m: Tuple[float, float] = (52.5, 34.0)
    voronoi_grid: Optional[np.ndarray] = None


class SpatialControl:
    """
    Calculates spatial dominance, team compactness, and Voronoi territory control
    from 2D real-world FIFA pitch coordinates.
    """

    def __init__(
        self,
        pitch_length_m: float = PITCH_LENGTH_M,
        pitch_width_m: float = PITCH_WIDTH_M,
        grid_resolution_m: float = 1.0,
    ):
        self.length_m = pitch_length_m
        self.width_m = pitch_width_m
        self.res = grid_resolution_m

        # Precompute metric grid (68, 105)
        xs = np.arange(0.0, self.length_m, self.res)
        ys = np.arange(0.0, self.width_m, self.res)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)
        self.grid_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))
        self.total_cells = len(self.grid_points)

    def compute_team_hull(self, positions_m: np.ndarray) -> Tuple[Optional[np.ndarray], float, Tuple[float, float]]:
        """
        Compute the Convex Hull polygon, surface area (m²), and centroid for a set of player positions.
        """
        if len(positions_m) == 0:
            return None, 0.0, (52.5, 34.0)

        centroid = (float(np.mean(positions_m[:, 0])), float(np.mean(positions_m[:, 1])))

        if len(positions_m) < 3:
            return None, 0.0, (round(centroid[0], 1), round(centroid[1], 1))

        try:
            hull = ConvexHull(positions_m)
            hull_vertices = positions_m[hull.vertices]
            area_m2 = float(hull.volume)  # In 2D, volume equals surface area in m²
            return hull_vertices, round(area_m2, 1), (round(centroid[0], 1), round(centroid[1], 1))
        except Exception:
            return None, 0.0, (round(centroid[0], 1), round(centroid[1], 1))

    def compute_voronoi_space_control(
        self,
        team_a_pos: np.ndarray,
        team_b_pos: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Calculate Voronoi pitch partition and space dominance percentages.
        """
        if len(team_a_pos) == 0 and len(team_b_pos) == 0:
            return 50.0, 50.0, np.zeros(self.grid_x.shape, dtype=np.int32)
        if len(team_a_pos) == 0:
            return 0.0, 100.0, np.ones(self.grid_x.shape, dtype=np.int32)
        if len(team_b_pos) == 0:
            return 100.0, 0.0, np.zeros(self.grid_x.shape, dtype=np.int32)

        # Vectorized Euclidean distance from each grid cell to closest Team A player
        diff_a = self.grid_points[:, np.newaxis, :] - team_a_pos[np.newaxis, :, :]
        dist_sq_a = np.min(np.sum(diff_a ** 2, axis=2), axis=1)

        # Distance to closest Team B player
        diff_b = self.grid_points[:, np.newaxis, :] - team_b_pos[np.newaxis, :, :]
        dist_sq_b = np.min(np.sum(diff_b ** 2, axis=2), axis=1)

        team_a_cells = dist_sq_a <= dist_sq_b
        pct_a = (np.sum(team_a_cells) / self.total_cells) * 100.0
        pct_b = 100.0 - pct_a

        voronoi_map = (~team_a_cells).astype(np.int32).reshape(self.grid_x.shape)
        return round(pct_a, 1), round(pct_b, 1), voronoi_map

    def analyze_frame(
        self,
        positions_m: np.ndarray,
        team_ids: np.ndarray,
    ) -> TacticalSpatialResult:
        """
        Compute full spatial tactical analytics for a frame.
        Includes Outfield players (0, 1) and Goalkeepers (3, 4).
        """
        if len(positions_m) == 0 or len(team_ids) == 0:
            return TacticalSpatialResult()

        # Team A: Outfield (0) + Goalkeeper (3)
        mask_a = np.isin(team_ids, [0, 3])
        # Team B: Outfield (1) + Goalkeeper (4)
        mask_b = np.isin(team_ids, [1, 4])

        team_a_pos = positions_m[mask_a]
        team_b_pos = positions_m[mask_b]

        # 1. Team Convex Hulls & Compactness
        hull_a, area_a, centroid_a = self.compute_team_hull(team_a_pos)
        hull_b, area_b, centroid_b = self.compute_team_hull(team_b_pos)

        # 2. Voronoi Pitch Space Control
        pct_a, pct_b, voronoi_map = self.compute_voronoi_space_control(team_a_pos, team_b_pos)

        return TacticalSpatialResult(
            team_a_control_pct=pct_a,
            team_b_control_pct=pct_b,
            team_a_hull=hull_a,
            team_b_hull=hull_b,
            team_a_area_m2=area_a,
            team_b_area_m2=area_b,
            team_a_centroid_m=centroid_a,
            team_b_centroid_m=centroid_b,
            voronoi_grid=voronoi_map,
        )
