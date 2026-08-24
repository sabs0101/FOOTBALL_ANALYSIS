"""
Unit tests for Advanced Tactical Analytics, Heatmaps & Voronoi Space Control (Milestone 7).
"""

import numpy as np
import pytest
from src.tactics.heatmaps import HeatmapGenerator
from src.tactics.spatial import SpatialControl, TacticalSpatialResult


def test_spatial_control_initialization():
    spatial = SpatialControl(pitch_length_m=105.0, pitch_width_m=68.0, grid_resolution_m=1.0)
    assert spatial.length_m == 105.0
    assert spatial.width_m == 68.0
    assert spatial.total_cells == 105 * 68


def test_team_convex_hull_calculation():
    """
    Test Convex Hull on a synthetic 20m x 20m square of players.
    Area should equal exactly 400.0 m² and centroid at (30.0, 30.0).
    """
    spatial = SpatialControl()

    # 4 corners of a 20m x 20m square
    pts = np.array([
        [20.0, 20.0],
        [40.0, 20.0],
        [40.0, 40.0],
        [20.0, 40.0],
    ], dtype=np.float32)

    hull, area_m2, centroid = spatial.compute_team_hull(pts)

    assert hull is not None
    assert len(hull) == 4
    assert pytest.approx(area_m2, rel=0.01) == 400.0
    assert pytest.approx(centroid[0], rel=0.01) == 30.0
    assert pytest.approx(centroid[1], rel=0.01) == 30.0


def test_voronoi_space_dominance_symmetric():
    """
    Test Voronoi partition with symmetric team placements:
    Team A on left half (X=25), Team B on right half (X=80).
    Territory dominance should equal ~50% each.
    """
    spatial = SpatialControl()

    team_a_pos = np.array([
        [25.0, 20.0],
        [25.0, 48.0],
    ], dtype=np.float32)

    team_b_pos = np.array([
        [80.0, 20.0],
        [80.0, 48.0],
    ], dtype=np.float32)

    pct_a, pct_b, voronoi_grid = spatial.compute_voronoi_space_control(team_a_pos, team_b_pos)

    assert pytest.approx(pct_a, abs=2.0) == 50.0
    assert pytest.approx(pct_b, abs=2.0) == 50.0
    assert voronoi_grid.shape == (68, 105)


def test_heatmap_generation():
    """
    Test 2D Gaussian density heatmap accumulation and canvas rendering.
    """
    heatmap_gen = HeatmapGenerator(grid_width=525, grid_height=340)

    # Accumulate 50 sample positions around center circle (52.5, 34.0)
    track_ids = np.array([10], dtype=np.int32)
    for _ in range(50):
        pos = np.array([[52.5 + np.random.randn() * 3.0, 34.0 + np.random.randn() * 3.0]], dtype=np.float32)
        heatmap_gen.add_positions(track_ids, pos, team_ids=np.array([0], dtype=np.int32))

    img = heatmap_gen.generate_player_heatmap(10)
    assert img.shape == (340, 525, 3)
    assert np.mean(img) > 0
