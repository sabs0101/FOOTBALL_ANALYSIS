"""
Unit Tests for Player Re-Identification (Re-ID) Engine (Milestone 10).
Validates multi-zone spatial appearance embeddings, L2 normalization,
EMA gallery maintenance, Hungarian cross-cut matching, and team constraint enforcement.
"""

import cv2
import numpy as np
import pytest

from src.detection.detector import DetectionResult
from src.tracking.reid import (
    PlayerReID,
    ReIDFeatureExtractor,
    ReIDGallery,
    ReIDMatcher,
    ReIDMatchResult,
)
from src.tracking.tracker import PlayerTracker


def create_synthetic_player_crop(jersey_bgr: tuple, shorts_bgr: tuple, height: int = 100, width: int = 40) -> np.ndarray:
    """Generate a synthetic player crop with specified jersey and shorts colors."""
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    # Zone 1: Jersey (y: 10% to 45%)
    crop[int(0.10 * height) : int(0.45 * height), :] = jersey_bgr
    # Zone 2: Shorts (y: 45% to 70%)
    crop[int(0.45 * height) : int(0.70 * height), :] = shorts_bgr
    # Zone 3: Socks (y: 70% to 95%)
    crop[int(0.70 * height) : int(0.95 * height), :] = jersey_bgr
    return crop


def create_synthetic_frame_with_players():
    """Create a synthetic frame with 2 players (one white jersey, one neon green)."""
    frame = np.full((720, 1280, 3), (40, 140, 50), dtype=np.uint8)  # Grass

    # Player 1 (Team A - White kit) at [100, 200, 140, 300]
    p1_crop = create_synthetic_player_crop(jersey_bgr=(240, 240, 240), shorts_bgr=(20, 20, 20), height=100, width=40)
    frame[200:300, 100:140] = p1_crop

    # Player 2 (Team B - Neon Green kit) at [300, 200, 340, 300]
    p2_crop = create_synthetic_player_crop(jersey_bgr=(30, 220, 70), shorts_bgr=(240, 240, 240), height=100, width=40)
    frame[200:300, 300:340] = p2_crop

    return frame


def test_reid_feature_extractor_dimension_and_norm():
    extractor = ReIDFeatureExtractor(embedding_dim=128)
    frame = create_synthetic_frame_with_players()
    box = np.array([100, 200, 140, 300], dtype=np.float32)

    feat = extractor.extract(frame, box)
    assert feat.shape == (128,)
    assert feat.dtype == np.float32
    # Check L2 unit length
    norm = np.linalg.norm(feat)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_reid_similarity_on_same_player_vs_different_player():
    extractor = ReIDFeatureExtractor(embedding_dim=128)
    frame = create_synthetic_frame_with_players()

    box1 = np.array([100, 200, 140, 300], dtype=np.float32)  # Team A (White)
    box2 = np.array([300, 200, 340, 300], dtype=np.float32)  # Team B (Green)

    # Shifted crop of same Team A player
    box1_shifted = np.array([105, 202, 145, 302], dtype=np.float32)

    f1 = extractor.extract(frame, box1)
    f1_shifted = extractor.extract(frame, box1_shifted)
    f2 = extractor.extract(frame, box2)

    sim_same = float(np.dot(f1, f1_shifted))
    sim_diff = float(np.dot(f1, f2))

    # Same player crop should have very high cosine similarity (> 0.85)
    assert sim_same > 0.85
    # Different kit color should have significantly lower similarity
    assert sim_diff < sim_same - 0.20


def test_reid_gallery_ema_updates():
    gallery = ReIDGallery(ema_alpha=0.80, max_age_frames=100)
    feat1 = np.random.randn(128).astype(np.float32)
    feat1 /= np.linalg.norm(feat1)

    gallery.update_track(track_id=10, feature=feat1, team_id=0, role_id=0, pitch_pos=(25.0, 10.0), frame_idx=1)
    assert 10 in gallery.features
    assert np.allclose(gallery.features[10], feat1)
    assert gallery.team_ids[10] == 0
    assert gallery.last_seen_frames[10] == 1

    # Second observation: check EMA update
    feat2 = np.random.randn(128).astype(np.float32)
    feat2 /= np.linalg.norm(feat2)
    gallery.update_track(track_id=10, feature=feat2, frame_idx=5)

    expected = 0.80 * feat1 + 0.20 * feat2
    expected /= np.linalg.norm(expected)
    assert np.allclose(gallery.features[10], expected, atol=1e-4)
    assert gallery.last_seen_frames[10] == 5


def test_reid_hungarian_matching_with_team_constraints():
    reid = PlayerReID(distance_threshold=0.45)
    frame = create_synthetic_frame_with_players()

    boxes = np.array([
        [100, 200, 140, 300],  # Team A (White)
        [300, 200, 340, 300],  # Team B (Green)
    ], dtype=np.float32)

    # Initialize gallery with known tracks: TID 10 (Team A), TID 20 (Team B)
    tids = np.array([10, 20], dtype=int)
    team_ids = np.array([0, 1], dtype=int)
    reid.update_gallery(frame, boxes, tids, team_ids=team_ids, frame_idx=0)

    # Simulate reappearance of players in new positions
    new_boxes = np.array([
        [120, 210, 160, 310],  # Team A player shifted
        [320, 210, 360, 310],  # Team B player shifted
    ], dtype=np.float32)

    match_res = reid.reassociate(
        frame=frame,
        boxes=new_boxes,
        candidate_teams=[0, 1],
        frame_idx=10,
    )

    assert match_res.reassigned_count == 2
    assert match_res.matches[0] == 10  # Match index 0 back to persistent ID 10
    assert match_res.matches[1] == 20  # Match index 1 back to persistent ID 20


def test_player_tracker_cut_and_reid_persistence():
    tracker = PlayerTracker()
    reid = PlayerReID(distance_threshold=0.45)
    frame = create_synthetic_frame_with_players()

    boxes = np.array([
        [100, 200, 140, 300],
        [300, 200, 340, 300],
    ], dtype=np.float32)

    # Frame 0: Normal tracking
    det0 = DetectionResult(
        xyxy=boxes,
        confidences=np.array([0.9, 0.9], dtype=np.float32),
        class_ids=np.array([0, 0], dtype=int),
        class_names=["person", "person"],
        frame_idx=0,
    )
    res0 = tracker.update(det0, is_cut=False, reid=reid, frame=frame, candidate_teams=[0, 1])
    assert len(res0.tracker_ids) == 2
    initial_tids = res0.tracker_ids.copy()
    assert initial_tids[0] >= 0
    assert initial_tids[1] >= 0

    # Frame 1: Camera Cut occurs -> Tracker resets Kalman and uses ReID to preserve track IDs
    det1 = DetectionResult(
        xyxy=boxes + 5.0,
        confidences=np.array([0.9, 0.9], dtype=np.float32),
        class_ids=np.array([0, 0], dtype=int),
        class_names=["person", "person"],
        frame_idx=1,
    )
    res1 = tracker.update(det1, is_cut=True, reid=reid, frame=frame, candidate_teams=[0, 1])

    # Persistent IDs should match initial IDs via Re-ID matching
    assert res1.tracker_ids[0] == initial_tids[0]
    assert res1.tracker_ids[1] == initial_tids[1]
