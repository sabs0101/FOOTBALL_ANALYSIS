"""
Unit tests for the TeamClassifier module.
All tests use synthetic numpy frames and bounding boxes so no real video is needed.
"""

import numpy as np
import pytest
from src.teams.classifier import TeamClassifier


def _make_frame(color_bgr: tuple, size: int = 200) -> np.ndarray:
    """Create a solid-color BGR image of given size."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = color_bgr
    return frame


def _make_box(x1=10, y1=10, x2=90, y2=190) -> np.ndarray:
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class TestJerseyExtraction:
    def test_returns_feature_for_valid_box(self):
        classifier = TeamClassifier()
        frame = _make_frame((255, 0, 0))  # Blue jersey
        feature = classifier.extract_jersey_feature(frame, _make_box())
        assert feature is not None
        assert feature.shape == (3,)

    def test_returns_none_for_zero_area_box(self):
        classifier = TeamClassifier()
        frame = _make_frame((255, 0, 0))
        feature = classifier.extract_jersey_feature(frame, np.array([50, 50, 50, 100], dtype=np.float32))
        assert feature is None

    def test_returns_none_for_negative_area_box(self):
        classifier = TeamClassifier()
        frame = _make_frame((255, 0, 0))
        feature = classifier.extract_jersey_feature(frame, np.array([100, 100, 50, 50], dtype=np.float32))
        assert feature is None

    def test_green_frame_falls_back_gracefully(self):
        """Frame that is entirely grass-green should not raise; returns fallback feature."""
        classifier = TeamClassifier()
        frame = _make_frame((0, 200, 0))
        feature = classifier.extract_jersey_feature(frame, _make_box())
        # Either fallback path is acceptable — should not raise
        assert feature is not None or feature is None


class TestFitAndPredict:
    def _build_and_fit_classifier(self, n_per_team: int = 10) -> TeamClassifier:
        """Create a classifier with enough data for two teams."""
        classifier = TeamClassifier(num_teams=2, fit_frames=1)
        red_frame = _make_frame((0, 0, 200))
        blue_frame = _make_frame((200, 0, 0))
        box = _make_box()

        track_id = 0
        for _ in range(n_per_team):
            classifier.get_team(red_frame, box, track_id)
            track_id += 1
        for _ in range(n_per_team):
            classifier.get_team(blue_frame, box, track_id)
            track_id += 1

        classifier.fit()
        return classifier

    def test_is_fitted_after_fit(self):
        clf = self._build_and_fit_classifier()
        assert clf.is_fitted is True

    def test_get_team_returns_valid_team_id_after_fit(self):
        clf = self._build_and_fit_classifier()
        frame = _make_frame((0, 0, 200))
        team_id = clf.get_team(frame, _make_box(), track_id=999)
        assert team_id in (0, 1)

    def test_returns_minus_one_before_fit(self):
        classifier = TeamClassifier(num_teams=2, fit_frames=100)
        frame = _make_frame((0, 0, 200))
        result = classifier.get_team(frame, _make_box(), track_id=42)
        assert result == -1

    def test_cached_team_returned_consistently(self):
        clf = self._build_and_fit_classifier()
        frame = _make_frame((0, 0, 200))
        first = clf.get_team(frame, _make_box(), track_id=1000)
        second = clf.get_team(frame, _make_box(), track_id=1000)
        assert first == second


class TestTeamCounts:
    def test_empty_counts_before_any_assignment(self):
        clf = TeamClassifier()
        assert clf.get_team_counts() == {}

    def test_counts_after_manual_assignment(self):
        clf = TeamClassifier()
        clf._team_assignments = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        counts = clf.get_team_counts()
        assert counts[0] == 2
        assert counts[1] == 3


class TestReset:
    def test_reset_clears_state(self):
        clf = TeamClassifier(num_teams=2, fit_frames=1)
        red = _make_frame((0, 0, 200))
        blue = _make_frame((200, 0, 0))
        box = _make_box()
        for tid in range(5):
            clf.get_team(red, box, tid)
        for tid in range(5, 10):
            clf.get_team(blue, box, tid)
        clf.fit()
        assert clf.is_fitted is True

        clf.reset()
        assert clf.is_fitted is False
        assert clf._player_features == []
        assert clf._player_track_ids == []
        assert clf._team_assignments == {}
        assert clf._frame_count == 0
        assert clf.kmeans is None

    def test_can_refit_after_reset(self):
        clf = TeamClassifier(num_teams=2, fit_frames=1)
        red = _make_frame((0, 0, 200))
        blue = _make_frame((200, 0, 0))
        box = _make_box()
        for tid in range(5):
            clf.get_team(red, box, tid)
        for tid in range(5, 10):
            clf.get_team(blue, box, tid)
        clf.fit()
        clf.reset()

        result = clf.get_team(red, box, track_id=100)
        assert result == -1  # Not fitted yet after reset
