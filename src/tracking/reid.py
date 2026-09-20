"""
Player Re-Identification (Re-ID) Module for Football Analysis (Milestone 10).
Extracts multi-zone spatial appearance embeddings (jersey, shorts, socks) and maintains
a long-term identity gallery with Hungarian cosine matching to re-identify players across
camera cuts, scene transitions, and track dropouts.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class ReIDMatchResult:
    """
    Container for Re-ID cross-cut matching outputs.

    Attributes:
        matches: Dict mapping current detection index -> matched persistent track ID.
        unmatched_detections: List of detection indices that did not match any gallery track.
        unmatched_tracks: List of gallery track IDs that were not matched.
        match_scores: Dict mapping detection index -> similarity score [0.0 - 1.0].
        reassigned_count: Number of successful identity reassignments.
    """
    matches: Dict[int, int]
    unmatched_detections: List[int]
    unmatched_tracks: List[int]
    match_scores: Dict[int, float]
    reassigned_count: int


class ReIDFeatureExtractor:
    """
    Multi-Zone Spatial Appearance & Color Feature Extractor.
    Extracts L2-normalized embeddings from vertical player body slices:
    - Upper Torso (Jersey): 10% - 45% of height
    - Lower Torso (Shorts): 45% - 70% of height
    - Lower Body (Socks): 70% - 95% of height
    """

    def __init__(
        self,
        h_bins: int = 16,
        s_bins: int = 8,
        v_bins: int = 8,
        embedding_dim: int = 128,
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.embedding_dim = embedding_dim

    def extract_crop_feature(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract normalized color and gradient signature from an image patch.
        """
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return np.zeros(32, dtype=np.float32)

        # 1. Color in HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

        # 2. Color in LAB
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        mean_lab = np.mean(lab, axis=(0, 1)) / 255.0
        std_lab = np.std(lab, axis=(0, 1)) / 255.0

        vec = np.concatenate([h_hist, s_hist, v_hist, mean_lab, std_lab])
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.astype(np.float32)

    def extract(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Extract a comprehensive multi-zone feature vector from a player bounding box.

        Args:
            frame: Full BGR frame (H, W, 3).
            box: Bounding box [x1, y1, x2, y2].

        Returns:
            L2-normalized feature embedding vector of shape (embedding_dim,).
        """
        h_img, w_img, _ = frame.shape
        x1 = max(0, min(w_img - 1, int(box[0])))
        y1 = max(0, min(h_img - 1, int(box[1])))
        x2 = max(0, min(w_img, int(box[2])))
        y2 = max(0, min(h_img, int(box[3])))

        bw = x2 - x1
        bh = y2 - y1

        if bw < 6 or bh < 12:
            dummy = np.zeros(self.embedding_dim, dtype=np.float32)
            dummy[0] = 1.0
            return dummy

        # Slice anatomical zones with horizontal margin to exclude grass background
        margin_x = int(0.15 * bw)
        sx1 = x1 + margin_x
        sx2 = x2 - margin_x

        # Zone 1: Upper Torso (Jersey)
        z1_y1 = y1 + int(0.10 * bh)
        z1_y2 = y1 + int(0.45 * bh)
        crop_jersey = frame[z1_y1:z1_y2, sx1:sx2]

        # Zone 2: Lower Torso (Shorts)
        z2_y1 = y1 + int(0.45 * bh)
        z2_y2 = y1 + int(0.70 * bh)
        crop_shorts = frame[z2_y1:z2_y2, sx1:sx2]

        # Zone 3: Lower Body (Socks / Cleats)
        z3_y1 = y1 + int(0.70 * bh)
        z3_y2 = y1 + int(0.95 * bh)
        crop_socks = frame[z3_y1:z3_y2, sx1:sx2]

        f_jersey = self.extract_crop_feature(crop_jersey)
        f_shorts = self.extract_crop_feature(crop_shorts)
        f_socks = self.extract_crop_feature(crop_socks)

        # Spatial aspect ratio & relative height
        aspect_ratio = np.array([float(bw) / float(bh), float(bh) / float(h_img)], dtype=np.float32)

        raw_feature = np.concatenate([f_jersey, f_shorts, f_socks, aspect_ratio])

        # Pad or project to embedding_dim
        if len(raw_feature) < self.embedding_dim:
            padded = np.zeros(self.embedding_dim, dtype=np.float32)
            padded[: len(raw_feature)] = raw_feature
            feature = padded
        else:
            feature = raw_feature[: self.embedding_dim].astype(np.float32)

        # L2 Normalization
        norm = np.linalg.norm(feature)
        if norm > 1e-6:
            feature = feature / norm
        else:
            feature[0] = 1.0

        return feature


class ReIDGallery:
    """
    Persistent Long-Term Identity Gallery for Football Players.
    Maintains rolling exponential moving average (EMA) embeddings and metadata
    (team, role, position) for persistent tracks across the entire match.
    """

    def __init__(self, ema_alpha: float = 0.85, max_age_frames: int = 300):
        self.ema_alpha = ema_alpha
        self.max_age_frames = max_age_frames

        self.features: Dict[int, np.ndarray] = {}
        self.team_ids: Dict[int, int] = {}
        self.roles: Dict[int, int] = {}
        self.pitch_positions: Dict[int, Tuple[float, float]] = {}
        self.last_seen_frames: Dict[int, int] = {}
        self.sample_counts: Dict[int, int] = defaultdict(int)

    def update_track(
        self,
        track_id: int,
        feature: np.ndarray,
        team_id: Optional[int] = None,
        role_id: Optional[int] = None,
        pitch_pos: Optional[Tuple[float, float]] = None,
        frame_idx: int = 0,
    ):
        """
        Update the gallery entry for a track with a new appearance observation.
        """
        if track_id < 0:
            return

        if track_id not in self.features:
            self.features[track_id] = feature.copy()
        else:
            # Exponential Moving Average update
            updated = self.ema_alpha * self.features[track_id] + (1.0 - self.ema_alpha) * feature
            norm = np.linalg.norm(updated)
            if norm > 1e-6:
                updated = updated / norm
            self.features[track_id] = updated

        if team_id is not None and team_id >= 0:
            self.team_ids[track_id] = team_id

        if role_id is not None and role_id >= 0:
            self.roles[track_id] = role_id

        if pitch_pos is not None:
            self.pitch_positions[track_id] = pitch_pos

        self.last_seen_frames[track_id] = frame_idx
        self.sample_counts[track_id] += 1

    def get_active_tracks(self, current_frame_idx: int) -> List[int]:
        """
        Retrieve list of track IDs that have been observed recently.
        """
        active = []
        for tid, last_seen in self.last_seen_frames.items():
            if (current_frame_idx - last_seen) <= self.max_age_frames:
                active.append(tid)
        return active


class ReIDMatcher:
    """
    Hungarian Identity Matcher across Camera Cuts and Track Dropouts.
    Solves global bipartite matching using cosine appearance distance,
    team/role consistency penalties, and pitch spatial priors.
    """

    def __init__(
        self,
        distance_threshold: float = 0.42,
        pos_weight: float = 0.20,
    ):
        self.distance_threshold = distance_threshold
        self.pos_weight = pos_weight

    def match(
        self,
        candidate_features: List[np.ndarray],
        candidate_teams: Optional[List[int]],
        candidate_roles: Optional[List[int]],
        candidate_positions: Optional[List[Optional[Tuple[float, float]]]],
        gallery: ReIDGallery,
        current_frame_idx: int,
    ) -> ReIDMatchResult:
        """
        Match detected candidate players against known gallery tracks.

        Returns:
            ReIDMatchResult containing optimal ID assignments.
        """
        gallery_tids = gallery.get_active_tracks(current_frame_idx)
        n_candidates = len(candidate_features)
        n_gallery = len(gallery_tids)

        if n_candidates == 0 or n_gallery == 0:
            return ReIDMatchResult(
                matches={},
                unmatched_detections=list(range(n_candidates)),
                unmatched_tracks=gallery_tids,
                match_scores={},
                reassigned_count=0,
            )

        # Build Cost Matrix (n_candidates x n_gallery)
        cost_matrix = np.full((n_candidates, n_gallery), 10.0, dtype=np.float32)

        for i, c_feat in enumerate(candidate_features):
            c_team = candidate_teams[i] if candidate_teams is not None else -1
            c_role = candidate_roles[i] if candidate_roles is not None else -1
            c_pos = candidate_positions[i] if candidate_positions is not None else None

            for j, g_tid in enumerate(gallery_tids):
                g_feat = gallery.features[g_tid]
                g_team = gallery.team_ids.get(g_tid, -1)
                g_role = gallery.roles.get(g_tid, -1)
                g_pos = gallery.pitch_positions.get(g_tid, None)

                # Hard Constraint: Team Incompatibility
                if c_team >= 0 and g_team >= 0 and c_team != g_team:
                    cost_matrix[i, j] = 99.0
                    continue

                # Hard Constraint: Role Incompatibility (e.g. Outfield vs Goalkeeper vs Ref)
                if c_role >= 0 and g_role >= 0 and c_role != g_role:
                    cost_matrix[i, j] = 99.0
                    continue

                # 1. Cosine Distance
                sim = float(np.dot(c_feat, g_feat))
                sim = np.clip(sim, -1.0, 1.0)
                d_app = 1.0 - sim

                # 2. Pitch Position Proximity (if available)
                d_pos = 0.0
                if c_pos is not None and g_pos is not None:
                    dist_m = float(np.sqrt((c_pos[0] - g_pos[0]) ** 2 + (c_pos[1] - g_pos[1]) ** 2))
                    d_pos = min(1.0, dist_m / 40.0)

                cost = d_app + self.pos_weight * d_pos
                cost_matrix[i, j] = cost

        # Solve Linear Sum Assignment (Hungarian Algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches: Dict[int, int] = {}
        match_scores: Dict[int, float] = {}
        matched_candidates: Set[int] = set()
        matched_gallery: Set[int] = set()

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost <= self.distance_threshold:
                g_tid = gallery_tids[c]
                matches[r] = g_tid
                match_scores[r] = float(1.0 - cost)
                matched_candidates.add(r)
                matched_gallery.add(g_tid)

        unmatched_dets = [i for i in range(n_candidates) if i not in matched_candidates]
        unmatched_tracks = [tid for tid in gallery_tids if tid not in matched_gallery]

        return ReIDMatchResult(
            matches=matches,
            unmatched_detections=unmatched_dets,
            unmatched_tracks=unmatched_tracks,
            match_scores=match_scores,
            reassigned_count=len(matches),
        )


class PlayerReID:
    """
    Unified Player Re-Identification Pipeline.
    Encapsulates feature extraction, gallery maintenance, and cross-cut Hungarian re-association.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        distance_threshold: float = 0.42,
        ema_alpha: float = 0.85,
        max_age_frames: int = 300,
    ):
        self.extractor = ReIDFeatureExtractor(embedding_dim=embedding_dim)
        self.gallery = ReIDGallery(ema_alpha=ema_alpha, max_age_frames=max_age_frames)
        self.matcher = ReIDMatcher(distance_threshold=distance_threshold)
        self.total_reassignments: int = 0

    def extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """
        Extract Re-ID embeddings for a batch of bounding boxes.
        """
        features = []
        for box in boxes:
            features.append(self.extractor.extract(frame, box))
        return features

    def update_gallery(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        track_ids: np.ndarray,
        team_ids: Optional[np.ndarray] = None,
        role_ids: Optional[np.ndarray] = None,
        pitch_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
        frame_idx: int = 0,
    ):
        """
        Update gallery embeddings for currently tracked players.
        """
        for i, tid in enumerate(track_ids):
            if tid >= 0 and i < len(boxes):
                feat = self.extractor.extract(frame, boxes[i])
                t_id = team_ids[i] if team_ids is not None and i < len(team_ids) else None
                r_id = role_ids[i] if role_ids is not None and i < len(role_ids) else None
                p_pos = pitch_positions[i] if pitch_positions is not None and i < len(pitch_positions) else None
                self.gallery.update_track(
                    track_id=tid,
                    feature=feat,
                    team_id=t_id,
                    role_id=r_id,
                    pitch_pos=p_pos,
                    frame_idx=frame_idx,
                )

    def reassociate(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        candidate_teams: Optional[List[int]] = None,
        candidate_roles: Optional[List[int]] = None,
        candidate_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
        frame_idx: int = 0,
    ) -> ReIDMatchResult:
        """
        Perform cross-cut Re-ID association for candidate detections against the gallery.
        """
        features = self.extract_features(frame, boxes)
        result = self.matcher.match(
            candidate_features=features,
            candidate_teams=candidate_teams,
            candidate_roles=candidate_roles,
            candidate_positions=candidate_positions,
            gallery=self.gallery,
            current_frame_idx=frame_idx,
        )
        self.total_reassignments += result.reassigned_count
        return result
