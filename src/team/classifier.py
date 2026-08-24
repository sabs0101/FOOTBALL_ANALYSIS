"""
Automated Team, Goalkeeper, Referee, and Coach Identification (Polished & Hardened).
Accurately separates:
- Team A Outfield [A] (White kits)
- Team B Outfield [B] (Neon Green kits)
- Team Goalkeepers [A-GK] and [B-GK] (Furthest defensive anchors in goal areas)
- Match Officials [REF] (Outfield referee / touchline linesmen)
- Technical Dugout Staff [COACH]
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class TeamResult:
    """
    Container for team classification outputs across detections in a frame.

    Attributes:
        team_ids: Array of shape (N,) with integer role indices:
                  0 = Team A Outfield
                  1 = Team B Outfield
                  2 = Referee / Linesman
                  3 = Team A Goalkeeper [A-GK]
                  4 = Team B Goalkeeper [B-GK]
                  5 = Coach / Dugout Staff
                 -1 = Football
        team_names: List of string labels ("Team A", "Team B", "Referee", "Team A GK", "Team B GK", "Coach", "Ball").
        team_colors: List of BGR color tuples for rendering each bounding box/dot.
        team_counts: Dict mapping team label to number of active players.
    """
    team_ids: np.ndarray
    team_names: List[str]
    team_colors: List[Tuple[int, int, int]]
    team_counts: Dict[str, int]


class TeamClassifier:
    """
    AI Football Kit & Role Classifier.
    Combines upper-torso jersey color extraction with pitch spatial kinematics
    and temporal majority voting to robustly separate Goalkeepers, Outfield Players, Referees, and Coaches.
    """

    def __init__(
        self,
        n_teams: int = 2,
        history_window: int = 15,
        referee_dist_threshold: float = 85.0,
    ):
        self.n_teams = n_teams
        self.history_window = history_window
        self.referee_dist_threshold = referee_dist_threshold

        # High-contrast tactical role palette in BGR
        self.palette = {
            "team_a": (240, 240, 240),       # Team A Outfield: Crisp White (BGR)
            "team_b": (30, 220, 70),         # Team B Outfield: Vibrant Neon Green (BGR)
            "referee": (0, 215, 255),        # Referee: Amber/Gold (BGR)
            "gk_a": (0, 140, 255),           # Team A Goalkeeper: Vibrant Orange (BGR)
            "gk_b": (220, 50, 200),          # Team B Goalkeeper: Magenta/Purple (BGR)
            "coach": (100, 100, 110),        # Coach / Staff: Sleek Dark Slate (BGR)
            "ball": (0, 60, 255),            # Ball: Red
        }

        # Persistent track ID vote histories
        self.track_votes: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_window))

    def extract_torso_hsv(self, frame: np.ndarray, box: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Extract the (Hue, Saturation, Value) profile of the upper torso.
        """
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = frame.shape

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 8:
            return None

        # Isolate central upper torso
        tx1 = int(x1 + 0.20 * bw)
        tx2 = int(x2 - 0.20 * bw)
        ty1 = int(y1 + 0.15 * bh)
        ty2 = int(y1 + 0.50 * bh)

        torso = frame[ty1:ty2, tx1:tx2]
        if torso.size == 0 or torso.shape[0] < 2 or torso.shape[1] < 2:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = np.median(hsv.reshape(-1, 3), axis=0)
        return float(mean_h), float(mean_s), float(mean_v)

    def predict_role(
        self,
        hsv_profile: Optional[Tuple[float, float, float]],
        pos_m: Optional[Tuple[float, float]] = None,
        box_y2: Optional[float] = None,
        is_deepest_a: bool = False,
        is_deepest_b: bool = False,
    ) -> int:
        """
        Classify a person into:
        0: Team A Outfield
        1: Team B Outfield
        2: Referee
        3: Team A Goalkeeper [A-GK]
        4: Team B Goalkeeper [B-GK]
        5: Coach / Staff
        """
        # 1. Coach / Dugout Staff Detection: Positioned below the bottom pitch boundary
        if box_y2 is not None and box_y2 > 880.0:
            return 5  # Coach / Staff

        if hsv_profile is None:
            return 0

        hue, sat, val = hsv_profile
        xm, ym = pos_m if pos_m is not None else (52.5, 34.0)

        # 2. Goalkeeper Detection: Deepest defensive anchor positioned in the defensive 32m of the pitch
        if is_deepest_a and xm < 32.0 and (10.0 <= ym <= 58.0):
            return 3  # Team A Goalkeeper [A-GK]
        if is_deepest_b and xm > 73.0 and (10.0 <= ym <= 58.0):
            return 4  # Team B Goalkeeper [B-GK]

        # 3. Referee Detection: Central outfield or touchlines with dark / amber kit
        if val < 110:
            return 2  # Referee

        # 4. Team B Outfield (Neon Green): High saturation green kit
        if 40 <= hue <= 72 and sat >= 68:
            return 1  # Team B Outfield

        # 5. Team A Outfield (White): Low saturation light kit
        if sat < 70 and val >= 120:
            return 0  # Team A Outfield

        if 35 <= hue <= 75 and sat >= 55:
            return 1
        return 0

    def update_track_role(self, track_id: int, raw_role_id: int) -> int:
        """
        Apply temporal sliding window majority voting to prevent frame flickering.
        """
        if track_id < 0:
            return raw_role_id

        votes = self.track_votes[track_id]
        votes.append(raw_role_id)

        counts: Dict[int, int] = defaultdict(int)
        for v in votes:
            counts[v] += 1

        return max(counts.keys(), key=lambda k: counts[k])

    def classify_frame(
        self,
        frame: np.ndarray,
        detections: Any,
        positions_m: Optional[np.ndarray] = None,
    ) -> TeamResult:
        """
        Classify all persons in the current frame into Teams, Goalkeepers, Referees, and Coaches.
        """
        num_items = len(detections.xyxy)
        tracker_ids = getattr(detections, "tracker_ids", None)

        team_ids = []
        team_names = []
        team_colors = []
        counts = {
            "Team A": 0,
            "Team B": 0,
            "Referee": 0,
            "Team A GK": 0,
            "Team B GK": 0,
            "Coach": 0,
        }

        # Identify deepest players (Goalkeepers) across pitch
        deepest_a_idx = -1
        deepest_b_idx = -1

        if positions_m is not None and len(positions_m) > 0:
            valid_p_indices = [
                i for i in range(len(positions_m))
                if detections.class_ids[i] == 0 and -2.0 <= positions_m[i, 0] <= 107.0 and -2.0 <= positions_m[i, 1] <= 70.0
            ]
            if valid_p_indices:
                p_xs = [positions_m[i, 0] for i in valid_p_indices]
                min_x_idx = valid_p_indices[int(np.argmin(p_xs))]
                max_x_idx = valid_p_indices[int(np.argmax(p_xs))]

                if positions_m[min_x_idx, 0] < 32.0:
                    deepest_a_idx = min_x_idx
                if positions_m[max_x_idx, 0] > 73.0:
                    deepest_b_idx = max_x_idx

        for i in range(num_items):
            cid = detections.class_ids[i]
            tid = tracker_ids[i] if (tracker_ids is not None and i < len(tracker_ids)) else -1

            if cid == 32:  # Ball
                team_ids.append(-1)
                team_names.append("Ball")
                team_colors.append(self.palette["ball"])
                continue

            box = detections.xyxy[i]
            pos = (float(positions_m[i, 0]), float(positions_m[i, 1])) if (positions_m is not None and i < len(positions_m)) else None
            hsv_profile = self.extract_torso_hsv(frame, box)

            raw_role = self.predict_role(
                hsv_profile,
                pos_m=pos,
                box_y2=box[3],
                is_deepest_a=(i == deepest_a_idx),
                is_deepest_b=(i == deepest_b_idx),
            )
            final_role = self.update_track_role(tid, raw_role)
            team_ids.append(final_role)

            if final_role == 0:
                name = "Team A"
                color = self.palette["team_a"]
                counts["Team A"] += 1
            elif final_role == 1:
                name = "Team B"
                color = self.palette["team_b"]
                counts["Team B"] += 1
            elif final_role == 2:
                name = "Referee"
                color = self.palette["referee"]
                counts["Referee"] += 1
            elif final_role == 3:
                name = "Team A GK"
                color = self.palette["gk_a"]
                counts["Team A GK"] += 1
                counts["Team A"] += 1
            elif final_role == 4:
                name = "Team B GK"
                color = self.palette["gk_b"]
                counts["Team B GK"] += 1
                counts["Team B"] += 1
            else:
                name = "Coach"
                color = self.palette["coach"]
                counts["Coach"] += 1

            team_names.append(name)
            team_colors.append(color)

        return TeamResult(
            team_ids=np.array(team_ids, dtype=np.int32),
            team_names=team_names,
            team_colors=team_colors,
            team_counts=counts,
        )
