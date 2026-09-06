import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Tuple

class TeamClassifier:
    """
    Classifies players into teams based on jersey colors using KMeans clustering.
    """
    def __init__(self, num_teams: int = 2, fit_frames: int = 50):
        self.num_teams = num_teams
        self.fit_frames = fit_frames
        self.kmeans: Optional[KMeans] = None
        
        self._frame_count = 0
        self._player_features: List[np.ndarray] = []
        self._player_track_ids: List[int] = []
        self._team_assignments: Dict[int, int] = {}
        self.is_fitted = False

    def extract_jersey_feature(self, image: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts the dominant color feature from the upper half of the player's bounding box.
        Ignores green background (grass) pixels.
        """
        x1, y1, x2, y2 = map(int, box)
        # Ensure coordinates are within image bounds
        h_img, w_img = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)
        
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return None

        # Take the upper half for the jersey
        crop = image[y1 : y1 + (y2 - y1) // 2, x1 : x2]
        if crop.size == 0:
            return None

        # Convert to HSV to easily filter out green grass
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Define grass green range in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv_crop, lower_green, upper_green)
        not_green_mask = cv2.bitwise_not(mask)
        
        # Get pixels that are not green grass
        fg_pixels = crop[not_green_mask > 0]
        
        if len(fg_pixels) == 0:
            # Fallback to center region if everything is classified as grass or empty
            center_x = (x2 - x1) // 2
            center_y = (y2 - y1) // 4
            if center_x > 0 and center_y > 0:
                return crop[center_y, center_x].astype(np.float32)
            else:
                return np.array([0, 0, 0], dtype=np.float32)

        # Average color of the jersey
        mean_color = np.mean(fg_pixels, axis=0)
        return mean_color.astype(np.float32)

    def fit(self):
        """Fit the KMeans model using accumulated features."""
        if len(self._player_features) < self.num_teams:
            return # Not enough data
        
        X = np.vstack(self._player_features)
        self.kmeans = KMeans(n_clusters=self.num_teams, n_init=10, random_state=42)
        self.kmeans.fit(X)
        self.is_fitted = True
        
        # Assign existing track IDs based on fitted model
        preds = self.kmeans.predict(X)
        for track_id, team_id in zip(self._player_track_ids, preds):
            self._team_assignments[track_id] = int(team_id)

    def get_team(self, image: np.ndarray, box: np.ndarray, track_id: int) -> int:
        """
        Returns the team ID (0 or 1) for a tracked player.
        If the model is not fitted, collects data and returns -1.
        """
        # Return cached team if already assigned
        if track_id in self._team_assignments:
            return self._team_assignments[track_id]

        feature = self.extract_jersey_feature(image, box)
        if feature is None:
            return -1

        if not self.is_fitted:
            self._player_features.append(feature)
            self._player_track_ids.append(track_id)
            return -1

        # Model is fitted, predict team
        team_id = self.kmeans.predict([feature])[0]
        self._team_assignments[track_id] = int(team_id)
        return int(team_id)

    def increment_frame(self):
        """Increment frame counter and fit model if threshold reached."""
        self._frame_count += 1
        if not self.is_fitted and self._frame_count >= self.fit_frames:
            self.fit()

    def reset(self):
        """
        Reset all accumulated state.

        Useful when switching to a new video clip or match segment
        so the classifier can re-learn team colors from scratch.
        """
        self._frame_count = 0
        self._player_features = []
        self._player_track_ids = []
        self._team_assignments = {}
        self.kmeans = None
        self.is_fitted = False

    def get_team_counts(self) -> Dict[int, int]:
        """
        Return the number of tracked players assigned to each team.

        Returns:
            Dict mapping team_id -> player count, e.g. {0: 11, 1: 10}.
        """
        counts: Dict[int, int] = {}
        for team_id in self._team_assignments.values():
            counts[team_id] = counts.get(team_id, 0) + 1
        return counts
