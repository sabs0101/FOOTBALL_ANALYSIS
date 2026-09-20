# Football Analytics & Tactical Reconstruction - Project Progress Log

**Project**: AI-Based Football Broadcast Analysis and Tactical Reconstruction  
**Engineer**: Sabari Sundaresan (3rd-Year BTech CSE)  
**Mentor**: Senior ML & Computer Vision Engineer  
**Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.4, Ada Lovelace)  
**Environment**: Python 3.12 (`.venv`), PyTorch 2.6.0+cu124, Ultralytics YOLOv8, ByteTrack, Supervision, OpenCV  

---

## Roadmap & Milestones Status

| Milestone | Description | Status | Verification & Metrics |
|:---|:---|:---:|:---|
| **Milestone 1** | Player & Ball Detection + Tactical HUD | **COMPLETED** | **26.54 FPS** on 1080p, 22.0 avg players/frame |
| **Milestone 2** | Multi-Object Tracking (ByteTrack) & Persistent IDs | **COMPLETED** | **25.68 FPS** on 1080p, 19-22 active tracks, smooth trails |
| **Milestone 3** | Football Pitch & Field Line Detection | **COMPLETED** | **100% Pitch Lock**, touchline coach exclusion |
| **Milestone 4** | Pitch Keypoints $\rightarrow$ Homography $\rightarrow$ 2D Pitch Radar Model | **COMPLETED** | $3 \times 3$ Homography $H$, sub-pixel metric projection ($<0.01$ px error) |
| **Milestone 5** | Player Image $\rightarrow$ Standardized 2D Pitch Coordinates (Speed & Distance Kinematics) | **COMPLETED** | **Regression velocity**, **38.0 km/h peak sprint**, **1.92 km total distance** |
| **Milestone 6** | Automated Team & Referee Identification (Jersey Color Clustering) | **COMPLETED** | **Team A (White: 11) vs Team B (Neon Green: 10) vs Referees**, 2D color-coded minimap |
| **Milestone 7** | Tactical Metrics (Positional Heatmaps, Team Convex Hull / Compactness, Voronoi Space Control) | **COMPLETED** | **62.7% vs 37.3% Space Dominance**, **944 m² Team Compactness**, 2D Gaussian Heatmaps |
| **Milestone 8** | Ball Tracking, Trajectory Smoothing & Possession Assignment | **COMPLETED** | **Kalman State Estimator**, **67.4% vs 32.6% On-Ball Possession**, **3 Turnovers**, Comet Trail |
| **Milestone 9** | Camera Movement & Zoom Compensation (GME & CMC) | **COMPLETED** | **Lucas-Kanade Pyramidal Flow**, **RANSAC Affine Decomposition**, **Warped Kalman State CMC** |
| **Milestone 10** | Camera Cut Detection & Re-Identification across Cuts | **COMPLETED** | **Multi-Cue HSV/Edge Cut Detector**, **Multi-Zone Spatial Re-ID**, **Hungarian Matching** |
| **Milestone 11** | Basic Football Event Recognition (Passes, Shots, Interceptions, Tackles) | **COMPLETED** | **Spatial-Temporal Event FSM**, **HUD Broadcast Toasts**, **JSON Timeline Export**, **66 / 66 Unit Tests Passing** |
| **Milestone 12** | Interactive Web Dashboard & Polished Demonstration | **COMPLETED** | **Glassmorphic Web App**, Universal H.264 MP4 streaming, Live drag-and-drop match report, **72 / 72 Unit Tests Passing** |

---

## Milestone 12 Details: Interactive Web Dashboard & Polished Demonstration Integration (Completed)

### 1. What We Built
- **`web/index.html`**:
  - **Comprehensive Multi-Feature Configuration Panel**: Toggles for 2D Tactical Minimap Radar, Kalman Ball Tracking & Possession, Velocity Regression Kinematics, Voronoi Space Dominance, 2D Positional Heatmaps, Camera Motion Compensation (GME/CMC), Camera Cut & Re-ID Persistence, Match Event Recognition, and CLAHE Luminance Enhancement.
  - **Glassmorphic Hero Dashboard**: 6 responsive KPI telemetry cards (Possession Split, Turnovers, Match Events, Top Sprint Speed, Homography Lock, Camera Cuts & Re-ID Recovery).
  - **Broadcast-Annotated Video Player**: Embedded HTML5 H.264 player with hardware acceleration, seek bar, and download links.
  - **Multi-Tab Tactical Intelligence Card**: Switchable views between Team A Heatmap, Team B Heatmap, Full Match Density Map, and Minimap Radar.
  - **Player Kinematics Leaderboard**: Displays top sprint speeds, distance run in meters, squad badges, and sprint intensity alerts.
  - **Match Event Feed & Discrete Action Timeline**: Real-time event log with color-coded badges (PASS, SHOT, INTERCEPTION, TACKLE), timestamps (`MM:SS (frame)`), involved player transitions (`#19 → #15`), and ball flight dynamics.
  - **Data Export Suite**: Direct downloads for annotated video (`.mp4`), match events (`.json`), and dataset annotations (`.tsv`).
- **`web/styles.css`**:
  - Dark-mode luxury HUD aesthetic with CSS custom properties, glassmorphism blur filters, animated progress rings, responsive CSS grid layouts, and color-coded neon badges.
- **`web/app.js`**:
  - Single-page application controller with drag-and-drop video upload (`/api/upload`), asynchronous task dispatch (`/api/process`), progress polling loop (`/api/progress`), and dynamic results rendering.
- **`app.py`**:
  - Multithreaded Python HTTP server and REST API handler (`ThreadingHTTPServer`) with HTTP Byte-Range support (`206 Partial Content`) for instantaneous video seeking.
- **`tests/test_dashboard_api.py`**:
  - 6 comprehensive unit tests covering static asset delivery, task triggering, progress polling states, and HTTP byte-range video streaming. Complete test suite: **72 / 72 tests passing**.

---

## Milestone 11 Details: Basic Football Event Recognition (Completed)

### 1. What We Built
- **`src/analytics/events.py`**:
  - `MatchEvent`: Structured representation of discrete events (`event_id`, `event_type`, `frame_idx`, `timestamp_s`, `team_id`, `team_name`, `primary_player_id`, `secondary_player_id`, `start_pos_m`, `end_pos_m`, `speed_kmh`, `distance_m`, `is_successful`, `description`, `details`).
  - `EventSummary`: Cumulative summary containing total events, pass counts & pass accuracy % per team, shots on goal, defensive interceptions, 1v1 duel tackles, total turnovers, and chronological match timeline.
  - `EventDetector`:
    - **Spatial-Temporal Finite State Machine (FSM)**: Tracks possession release ($v \ge 12\text{ km/h}$), ballistic flight vectors, and target acquisitions.
    - **Pass Completion Engine**: Detects completed passes between teammates ($T_1 \to T_1$, $d \ge 3.0\text{m}$) and calculates pass length and flight velocity.
    - **Interception Classifier**: Detects opponent pass cut-offs ($T_1 \to T_2$, $d \ge 3.0\text{m}$) where ball is intercepted mid-flight.
    - **Shot on Goal Classifier**: Detects attacking strikes in the final third ($X \ge 85\text{m}$ or $X \le 20\text{m}$, $Y \in [20, 48]\text{m}$, $v \ge 36\text{ km/h}$) targeted towards the goalmouth.
    - **Defensive Tackles & Contested Duels**: Detects close-quarters 1v1 possession transitions ($\Delta t \le 6\text{ frames}$, without free flight).
    - **Real-Time Broadcast HUD Toast Alert**: Maintains active event toast for 40 frames ($~1.6\text{s}$) with automatic expiration.
- **`src/visualization/annotator.py`**:
  - Added `draw_event_toast(frame, active_event)`: Renders glowing glassmorphic event notification pill beneath the top HUD with color-coded event badges (Emerald: Pass, Rose: Shot, Amber: Interception, Purple: Tackle).
- **`main.py` & `app.py`**:
  - Added `--no-events` CLI flag.
  - Integrates `EventDetector` in the main frame loop.
  - Generates comprehensive event summary in terminal and saves JSON export to `outputs/logs/<name>_match_events.json`.
  - Added event timeline payload to web app API.
- **`web/` (`index.html`, `styles.css`, `app.js`)**:
  - Added **Match Events Detected** KPI Card to the dashboard grid.
  - Added **Match Event Feed & Timeline** card with dynamic tabular feed, color badges, and speed/distance metrics.
- **`tests/test_events.py`**:
  - 8 automated unit tests covering pass completion, interception, shots on goal, tackle duels, dribbles, toast banner expiration, and summary aggregation. Complete test suite: **66 / 66 tests passing**.

---

## Milestone 10 Details: Camera Cut Detection & Re-Identification across Cuts (Completed)

### 1. What We Built
- **`src/tracking/cut_detector.py`**:
  - `CutDetectionResult`: Dataclass containing `is_cut`, `cut_type`, `hist_distance`, `edge_diff`, `combined_score`, `frame_idx`, and `cut_count`.
  - `CameraCutDetector`:
    - **Multi-Cue Dissimilarity**: Combines normalized 2D Hue-Saturation Bhattacharyya distance with Canny structural edge change ratio.
    - **Optical Flow Penalty**: Automatically penalizes RANSAC inlier collapse during abrupt scene changes.
    - **Temporal Debouncing**: Enforces minimum shot duration interval ($15\text{ frames}$) to prevent multi-triggering on cross-dissolves and wipes.
- **`src/tracking/reid.py`**:
  - `ReIDFeatureExtractor`: Extracts 128D L2-normalized spatial appearance embeddings across 3 anatomical zones: Upper Torso (Jersey, 10-45%), Lower Torso (Shorts, 45-70%), and Lower Legs (Socks, 70-95%).
  - `ReIDGallery`: Long-term persistent identity bank with Exponential Moving Average (EMA) appearance smoothing ($\alpha=0.85$), role, team affiliation, and pitch coordinate histories.
  - `ReIDMatcher`: Solves optimal linear sum assignment (Hungarian algorithm) using cosine distance, pitch proximity priors, and hard team/role consistency constraints.
  - `PlayerReID`: High-level engine coordinating appearance extraction, gallery maintenance, and cross-cut re-association.
- **`src/tracking/tracker.py`**:
  - Cut-aware state resetting: Clears Kalman coasting buffers and motion trails upon camera cuts, preventing screen-spanning line streaks and teleportation jumps.
  - Seamlessly re-associates gallery track IDs to detections appearing after a scene transition.
- **`src/visualization/annotator.py`**:
  - HUD scene status badge: Displays live cut alerts (e.g. `[SCENE: CUT #1 (Re-ID)]`) and flashes an amber top border upon scene transitions.
- **`web/` (`index.html`, `styles.css`, `app.js`)**:
  - Added **Scene Cuts & Re-ID Match** KPI card into the 5-column dashboard grid.
- **`tests/test_cut_detector.py` & `tests/test_reid.py`**:
  - 11 new automated unit tests covering cut triggers, debouncing, optical flow collapse, multi-zone feature extraction, EMA gallery updates, and Hungarian identity recovery across cuts. Complete test suite: **58 / 58 tests passing**.

---

## Milestone 9 Details: Camera Movement & Zoom Compensation (Completed)

### 1. What We Built
- **`src/calibration/camera_motion.py`**:
  - `CameraMotionResult`: Typed dataclass storing affine transformation matrix $M$ ($2\times 3$), pan $(\Delta x)$, tilt $(\Delta y)$, zoom scale factor $(s)$, rotation angle $(\theta)$, and motion category (`PAN_LEFT`, `PAN_RIGHT`, `TILT_UP`, `TILT_DOWN`, `ZOOM_IN`, `ZOOM_OUT`, `STATIC`).
  - `CameraMotionCompensator`:
    - **Foreground Player Exclusion Masking**: Dilates detected player bounding boxes by $+15\text{ px}$ to create a static background pitch mask, preventing optical flow contamination from moving humans.
    - **Pyramidal Lucas-Kanade Sparse Optical Flow**: Tracks Shi-Tomasi corners (`cv2.goodFeaturesToTrack`) across frame pairs $(I_{t-1}, I_t)$.
    - **RANSAC Rigid / Partial Affine Model**: Computes robust global transformation matrix $M$ with outlier rejection (threshold $< 2.5\text{ px}$).
    - **Kinematic Decomposition**: Decomposes $M$ into instantaneous pan velocity, tilt velocity, zoom factor, and rotational drift.
    - **Coordinate Warping**: `warp_points(points, transform)` and `warp_boxes(boxes, transform)` to transform coordinate spaces across camera frames.
- **`src/tracking/tracker.py`**:
  - **Camera Motion Compensation (CMC)**: Integrated `_apply_camera_motion_compensation(camera_transform)` into `PlayerTracker.update(...)`. Warps prior track bounding boxes by affine matrix $M_{t-1 \to t}$ before ByteTrack association, eliminating track ID switches during aggressive camera pans.
- **`src/visualization/annotator.py`**:
  - **Live Camera Telemetry HUD**: Renders real-time camera state badge (e.g. `[CAM: PAN R (+14px) | ZOOM 1.00x]`) in the broadcast HUD.
- **`main.py` & `app.py`**:
  - Added `--no-cmc` command-line toggle to enable/disable camera compensation.
  - Aggregates camera motion distribution (`pan_left_frames`, `pan_right_frames`, `zoom_in_frames`, `zoom_out_frames`, `static_frames`, `mean_pan_speed`) in match telemetry summary JSON.
- **`tests/test_camera_motion.py`**:
  - 5 comprehensive unit tests covering compensator initialization, synthetic translation estimation, synthetic zoom estimation, foreground player masking, and point/box coordinate warping. Complete test suite: **47 / 47 tests passing**.

---

## Milestone 8 Details: Ball Tracking, Trajectory Smoothing & Player Possession Assignment (Completed)

### 1. What We Built
- **`src/tracking/ball_tracker.py`**:
  - `BallKalmanFilter`: 2D Constant-Velocity Kalman Filter for $(x, y, v_x, v_y)$ state estimation in both pixel coordinates and FIFA metric pitch space.
  - `BallTracker`:
    - **Physical Outlier Rejection**: Rejects false positive ball detections with unrealistic velocities ($>140\text{ km/h}$) or sudden displacement jumps ($>380\text{ px}$).
    - **Trajectory Interpolation**: Automatically bridges $1\text{--}6$ frame detection dropouts with Kalman forward projection and smooths trajectories.
    - **Player Possession Assignment**: Computes Euclidean distance $d \le 1.8\text{m}$ between ball position $(X_m, Y_m)$ and player foot positions.
    - **Debouncing & Hysteresis**: Prevents high-frequency turnover flickering in contested 50-50 duels, tracks match possession percentages and turnover counts.
- **`src/visualization/annotator.py`**:
  - **Ball Comet Trail**: Renders glowing luminous comet tail with fading thickness along the ball's movement vector.
  - **Possession Carrier Beacon**: Renders glowing chevron and illuminated halo around the player holding possession (`[CARRIER: #<id>]`).
  - **HUD Telemetry Update**: Real-time possession split bar and live ball carrier telemetry.
- **`src/visualization/radar.py`**:
  - Highlights ball carrier dot with glowing radiant beacon ring and displays ball pulse dot on 2D tactical radar.
- **`web/` (`index.html`, `app.js`)**:
  - Live On-Ball Possession Split Card (Team A % vs Team B %), Possession Turnovers Counter, and Top Ball Carrier badge.
- **`tests/test_ball_tracker.py`**:
  - 6 unit tests covering Kalman convergence, outlier rejection, gap interpolation, proximity assignment, contested duels, and turnovers.

### 2. Experimental Results & Verification
- **Test Video**: 1080p Bundesliga Broadcast Match (`data/videos/sample_broadcast.mp4`, 750 frames).
- **Match Possession Split**: Team B (67.4%) vs Team A (32.6%).
- **Turnover Count**: 3 turnovers registered.
- **Top Ball Carrier**: Player `#106`.
- **Ball Detection & Interpolation Rate**: 482 / 750 frames (64.3%) with continuous trajectory tracking.
- **Unit Tests**: `42 passed in 5.52s` (100% pass rate).
- **Outputs Produced**:
  - Video: `outputs/tracks/sample_broadcast_tactical_master.mp4`
  - Snapshot: `outputs/tracks/sample_frame_milestone8.jpg`
  - Summary Log: `outputs/logs/sample_broadcast_tactical_master_summary.json`

---

## How to Run & Verify

```powershell
# Run all 42 automated unit tests
.venv\Scripts\pytest.exe tests/ -v

# Run complete Tactical Master pipeline (Milestones 1-8)
.venv\Scripts\python.exe main.py --source data/videos/sample_broadcast.mp4 --device 0

# Start interactive Web Dashboard server
.venv\Scripts\python.exe app.py 8000
```
