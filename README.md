<div align="center">
  <h1>AI Football Tactical Analysis & Kinematics Engine</h1>
  <p><strong>Real-Time Broadcast Football Computer Vision, Multi-Object Tracking, Homography Calibration & Spatial Analytics</strong></p>

  <!-- Badges -->
  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python 3.12" /></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white" alt="PyTorch CUDA" /></a>
    <a href="https://docs.ultralytics.com/"><img src="https://img.shields.io/badge/YOLO-v8%20%7C%2011-00FFFF.svg?style=flat" alt="Ultralytics YOLO" /></a>
    <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat&logo=opencv&logoColor=white" alt="OpenCV" /></a>
    <a href="https://pytest.org/"><img src="https://img.shields.io/badge/Tests-30%2F30%20Passing-brightgreen.svg?style=flat&logo=pytest&logoColor=white" alt="Tests" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat" alt="License: MIT" /></a>
  </p>
</div>

---

## 📌 Overview

**AI Football Analysis Engine** is an end-to-end computer vision and tactical analytics system designed for standard single-camera broadcast match footage. Transforming raw monocular video streams into actionable pitch intelligence, the system extracts 2D metric player coordinates, estimates physical velocity kinetics, identifies team roles (Outfield, Goalkeepers, Referees, Coaches), computes spatial pitch dominance (Voronoi & Convex Hulls), and renders a real-time top-down 2D Tactical Minimap Radar.

---

## 🚀 Core Features

- **Multi-Class Dual-Threshold Detection**: YOLOv8m with calibrated thresholds (`conf=0.18` for players, `conf=0.12` for the football) with CUDA FP16 half-precision acceleration.
- **Persistent Multi-Object Tracking**: ByteTrack two-stage association with **Dimension-Locked Kalman Track Coasting** to eliminate dropout flickering across occlusions.
- **Dynamic Field Turf Segmentation**: HSV color masking and Hough line transforms with temporal EMA smoothing ($\alpha = 0.80$) to isolate the playable pitch ($y \le 855\text{px}$) and filter out stadium spectators and dugout benches.
- **Planar Homography Projective Geometry ($H \in \mathbb{R}^{3 \times 3}$)**: Direct Linear Transformation (DLT) mapping image pixel coordinates $(u, v)$ to standardized FIFA metric pitch coordinates $(X, Y)$ in meters.
- **Least-Squares Linear Regression Kinematics**: Closed-form velocity estimation ($\text{km/h}$) over an 18-frame sliding window with $3\sigma$ outlier residual rejection, canceling leg-swing oscillation and camera noise.
- **Automated Jersey Kit & Role Classification**: Upper-torso HSV clustering + spatial goal-area priors to distinguish Team A (White), Team B (Neon Green), Goalkeepers (`[A-GK]` in Orange), Match Officials (`[REF]` in Amber), and Dugout Staff (`[COACH]` in Slate) with 15-frame rolling majority voting.
- **Spatial Tactical Metrics & Compactness**: Computes Voronoi Pitch Space Dominance (e.g. 64% vs 36%), Team Convex Hull Compactness ($m^2$), and 2D Gaussian Kernel Density Estimation (KDE) positional heatmaps.
- **2D Top-Down Tactical Minimap Radar**: Real-time HUD broadcast overlay depicting team formations, goalkeeper anchors, and shaded team convex hulls.

---

## 🏗️ Project Architecture & Pipeline

```
Broadcast Match Video (1080p @ 25 FPS)
  │
  ├──► [1] YOLOv8 Object Detection (Players & Football)
  │       └── Dual Confidence Calibration + Coordinate Sanitization
  │
  ├──► [2] ByteTrack Multi-Object Tracking + Kalman Track Coasting
  │       └── Trajectory Trails & Lost-Track Interpolation
  │
  ├──► [3] HSV Pitch Segmentation & Hough Line Extraction
  │       └── Temporal EMA Smoothing + Crowd/Dugout Filtering
  │
  ├──► [4] Planar Homography Transformation Matrix (H)
  │       └── Image Pixels (u, v) ──► FIFA Metric Pitch (X_m, Y_m)
  │
  ├──► [5] Least-Squares Linear Regression Kinematics
  │       └── Velocity (km/h), Sprint Classification & Total Distance (km)
  │
  ├──► [6] Torso Color Clustering & Spatial Role Identification
  │       └── Team A [A], Team B [B], Goalkeepers [A-GK], Referees [REF], Coaches [COACH]
  │
  ├──► [7] Spatial Tactical Analytics & Positional Density Heatmaps
  │       └── Voronoi Space Dominance (%), Convex Hulls (m²), Gaussian KDE
  │
  └──► [8] Tactical Minimap Radar & Broadcast HUD Telemetry Overlay
```

---

## 📊 Milestone Progress & Roadmap

### Phase 1: Vision, Tracking & Geometry (Completed)
- [x] **Milestone 1**: Deep Learning Player & Ball Detection (YOLOv8 FP16 CUDA).
- [x] **Milestone 2**: ByteTrack Multi-Object Tracking with Kalman State Estimation & Track Coasting.
- [x] **Milestone 3**: Dynamic Pitch Segmentation, Hough Line Detection & Dugout/Crowd Filtering.
- [x] **Milestone 4**: Planar Homography Calibration ($H \in \mathbb{R}^{3 \times 3}$) & 2D Tactical Minimap Radar.
- [x] **Milestone 5**: Least-Squares Linear Regression Kinematics, Velocity ($\text{km/h}$) & Distance Engine.

### Phase 2: Team Intelligence & Spatial Analytics (Completed)
- [x] **Milestone 6**: Automated Team, Goalkeeper (`[A-GK]`), Referee (`[REF]`), and Coach (`[COACH]`) Role Classification with 15-frame Majority Voting.
- [x] **Milestone 7**: Spatial Tactical Metrics, Team Convex Hull Compactness ($m^2$), Voronoi Pitch Dominance (%), and 2D Gaussian KDE Heatmaps.

### Phase 3: Advanced Analytics & Web Platform (Upcoming)
- [ ] **Milestone 8**: Ball Trajectory Smoothing & Proximity-Based Player Possession Assignment.
- [ ] **Milestone 9**: Optical Flow Camera Pan & Tilt Motion Compensation ($\mathbf{v}_{\text{cam}}$).
- [ ] **Milestone 10**: Camera Cut Detection & Cross-Cut Track Re-Identification.
- [ ] **Milestone 11**: Football Event Recognition (Passes, Interceptions, Shots, Turnovers).
- [ ] **Milestone 12**: Interactive Web Dashboard & Real-Time Match Analytics UI.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ (Recommended: Python 3.12)
- NVIDIA GPU with CUDA 12.0+ (Tested on NVIDIA RTX 4050 Laptop GPU)

### 1. Clone Repository & Setup Virtual Environment
```powershell
git clone https://github.com/sabs0101/FOOTBALL_ANALYSIS.git
cd FOOTBALL_ANALYSIS

python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics supervision opencv-python numpy scipy pyyaml pytest
```

---

## ⚡ Usage & Execution

### Run Master Tactical Analytics Pipeline
```powershell
python main.py --source data/videos/sample_broadcast.mp4 --device 0
```

### Optional CLI Arguments:
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--source` | `data/videos/sample_broadcast.mp4` | Path to input match video file. |
| `--device` | `0` | GPU device index (`0`, `cuda`, or `cpu`). |
| `--output` | `outputs/tracks/sample_broadcast_tactical_master.mp4` | Destination path for annotated master video. |
| `--imgsz` | `1280` | Inference resolution for YOLO detector. |
| `--conf` | `0.18` | Base confidence threshold for player detections. |

### Fine-Tune Custom YOLO Model (Transfer Learning)
```powershell
python train.py --data data/dataset/data.yaml --model yolov8m.pt --epochs 50 --batch 16 --device 0
```

---

## 🧪 Automated Testing Suite

The repository includes a comprehensive 30-case test suite covering all modules:

```powershell
pytest tests/ -v
```

```
============================= test session starts =============================
tests/test_analytics.py::test_speed_estimator_initialization PASSED      [  3%]
tests/test_analytics.py::test_speed_category_classification PASSED       [  6%]
tests/test_analytics.py::test_constant_velocity_regression_speed PASSED  [ 10%]
tests/test_analytics.py::test_cumulative_distance_accumulation PASSED    [ 13%]
tests/test_analytics.py::test_unrealistic_speed_teleportation_clamping PASSED [ 16%]
tests/test_analytics.py::test_team_summary_aggregation PASSED            [ 20%]
tests/test_detection.py::test_config_loading PASSED                      [ 23%]
tests/test_detection.py::test_device_selection PASSED                    [ 26%]
tests/test_detection.py::test_detector_on_synthetic_frame PASSED         [ 30%]
tests/test_detection.py::test_annotator_rendering PASSED                 [ 33%]
tests/test_homography.py::test_pitch_template_dimensions PASSED          [ 36%]
tests/test_homography.py::test_homography_computation PASSED             [ 40%]
tests/test_homography.py::test_point_projection_roundtrip PASSED         [ 43%]
tests/test_homography.py::test_tactical_radar_rendering PASSED           [ 46%]
tests/test_pitch.py::test_pitch_detector_initialization PASSED           [ 50%]
tests/test_pitch.py::test_pitch_mask_on_synthetic_grass_field PASSED     [ 53%]
tests/test_pitch.py::test_line_detection_on_synthetic_pitch PASSED       [ 56%]
tests/test_pitch.py::test_crowd_filtering PASSED                         [ 60%]
tests/test_tactics.py::test_spatial_control_initialization PASSED        [ 63%]
tests/test_tactics.py::test_team_convex_hull_calculation PASSED          [ 66%]
tests/test_tactics.py::test_voronoi_space_dominance_symmetric PASSED     [ 70%]
tests/test_tactics.py::test_heatmap_generation PASSED                    [ 73%]
tests/test_team.py::test_team_classifier_initialization PASSED           [ 76%]
tests/test_team.py::test_role_prediction_logic PASSED                    [ 80%]
tests/test_team.py::test_rolling_majority_vote_persistence PASSED        [ 83%]
tests/test_tracking.py::test_tracker_initialization PASSED               [ 86%]
tests/test_tracking.py::test_tracking_persistence_on_moving_boxes PASSED [ 90%]
tests/test_tracking.py::test_kalman_coasting_during_temporary_dropout PASSED [ 93%]
tests/test_tracking.py::test_trail_accumulation PASSED                   [ 96%]
tests/test_tracking.py::test_ball_passthrough_without_tracking PASSED    [100%]
============================= 30 passed in 3.98s ==============================
```

---

## 📁 Repository Directory Structure

```
FOOTBALL_ANALYSIS/
├── config.yaml                    # Centralized system hyperparameters
├── main.py                        # Master pipeline orchestration script
├── train.py                       # YOLO fine-tuning & transfer learning script
├── pytest.ini                     # Pytest test suite configuration
├── README.md                      # Comprehensive project documentation
├── VIVA_PREPARATION_GUIDE.md      # Detailed Viva defense, mathematical theory & Q&A
│
├── data/
│   ├── dataset/                   # Fine-tuning dataset schema (train/val splits)
│   │   ├── images/
│   │   ├── labels/
│   │   └── data.yaml
│   └── videos/                    # Broadcast evaluation match videos
│
├── models/                        # Pretrained & fine-tuned model checkpoints
│
├── src/
│   ├── analytics/                 # Kinematics, linear regression speed & distance
│   ├── calibration/               # Homography transformation (H) & 2D pitch models
│   ├── detection/                 # YOLO player & ball detector with dual thresholds
│   ├── pitch/                     # HSV pitch segmentation, Hough lines & boundary EMA
│   ├── tactics/                   # Convex hulls, Voronoi pitch dominance & KDE heatmaps
│   ├── team/                      # Kit color clustering & spatial role identification
│   ├── tracking/                  # ByteTrack multi-object tracking & Kalman coasting
│   ├── utils/                     # Device configuration & file utilities
│   └── visualization/             # Bounding box annotator, badges, HUD & 2D Radar
│
└── tests/                         # 30 Unit test suites for all pipeline modules
```

---

## 📜 License & Citation

Distributed under the **MIT License**. See `LICENSE` for more information.

Developed by **Sabari Sundaresan**.
