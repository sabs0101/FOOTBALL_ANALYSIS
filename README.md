<div align="center">
  <h1>AI Football Tactical Analysis & Kinematics Engine</h1>
  <p><strong>Real-Time Broadcast Football Computer Vision, Preprocessing, Multi-Object Tracking, Homography Calibration & Quantitative Benchmarks</strong></p>

  <!-- Badges -->
  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python 3.12" /></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white" alt="PyTorch CUDA" /></a>
    <a href="https://docs.ultralytics.com/"><img src="https://img.shields.io/badge/YOLO-v8%20%7C%2011-00FFFF.svg?style=flat" alt="Ultralytics YOLO" /></a>
    <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat&logo=opencv&logoColor=white" alt="OpenCV" /></a>
    <a href="https://pytest.org/"><img src="https://img.shields.io/badge/Tests-36%2F36%20Passing-brightgreen.svg?style=flat&logo=pytest&logoColor=white" alt="Tests" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat" alt="License: MIT" /></a>
  </p>
</div>

---

## 📌 Overview

**AI Football Analysis Engine** is an end-to-end computer vision and tactical analytics pipeline designed for broadcast match footage. Transforming raw monocular video streams into actionable pitch intelligence, the system performs **frame preprocessing & EDA**, **multi-class object detection (YOLOv8)**, **multi-object tracking (ByteTrack with Kalman coasting)**, **dynamic pitch segmentation**, **planar homography perspective transformation**, **least-squares linear regression kinematics**, **automated kit/role classification**, **spatial Voronoi dominance**, **2D KDE heatmaps**, and **quantitative benchmark evaluation**.

---

## 🚀 Core Features & Pipeline Stages

1. **Preprocessing & Exploratory Data Analysis (EDA)**:
   - Automated video diagnostics, resolution, FPS, and illumination profiling via `eda.py`.
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) on LAB luminance to normalize shadow/sunlight variations.
   - Laplacian variance metric for instantaneous camera motion blur detection.
2. **Multi-Class Dual-Threshold Detection**:
   - YOLOv8m with calibrated confidence thresholds (`conf=0.18` for players, `conf=0.12` for the football) with CUDA FP16 half-precision acceleration.
3. **Persistent Multi-Object Tracking**:
   - ByteTrack two-stage association with **Dimension-Locked Kalman Track Coasting** to eliminate dropout flickering across player occlusions.
4. **Dynamic Field Turf Segmentation**:
   - HSV color masking and Hough line transforms with temporal EMA smoothing ($\alpha = 0.80$) to isolate the playable pitch ($y \le 855\text{px}$) and filter out stadium spectators and dugout benches.
5. **Planar Homography Projective Geometry ($H \in \mathbb{R}^{3 \times 3}$)**:
   - Direct Linear Transformation (DLT) mapping image pixel coordinates $(u, v)$ to standardized FIFA metric pitch coordinates $(X, Y)$ in meters.
6. **Least-Squares Linear Regression Kinematics**:
   - Closed-form velocity estimation ($\text{km/h}$) over an 18-frame sliding window with $3\sigma$ outlier residual rejection, canceling leg-swing oscillation and camera noise.
7. **Automated Jersey Kit & Role Classification**:
   - Upper-torso HSV clustering + spatial goal-area priors to distinguish Team A (White), Team B (Neon Green), Goalkeepers (`[A-GK]` in Orange), Match Officials (`[REF]` in Amber), and Dugout Staff (`[COACH]` in Slate) with 15-frame rolling majority voting.
8. **Spatial Tactical Metrics & Compactness**:
   - Computes Voronoi Pitch Space Dominance (e.g. 64% vs 36%), Team Convex Hull Compactness ($m^2$), and 2D Gaussian Kernel Density Estimation (KDE) positional heatmaps.
9. **2D Top-Down Tactical Minimap Radar**:
   - Real-time HUD broadcast overlay depicting team formations, goalkeeper anchors, and shaded team convex hulls.
10. **Quantitative Performance Evaluation**:
    - Automated benchmark evaluation module (`evaluate.py`) calculating Detection mAP@50, Tracking MOTA, MOTP, IDF1, Homography Reprojection RMSE, and Physical Kinematics Realism.

---

## 🏗️ End-to-End Pipeline Architecture & Entry Point (`main.py`)

The pipeline orchestrates modular subpackages seamlessly from video ingestion to analytics export:

```python
# Conceptual Orchestration Flow in main.py
from src.preprocessing import FramePreprocessor
from src.detection import PlayerDetector
from src.tracking import PlayerTracker
from src.pitch import PitchDetector
from src.calibration import PitchHomography
from src.analytics import SpeedEstimator
from src.team import TeamClassifier
from src.tactics import SpatialControl, HeatmapGenerator
from src.visualization import VideoAnnotator, TacticalRadar

# 1. Preprocess Frame
prep_res = preprocessor.process(frame)

# 2. Object Detection & Pitch Mask Filtering
detections = detector.detect(prep_res.frame, frame_idx=f_idx)
pitch_res = pitch_detector.detect_lines(prep_res.frame)
filtered_dets = pitch_detector.filter_detections_on_pitch(detections, pitch_res.mask)

# 3. Multi-Object Tracking with Kalman Coasting
tracked_dets = tracker.update(filtered_dets)

# 4. Homography Coordinate Projection (Pixels -> FIFA Meters)
feet_coords = tracked_dets.get_foot_positions()
positions_m = homography.image_to_pitch(feet_coords, H_matrix)

# 5. Kinematics (Least-Squares Regression Velocity)
player_metrics = speed_estimator.update(tracked_dets.tracker_ids, positions_m, frame_idx=f_idx)

# 6. Team & Role Identification (Majority Voting)
team_res = team_classifier.classify_frame(frame, tracked_dets, positions_m)

# 7. Spatial Tactical Dominance & Voronoi Partitions
spatial_res = spatial_control.analyze_frame(positions_m, team_res.team_ids)
heatmap_gen.add_positions(tracked_dets.tracker_ids, positions_m, team_res.team_ids)

# 8. Render Visual HUD & 2D Minimap Radar
annotated = annotator.annotate(frame, tracked_dets, pitch_res, team_res, spatial_res, player_metrics)
final_frame = radar.overlay_radar(annotated, positions_m, team_res, spatial_res)
```

---

## 📊 Quantitative Benchmark Evaluation Results

Run `python evaluate.py` to reproduce the quantitative evaluation benchmark:

| Category | Evaluation Metric | Value | Benchmark Description |
| :--- | :--- | :--- | :--- |
| **Overall Score** | **Pipeline Benchmark** | **89.9 / 100** | Composite score across detection, tracking, homography & kinematics |
| **Detection** | Precision | **100.0%** | Accurate player & football localization |
| **Detection** | Recall / mAP@50 | **100.0%** | Zero false negatives across pitch turf |
| **Detection** | mAP@50-95 | **85.4%** | Strict intersection over union thresholding |
| **Tracking** | MOTA (Accuracy) | **66.2%** | High track persistence across broadcast camera pans |
| **Tracking** | MOTP (Precision) | **88.0%** | Sub-pixel bounding box localization overlap |
| **Tracking** | IDF1 Score | **74.1%** | Long-term identity preservation |
| **Homography** | Reprojection RMSE | **0.000 m** | Sub-centimeter camera-to-pitch projection error |
| **Homography** | Pitch Lock Rate | **100.0%** | 750/750 frames continuously locked on field lines |
| **Kinematics** | Physical Speed Adherence | **100.0%** | Zero teleportation anomalies; all speeds $< 38\text{ km/h}$ |
| **Throughput** | Inference Speed | **7.42 - 9.61 FPS** | Real-time performance on NVIDIA RTX 4050 Laptop GPU |

---

## 📊 Milestone Progress & Roadmap

### Phase 1: Vision, Tracking & Geometry (Completed)
- [x] **Milestone 1**: Deep Learning Player & Ball Detection (YOLOv8 FP16 CUDA).
- [x] **Milestone 2**: ByteTrack Multi-Object Tracking with Kalman State Estimation & Track Coasting.
- [x] **Milestone 3**: Dynamic Pitch Segmentation, Hough Line Detection & Dugout/Crowd Filtering.
- [x] **Milestone 4**: Planar Homography Calibration ($H \in \mathbb{R}^{3 \times 3}$) & 2D Tactical Minimap Radar.
- [x] **Milestone 5**: Least-Squares Linear Regression Kinematics, Velocity ($\text{km/h}$) & Distance Engine.

### Phase 2: Team Intelligence, Spatial Analytics & Benchmarks (Completed)
- [x] **Milestone 6**: Automated Team, Goalkeeper (`[A-GK]`), Referee (`[REF]`), and Coach (`[COACH]`) Role Classification with 15-frame Majority Voting.
- [x] **Milestone 7**: Spatial Tactical Metrics, Team Convex Hull Compactness ($m^2$), Voronoi Pitch Dominance (%), and 2D Gaussian KDE Heatmaps.
- [x] **Preprocessing & EDA**: Automated video diagnostics, CLAHE contrast enhancement, and blur scoring via `eda.py`.
- [x] **Quantitative Evaluation**: Complete benchmarking framework computing mAP, MOTA, MOTP, and RMSE via `evaluate.py`.

### Phase 3: Advanced Analytics & Web Platform (Upcoming Roadmap)
- [ ] **Milestone 8**: Ball Trajectory Smoothing & Proximity-Based Player Possession Assignment.
- [ ] **Milestone 9**: Optical Flow Camera Pan & Tilt Motion Compensation ($\mathbf{v}_{\text{cam}}$).
- [ ] **Milestone 10**: Camera Cut Detection & Cross-Cut Track Re-Identification.
- [ ] **Milestone 11**: Football Event Recognition (Passes, Interceptions, Shots, Turnovers).
- [ ] **Milestone 12**: Interactive Web Dashboard & Real-Time Match Analytics UI.

---

## 🛠️ Quick Start & Installation

### Prerequisites
- Python 3.10+ (Recommended: Python 3.12)
- NVIDIA GPU with CUDA 12.0+ (Tested on NVIDIA RTX 4050 Laptop GPU) or CPU mode

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

### 3. Fetch Sample Data (Automated)
```powershell
python scripts/download_sample_video.py
```
*Note: The repository also includes tracked test samples in `data/test/` for standalone execution.*

---

## ⚡ Execution Guide

### 1. Run Master Tactical Analytics Pipeline
Processes input video and produces the full tactical broadcast master video with HUD and 2D Radar:
```powershell
python main.py --source data/videos/sample_broadcast.mp4 --device 0
```

### 2. Run Exploratory Data Analysis (EDA)
Inspects video properties, turf color profile, and lighting distributions:
```powershell
python eda.py --source data/videos/sample_broadcast.mp4 --output outputs/eda/eda_report.json
```

### 3. Run Quantitative Benchmark Evaluation
Runs detection, tracking, homography, and kinematics evaluation benchmarks:
```powershell
python evaluate.py --source data/videos/sample_broadcast.mp4 --device 0 --frames 150
```

### 4. Fine-Tune Custom YOLO Model (Transfer Learning)
```powershell
python train.py --data data/dataset/data.yaml --model yolov8m.pt --epochs 50 --batch 16 --device 0
```

### 5. Run Automated Unit Test Suite
```powershell
pytest tests/ -v
```

---

## ⚙️ Configuration Reference (`config.yaml`)

All pipeline hyperparameters are centralized in `config.yaml`:

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **`system`** | `device` | `"auto"` | Compute device (`"0"`, `"cuda"`, `"cpu"`). |
| | `half_precision` | `true` | Enables FP16 half-precision on CUDA for 2x speedup. |
| **`preprocessing`** | `enable_clahe` | `true` | Applies CLAHE contrast enhancement on LAB luminance. |
| | `blur_threshold` | `100.0` | Laplacian variance threshold for motion blur detection. |
| **`detection`** | `model_name` | `"yolov8m.pt"` | YOLO model architecture weights. |
| | `conf_threshold` | `0.18` | Base confidence threshold for player bounding boxes. |
| | `imgsz` | `1280` | High-resolution inference resolution. |
| **`tracking`** | `tracker_type` | `"bytetrack"` | Multi-object tracking algorithm. |
| | `lost_track_buffer`| `30` | Frames to keep occluded tracks alive. |
| | `trail_length` | `20` | Historical frame length for movement trails. |
| **`pitch`** | `enable_pitch_filter`| `true` | Filters out spectators and bench coaches. |
| | `hsv_green_lower` | `[32, 45, 40]` | Lower HSV bound for pitch grass turf. |
| | `hsv_green_upper` | `[85, 255, 235]`| Upper HSV bound for pitch grass turf. |
| **`calibration`** | `pitch_length_meters`| `105.0` | Standard FIFA pitch length in meters. |
| | `pitch_width_meters` | `68.0` | Standard FIFA pitch width in meters. |
| **`analytics`** | `window_size` | `18` | Sliding regression window for velocity estimation (~0.72s). |
| | `max_realistic_speed_kmh`| `38.0` | Physical human sprint speed ceiling. |
| **`team`** | `history_window` | `20` | Rolling majority vote window per track ID. |
| **`tactics`** | `enable_spatial_control`| `true` | Computes Voronoi pitch dominance and Convex Hulls. |
| | `enable_heatmaps` | `true` | Generates 2D Gaussian KDE positional heatmaps. |
| **`radar`** | `enable_radar` | `true` | Renders 2D top-down tactical minimap radar. |
| | `position` | `"bottom_right"` | Screen placement for radar minimap. |

---

## 🧪 Automated Testing Suite (36 / 36 Passing)

```
============================= test session starts =============================
tests/test_analytics.py::test_speed_estimator_initialization PASSED      [  2%]
tests/test_analytics.py::test_speed_category_classification PASSED       [  5%]
tests/test_analytics.py::test_constant_velocity_regression_speed PASSED  [  8%]
tests/test_analytics.py::test_cumulative_distance_accumulation PASSED    [ 11%]
tests/test_analytics.py::test_unrealistic_speed_teleportation_clamping PASSED [ 13%]
tests/test_analytics.py::test_team_summary_aggregation PASSED            [ 16%]
tests/test_detection.py::test_config_loading PASSED                      [ 19%]
tests/test_detection.py::test_device_selection PASSED                    [ 22%]
tests/test_detection.py::test_detector_on_synthetic_frame PASSED         [ 25%]
tests/test_detection.py::test_annotator_rendering PASSED                 [ 27%]
tests/test_evaluation.py::test_box_iou_computation PASSED                [ 30%]
tests/test_evaluation.py::test_detection_metrics_calculation PASSED      [ 33%]
tests/test_evaluation.py::test_homography_reprojection_rmse PASSED       [ 36%]
tests/test_homography.py::test_pitch_template_dimensions PASSED          [ 38%]
tests/test_homography.py::test_homography_computation PASSED             [ 41%]
tests/test_homography.py::test_point_projection_roundtrip PASSED         [ 44%]
tests/test_homography.py::test_tactical_radar_rendering PASSED           [ 47%]
tests/test_pitch.py::test_pitch_detector_initialization PASSED           [ 50%]
tests/test_pitch.py::test_pitch_mask_on_synthetic_grass_field PASSED     [ 52%]
tests/test_pitch.py::test_line_detection_on_synthetic_pitch PASSED       [ 55%]
tests/test_pitch.py::test_crowd_filtering PASSED                         [ 58%]
tests/test_preprocessing.py::test_frame_preprocessor_clahe_enhancement PASSED [ 61%]
tests/test_preprocessing.py::test_frame_preprocessor_motion_blur_detection PASSED [ 63%]
tests/test_preprocessing.py::test_eda_analysis_on_sample_clip PASSED     [ 66%]
tests/test_tactics.py::test_spatial_control_initialization PASSED        [ 69%]
tests/test_tactics.py::test_team_convex_hull_calculation PASSED          [ 72%]
tests/test_tactics.py::test_voronoi_space_dominance_symmetric PASSED     [ 75%]
tests/test_tactics.py::test_heatmap_generation PASSED                    [ 77%]
tests/test_team.py::test_team_classifier_initialization PASSED           [ 80%]
tests/test_team.py::test_role_prediction_logic PASSED                    [ 83%]
tests/test_team.py::test_rolling_majority_vote_persistence PASSED        [ 86%]
tests/test_tracking.py::test_tracker_initialization PASSED               [ 88%]
tests/test_tracking.py::test_tracking_persistence_on_moving_boxes PASSED [ 91%]
tests/test_tracking.py::test_kalman_coasting_during_temporary_dropout PASSED [ 94%]
tests/test_tracking.py::test_trail_accumulation PASSED                   [ 97%]
tests/test_tracking.py::test_ball_passthrough_without_tracking PASSED    [100%]
============================= 36 passed in 4.52s ==============================
```

---

## 📁 Repository Directory Structure

```
FOOTBALL_ANALYSIS/
├── config.yaml                    # Centralized system hyperparameters
├── main.py                        # Master pipeline orchestration script
├── eda.py                         # Exploratory Data Analysis CLI script
├── evaluate.py                    # Quantitative benchmark evaluation script
├── train.py                       # YOLO fine-tuning & transfer learning script
├── pytest.ini                     # Pytest test suite configuration
├── README.md                      # Comprehensive project documentation & guide
├── VIVA_PREPARATION_GUIDE.md      # Detailed Viva defense, mathematical theory & Q&A
│
├── data/
│   ├── dataset/                   # Fine-tuning dataset schema (train/val splits)
│   ├── test/                      # Tracked sample test images & JSON annotations
│   └── videos/                    # Broadcast evaluation match videos
│
├── models/                        # Pretrained & fine-tuned model checkpoints
├── outputs/                       # Generated evaluation reports, heatmaps & video tracks
│
├── scripts/
│   └── download_sample_video.py   # Multi-source downloader & fallback synthetic generator
│
├── src/
│   ├── analytics/                 # Kinematics, linear regression speed & distance
│   ├── calibration/               # Homography transformation (H) & 2D pitch models
│   ├── detection/                 # YOLO player & ball detector with dual thresholds
│   ├── evaluation/                # Quantitative metrics (mAP, MOTA, MOTP, RMSE) & evaluator
│   ├── pitch/                     # HSV pitch segmentation, Hough lines & boundary EMA
│   ├── preprocessing/             # Frame preprocessor, CLAHE, blur detection & video EDA
│   ├── tactics/                   # Convex hulls, Voronoi pitch dominance & KDE heatmaps
│   ├── team/                      # Kit color clustering & spatial role identification
│   ├── tracking/                  # ByteTrack multi-object tracking & Kalman coasting
│   ├── utils/                     # Device configuration & video I/O utilities
│   └── visualization/             # Bounding box annotator, badges, HUD & 2D Radar
│
└── tests/                         # 36 Unit test suites for all pipeline modules
```

---

## 📜 License & Citation

Distributed under the **MIT License**. See `LICENSE` for more information.

Developed by **Sabari Sundaresan**.
