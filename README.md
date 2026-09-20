<div align="center">
  <h1>AI Football Tactical Analysis & Kinematics Engine</h1>
  <p><strong>Real-Time Broadcast Football Computer Vision, Preprocessing, Multi-Object Tracking, Homography Calibration & Quantitative Benchmarks</strong></p>

  <!-- Badges -->
  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python 3.12" /></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white" alt="PyTorch CUDA" /></a>
    <a href="https://docs.ultralytics.com/"><img src="https://img.shields.io/badge/YOLO-v8%20%7C%2011-00FFFF.svg?style=flat" alt="Ultralytics YOLO" /></a>
    <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat&logo=opencv&logoColor=white" alt="OpenCV" /></a>
    <a href="https://pytest.org/"><img src="https://img.shields.io/badge/Tests-72%2F72%20Passing-brightgreen.svg?style=flat&logo=pytest&logoColor=white" alt="Tests" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat" alt="License: MIT" /></a>
  </p>
</div>

---

## 📌 Overview

**AI Football Analysis Engine** is an end-to-end computer vision and tactical analytics pipeline designed for broadcast match footage. Transforming raw monocular video streams into actionable pitch intelligence, the system performs **frame preprocessing & EDA**, **multi-class object detection (YOLOv8)**, **multi-object tracking (ByteTrack with Kalman coasting)**, **dynamic pitch segmentation**, **planar homography perspective transformation**, **least-squares linear regression kinematics**, **automated kit/role classification**, **spatial Voronoi dominance**, **2D KDE heatmaps**, **ball Kalman state estimation & possession**, **camera motion compensation (GME/CMC)**, **multi-cue camera cut detection & cross-cut Re-ID**, **discrete football event recognition (passes, shots, interceptions, tackles)**, and an **interactive glassmorphic web application**.

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
9. **Ball Kalman State Estimation & Possession Engine**:
   - 2D constant-velocity Kalman Filter, physical outlier rejection ($v > 140\text{ km/h}$), trajectory gap interpolation, and Euclidean foot proximity possession tagging ($d \le 1.8\text{m}$).
10. **Camera Motion Compensation (GME/CMC)**:
    - Lucas-Kanade sparse pyramidal optical flow with foreground player dilation masking and RANSAC affine decomposition ($M_{t-1 \to t}$) to eliminate track drift during camera pans.
11. **Camera Cut Detection & Cross-Cut Re-Identification**:
    - Multi-cue Hue-Saturation histogram distance + Canny edge change ratio with Hungarian assignment on 128D anatomical multi-zone appearance embeddings.
12. **Discrete Football Event Recognition**:
    - Spatial-temporal FSM classifying completed passes ($T_1 \to T_1, d \ge 3\text{m}$), interceptions ($T_1 \to T_2$), goal-bound shots ($v \ge 36\text{ km/h}$), and close 1v1 duel tackles.
13. **Interactive Web Dashboard & Video Player**:
    - Glassmorphic frontend with drag-and-drop video processing, animated progress HUD, H.264 video streaming, 6 KPI cards, speed leaderboard, event timeline, and CSV/JSON/TSV dataset exports.

---

## 📊 Milestone Progress & Roadmap (100% Completed)

| Phase | Milestone | Focus Area | Status |
|---|---|---|:---:|
| **Phase 1** | **Milestone 1** | YOLOv8 Player & Ball Detection + Tactical HUD | **COMPLETED** |
| | **Milestone 2** | Multi-Object Tracking (ByteTrack) & Persistent IDs | **COMPLETED** |
| | **Milestone 3** | Football Pitch & Field Line Segmentation | **COMPLETED** |
| | **Milestone 4** | Planar Homography ($H \in \mathbb{R}^{3 \times 3}$) & 2D Minimap Radar | **COMPLETED** |
| | **Milestone 5** | Player Kinematics, Linear Regression Speed & Distance | **COMPLETED** |
| **Phase 2** | **Milestone 6** | Jersey Kit Clustering & Spatial Role Classification | **COMPLETED** |
| | **Milestone 7** | Voronoi Space Dominance & 2D Gaussian Heatmaps | **COMPLETED** |
| | **Milestone 8** | Ball Kalman State Estimation & Possession Engine | **COMPLETED** |
| **Phase 3** | **Milestone 9** | Camera Movement & Zoom Compensation (GME/CMC) | **COMPLETED** |
| | **Milestone 10** | Camera Cut Detection & Cross-Cut Player Re-ID | **COMPLETED** |
| | **Milestone 11** | Basic Football Event Recognition (Passes, Shots, Interceptions) | **COMPLETED** |
| | **Milestone 12** | Interactive Web Dashboard & Polished Demonstration Integration | **COMPLETED** |

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

### 1. Launch Interactive Tactical Web Dashboard
Starts the local web server with drag-and-drop video upload, real-time telemetry, and match reports:
```powershell
python app.py 8000
```
Open **`http://localhost:8000`** in your browser.

### 2. Run Master Tactical Analytics Pipeline (CLI)
Processes input video and produces the full tactical broadcast master video with HUD and 2D Radar:
```powershell
python main.py --source data/videos/sample_broadcast.mp4 --device 0
```

### 3. Run Exploratory Data Analysis (EDA)
Inspects video properties, turf color profile, and lighting distributions:
```powershell
python eda.py --source data/videos/sample_broadcast.mp4 --output outputs/eda/eda_report.json
```

### 4. Run Quantitative Benchmark Evaluation
Runs detection, tracking, homography, and kinematics evaluation benchmarks:
```powershell
python evaluate.py --source data/videos/sample_broadcast.mp4 --device 0 --frames 150
```

### 5. Fine-Tune Custom YOLO Model (Transfer Learning)
```powershell
python train.py --data data/dataset/data.yaml --model yolov8m.pt --epochs 50 --batch 16 --device 0
```

### 6. Run Automated Unit Test Suite
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
## 🧪 Automated Testing Suite (72 / 72 Passing)

```
============================= test session starts =============================
platform win32 -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
collected 72 items

tests/test_analytics.py (6 tests) ...................... PASSED [  8%]
tests/test_ball_tracker.py (6 tests) .................. PASSED [ 16%]
tests/test_camera_motion.py (5 tests) ................. PASSED [ 23%]
tests/test_cut_detector.py (6 tests) .................. PASSED [ 31%]
tests/test_dashboard_api.py (6 tests) ................. PASSED [ 40%]
tests/test_detection.py (4 tests) ..................... PASSED [ 45%]
tests/test_evaluation.py (3 tests) .................... PASSED [ 50%]
tests/test_events.py (8 tests) ........................ PASSED [ 61%]
tests/test_homography.py (4 tests) .................... PASSED [ 66%]
tests/test_pitch.py (4 tests) ......................... PASSED [ 72%]
tests/test_preprocessing.py (3 tests) ................. PASSED [ 76%]
tests/test_reid.py (5 tests) .......................... PASSED [ 83%]
tests/test_tactics.py (4 tests) ....................... PASSED [ 88%]
tests/test_team.py (3 tests) .......................... PASSED [ 93%]
tests/test_tracking.py (5 tests) ...................... PASSED [100%]

======================== 72 passed in 9.68s ========================
```

---

## 📁 Repository Directory Structure

```
FOOTBALL_ANALYSIS/
├── config.yaml                    # Centralized system hyperparameters
├── main.py                        # Master pipeline CLI orchestration script
├── app.py                         # Interactive Web Dashboard HTTP & REST API server
├── eda.py                         # Exploratory Data Analysis CLI script
├── evaluate.py                    # Quantitative benchmark evaluation script
├── train.py                       # YOLO fine-tuning & transfer learning script
├── pytest.ini                     # Pytest test suite configuration
├── README.md                      # Comprehensive project documentation & guide
├── PROJECT_PROGRESS.md            # Detailed milestone progression log
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
│   ├── analytics/                 # Kinematics, linear regression speed & discrete events
│   ├── calibration/               # Homography transformation (H) & camera motion (CMC)
│   ├── detection/                 # YOLO player & ball detector with dual thresholds
│   ├── evaluation/                # Quantitative metrics (mAP, MOTA, MOTP, RMSE)
│   ├── pitch/                     # HSV pitch segmentation, Hough lines & boundary EMA
│   ├── preprocessing/             # Frame preprocessor, CLAHE, blur detection & video EDA
│   ├── tactics/                   # Convex hulls, Voronoi pitch dominance & KDE heatmaps
│   ├── team/                      # Kit color clustering & spatial role identification
│   ├── tracking/                  # ByteTrack tracking, Re-ID & camera cut detection
│   ├── utils/                     # Device configuration & video I/O utilities
│   └── visualization/             # Bounding box annotator, HUD toasts & 2D Radar
│
├── web/
│   ├── index.html                 # Glassmorphic drag-and-drop tactical dashboard UI
│   ├── styles.css                 # Dark-mode glowing HUD CSS design system
│   └── app.js                     # REST API polling, video streaming & chart controller
│
└── tests/                         # 72 Unit test suites for all 16 pipeline modules
```

---

## 📜 License & Citation

Distributed under the **MIT License**. See `LICENSE` for more information.

Developed by **Sabari Sundaresan**.

