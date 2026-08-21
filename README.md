<div align="center">
  <h1>⚽ Football Analysis & Tactical Reconstruction</h1>
  <p><strong>An End-to-End Computer Vision System for Broadcast Football Footage</strong></p>

  <!-- Badges -->
  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12" /></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-CUDA%2012.4-green.svg" alt="PyTorch CUDA" /></a>
    <a href="https://docs.ultralytics.com/"><img src="https://img.shields.io/badge/YOLO-v8%20%7C%2011-orange.svg" alt="Ultralytics YOLO" /></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-1.5+-blue.svg" alt="Scikit-Learn" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
  </p>
</div>

---

## 📖 Overview

**Football Analysis** is a comprehensive tactical analytics pipeline built to process raw, broadcast-angle football footage. Leveraging state-of-the-art deep learning (YOLO) and computer vision techniques, it extracts high-value data such as player coordinates, tracking IDs, and team classifications, rendering them onto a dynamic tactical HUD.

### ✨ Core Features

- 🎯 **High-Fidelity Detection:** Detect players, referees, and the ball with sub-pixel accuracy.
- 🔗 **Persistent Multi-Object Tracking:** Assign and maintain unique tracking IDs seamlessly across frames using `ByteTrack`.
- 🎨 **Automated Team Classification:** Dynamically cluster jersey colors via K-Means to intelligently assign players to their respective teams (Team 1, Team 2) on-the-fly.
- 📺 **Tactical HUD Visualization:** Real-time analytics overlay, displaying frame rates, live player counts, tracking IDs, and team-colored bounding boxes.

---

## 🗺️ Roadmap & Milestones

<details open>
<summary><b>Phase 1: Vision & Tracking (Current Focus)</b></summary>
<br>

- [x] **Milestone 1**: Video $\rightarrow$ YOLO Player/Ball Detection $\rightarrow$ Annotated Output Video with Tactical HUD.
- [x] **Milestone 2**: Multi-Object Tracking (ByteTrack / BoT-SORT) with Persistent Player IDs.
- [x] **Milestone 6**: Automated Team & Referee Classification via Jersey Color Clustering.
</details>

<details>
<summary><b>Phase 2: Spatial & Pitch Mapping (Upcoming)</b></summary>
<br>

- [ ] **Milestone 3**: Pitch & Field Line Detection.
- [ ] **Milestone 4**: Pitch Keypoints $\rightarrow$ Homography $\rightarrow$ Standardized 2D Pitch Model.
- [ ] **Milestone 5**: Camera-to-Pitch Coordinate Transformation (2D Radar).
</details>

<details>
<summary><b>Phase 3: Advanced Analytics & Web Dashboard</b></summary>
<br>

- [ ] **Milestone 7**: Tactical Analytics (Movement Trajectories, Speed, Distance, Heatmaps, Voronoi).
- [ ] **Milestone 8**: Ball Tracking, Trajectory Smoothing, and Possession Assignment.
- [ ] **Milestone 9**: Camera Movement & Zoom Compensation.
- [ ] **Milestone 10**: Camera Cut Detection & Re-Identification across Cuts.
- [ ] **Milestone 11**: Football Event Recognition (Passes, Shots, Interceptions).
- [ ] **Milestone 12**: Interactive Web Dashboard (Streamlit / Next.js).
</details>

---

## 🚀 Quick Start

### 1. Setup Environment
It is highly recommended to use a Python virtual environment. A CUDA-capable GPU is heavily recommended for inference speed.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Pipeline
Edit `config.yaml` to toggle tracking and team classification features according to your needs:
```yaml
tracking:
  enabled: true
  tracker_type: "bytetrack.yaml"

teams:
  enabled: true
  num_teams: 2
```

### 3. Run Inference Pipeline
Execute the main script to process a video and generate a tactically annotated output video with all features enabled.

```powershell
# Run on sample video using NVIDIA GPU (Default Config)
python main.py --source data/videos/sample_broadcast.mp4 --device 0

# Run with custom YOLO model and confidence threshold
python main.py --source data/videos/match.mp4 --model yolov8m.pt --conf 0.35
```

### 4. Run Automated Tests
Ensure everything in the pipeline is working correctly:
```powershell
pytest tests/ -v
```

---

## 📂 Project Structure

```text
FOOTBALL_ANALYSIS/
├── config.yaml                 # Central configuration (Tracking, Teams, YOLO, HUD)
├── requirements.txt            # Dependencies
├── data/
│   ├── videos/                 # Raw input broadcast footage
│   └── test/                   # Short validation clips
├── src/
│   ├── detection/              # Player & Ball Detection (YOLO)
│   ├── tracking/               # Multi-Object Tracking logic
│   ├── teams/                  # Team Jersey Color Clustering (K-Means)
│   ├── visualization/          # Annotator & Tactical HUD
│   └── utils/                  # Video IO & Config utilities
├── outputs/                    # Exported HUD videos and JSON metrics
├── tests/                      # Pytest suite
└── main.py                     # CLI pipeline entry point
```
