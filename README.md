# AI-Based Football Broadcast Analysis and Tactical Reconstruction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6 CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.4-green.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%2011-orange.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision and tactical analytics system for normal broadcast football footage.

---

## Roadmap & Milestones

- [x] **Milestone 1**: Video $\rightarrow$ YOLO Player/Ball Detection $\rightarrow$ Annotated Output Video with Tactical HUD.
- [x] **Milestone 2**: Multi-Object Tracking (ByteTrack / BoT-SORT) with Persistent Player IDs.
- [ ] **Milestone 3**: Pitch & Field Line Detection.
- [ ] **Milestone 4**: Pitch Keypoints $\rightarrow$ Homography $\rightarrow$ Standardized 2D Pitch Model.
- [ ] **Milestone 5**: Camera-to-Pitch Coordinate Transformation (2D Radar).
- [x] **Milestone 6**: Automated Team & Referee Classification via Jersey Color Clustering.
- [ ] **Milestone 7**: Tactical Analytics (Movement Trajectories, Speed, Distance, Heatmaps, Voronoi).
- [ ] **Milestone 8**: Ball Tracking, Trajectory Smoothing, and Possession Assignment.
- [ ] **Milestone 9**: Camera Movement & Zoom Compensation.
- [ ] **Milestone 10**: Camera Cut Detection & Re-Identification across Cuts.
- [ ] **Milestone 11**: Football Event Recognition (Passes, Shots, Interceptions).
- [ ] **Milestone 12**: Interactive Web Dashboard (Streamlit / Next.js).

---

## Project Structure

```
FOOTBALL_ANALYSIS/
├── .venv/                      # Python 3.12 virtual environment (CUDA enabled)
├── config.yaml                 # Central configuration
├── requirements.txt            # Pinned dependencies
├── data/
│   ├── videos/                 # Raw input broadcast footage
│   └── test/                   # Short validation clips
├── models/                     # YOLO checkpoints (yolov8m.pt, etc.)
├── src/
│   ├── detection/              # Player & Ball Detection (Milestone 1)
│   ├── tracking/               # Multi-Object Tracking (Milestone 2)
│   ├── pitch/                  # Pitch Segmentation & Lines (Milestone 3)
│   ├── calibration/            # Homography & Pitch Mapping (Milestones 4-5)
│   ├── teams/                  # Team Jersey Color Clustering (Milestone 6)
│   ├── analytics/              # Speed, Distance, Heatmaps, Voronoi (Milestone 7)
│   ├── ball/                   # Ball State & Possession (Milestone 8)
│   ├── visualization/          # Annotator & 2D Tactical Radar
│   └── utils/                  # Video IO & Config utilities
├── outputs/
│   ├── detections/             # Processed videos
│   └── logs/                   # JSON analytical metrics
├── tests/                      # Pytest suite
└── main.py                     # CLI pipeline entry point
```

---

## Quick Start (Milestone 1)

### 1. Setup Virtual Environment
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Player Detection Pipeline
```powershell
# Run on sample video using NVIDIA RTX 4050 GPU
python main.py --source data/videos/sample_broadcast.mp4 --device 0

# Run with custom model and confidence
python main.py --source data/videos/match.mp4 --model yolov8m.pt --conf 0.35
```

### 3. Run Automated Tests
```powershell
pytest tests/ -v
```
