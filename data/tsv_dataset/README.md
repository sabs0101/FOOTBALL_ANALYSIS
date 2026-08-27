# 10,000 Frame TSV Football Tactical Dataset

This directory contains the **10,000 Frame Tab-Separated Values (TSV) Football Tactical Dataset** generated from broadcast match footage.

### Directory Structure:
```
data/tsv_dataset/
├── images/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ... (10,000 full HD images)
├── dataset.tsv              # 10,000 Tab-Separated Records (17.2 MB)
├── dataset_summary.json     # Schema and category specifications
└── README.md                # Dataset documentation
```

### TSV Column Specifications (`dataset.tsv`):
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `image_name` | String | Corresponding image filename (e.g. `frame_000001.jpg`). |
| `frame_idx` | Integer | Sequential frame index (1 to 10,000). |
| `timestamp_s` | Float | Timestamp in seconds (e.g. `0.040`). |
| `num_players` | Integer | Total active on-pitch players detected. |
| `ball_detected` | Boolean | True if the football is visible in frame. |
| `team_a_count` | Integer | Active outfield players for Team A. |
| `team_b_count` | Integer | Active outfield players for Team B. |
| `top_speed_kmh` | Float | Instantaneous peak player sprint speed in km/h. |
| `team_a_space_dominance_pct` | Float | Percentage of pitch territory controlled by Team A (Voronoi). |
| `team_b_space_dominance_pct` | Float | Percentage of pitch territory controlled by Team B (Voronoi). |
| `team_a_compactness_m2` | Float | Surface area of Team A's 2D Convex Hull in square meters ($m^2$). |
| `bounding_boxes_json` | JSON | Array of all bounding boxes `[[x1, y1, x2, y2], ...]`. |
| `player_roles_json` | JSON | Array of role classifications (`"Team A"`, `"Team B"`, `"Referee"`, `"Goalkeeper"`). |
| `player_speeds_json` | JSON | Dictionary mapping tracking ID to live speed in km/h. |

### How to Load in Python:
```python
import pandas as pd

# Load TSV dataset
df = pd.read_csv("data/tsv_dataset/dataset.tsv", sep="\t")
print(df.head())
print(f"Total Annotated Samples: {len(df)}")
```
