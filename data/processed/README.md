# Processed Match Annotations Dataset (TSV Format)

This folder contains processed tactical annotations in **Tab-Separated Values (`.tsv`)** format.

### File: `match_annotations.tsv`
- **Delimiter**: Tab (`\t`)
- **Compatibility**: Microsoft Excel, Google Sheets, Pandas (`pd.read_csv('match_annotations.tsv', sep='\t')`)

### Schema:
| Column | Description |
| :--- | :--- |
| `frame_id` | Sequential frame index. |
| `image_path` | Relative path to corresponding raw frame image. |
| `timestamp_s` | Match video timestamp in seconds. |
| `num_players` | Total count of players on the field. |
| `ball_detected` | `True` / `False` indicator for football detection. |
| `team_a_count` | Number of outfield players in Team A. |
| `team_b_count` | Number of outfield players in Team B. |
| `top_speed_kmh` | Peak player sprint speed recorded in frame (km/h). |
| `team_a_space_control_pct` | Voronoi spatial pitch dominance % for Team A. |
| `team_b_space_control_pct` | Voronoi spatial pitch dominance % for Team B. |
| `bounding_boxes` | JSON array of bounding box coordinates `[[x1, y1, x2, y2], ...]`. |
| `player_roles` | JSON array of detected roles (`Team A`, `Team B`, `Referee`, `Goalkeeper`). |
| `player_speeds` | JSON dictionary mapping Player Track ID to live metric speed in km/h. |
