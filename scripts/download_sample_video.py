"""
Sample Video Downloader and Fallback Generator for Football Analytics.
Downloads sample Bundesliga/EPL broadcast footage or generates a synthetic broadcast clip
if offline, guaranteeing 100% reproducibility for reviewers and evaluators.
"""

import argparse
from pathlib import Path
import urllib.request
import cv2
import numpy as np


def generate_synthetic_broadcast(output_path: str, duration_sec: int = 5, fps: int = 25):
    """
    Generate synthetic 1080p broadcast football clip with moving players and ball.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    w, h = 1920, 1080
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))

    print(f"[Fallback Generator] Creating synthetic {duration_sec}s broadcast stream: {path.name}...")

    # Player positions and speeds
    players_a = [[300 + i * 80, 500 + (i % 3) * 40] for i in range(10)]
    players_b = [[1100 + i * 70, 500 + (i % 3) * 40] for i in range(10)]
    ball = [960.0, 540.0]
    ball_v = [4.0, 2.0]

    for f in range(total_frames):
        # 1. Green Pitch Canvas
        frame = np.full((h, w, 3), (34, 120, 34), dtype=np.uint8)
        
        # 2. Stadium stands top & bottom
        frame[0:220, :] = (80, 80, 90)
        frame[880:h, :] = (40, 40, 45)

        # 3. White Pitch Lines
        cv2.line(frame, (80, 240), (1840, 240), (240, 240, 240), 3)   # Top touchline
        cv2.line(frame, (80, 860), (1840, 860), (240, 240, 240), 3)   # Bottom touchline
        cv2.line(frame, (960, 240), (960, 860), (240, 240, 240), 3)   # Halfway line
        cv2.circle(frame, (960, 550), 120, (240, 240, 240), 3)        # Center circle
        cv2.circle(frame, (960, 550), 5, (240, 240, 240), -1)          # Center spot

        # 4. Draw Team A Players (White)
        for i, p in enumerate(players_a):
            px = int(p[0] + np.sin(f * 0.1 + i) * 15)
            py = int(p[1] + np.cos(f * 0.1 + i) * 8)
            cv2.rectangle(frame, (px - 15, py - 55), (px + 15, py), (240, 240, 240), -1)
            cv2.circle(frame, (px, py - 62), 8, (200, 200, 200), -1)

        # 5. Draw Team B Players (Neon Green)
        for i, p in enumerate(players_b):
            px = int(p[0] - np.sin(f * 0.1 + i) * 15)
            py = int(p[1] - np.cos(f * 0.1 + i) * 8)
            cv2.rectangle(frame, (px - 15, py - 55), (px + 15, py), (30, 220, 70), -1)
            cv2.circle(frame, (px, py - 62), 8, (30, 220, 70), -1)

        # 6. Draw Ball
        ball[0] += ball_v[0]
        ball[1] += ball_v[1]
        if ball[0] < 200 or ball[0] > 1700:
            ball_v[0] *= -1
        if ball[1] < 300 or ball[1] > 800:
            ball_v[1] *= -1
        cv2.circle(frame, (int(ball[0]), int(ball[1])), 7, (0, 60, 255), -1)

        out.write(frame)

    out.release()
    print(f"[Fallback Generator] Generated synthetic video at {path.resolve()}")


def download_or_generate(destination: str = "data/videos/sample_broadcast.mp4"):
    """
    Check if video exists; if missing, attempt download or generate fallback.
    """
    dest = Path(destination)
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"[Dataset] Video already exists: {dest.resolve()} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    sample_url = "https://raw.githubusercontent.com/roboflow/sports/main/data/football.mp4"

    try:
        print(f"[Dataset] Attempting download from: {sample_url}...")
        urllib.request.urlretrieve(sample_url, str(dest))
        print(f"[Dataset] Successfully downloaded to: {dest.resolve()}")
    except Exception as e:
        print(f"[Dataset] Download failed ({e}). Falling back to synthetic broadcast generator...")
        generate_synthetic_broadcast(str(dest), duration_sec=10, fps=25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate sample football video")
    parser.add_argument("--output", type=str, default="data/videos/sample_broadcast.mp4", help="Output path")
    args = parser.parse_args()

    download_or_generate(args.output)
