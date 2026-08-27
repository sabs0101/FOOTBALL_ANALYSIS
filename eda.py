"""
Exploratory Data Analysis (EDA) CLI Script.
Inspects video properties, turf color profiles, lighting distribution, and motion metrics.

Usage:
    python eda.py --source data/videos/sample_broadcast.mp4 --output outputs/eda/eda_report.json
"""

import argparse
from pathlib import Path
from src.preprocessing.eda import VideoEDA


def main():
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis on Football Video")
    parser.add_argument("--source", type=str, default="data/videos/sample_broadcast.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/eda/eda_report.json", help="Path to output JSON report")
    parser.add_argument("--samples", type=int, default=40, help="Number of sampled frames")

    args = parser.parse_args()

    print("=" * 60)
    print(" AI Football Analysis - Exploratory Data Analysis (EDA)")
    print("=" * 60)
    print(f" Source Video : {args.source}")
    print(f" Sample Frames: {args.samples}")
    print(f" Output File  : {args.output}")
    print("=" * 60)

    eda = VideoEDA()
    report = eda.analyze_video(args.source, max_samples=args.samples)
    eda.save_report(report, args.output)

    print("\n--- Summary of Findings ---")
    print(f"Resolution       : {report.metadata.width}x{report.metadata.height} ({report.metadata.aspect_ratio})")
    print(f"Total Duration   : {report.metadata.duration_seconds}s ({report.metadata.total_frames} frames @ {report.metadata.fps} FPS)")
    print(f"Pitch Turf Cover : {report.grass_coverage_pct}%")
    print(f"Mean Illumination: {report.mean_brightness:.1f}/255")
    print(f"Blurry Frames    : {report.blurry_frames_pct}%")
    print("Recommendations  :")
    for rec in report.recommendations:
        print(f"  • {rec}")
    print("=" * 60)


if __name__ == "__main__":
    main()
