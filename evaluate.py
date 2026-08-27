"""
Quantitative Model & Pipeline Evaluation CLI Script.
Computes mAP, MOTA, MOTP, Reprojection RMSE, and Physical Kinematics Realism.

Usage:
    python evaluate.py --source data/videos/sample_broadcast.mp4 --output outputs/evaluation/evaluation_report.json --frames 150
"""

import argparse
from src.evaluation.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run Quantitative Benchmark Evaluation on Football Video Pipeline")
    parser.add_argument("--source", type=str, default="data/videos/sample_broadcast.mp4", help="Path to evaluation video")
    parser.add_argument("--model", type=str, default="models/yolov8m.pt", help="Path to YOLO model checkpoint")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    parser.add_argument("--output", type=str, default="outputs/evaluation/evaluation_report.json", help="Output report JSON")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames to evaluate")

    args = parser.parse_args()

    print("=" * 65)
    print(" AI Football Analysis - Quantitative Benchmark Evaluator")
    print("=" * 65)
    print(f" Target Video   : {args.source}")
    print(f" Model Weights  : {args.model}")
    print(f" Device         : CUDA:{args.device}")
    print(f" Eval Frames    : {args.frames}")
    print(f" Output Report  : {args.output}")
    print("=" * 65)

    evaluator = ModelEvaluator(model_name=args.model, device=args.device)
    report = evaluator.evaluate_video(args.source, max_eval_frames=args.frames)
    evaluator.save_report(report, args.output)

    print("\n" + "=" * 65)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f" Overall Score        : {report.overall_pipeline_score:.1f} / 100")
    print(f" Processing Speed     : {report.fps_throughput:.2f} FPS")
    print(f" Detection Precision  : {report.detection.precision * 100:.2f}% (mAP@50: {report.detection.map_50 * 100:.2f}%)")
    print(f" Tracking MOTA        : {report.tracking.mota * 100:.2f}% (IDF1: {report.tracking.idf1 * 100:.2f}%)")
    print(f" Homography RMSE      : {report.homography.reprojection_rmse_m:.3f} meters")
    print(f" Valid Speed Adherence: {report.kinematics.valid_speed_pct:.2f}%")
    print(f" Peak Sprint Speed    : {report.kinematics.max_sprint_kmh:.1f} km/h")
    print("=" * 65)


if __name__ == "__main__":
    main()
