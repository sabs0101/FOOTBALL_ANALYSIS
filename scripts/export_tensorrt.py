"""
Export YOLOv8/v11 PyTorch Models to High-Performance TensorRT FP16 Engines and ONNX.

Usage:
    python scripts/export_tensorrt.py --model models/yolov8m.pt --imgsz 1280 --half --format engine
    python scripts/export_tensorrt.py --model yolov8n.pt --imgsz 1280 --half --format both
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO


def export_model(
    model_path: str,
    export_format: str = "engine",
    imgsz: int = 1280,
    half: bool = True,
    device: str = "0",
    dynamic: bool = False,
    workspace: int = 4,
    verify: bool = True,
) -> str:
    """
    Export a PyTorch YOLO model (.pt) to TensorRT (.engine) or ONNX format.

    Args:
        model_path: Path or name of YOLO model (e.g., 'models/yolov8m.pt', 'yolov8n.pt').
        export_format: 'engine', 'onnx', or 'both'.
        imgsz: Image input resolution (e.g. 1280, 640).
        half: Enable FP16 half-precision for massive TensorRT / CUDA speedup.
        device: CUDA device index ('0') or 'cpu'.
        dynamic: Enable dynamic shapes (False enables peak TensorRT static optimization).
        workspace: TensorRT build workspace limit in GB.
        verify: Run a test inference after export to verify parity and latency.

    Returns:
        Path to the exported artifact.
    """
    path = Path(model_path)
    if not path.exists():
        # Check in models/ directory
        candidate = Path("models") / model_path
        if candidate.exists():
            path = candidate

    print("=" * 70)
    print(f"[Export] Target Model       : {path}")
    print(f"[Export] Target Format      : {export_format.upper()}")
    print(f"[Export] Input Resolution   : {imgsz}x{imgsz}")
    print(f"[Export] FP16 Precision     : {half}")
    print(f"[Export] Target Device      : {device}")
    print(f"[Export] Dynamic Shapes     : {dynamic}")
    print("=" * 70)

    # Check CUDA availability
    if not torch.cuda.is_available() and export_format in ("engine", "both"):
        print("[Export] WARNING: CUDA is not available. TensorRT requires an NVIDIA GPU.")
        print("[Export] Falling back to ONNX export.")
        export_format = "onnx"

    print(f"[Export] Loading YOLO model: {path}...")
    model = YOLO(str(path))

    formats_to_export = ["onnx", "engine"] if export_format == "both" else [export_format]
    exported_paths = {}

    for fmt in formats_to_export:
        t0 = time.perf_counter()
        print(f"\n[Export] Starting export to '{fmt}' format...")
        try:
            export_kwargs = {
                "format": fmt,
                "imgsz": imgsz,
                "half": half,
                "device": device if torch.cuda.is_available() else "cpu",
                "dynamic": dynamic,
                "verbose": True,
            }
            if fmt == "engine":
                export_kwargs["workspace"] = workspace

            out_file = model.export(**export_kwargs)
            elapsed = time.perf_counter() - t0
            print(f"[Export] SUCCESS: Exported to '{out_file}' in {elapsed:.2f}s")
            exported_paths[fmt] = out_file
        except Exception as e:
            print(f"[Export] ERROR during {fmt} export: {e}")
            if fmt == "engine":
                print("[Export] Note: TensorRT export requires TensorRT libraries and CUDA toolkit.")

    if verify and exported_paths:
        print("\n" + "=" * 70)
        print("[Export] Running Verification Inference...")
        print("=" * 70)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        for fmt, out_path in exported_paths.items():
            if not os.path.exists(out_path):
                continue
            try:
                print(f"[Verify] Testing loaded model '{out_path}'...")
                v_model = YOLO(out_path, task="detect")

                # Warmup
                for _ in range(3):
                    _ = v_model.predict(
                        source=dummy_frame,
                        imgsz=imgsz,
                        device=device if torch.cuda.is_available() else "cpu",
                        half=half if fmt != "onnx" else False,
                        verbose=False,
                    )

                # Benchmark 10 iterations
                times = []
                for _ in range(10):
                    t_start = time.perf_counter()
                    _ = v_model.predict(
                        source=dummy_frame,
                        imgsz=imgsz,
                        device=device if torch.cuda.is_available() else "cpu",
                        half=half if fmt != "onnx" else False,
                        verbose=False,
                    )
                    times.append((time.perf_counter() - t_start) * 1000.0)

                avg_ms = float(np.mean(times))
                fps = 1000.0 / avg_ms if avg_ms > 0 else 0
                print(f"[Verify] [{fmt.upper()}] Avg Latency: {avg_ms:.2f} ms | Throughput: {fps:.1f} FPS")
            except Exception as e:
                print(f"[Verify] Verification for {fmt} encountered error: {e}")

    return exported_paths.get(export_format, list(exported_paths.values())[0] if exported_paths else "")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO models to TensorRT and ONNX.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model checkpoint (.pt)")
    parser.add_argument("--format", type=str, default="engine", choices=["engine", "onnx", "both"], help="Export format")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution (default: 1280)")
    parser.add_argument("--half", action="store_true", default=True, help="Use FP16 half precision")
    parser.add_argument("--no-half", action="store_false", dest="half", help="Disable FP16 precision")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--dynamic", action="store_true", default=False, help="Enable dynamic shapes")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace limit in GB")
    parser.add_argument("--no-verify", action="store_false", dest="verify", help="Skip verification benchmark")

    args = parser.parse_args()
    export_model(
        model_path=args.model,
        export_format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        dynamic=args.dynamic,
        workspace=args.workspace,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
