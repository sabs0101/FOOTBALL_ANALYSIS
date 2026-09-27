"""
High-Performance Inference Benchmark Suite for Football Tactical Analytics.
Evaluates PyTorch FP32, PyTorch FP16, ONNX Runtime, and TensorRT FP16 across resolutions and batch sizes.

Usage:
    python scripts/benchmark_inference.py --model models/yolov8m.pt --iterations 50
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from ultralytics import YOLO


def get_vram_usage_mb() -> float:
    """Return current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_model(
    model_source: str,
    imgsz: int = 1280,
    batch_size: int = 1,
    half: bool = True,
    device: str = "0",
    iterations: int = 50,
    warmup: int = 10,
) -> Dict[str, Any]:
    """
    Run an isolated inference benchmark on a specific model target.

    Args:
        model_source: Path to .pt, .onnx, or .engine file.
        imgsz: Image resolution.
        batch_size: Number of concurrent frames.
        half: Enable FP16 inference.
        device: CUDA device or 'cpu'.
        iterations: Number of timed passes.
        warmup: Number of untimed warmup passes.

    Returns:
        Dict with benchmark metrics (avg_ms, p95_ms, fps, vram_mb).
    """
    target_device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
    print(f"\n[Benchmark] Loading '{model_source}' (imgsz={imgsz}, B={batch_size}, FP16={half}, device={target_device})...")

    try:
        model = YOLO(model_source, task="detect")
    except Exception as e:
        print(f"[Benchmark] ERROR loading '{model_source}': {e}")
        return {"status": "error", "error": str(e)}

    # Create dummy frames
    if batch_size == 1:
        dummy_input = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    else:
        dummy_input = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(batch_size)]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        # Warmup passes
        for _ in range(warmup):
            _ = model.predict(
                source=dummy_input,
                imgsz=imgsz,
                device=target_device,
                half=half and (target_device != "cpu") and not str(model_source).endswith(".onnx"),
                verbose=False,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        latencies = []
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model.predict(
                source=dummy_input,
                imgsz=imgsz,
                device=target_device,
                half=half and (target_device != "cpu") and not str(model_source).endswith(".onnx"),
                verbose=False,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(elapsed)

        latencies = np.array(latencies)
        avg_ms = float(np.mean(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        fps = (batch_size * 1000.0) / avg_ms if avg_ms > 0 else 0.0
        vram_peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        )

        return {
            "status": "ok",
            "model": str(model_source),
            "imgsz": imgsz,
            "batch_size": batch_size,
            "half": half,
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
            "fps": fps,
            "vram_peak_mb": vram_peak_mb,
        }
    except Exception as e:
        print(f"[Benchmark] Execution failed for '{model_source}': {e}")
        return {"status": "error", "error": str(e)}


def run_benchmark_suite(
    model_name: str = "models/yolov8m.pt",
    iterations: int = 50,
) -> List[Dict[str, Any]]:
    """Run full matrix comparison of PyTorch FP32, PyTorch FP16, and Engines."""
    pt_path = Path(model_name)
    if not pt_path.exists():
        pt_path = Path("models") / model_name
        if not pt_path.exists():
            pt_path = Path("yolov8n.pt")

    stem = pt_path.stem
    parent = pt_path.parent
    engine_1280 = parent / f"{stem}.engine"
    onnx_1280 = parent / f"{stem}.onnx"

    configs = [
        # (name, path, imgsz, batch, half, device)
        ("PyTorch CPU FP32 (Baseline)", str(pt_path), 640, 1, False, "cpu"),
        ("PyTorch GPU FP32 (640px)", str(pt_path), 640, 1, False, "0"),
        ("PyTorch GPU FP16 (640px)", str(pt_path), 640, 1, True, "0"),
        ("PyTorch GPU FP32 (1280px)", str(pt_path), 1280, 1, False, "0"),
        ("PyTorch GPU FP16 (1280px)", str(pt_path), 1280, 1, True, "0"),
        ("PyTorch GPU FP16 (1280px, Batch=2)", str(pt_path), 1280, 2, True, "0"),
        ("PyTorch GPU FP16 (1280px, Batch=4)", str(pt_path), 1280, 4, True, "0"),
    ]

    if onnx_1280.exists():
        configs.append(("ONNX Runtime (1280px)", str(onnx_1280), 1280, 1, True, "0"))

    if engine_1280.exists():
        configs.append(("TensorRT FP16 Engine (1280px)", str(engine_1280), 1280, 1, True, "0"))
        configs.append(("TensorRT FP16 Engine (1280px, Batch=2)", str(engine_1280), 1280, 2, True, "0"))

    results = []
    print("\n" + "=" * 95)
    print("                      FOOTBALL VISION INFERENCE BENCHMARK MATRIX")
    print(f" Target Model: {pt_path} | Iterations: {iterations}")
    if torch.cuda.is_available():
        print(f" GPU Device  : {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    print("=" * 95)

    for label, path, imgsz, batch, half, dev in configs:
        if not os.path.exists(path) and not path.endswith(".pt"):
            continue
        res = benchmark_model(
            model_source=path,
            imgsz=imgsz,
            batch_size=batch,
            half=half,
            device=dev,
            iterations=iterations,
        )
        if res.get("status") == "ok":
            res["label"] = label
            results.append(res)
            print(
                f" -> {label:<40} | Latency: {res['avg_ms']:6.2f} ms (p95: {res['p95_ms']:6.2f} ms) | "
                f"Throughput: {res['fps']:6.1f} FPS | Peak VRAM: {res['vram_peak_mb']:6.1f} MB"
            )

    print("\n" + "=" * 95)
    print("                                SUMMARY REPORT")
    print("=" * 95)
    print(f"{'Configuration':<42} | {'ImgSz':<6} | {'Batch':<5} | {'Avg Latency':<12} | {'Throughput':<12} | {'Peak VRAM'}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['label']:<42} | {r['imgsz']:<6} | {r['batch_size']:<5} | "
            f"{r['avg_ms']:>8.2f} ms | {r['fps']:>8.1f} FPS | {r['vram_peak_mb']:>7.1f} MB"
        )
    print("=" * 95)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO detection inference latency and throughput.")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Path to YOLO model (.pt)")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    args = parser.parse_args()

    run_benchmark_suite(model_name=args.model, iterations=args.iterations)


if __name__ == "__main__":
    main()
