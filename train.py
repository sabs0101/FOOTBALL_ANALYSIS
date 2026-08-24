"""
Fine-Tuning Script for YOLOv8 on Custom Football Dataset (Transfer Learning).
Allows training or fine-tuning yolov8m.pt on custom labeled football match frames.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def train_yolo(
    data_yaml: str = "data/dataset/data.yaml",
    model_name: str = "yolov8m.pt",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 1280,
    device: str = "0",
    learning_rate: float = 0.01,
    optimizer: str = "AdamW",
    save_dir: str = "models/fine_tuned",
):
    """
    Fine-tune YOLOv8 on football dataset using Transfer Learning.
    """
    print("=" * 60)
    print(" AI Football YOLOv8 Fine-Tuning Pipeline (Transfer Learning)")
    print("=" * 60)
    print(f" Base Weights      : {model_name}")
    print(f" Dataset Config    : {data_yaml}")
    print(f" Target Epochs     : {epochs}")
    print(f" Batch Size        : {batch_size}")
    print(f" Image Size        : {imgsz}x{imgsz}")
    print(f" Optimizer / LR    : {optimizer} (lr0={learning_rate})")
    print(f" Compute Device    : CUDA:{device}" if torch.cuda.is_available() else "CPU")
    print("=" * 60)

    model = YOLO(model_name)

    # Launch fine-tuning training loop
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device if torch.cuda.is_available() else "cpu",
        lr0=learning_rate,
        optimizer=optimizer,
        project=save_dir,
        name="football_yolov8m",
        pretrained=True,
        verbose=True,
    )

    print("\n[Training Complete] Best weights saved to:", Path(save_dir) / "football_yolov8m" / "weights" / "best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Football Dataset")
    parser.add_argument("--data", type=str, default="data/dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")

    args = parser.parse_args()
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
