"""Train EGCA-Net after applying scripts/install_ultralytics_patch.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to a YOLO dataset YAML")
    parser.add_argument("--model", default="configs/egca_net.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/egca_net")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    model = YOLO(Path(args.model))
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="SGD",
        seed=0,
        deterministic=True,
        patience=0,
        close_mosaic=15,
        mosaic=0.08,
        mixup=0.09,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
