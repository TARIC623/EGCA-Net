"""Evaluate a trained EGCA-Net checkpoint."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    YOLO(args.weights).val(data=args.data, split="val", imgsz=args.imgsz, batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
