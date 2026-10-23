from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO11s for football object detection."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the football dataset YAML file.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="Training device, e.g. 0 for GPU or cpu.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO("yolo11s.pt")

    model.train(
        data=args.data,
        epochs=30,
        imgsz=1280,
        batch=4,
        device=args.device,
        seed=42,
        deterministic=True,
        project="runs",
        name="football_detector",
    )


if __name__ == "__main__":
    main()