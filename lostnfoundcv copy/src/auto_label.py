from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2

from src.detector import DetectionRecord, LostFoundDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap YOLO labels from unlabeled lost-and-found photos."
    )
    parser.add_argument(
        "--incoming",
        default="training_data/incoming_photos",
        help="Folder containing unlabeled photos.",
    )
    parser.add_argument(
        "--output",
        default="training_data/labeled",
        help="Output folder containing YOLO train/val images and labels.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio between 0 and 1.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/val split.",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "train_images": base / "images" / "train",
        "val_images": base / "images" / "val",
        "train_labels": base / "labels" / "train",
        "val_labels": base / "labels" / "val",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def clear_previous_labels(paths: dict[str, Path]) -> None:
    for path in paths.values():
        for item in path.iterdir():
            if item.is_file():
                item.unlink()


def to_yolo_bbox(
    bbox: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / image_width
    y_center = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height


def write_label_file(
    label_path: Path,
    detections: list[DetectionRecord],
    class_to_id: dict[str, int],
    image_width: int,
    image_height: int,
) -> None:
    lines: list[str] = []
    for detection in detections:
        class_id = class_to_id[detection.subgroup]
        x_center, y_center, width, height = to_yolo_bbox(
            detection.bbox, image_width=image_width, image_height=image_height
        )
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_data_yaml(output_dir: Path, class_names: list[str]) -> None:
    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for idx, class_name in enumerate(class_names):
        yaml_lines.append(f"  {idx}: {class_name}")
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (0 < args.val_ratio < 1):
        raise ValueError("--val-ratio must be between 0 and 1")

    incoming_dir = Path(args.incoming)
    output_dir = Path(args.output)
    if not incoming_dir.exists():
        raise FileNotFoundError(f"Incoming folder not found: {incoming_dir}")

    image_paths = sorted(
        p for p in incoming_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(
            f"No photos found in {incoming_dir}. Add .jpg/.jpeg/.png/.webp files first."
        )

    detector = LostFoundDetector(confidence_threshold=args.conf)
    split_paths = ensure_dirs(output_dir)
    clear_previous_labels(split_paths)

    random.seed(args.seed)
    random.shuffle(image_paths)
    val_count = max(1, int(len(image_paths) * args.val_ratio))
    val_set = {p for p in image_paths[:val_count]}

    discovered_classes: set[str] = set()
    total_boxes = 0
    split_counts = {"train": 0, "val": 0}

    staged: list[tuple[Path, str, list[DetectionRecord], int, int]] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable file: {image_path}")
            continue
        image_height, image_width = image.shape[:2]
        detections = detector.detect(image_path)
        for detection in detections:
            discovered_classes.add(detection.subgroup)
        split = "val" if image_path in val_set else "train"
        staged.append((image_path, split, detections, image_width, image_height))
        split_counts[split] += 1
        total_boxes += len(detections)

    class_names = sorted(discovered_classes)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    for image_path, split, detections, image_width, image_height in staged:
        image_target = split_paths[f"{split}_images"] / image_path.name
        shutil.copy2(image_path, image_target)

        label_target = split_paths[f"{split}_labels"] / f"{image_path.stem}.txt"
        write_label_file(
            label_target,
            detections=detections,
            class_to_id=class_to_id,
            image_width=image_width,
            image_height=image_height,
        )

    write_data_yaml(output_dir, class_names)

    print(f"Processed images: {len(staged)}")
    print(f"Train images: {split_counts['train']}")
    print(f"Val images: {split_counts['val']}")
    print(f"Total boxes: {total_boxes}")
    print(f"Discovered classes: {len(class_names)}")
    print(f"YOLO data config: {(output_dir / 'data.yaml').as_posix()}")
    if not class_names:
        print(
            "No classes discovered. Try lowering --conf or add clearer photos with visible items."
        )
    else:
        print("Classes:", ", ".join(class_names))


if __name__ == "__main__":
    main()

