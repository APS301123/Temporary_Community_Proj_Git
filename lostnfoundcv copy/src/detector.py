from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld

from src.taxonomy import map_label_to_taxonomy


LOST_FOUND_PROMPTS = [
    "water bottle",
    "insulated bottle",
    "coffee tumbler",
    "cup",
    "lunch box",
    "food container",
    "lunch bag",
    "hoodie",
    "jacket",
    "sweater",
    "t-shirt",
    "school uniform shirt",
    "pants",
    "shorts",
    "backpack",
    "bag",
    "pencil pouch",
    "notebook",
    "textbook",
    "folder",
    "calculator",
    "smartphone",
    "headphones",
    "charger",
    "umbrella",
    "hat",
    "keys",
    "wallet",
    "eyeglasses",
    "shoes",
    "sneakers",
]

APPAREL_FOCUSED_PROMPTS = [
    "jacket",
    "hoodie",
    "sweater",
    "sweatshirt",
    "shirt",
    "polo shirt",
    "t-shirt",
    "long sleeve shirt",
    "school uniform shirt",
    "button-down shirt",
    "coat",
    "jersey",
    "jacket on hanger",
    "sweater on hanger",
    "shirt on hanger",
    "hoodie on hanger",
]

COLOR_PALETTE = {
    "black": np.array([30, 30, 30], dtype=np.float32),
    "white": np.array([230, 230, 230], dtype=np.float32),
    "gray": np.array([130, 130, 130], dtype=np.float32),
    "red": np.array([50, 50, 200], dtype=np.float32),
    "orange": np.array([50, 120, 220], dtype=np.float32),
    "yellow": np.array([90, 210, 230], dtype=np.float32),
    "green": np.array([70, 170, 70], dtype=np.float32),
    "blue": np.array([190, 90, 40], dtype=np.float32),
    "purple": np.array([150, 90, 160], dtype=np.float32),
    "pink": np.array([175, 140, 225], dtype=np.float32),
    "brown": np.array([60, 90, 140], dtype=np.float32),
}


@dataclass
class DetectionRecord:
    label: str
    group: str
    subgroup: str
    descriptive_subgroup: str
    confidence: float
    bbox: tuple[int, int, int, int]
    color: list[str]
    size_bucket: str
    foreground_score: float = 0.0


class LostFoundDetector:
    def __init__(self, confidence_threshold: float = 0.2) -> None:
        self.confidence_threshold = confidence_threshold
        self.model, self.is_world_model = self._load_model()

    def _load_model(self):
        try:
            model = YOLOWorld("yolov8s-worldv2.pt")
            model.set_classes(LOST_FOUND_PROMPTS)
            return model, True
        except Exception:
            # Fallback to broad COCO detector if YOLO-World is unavailable.
            return YOLO("yolov8n.pt"), False

    def detect(self, image_path: str | Path) -> list[DetectionRecord]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")

        height, width = image.shape[:2]
        raw_detections: list[DetectionRecord] = []

        if self.is_world_model:
            self.model.set_classes(LOST_FOUND_PROMPTS)
            baseline_result = self.model.predict(
                source=str(image_path),
                conf=self.confidence_threshold,
                verbose=False,
            )[0]
            raw_detections.extend(
                parse_detections(
                    result=baseline_result,
                    image=image,
                    image_width=width,
                    image_height=height,
                    min_confidence=self.confidence_threshold,
                )
            )

            apparel_confidence = max(0.08, self.confidence_threshold * 0.6)
            self.model.set_classes(APPAREL_FOCUSED_PROMPTS)
            apparel_result = self.model.predict(
                source=str(image_path),
                conf=apparel_confidence,
                verbose=False,
            )[0]
            raw_detections.extend(
                parse_detections(
                    result=apparel_result,
                    image=image,
                    image_width=width,
                    image_height=height,
                    min_confidence=apparel_confidence,
                )
            )
        else:
            result = self.model.predict(
                source=str(image_path),
                conf=self.confidence_threshold,
                verbose=False,
            )[0]
            raw_detections.extend(
                parse_detections(
                    result=result,
                    image=image,
                    image_width=width,
                    image_height=height,
                    min_confidence=self.confidence_threshold,
                )
            )

        deduped = dedupe_detections(raw_detections)
        return suppress_generic_clothing(deduped)


def parse_detections(
    result,
    image: np.ndarray,
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> list[DetectionRecord]:
    detections: list[DetectionRecord] = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes:
        confidence = float(box.conf.item())
        if confidence < min_confidence:
            continue

        cls_idx = int(box.cls.item())
        label = str(names[cls_idx]).strip()
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_width - 1, x2), min(image_height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        colors = dominant_color_names(crop)
        size_bucket = describe_box_size((x1, y1, x2, y2), image_width, image_height)
        taxonomy = map_label_to_taxonomy(label)
        descriptive_subgroup = f"{taxonomy.subgroup} | {'/'.join(colors)} | {size_bucket}"
        fg_score = _foreground_score(confidence, (x1, y1, x2, y2), image_width, image_height)
        detections.append(
            DetectionRecord(
                label=label,
                group=taxonomy.group,
                subgroup=taxonomy.subgroup,
                descriptive_subgroup=descriptive_subgroup,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                color=colors,
                size_bucket=size_bucket,
                foreground_score=fg_score,
            )
        )
    return detections


def _foreground_score(
    confidence: float,
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> float:
    x1, y1, x2, y2 = bbox
    box_area = max(1, (x2 - x1) * (y2 - y1))
    image_area = max(1, image_width * image_height)
    area_ratio = box_area / image_area
    # Larger boxes (foreground) get up to 3x boost
    area_boost = min(3.0, 1.0 + 4.0 * area_ratio)
    # Centrality: 1.0 at center, lower toward edges
    cx = (x1 + x2) / (2 * image_width)
    cy = (y1 + y2) / (2 * image_height)
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    centrality = max(0.4, 1.0 - dist * 1.2)
    return confidence * area_boost * centrality


def dedupe_detections(
    detections: list[DetectionRecord], iou_threshold: float = 0.55
) -> list[DetectionRecord]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.foreground_score, reverse=True)
    kept: list[DetectionRecord] = []

    for candidate in ordered:
        duplicate = False
        for existing in kept:
            if candidate.subgroup != existing.subgroup:
                continue
            if bbox_iou(candidate.bbox, existing.bbox) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    return kept


def suppress_generic_clothing(detections: list[DetectionRecord]) -> list[DetectionRecord]:
    if not detections:
        return []

    specific_apparel = [
        item
        for item in detections
        if item.group == "Apparel" and item.subgroup not in {"Clothing", "Uncategorized"}
    ]
    if not specific_apparel:
        return detections

    filtered: list[DetectionRecord] = []
    for item in detections:
        if item.group != "Apparel" or item.subgroup != "Clothing":
            filtered.append(item)
            continue

        has_specific_overlap = any(
            bbox_iou(item.bbox, specific.bbox) >= 0.12 for specific in specific_apparel
        )
        if not has_specific_overlap:
            filtered.append(item)

    return filtered


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union_area = max(1, area_a + area_b - inter_area)
    return inter_area / union_area


def dominant_color_names(crop_bgr: np.ndarray, max_colors: int = 2) -> list[str]:
    """Return up to max_colors best-guess color names using HSV per-pixel analysis."""
    if crop_bgr.size == 0:
        return ["unknown"]

    # Resize for performance
    h_px, w_px = crop_bgr.shape[:2]
    if h_px * w_px > 10000:
        scale = (10000.0 / (h_px * w_px)) ** 0.5
        crop_bgr = cv2.resize(
            crop_bgr, (max(1, int(w_px * scale)), max(1, int(h_px * scale)))
        )

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].flatten().astype(np.int32)
    S = hsv[:, :, 1].flatten().astype(np.int32)
    V = hsv[:, :, 2].flatten().astype(np.int32)
    n = len(H)

    # Achromatic masks (low saturation or very dark)
    is_black = V < 40
    is_white = (S < 25) & (V > 190)
    is_gray = (S < 25) & ~is_black & ~is_white
    is_chroma = ~is_black & ~is_white & ~is_gray  # S >= 25 and V >= 40

    # Chromatic hue ranges (OpenCV hue is 0–179)
    is_red = is_chroma & ((H < 10) | (H >= 165))
    is_orange = is_chroma & (H >= 10) & (H < 22) & (V >= 150)
    is_brown = is_chroma & (H >= 10) & (H < 25) & (V < 150)
    is_yellow = is_chroma & (H >= 22) & (H < 40) & (V >= 100)
    is_brown |= is_chroma & (H >= 22) & (H < 40) & (V < 100)
    is_green = is_chroma & (H >= 40) & (H < 85)
    # Teal/cyan/blue all mapped to blue for broad matching
    is_blue = is_chroma & (H >= 85) & (H < 130)
    is_purple = is_chroma & (H >= 130) & (H < 155)
    is_pink = is_chroma & (H >= 155) & (H < 165)

    counts = {
        "black": int(np.sum(is_black)),
        "white": int(np.sum(is_white)),
        "gray": int(np.sum(is_gray)),
        "red": int(np.sum(is_red)),
        "orange": int(np.sum(is_orange)),
        "yellow": int(np.sum(is_yellow)),
        "green": int(np.sum(is_green)),
        "blue": int(np.sum(is_blue)),
        "purple": int(np.sum(is_purple)),
        "pink": int(np.sum(is_pink)),
        "brown": int(np.sum(is_brown)),
    }

    sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    result: list[str] = []
    for name, count in sorted_colors:
        if count == 0:
            continue
        if not result:
            result.append(name)
        elif count / n >= 0.20:
            result.append(name)
        if len(result) >= max_colors:
            break

    return result if result else ["unknown"]


def nearest_color(avg_bgr: np.ndarray) -> str:
    best_name = "unknown"
    best_dist = float("inf")
    for name, reference_bgr in COLOR_PALETTE.items():
        dist = np.linalg.norm(avg_bgr - reference_bgr)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def describe_box_size(
    bbox: tuple[int, int, int, int], image_width: int, image_height: int
) -> str:
    x1, y1, x2, y2 = bbox
    box_area = max(1, (x2 - x1) * (y2 - y1))
    image_area = max(1, image_width * image_height)
    ratio = box_area / image_area
    if ratio < 0.03:
        return "small"
    if ratio < 0.12:
        return "medium"
    return "large"


def count_by(items: Iterable[DetectionRecord], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = getattr(item, key)
        counts[value] = counts.get(value, 0) + 1
    return counts

