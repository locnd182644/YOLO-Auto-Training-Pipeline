"""Convert labelme JSON files to YOLO txt.

Only reads from the ``auto_accepted/`` and ``reviewed/`` buckets — the
``needs_review/`` bucket is deliberately skipped (Hard rule #3). Output
is one ``.txt`` per input JSON, plus the matching image copied alongside,
in ``data/labeled/yolo_txt/``.

Each YOLO line is ``<cls_id> <cx> <cy> <w> <h>`` with all spatial values
normalized to [0, 1].
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("convert")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_BUCKETS = ("auto_accepted", "reviewed")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def labelme_shape_to_yolo(
    shape: dict[str, Any],
    img_w: int,
    img_h: int,
    class_to_id: dict[str, int],
) -> str:
    """Convert one labelme rectangle shape to a YOLO line.

    Labelme rectangles store two arbitrary corner points; normalize to
    top-left / bottom-right before converting to YOLO's normalized
    center-x/center-y/width/height.
    """
    if shape["shape_type"] != "rectangle":
        raise ValueError(
            f"Only 'rectangle' shapes supported, got '{shape['shape_type']}'"
        )
    label = shape["label"]
    if label not in class_to_id:
        raise ValueError(
            f"Label '{label}' is not in the configured class list "
            f"{list(class_to_id)}. Class list is immutable within a major "
            f"version (Hard rule #5)."
        )
    (x1, y1), (x2, y2) = shape["points"]
    x_min, x_max = sorted((float(x1), float(x2)))
    y_min, y_max = sorted((float(y1), float(y2)))
    if x_min < 0 or y_min < 0 or x_max > img_w or y_max > img_h:
        raise ValueError(
            f"Bounding box out of image bounds for '{label}': "
            f"points=({x_min:.1f},{y_min:.1f})→({x_max:.1f},{y_max:.1f}) "
            f"image={img_w}x{img_h}"
        )
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Degenerate bounding box for '{label}': "
            f"({x_min:.1f},{y_min:.1f})→({x_max:.1f},{y_max:.1f})"
        )
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    cls_id = class_to_id[label]
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_one(
    json_path: Path,
    out_dir: Path,
    class_to_id: dict[str, int],
    source_image_dir: Path,
) -> Path:
    """Convert a single labelme JSON, copy its image, return the txt path."""
    with json_path.open("r") as f:
        doc = json.load(f)
    img_w = int(doc["imageWidth"])
    img_h = int(doc["imageHeight"])
    lines = [
        labelme_shape_to_yolo(shape, img_w, img_h, class_to_id)
        for shape in doc["shapes"]
    ]
    out_txt = out_dir / f"{json_path.stem}.txt"
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))

    # Copy the image next to the txt — training needs paired files.
    image_name = doc.get("imagePath") or ""
    candidates = []
    if image_name:
        candidates.append(source_image_dir / image_name)
    for suf in IMAGE_SUFFIXES:
        candidates.append(source_image_dir / f"{json_path.stem}{suf}")
    for c in candidates:
        if c.exists():
            shutil.copy2(c, out_dir / c.name)
            break
    else:
        raise FileNotFoundError(
            f"No image found for {json_path.name} in {source_image_dir}"
        )
    return out_txt


def run(
    labelme_root: Path,
    out_dir: Path,
    config_path: Path,
) -> dict[str, int]:
    cfg = load_config(config_path)
    class_to_id = {name: idx for idx, name in enumerate(cfg["classes"])}
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"converted": 0, "failed": 0, "skipped_needs_review": 0}

    # Explicit whitelist: we only read from auto_accepted and reviewed.
    # needs_review is skipped here to guarantee Hard rule #3.
    for bucket in ALLOWED_BUCKETS:
        bucket_dir = labelme_root / bucket
        if not bucket_dir.is_dir():
            log.warning("Bucket %s does not exist — skipping", bucket_dir)
            continue
        for json_path in sorted(bucket_dir.glob("*.json")):
            try:
                convert_one(json_path, out_dir, class_to_id, bucket_dir)
                counts["converted"] += 1
            except Exception:
                log.exception("Failed to convert %s", json_path)
                counts["failed"] += 1

    needs_review = labelme_root / "needs_review"
    if needs_review.is_dir():
        counts["skipped_needs_review"] = len(list(needs_review.glob("*.json")))

    log.info(
        "Convert done: %d converted, %d failed, %d pending in needs_review",
        counts["converted"],
        counts["failed"],
        counts["skipped_needs_review"],
    )
    if counts["failed"] > 0:
        raise RuntimeError(
            f"{counts['failed']} conversions failed — see log for details"
        )
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labelme-root",
        type=Path,
        default=Path("data/labeled/labelme_json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/labeled/yolo_txt")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/autolabel.yaml")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    run(args.labelme_root, args.out, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
