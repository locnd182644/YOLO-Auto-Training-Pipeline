"""Autolabel incoming raw images and route them by confidence.

For every image in ``--input``:
  * run the current model
  * drop detections whose confidence is below ``CONF_LOW``
  * if any surviving detection is below ``CONF_HIGH`` → route to ``needs_review``
  * otherwise → route to ``auto_accepted``

Labels are written in labelme JSON format so a human reviewer can open the
``needs_review`` queue directly in labelme. Processed source images are
moved to ``data/raw/archive`` to keep ``incoming`` a pure work queue.

Cold start: if ``--model`` does not exist (no ``models/current/best.pt`` yet),
the script falls back to the ``pretrained_fallback`` named in the config
(ultralytics auto-downloads it). Those weights don't know the project's class
list, so per-image inference is skipped and every image is routed to
``needs_review`` for manual labeling — the honest first-run behavior.
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
from PIL import Image
from ultralytics import YOLO

log = logging.getLogger("autolabel")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_labelme_doc(
    image_path: Path,
    image_size: tuple[int, int],
    detections: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    """Return a labelme-compatible dict for one image."""
    width, height = image_size
    shapes = []
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cls_id = int(det["cls"])
        if cls_id < 0 or cls_id >= len(class_names):
            raise ValueError(
                f"Model produced class id {cls_id} outside config classes "
                f"(len={len(class_names)}). Class list must match (Hard rule #5)."
            )
        shapes.append(
            {
                "label": class_names[cls_id],
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": f"conf={det['conf']:.3f}",
                "shape_type": "rectangle",
                "flags": {},
            }
        )
    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def classify_image(
    detections: list[dict[str, Any]],
    conf_high: float,
    conf_low: float,
) -> tuple[str, list[dict[str, Any]]]:
    """Decide routing bucket and return filtered detections.

    Images with zero surviving detections are sent to ``needs_review`` —
    a blank label set is a strong drift signal we do not want silently
    merged into training data.
    """
    kept = [d for d in detections if d["conf"] >= conf_low]
    if not kept:
        return "needs_review", kept
    if any(d["conf"] < conf_high for d in kept):
        return "needs_review", kept
    return "auto_accepted", kept


def autolabel_image(
    image_path: Path,
    model: YOLO,
    cfg: dict[str, Any],
    class_names: list[str],
) -> tuple[str, dict[str, Any]]:
    results = model.predict(
        source=str(image_path),
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        iou=cfg["iou"],
        max_det=cfg["max_det"],
        conf=cfg["CONF_LOW"],  # prefilter at the model boundary
        verbose=False,
    )
    result = results[0]
    detections: list[dict[str, Any]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            detections.append(
                {
                    "xyxy": box.xyxy[0].tolist(),
                    "conf": float(box.conf[0]),
                    "cls": int(box.cls[0]),
                }
            )
    bucket, kept = classify_image(detections, cfg["CONF_HIGH"], cfg["CONF_LOW"])

    with Image.open(image_path) as im:
        size = im.size  # (width, height)
    doc = build_labelme_doc(image_path, size, kept, class_names)
    return bucket, doc


def run(
    input_dir: Path,
    model_path: Path,
    config_path: Path,
    cold_start: bool = False,
) -> dict[str, int]:
    cfg = load_config(config_path)
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    class_names = cfg["classes"]

    for p in (paths["auto_accepted"], paths["needs_review"], paths["archive"]):
        p.mkdir(parents=True, exist_ok=True)

    log.info("Loading model from %s", model_path)
    model = YOLO(str(model_path))

    counts = {"auto_accepted": 0, "needs_review": 0, "skipped": 0}
    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    log.info("Found %d images in %s", len(images), input_dir)

    for img in images:
        try:
            if cold_start:
                with Image.open(img) as im:
                    size = im.size
                doc = build_labelme_doc(img, size, [], class_names)
                bucket = "needs_review"
            else:
                bucket, doc = autolabel_image(img, model, cfg, class_names)
        except Exception:
            log.exception("Failed to autolabel %s — skipping", img)
            counts["skipped"] += 1
            continue

        out_json = paths[bucket] / f"{img.stem}.json"
        with out_json.open("w") as f:
            json.dump(doc, f, indent=2)
        # Also copy the image next to the JSON so the reviewer can open it.
        shutil.copy2(img, paths[bucket] / img.name)
        # Move original out of the incoming queue.
        shutil.move(str(img), paths["archive"] / img.name)
        counts[bucket] += 1
        log.debug("%s → %s", img.name, bucket)

    log.info(
        "Autolabel done: %d auto_accepted, %d needs_review, %d skipped",
        counts["auto_accepted"],
        counts["needs_review"],
        counts["skipped"],
    )
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/incoming"))
    parser.add_argument("--model", type=Path, default=Path("models/current/best.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/autolabel.yaml"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    model_path = args.model
    cold_start = False
    if not model_path.exists():
        cfg = load_config(args.config)
        fallback = Path(cfg.get("pretrained_fallback", "yolo11n.pt"))
        log.warning(
            "COLD START: %s not found. Falling back to pretrained %s "
            "(ultralytics will auto-download). Cold-start weights do not know "
            "project classes — every image will be routed to needs_review for "
            "manual labeling.",
            model_path,
            fallback,
        )
        model_path = fallback
        cold_start = True

    run(args.input, model_path, args.config, cold_start=cold_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
