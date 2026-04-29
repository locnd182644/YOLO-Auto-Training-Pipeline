from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image


CONFIG = {
    "CONF_HIGH": 0.6,
    "CONF_LOW": 0.25,
    "imgsz": 640,
    "device": "cpu",
    "iou": 0.5,
    "max_det": 300,
    "paths": {
        "incoming": "data/raw/incoming",
        "archive": "data/raw/archive",
        "auto_accepted": "data/labeled/labelme_json/auto_accepted",
        "needs_review": "data/labeled/labelme_json/needs_review",
        "reviewed": "data/labeled/labelme_json/reviewed",
    },
    "classes": ["cat", "dog"],
}


def write_config(tmp_path: Path) -> Path:
    p = tmp_path / "autolabel.yaml"
    with p.open("w") as f:
        yaml.safe_dump(CONFIG, f)
    return p


def make_labelme(
    tmp_path: Path,
    bucket: str,
    stem: str,
    shapes: list[dict],
    w: int = 100,
    h: int = 200,
) -> Path:
    bucket_dir = tmp_path / "labelme_json" / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{stem}.jpg",
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    json_path = bucket_dir / f"{stem}.json"
    with json_path.open("w") as f:
        json.dump(doc, f)
    Image.new("RGB", (w, h), color=(0, 0, 0)).save(bucket_dir / f"{stem}.jpg")
    return json_path


def test_rectangle_to_yolo_basic(convert_mod):
    shape = {
        "label": "cat",
        "points": [[10.0, 20.0], [50.0, 80.0]],
        "shape_type": "rectangle",
    }
    line = convert_mod.labelme_shape_to_yolo(shape, 100, 200, {"cat": 0, "dog": 1})
    cls_id, cx, cy, w, h = line.split()
    # center x = 30 / 100, center y = 50 / 200, w = 40/100, h = 60/200
    assert cls_id == "0"
    assert float(cx) == pytest.approx(0.30)
    assert float(cy) == pytest.approx(0.25)
    assert float(w) == pytest.approx(0.40)
    assert float(h) == pytest.approx(0.30)


def test_rectangle_to_yolo_swapped_corners(convert_mod):
    """Labelme stores arbitrary corner order — we must normalize."""
    shape = {
        "label": "dog",
        "points": [[50.0, 80.0], [10.0, 20.0]],  # bottom-right, top-left
        "shape_type": "rectangle",
    }
    line = convert_mod.labelme_shape_to_yolo(shape, 100, 200, {"cat": 0, "dog": 1})
    assert line.split()[0] == "1"
    assert float(line.split()[1]) == pytest.approx(0.30)


def test_reject_non_rectangle(convert_mod):
    shape = {"label": "cat", "points": [[1, 1]], "shape_type": "point"}
    with pytest.raises(ValueError, match="rectangle"):
        convert_mod.labelme_shape_to_yolo(shape, 100, 200, {"cat": 0})


def test_reject_unknown_label(convert_mod):
    shape = {"label": "elephant", "points": [[1, 1], [2, 2]], "shape_type": "rectangle"}
    with pytest.raises(ValueError, match="not in the configured class list"):
        convert_mod.labelme_shape_to_yolo(shape, 100, 200, {"cat": 0, "dog": 1})


def test_clamp_out_of_bounds(convert_mod, caplog):
    shape = {
        "label": "cat",
        "points": [[-5.0, 0.0], [50.0, 80.0]],
        "shape_type": "rectangle",
    }
    with caplog.at_level("WARNING"):
        line = convert_mod.labelme_shape_to_yolo(shape, 100, 200, {"cat": 0})
    assert "Clamping" in caplog.text
    cls_id, cx, cy, w, h = line.split()
    assert cls_id == "0"
    assert float(cx) == pytest.approx(25.0 / 100)
    assert float(w) == pytest.approx(50.0 / 100)


def test_reject_degenerate_after_clamp(convert_mod):
    shape = {
        "label": "cat",
        "points": [[-10.0, -10.0], [-1.0, 50.0]],
        "shape_type": "rectangle",
    }
    with pytest.raises(ValueError, match="Degenerate"):
        convert_mod.labelme_shape_to_yolo(shape, 100, 200, {"cat": 0})


def test_run_skips_needs_review(convert_mod, tmp_path: Path):
    """Hard rule #3: needs_review must never feed training data."""
    good_shape = {
        "label": "cat",
        "points": [[10.0, 20.0], [50.0, 80.0]],
        "shape_type": "rectangle",
    }
    make_labelme(tmp_path, "auto_accepted", "img_ok", [good_shape])
    make_labelme(tmp_path, "reviewed", "img_reviewed", [good_shape])
    # This one would fail validation if read:
    make_labelme(
        tmp_path,
        "needs_review",
        "img_pending",
        [{"label": "elephant", "points": [[0, 0], [10, 10]], "shape_type": "rectangle"}],
    )

    out_dir = tmp_path / "yolo_txt"
    cfg_path = write_config(tmp_path)

    counts = convert_mod.run(
        labelme_root=tmp_path / "labelme_json",
        out_dir=out_dir,
        config_path=cfg_path,
    )
    assert counts["converted"] == 2
    assert counts["failed"] == 0
    assert counts["skipped_needs_review"] == 1
    assert (out_dir / "img_ok.txt").exists()
    assert (out_dir / "img_reviewed.txt").exists()
    assert not (out_dir / "img_pending.txt").exists()
