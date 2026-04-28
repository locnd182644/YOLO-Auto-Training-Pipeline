from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image


def write_config(tmp_path: Path, classes: list[str]) -> Path:
    p = tmp_path / "autolabel.yaml"
    with p.open("w") as f:
        yaml.safe_dump({"classes": classes}, f)
    return p


def make_pair(dir_: Path, stem: str, line: str = "0 0.5 0.5 0.1 0.1") -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{stem}.txt").write_text(line + "\n")
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(dir_ / f"{stem}.jpg")


def make_prev_dataset(base: Path, version: str, classes: list[str]) -> Path:
    root = base / version
    make_pair(root / "images/train", "old_a")
    (root / "labels/train").mkdir(parents=True, exist_ok=True)
    (root / "labels/train/old_a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    with (root / "data.yaml").open("w") as f:
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val",
             "nc": len(classes), "names": classes},
            f, sort_keys=False,
        )
    return root


def test_version_parse(merge_mod):
    v = merge_mod.Version.parse("v1.2")
    assert (v.major, v.minor) == (1, 2)
    assert str(v) == "v1.2"
    with pytest.raises(ValueError):
        merge_mod.Version.parse("1.2")
    with pytest.raises(ValueError):
        merge_mod.Version.parse("v1")


def test_deterministic_split_is_stable(merge_mod, tmp_path: Path):
    pairs = [(tmp_path / f"{i}.jpg", tmp_path / f"{i}.txt") for i in range(100)]
    a_train, a_val = merge_mod.deterministic_split(pairs, 0.2, seed=42)
    b_train, b_val = merge_mod.deterministic_split(pairs, 0.2, seed=42)
    assert [p[0].name for p in a_val] == [p[0].name for p in b_val]
    # Roughly correct ratio (tolerant bounds for 100 samples)
    assert 10 <= len(a_val) <= 35
    assert len(a_train) + len(a_val) == 100


def test_validate_unchanged_minor_bump_ok(merge_mod, tmp_path: Path):
    prev = make_prev_dataset(tmp_path, "v1.1", ["cat"]) / "data.yaml"
    merge_mod.validate_class_list(
        ["cat"],
        prev,
        merge_mod.Version(1, 1),
        merge_mod.Version(1, 2),
        allow_major=False,
    )


def test_validate_unchanged_major_bump_rejected(merge_mod, tmp_path: Path):
    prev = make_prev_dataset(tmp_path, "v1.1", ["cat"]) / "data.yaml"
    with pytest.raises(ValueError, match="reserved for class-list changes"):
        merge_mod.validate_class_list(
            ["cat"],
            prev,
            merge_mod.Version(1, 1),
            merge_mod.Version(2, 0),
            allow_major=True,
        )


def test_validate_changed_minor_bump_rejected(merge_mod, tmp_path: Path):
    prev = make_prev_dataset(tmp_path, "v1.1", ["cat"]) / "data.yaml"
    with pytest.raises(ValueError, match="REQUIRES a major bump"):
        merge_mod.validate_class_list(
            ["cat", "dog"],
            prev,
            merge_mod.Version(1, 1),
            merge_mod.Version(1, 2),
            allow_major=False,
        )


def test_validate_changed_major_without_allow_rejected(merge_mod, tmp_path: Path):
    prev = make_prev_dataset(tmp_path, "v1.1", ["cat"]) / "data.yaml"
    with pytest.raises(ValueError, match="--allow-major"):
        merge_mod.validate_class_list(
            ["cat", "dog"],
            prev,
            merge_mod.Version(1, 1),
            merge_mod.Version(2, 0),
            allow_major=False,
        )


def test_validate_changed_major_with_allow_ok(merge_mod, tmp_path: Path):
    prev = make_prev_dataset(tmp_path, "v1.1", ["cat"]) / "data.yaml"
    merge_mod.validate_class_list(
        ["cat", "dog"],
        prev,
        merge_mod.Version(1, 1),
        merge_mod.Version(2, 0),
        allow_major=True,
    )


def test_run_end_to_end(merge_mod, tmp_path: Path):
    # Previous version with one existing train image
    datasets = tmp_path / "datasets"
    make_prev_dataset(datasets, "v1.1", ["cat"])

    # New labeled samples in flat yolo_txt dir
    yolo_dir = tmp_path / "yolo_txt"
    for i in range(20):
        make_pair(yolo_dir, f"new_{i:02d}")

    cfg = write_config(tmp_path, ["cat"])
    new_dir = merge_mod.run(
        prev="v1.1",
        new="v1.2",
        yolo_dir=yolo_dir,
        datasets_dir=datasets,
        config_path=cfg,
        val_fraction=0.2,
        seed=42,
        allow_major=False,
        run_dvc=False,
        run_tag=False,
    )
    assert new_dir == datasets / "v1.2"
    assert (new_dir / "data.yaml").exists()
    # Old sample preserved
    assert (new_dir / "images/train/old_a.jpg").exists()
    # New samples distributed across train+val
    train_new = list((new_dir / "images/train").glob("new_*.jpg"))
    val_new = list((new_dir / "images/val").glob("new_*.jpg"))
    assert len(train_new) + len(val_new) == 20
    assert len(val_new) > 0
    # Each image has its matching label
    for img in train_new + val_new:
        rel = img.relative_to(new_dir / "images")
        lbl = new_dir / "labels" / rel.with_suffix(".txt")
        assert lbl.exists(), f"missing label for {img}"


def test_run_refuses_existing_version(merge_mod, tmp_path: Path):
    datasets = tmp_path / "datasets"
    make_prev_dataset(datasets, "v1.1", ["cat"])
    (datasets / "v1.2").mkdir()  # already exists
    yolo_dir = tmp_path / "yolo_txt"
    make_pair(yolo_dir, "n")
    cfg = write_config(tmp_path, ["cat"])
    with pytest.raises(FileExistsError):
        merge_mod.run(
            prev="v1.1", new="v1.2",
            yolo_dir=yolo_dir, datasets_dir=datasets, config_path=cfg,
            val_fraction=0.1, seed=42,
            allow_major=False, run_dvc=False, run_tag=False,
        )
