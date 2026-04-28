"""Merge newly converted labels into a new dataset version.

Inputs:
  * ``data/labeled/yolo_txt/`` — image+txt pairs produced by 02_.
  * ``data/datasets/v{PREV}/`` — previous dataset snapshot (copied forward).

Output:
  * ``data/datasets/v{NEW}/`` with train/val subdirs, a data.yaml, and a
    deterministic split of the new samples seeded by ``--seed``.

Rules enforced here:
  * Class list must match the config. A changed class list requires a
    MAJOR bump + ``--allow-major`` (Hard rule #5).
  * Old train/val assignments are preserved — we never re-split existing
    data, because that would break cross-version comparisons.
  * ``data/datasets/holdout_test/`` is never touched (Hard rule #1).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("merge")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VERSION_RE = re.compile(r"^v(\d+)\.(\d+)$")


@dataclass(frozen=True)
class Version:
    major: int
    minor: int

    @classmethod
    def parse(cls, s: str) -> "Version":
        m = VERSION_RE.match(s)
        if not m:
            raise ValueError(f"Bad version '{s}', expected v{{MAJOR}}.{{MINOR}}")
        return cls(int(m.group(1)), int(m.group(2)))

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def pair_images_with_labels(
    yolo_dir: Path,
) -> list[tuple[Path, Path]]:
    """Return (image, txt) pairs from the flat yolo_txt directory."""
    pairs: list[tuple[Path, Path]] = []
    for txt in sorted(yolo_dir.glob("*.txt")):
        image = None
        for suf in IMAGE_SUFFIXES:
            candidate = yolo_dir / f"{txt.stem}{suf}"
            if candidate.exists():
                image = candidate
                break
        if image is None:
            raise FileNotFoundError(
                f"No image found alongside {txt}. Conversion step should "
                f"always emit them as pairs."
            )
        pairs.append((image, txt))
    return pairs


def deterministic_split(
    pairs: list[tuple[Path, Path]], val_fraction: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Deterministic train/val split based on a hash of the file stem.

    Using the stem (not random.shuffle) means the same file always lands
    in the same split across runs, even if new files are added later.
    """
    train, val = [], []
    for img, txt in pairs:
        digest = hashlib.sha256(f"{seed}:{img.stem}".encode()).hexdigest()
        if int(digest[:8], 16) / 0xFFFFFFFF < val_fraction:
            val.append((img, txt))
        else:
            train.append((img, txt))
    return train, val


def read_class_list(data_yaml: Path) -> list[str]:
    with data_yaml.open("r") as f:
        doc = yaml.safe_load(f)
    names = doc["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def copy_pair(src_img: Path, src_txt: Path, dst_images: Path, dst_labels: Path) -> None:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_images / src_img.name)
    shutil.copy2(src_txt, dst_labels / src_txt.name)


def copy_prev_snapshot(prev_dir: Path, new_dir: Path) -> None:
    if not prev_dir.exists():
        log.info("No previous dataset at %s — starting fresh", prev_dir)
        return
    log.info("Copying previous snapshot %s → %s", prev_dir, new_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        src = prev_dir / sub
        if src.is_dir():
            dst = new_dir / sub
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f.name)


def write_data_yaml(
    new_dir: Path, classes: list[str]
) -> None:
    data = {
        "path": str(new_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    with (new_dir / "data.yaml").open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def dvc_add(path: Path) -> None:
    log.info("dvc add %s", path)
    subprocess.run(["dvc", "add", str(path)], check=True)


def git_tag(tag: str) -> None:
    log.info("git tag %s", tag)
    subprocess.run(["git", "tag", tag], check=True)


def validate_class_list(
    config_classes: list[str],
    prev_data_yaml: Path | None,
    prev_v: Version,
    new_v: Version,
    allow_major: bool,
) -> None:
    if new_v.major < prev_v.major or (
        new_v.major == prev_v.major and new_v.minor <= prev_v.minor
    ):
        raise ValueError(
            f"New version {new_v} must be strictly greater than prev {prev_v}"
        )
    if prev_data_yaml is None or not prev_data_yaml.exists():
        log.info("No prev data.yaml — skipping class-list check (first version)")
        return
    prev_classes = read_class_list(prev_data_yaml)
    if prev_classes == config_classes:
        if new_v.major != prev_v.major:
            raise ValueError(
                f"Class list unchanged but major bumped {prev_v}→{new_v}. "
                f"Major bumps are reserved for class-list changes (Hard rule #5)."
            )
        return
    # Class list differs
    if new_v.major == prev_v.major:
        raise ValueError(
            f"Class list changed ({prev_classes} → {config_classes}) "
            f"but version is a minor bump {prev_v}→{new_v}. A class-list "
            f"change REQUIRES a major bump AND relabeling of old data "
            f"(Hard rule #5)."
        )
    if not allow_major:
        raise ValueError(
            f"Class list changed. Major bump {prev_v}→{new_v} needs "
            f"--allow-major and prior relabeling of old data."
        )
    log.warning(
        "Proceeding with MAJOR bump: classes %s → %s", prev_classes, config_classes
    )


def run(
    prev: str,
    new: str,
    yolo_dir: Path,
    datasets_dir: Path,
    config_path: Path,
    val_fraction: float,
    seed: int,
    allow_major: bool,
    run_dvc: bool,
    run_tag: bool,
) -> Path:
    prev_v = Version.parse(prev)
    new_v = Version.parse(new)
    cfg = load_config(config_path)
    config_classes: list[str] = list(cfg["classes"])

    prev_dir = datasets_dir / str(prev_v)
    new_dir = datasets_dir / str(new_v)
    prev_data_yaml = prev_dir / "data.yaml"

    if new_dir.exists():
        raise FileExistsError(
            f"{new_dir} already exists — refusing to overwrite. "
            f"Delete it or pick a new version."
        )

    validate_class_list(
        config_classes, prev_data_yaml, prev_v, new_v, allow_major
    )

    pairs = pair_images_with_labels(yolo_dir)
    log.info("Found %d new image/label pairs in %s", len(pairs), yolo_dir)

    copy_prev_snapshot(prev_dir, new_dir)

    train_pairs, val_pairs = deterministic_split(pairs, val_fraction, seed)
    log.info(
        "New split: %d train, %d val (val_fraction=%.2f, seed=%d)",
        len(train_pairs),
        len(val_pairs),
        val_fraction,
        seed,
    )
    for img, txt in train_pairs:
        copy_pair(img, txt, new_dir / "images/train", new_dir / "labels/train")
    for img, txt in val_pairs:
        copy_pair(img, txt, new_dir / "images/val", new_dir / "labels/val")

    write_data_yaml(new_dir, config_classes)

    if run_dvc:
        dvc_add(new_dir)
    if run_tag:
        git_tag(f"dataset-{new_v}")

    log.info("Dataset %s written to %s", new_v, new_dir)
    return new_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prev", required=True, help="e.g. v1.1")
    parser.add_argument("--new", required=True, help="e.g. v1.2")
    parser.add_argument("--yolo-dir", type=Path, default=Path("data/labeled/yolo_txt"))
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path("data/datasets")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/autolabel.yaml")
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-major",
        action="store_true",
        help="Acknowledge a MAJOR class-list change (Hard rule #5).",
    )
    parser.add_argument(
        "--no-dvc", action="store_true", help="Skip `dvc add` (useful in tests)"
    )
    parser.add_argument(
        "--no-tag", action="store_true", help="Skip `git tag` (useful in tests)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    run(
        prev=args.prev,
        new=args.new,
        yolo_dir=args.yolo_dir,
        datasets_dir=args.datasets_dir,
        config_path=args.config,
        val_fraction=args.val_fraction,
        seed=args.seed,
        allow_major=args.allow_major,
        run_dvc=not args.no_dvc,
        run_tag=not args.no_tag,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
