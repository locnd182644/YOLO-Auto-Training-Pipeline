"""Evaluate a candidate model against the current production model.

Both models are scored on the FIXED holdout test set at
``data/datasets/holdout_test/data.yaml`` (Hard rule #1). The script
exits 1 if ``new_mAP50 < old_mAP50 - threshold`` — CI treats that as a
blocked promotion (Hard rule #4).

If ``--old`` is missing (first ever run), the comparison is skipped and
the script exits 0 after logging the new model's absolute metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("evaluate")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def evaluate_one(weights: Path, data_yaml: Path, ecfg: dict[str, Any]) -> dict[str, float]:
    """Run val() on one checkpoint and return a flat metrics dict."""
    from ultralytics import YOLO  # lazy — heavy

    log.info("Evaluating %s on %s", weights, data_yaml)
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=ecfg["imgsz"],
        device=ecfg["device"],
        verbose=False,
    )
    box = metrics.box
    return {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }


def run(
    new_weights: Path,
    old_weights: Path | None,
    config_path: Path,
    metrics_out: Path,
) -> int:
    cfg = load_config(config_path)
    ecfg = cfg["evaluation"]
    data_yaml = Path(ecfg["holdout_data_yaml"])
    threshold = float(ecfg["threshold"])

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Holdout data.yaml missing: {data_yaml}. The fixed test set "
            f"is required to evaluate (Hard rule #1)."
        )

    new_metrics = evaluate_one(new_weights, data_yaml, ecfg)
    log.info("NEW %s: %s", new_weights, new_metrics)

    result: dict[str, Any] = {
        "new": {"weights": str(new_weights), **new_metrics},
        "threshold": threshold,
    }

    exit_code = 0
    if old_weights is not None and old_weights.exists():
        old_metrics = evaluate_one(old_weights, data_yaml, ecfg)
        log.info("OLD %s: %s", old_weights, old_metrics)
        delta = new_metrics["mAP50"] - old_metrics["mAP50"]
        result["old"] = {"weights": str(old_weights), **old_metrics}
        result["delta_mAP50"] = delta
        if new_metrics["mAP50"] < old_metrics["mAP50"] - threshold:
            log.error(
                "REGRESSION: new mAP50 %.4f < old %.4f - threshold %.4f "
                "(delta %.4f). Blocking promotion.",
                new_metrics["mAP50"],
                old_metrics["mAP50"],
                threshold,
                delta,
            )
            result["decision"] = "block"
            exit_code = 1
        else:
            log.info(
                "PASS: new mAP50 %.4f vs old %.4f (delta %.4f, threshold %.4f)",
                new_metrics["mAP50"],
                old_metrics["mAP50"],
                delta,
                threshold,
            )
            result["decision"] = "promote"
    else:
        log.warning(
            "No old model — skipping regression check. This is only "
            "acceptable on the very first run."
        )
        result["decision"] = "promote"
        result["old"] = None

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote metrics to %s", metrics_out)
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument(
        "--old", type=Path, default=Path("models/current/best.pt")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training.yaml")
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("runs/evaluate/metrics.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    old = args.old if args.old.exists() else None
    return run(args.new, old, args.config, args.metrics_out)


if __name__ == "__main__":
    sys.exit(main())
