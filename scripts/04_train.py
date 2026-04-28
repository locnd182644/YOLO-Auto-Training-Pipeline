"""Fine-tune a new YOLO model from the current production checkpoint.

Never trains from scratch (Hard rule #2). If ``models/current/best.pt``
is missing (first-ever run), falls back to the pretrained weights named
in ``configs/training.yaml`` and logs a loud cold-start warning.

Output: ``models/v{MAJOR}.{MINOR}/best.pt``. The ``models/current``
symlink is intentionally NOT updated here — promotion happens in CI
only after 05_evaluate passes the regression gate.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("train")

VERSION_RE = re.compile(r"v\d+\.\d+$")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve_start_weights(cfg: dict[str, Any]) -> tuple[Path, bool]:
    """Return (weights_path, cold_start). ``cold_start`` flags the fallback."""
    current = Path(cfg["paths"]["current_symlink"]) / "best.pt"
    if current.exists():
        log.info("Fine-tuning from %s", current)
        return current, False
    fallback = Path(cfg["training"]["pretrained_fallback"])
    log.warning(
        "COLD START: %s does not exist. Falling back to pretrained %s. "
        "This should only happen on the very first training run.",
        current,
        fallback,
    )
    return fallback, True


def extract_version(data_yaml: Path) -> str:
    """Pull the ``vMAJOR.MINOR`` token out of a dataset path."""
    for part in reversed(data_yaml.parts):
        if VERSION_RE.match(part):
            return part
    raise ValueError(
        f"Could not extract version from {data_yaml}; "
        f"expected a path segment like 'v1.2'."
    )


def _mlflow_log(cfg: dict[str, Any], version: str, cold_start: bool) -> Any:
    """Initialize MLflow run. Return the active run context manager.

    Import is lazy so tests that don't need MLflow can still import this
    module without the dependency installed.
    """
    import mlflow

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    run = mlflow.start_run(run_name=f"train-{version}")
    mlflow.log_params(
        {
            **cfg["training"],
            "version": version,
            "cold_start": cold_start,
        }
    )
    return mlflow, run


def run(
    data_yaml: Path,
    config_path: Path,
    epochs: int | None,
) -> Path:
    cfg = load_config(config_path)
    tcfg = cfg["training"]
    if epochs is not None:
        tcfg["epochs"] = epochs

    version = extract_version(data_yaml)
    weights, cold_start = resolve_start_weights(cfg)

    from ultralytics import YOLO  # lazy import — heavy

    mlflow, run_ctx = _mlflow_log(cfg, version, cold_start)
    with run_ctx:
        model = YOLO(str(weights))
        results = model.train(
            data=str(data_yaml),
            epochs=tcfg["epochs"],
            imgsz=tcfg["imgsz"],
            batch=tcfg["batch"],
            optimizer=tcfg["optimizer"],
            lr0=tcfg["lr0"],
            lrf=tcfg["lrf"],
            patience=tcfg["patience"],
            workers=tcfg["workers"],
            device=tcfg["device"],
            seed=tcfg["seed"],
            project=cfg["paths"]["runs_dir"],
            name=version,
            exist_ok=False,
        )

        run_best = Path(results.save_dir) / "weights" / "best.pt"
        if not run_best.exists():
            raise FileNotFoundError(
                f"Expected best.pt at {run_best} — training produced nothing."
            )

        out_dir = Path(cfg["paths"]["models_dir"]) / version
        out_dir.mkdir(parents=True, exist_ok=True)
        out_best = out_dir / "best.pt"
        shutil.copy2(run_best, out_best)

        metrics = getattr(results, "results_dict", None) or {}
        # print("Training metrics:", metrics) (B)
        # Clean: Replace '(' and ')' to '_' in metric keys to avoid MLflow issues.
        clean_metrics = {
            k.replace('(', '_').replace(')', ''): v 
            for k, v in metrics.items()
        }
        mlflow.log_metrics({k: float(v) for k, v in clean_metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_artifact(str(out_best))

    log.info("Training done. New weights at %s", out_best)
    return out_best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training.yaml")
    )
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    run(args.data, args.config, args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
