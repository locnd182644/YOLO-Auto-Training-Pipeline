# YOLO Auto-Training Pipeline

This project builds an automated pipeline to retrain a YOLO model whenever new data arrives. Input is raw images; output is a verified `.pt` model promoted to production.

## Goal

Each time a new batch of images arrives:
1. Autolabel using the current model → filter by confidence → reviewer confirms low-confidence images
2. Convert Labelme JSON → YOLO txt
3. Merge into the existing dataset, bump version (DVC tag)
4. Fine-tune from the previous model (never train from scratch)
5. Evaluate on a **fixed** test set → only promote if no regression
6. The new model becomes the autolabeler for the next cycle

## Stack

- **Python 3.10+**, `ultralytics` (YOLOv11), `labelme`
- **DVC** for data versioning, S3 backend (or local remote for dev)
- **MLflow** for experiment tracking (metrics, hyperparams, artifacts)
- **GitHub Actions** self-hosted runner with GPU to trigger retraining
- No Docker during development — production packaging comes later

## Directory structure (immutable)

```
yolo-pipeline/
├── data/
│   ├── raw/{incoming,archive}/         # raw images; incoming = unprocessed
│   ├── labeled/{labelme_json,yolo_txt}/
│   └── datasets/
│       ├── holdout_test/               # FIXED TEST SET — never merge new data here
│       └── v{MAJOR}.{MINOR}/           # training data snapshot per version
│           ├── images/{train,val}/
│           ├── labels/{train,val}/
│           └── data.yaml
├── models/
│   ├── v{MAJOR}.{MINOR}/best.pt
│   └── current -> v{MAJOR}.{MINOR}/    # symlink → production model
├── configs/{autolabel.yaml,training.yaml}
├── scripts/
│   ├── 01_autolabel.py
│   ├── 02_convert_labelme_to_yolo.py
│   ├── 03_merge_and_version.py
│   ├── 04_train.py
│   └── 05_evaluate.py
├── tests/                              # pytest for each script
├── dvc.yaml
└── .github/workflows/retrain.yml
```

Do not change this structure. If a new directory is needed, ask first.

## Hard rules — do not violate

1. **Fixed test set** — `data/datasets/holdout_test/` must never have images added or removed for any reason. It is the baseline for comparing model versions across time. Every evaluation script uses this path.

2. **Never train from scratch** — `04_train.py` always loads weights from `models/current/best.pt` for fine-tuning. If `current` does not exist (first run), fall back to pretrained `yolov11n.pt` and log a clear "cold start" message.

3. **Human review gate** — images with any detection where `confidence < CONF_HIGH` (default 0.6) MUST be routed to `data/labeled/labelme_json/needs_review/`. The merge script (`03_`) only reads from `auto_accepted/` and `reviewed/`, NEVER from `needs_review/`. This is the label-drift guard.

4. **Evaluation gate** — `05_evaluate.py` must exit with code 1 if `new_mAP50 < old_mAP50 - THRESHOLD` (default threshold 0.01). CI will not promote a model that fails the gate.

5. **Class list immutable within a major version** — adding a new class requires a major bump (`v1.x` → `v2.0`) AND relabeling of all old data for that class. Never silently merge a new class into an in-flight dataset.

6. **Holdout test set uses its own data.yaml** — `data/datasets/holdout_test/data.yaml` is separate and must not point to the training set.

## Conventions

- **Version naming:** `v{MAJOR}.{MINOR}`. MAJOR increments when the class list changes; MINOR increments every time new data is added.
- **DVC tag format:** `dataset-v1.2`, `model-v1.2` (tags are paired).
- **Logging:** use the `logging` module, not `print`. INFO level for main flow, DEBUG for details.
- **Config:** hyperparameters live in `configs/*.yaml`, never hardcoded in scripts. Load with `pyyaml`.
- **Main function per script** must be importable and testable — not only reachable through `if __name__ == "__main__"`.
- **Error handling:** fail loud — raise exceptions with clear messages instead of silently skipping.

## Common commands

```bash
# Pull the current data version
dvc pull

# Run a single step
python scripts/01_autolabel.py --input data/raw/incoming --model models/current/best.pt
python scripts/02_convert_labelme_to_yolo.py
python scripts/03_merge_and_version.py --prev v1.1 --new v1.2
python scripts/04_train.py --data data/datasets/v1.2/data.yaml --epochs 50
python scripts/05_evaluate.py --new runs/train/v1.2/weights/best.pt --old models/current/best.pt

# Run the full pipeline via DVC (used in CI)
dvc repro

# Push new artifacts
dvc push
git push --tags

# Test
pytest tests/ -v
```

## Do NOT

- **Do not** create files at the project root outside the defined structure (no notes, temp files, or experimental notebooks dumped at root).
- **Do not** commit `.pt` weights or images directly to git — use DVC.
- **Do not** add a new class without explicit user confirmation — this is a breaking change.
- **Do not** randomly re-split the test set on every run — use a fixed seed or split once and keep the file list.
- **Do not** change `CONF_HIGH`, `CONF_LOW`, or the regression threshold without updating the config file and noting the reason.
- **Do not** write filler comments like `# loop through items` — only comment when the logic is non-obvious.
- **Do not** merge multiple responsibilities into one script — keep exactly 5 scripts in numbered order.

## Additional context files

Reference when needed (do not read all at once — lazy-load):
- `@docs/architecture.md` — decision log: why DVC over Git-LFS, why Ultralytics, etc.
- `@docs/runbook.md` — rollback procedure, what to do when CI fails or mAP drops.
- `@docs/data_policy.md` — privacy rules, label spec, acceptance criteria for new images.

## Current status

- [x] Directory scaffold initialized
- [x] Script `01_autolabel.py` (autolabel + confidence routing)
- [x] Script `02_convert_labelme_to_yolo.py`
- [x] Script `03_merge_and_version.py` (merge + DVC add + tag)
- [x] Script `04_train.py` (fine-tune from `models/current`)
- [x] Script `05_evaluate.py` (regression gate)
- [x] DVC pipeline (`dvc.yaml`)
- [x] GitHub Actions workflow
- [ ] Holdout test set finalized and versioned
- [x] Unit tests for `02_` and `03_` (convert + merge logic)
- [x] Rollback runbook

Check off items as they are completed.

## When unsure

If a request may violate any rule in "Hard rules" or "Do NOT", STOP and ask the user before proceeding. Do not guess.
