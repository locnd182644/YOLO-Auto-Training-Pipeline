# User manual

Practical guide for running one retraining cycle end-to-end. If you want
to know *why* the pipeline is shaped this way, read `architecture.md`.
If something is on fire, read `runbook.md`.

---

## What this pipeline does, in one sentence

Drop new images into `data/raw/incoming/`, review what the current model
isn't sure about, run `dvc repro`, get a new `.pt` model promoted to
`models/current/` — only if it didn't regress on the fixed holdout set.

## Who does what

| Role | Touches |
|---|---|
| Data operator | drops images, reviews `needs_review/`, edits `params.yaml` |
| CI (GitHub Actions) | runs `dvc repro`, promotes on success |
| You (the reader) | usually the data operator |

The only manual step per cycle is **human review** — everything else is
scripted.

---

## One-time setup

You only do this on a fresh checkout. Skip to "Per-cycle workflow" if
the repo is already initialized.

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. A GPU is strongly recommended for training (it's required
on the CI runner; locally you can still smoke-test with `device: cpu` in
`configs/training.yaml`).

### 2. Configure the DVC remote

The dev default is a local directory remote (`/root/Programming/Automation/dvc-remote`).
For your machine, point it somewhere writable:

```bash
dvc remote modify localremote url /path/to/your/dvc-remote
# or, for S3:
# dvc remote add -d s3remote s3://your-bucket/yolo-pipeline
```

Then pull whatever the team has already pushed:

```bash
dvc pull
```

If `dvc pull` fails on a fresh clone with "no remote", that means the
remote isn't configured yet — set it as above before continuing.

### 3. Finalize the holdout test set (one-time, IRREVERSIBLE)

`data/datasets/holdout_test/` is the **fixed** baseline used for every
cross-version comparison. It must be assembled once and never touched
again (Hard rule #1).

1. Hand-label a representative set of images covering every class and
   every important scene type. Aim for hundreds, not dozens.
2. Place them at:
   ```
   data/datasets/holdout_test/
     images/      # *.jpg|png|bmp
     labels/      # YOLO *.txt, one per image
     data.yaml    # points at images/, lists classes (own copy, NOT a symlink to a versioned data.yaml)
   ```
3. `dvc add data/datasets/holdout_test`
4. `git add data/datasets/holdout_test.dvc && git commit -m "freeze holdout test set"`
5. `dvc push`

**After this point, do not add or remove a single image** — every mAP50
number ever produced for this project is anchored to the contents of
this directory. If you must change it, treat it as a major incident,
document the reason, and accept that all historical metrics become
incomparable.

### 4. First-ever model (cold start)

You have no production model yet, so there's nothing to autolabel
*with*. Two options:

**Option A — let `01_autolabel.py` cold-start it for you (recommended).**
On first run, it falls back to `yolov11n.pt` (auto-downloaded by
ultralytics). Because COCO weights don't know your project's classes,
every image is routed to `needs_review/` for manual labeling. You
then proceed through the normal cycle from step "Review". `04_train.py`
will also cold-start from `yolov11n.pt` and log a loud `COLD START`
warning in MLflow.

**Option B — hand-label a seed set first.**
Skip `01_autolabel.py`, label some images directly in labelme, drop the
JSON+image pairs into `data/labeled/labelme_json/reviewed/`, then run
the pipeline from `02_` onward. Use `--prev v0.0 --new v1.0` for the
merge step (the script tolerates a missing prev directory).

After either option, once `models/v1.0/best.pt` exists:

```bash
ln -sfn v1.0 models/current
git add models/current && git commit -m "bootstrap models/current → v1.0"
git tag dataset-v1.0 model-v1.0
dvc push && git push --follow-tags
```

---

## Per-cycle workflow

This is the loop you run every time a fresh batch of images arrives.

### Step 1 — Drop new images

Put raw images directly in `data/raw/incoming/`:

```bash
cp /path/to/new/batch/*.jpg data/raw/incoming/
```

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`. Resolution at least
320px on the short side. No HEIC, no PII unless documented (see
`data_policy.md`).

### Step 2 — Update `params.yaml`

Tell the pipeline which version it's producing:

```yaml
# params.yaml
versions:
  prev: v1.1   # whatever models/current → today
  new:  v1.2   # the version this run will create
```

- **MINOR bump** (`v1.1 → v1.2`) — normal case: more data, same class
  list. Use this 99% of the time.
- **MAJOR bump** (`v1.1 → v2.0`) — only when the class list itself
  changes. Requires relabeling all historical data and passing
  `--allow-major` to `03_merge_and_version.py`. Read `data_policy.md`
  before attempting.

### Step 3 — Autolabel

```bash
python scripts/01_autolabel.py
```

The script reads the current model from `models/current/best.pt`, runs
inference on every file in `data/raw/incoming/`, and routes each image
into one of:

- `data/labeled/labelme_json/auto_accepted/` — every detection ≥
  `CONF_HIGH` (default 0.6). These are trusted; the human doesn't see
  them.
- `data/labeled/labelme_json/needs_review/` — anything uncertain (or
  zero detections). **You must review these by hand.**

Originals are moved to `data/raw/archive/` so `incoming/` stays a clean
work queue. If `models/current/best.pt` doesn't exist, see Step 4 of
"One-time setup" above — the script will cold-start with
`yolov11n.pt`.

### Step 4 — Review (the only manual step)

Open the review queue in labelme:

```bash
labelme data/labeled/labelme_json/needs_review/ \
    --labels configs/autolabel.yaml \
    --autosave
```

For each file:

1. Fix or add bounding boxes. Rectangles only — polygons are rejected
   downstream. Boxes must be fully inside the image (no clipping).
2. Save (Ctrl+S, autosaved if you used `--autosave`).
3. **Move the image+JSON pair from `needs_review/` to `reviewed/`:**
   ```bash
   mv data/labeled/labelme_json/needs_review/foo.{jpg,json} \
      data/labeled/labelme_json/reviewed/
   ```

The pipeline will not look at `needs_review/` — files only become
training data after they leave that bucket. This is the label-drift
guard (Hard rule #3).

If you see a totally novel class in `needs_review/` that isn't in
`configs/autolabel.yaml`, **stop**. Adding a class is a MAJOR-bump,
relabel-the-world event. Read `data_policy.md` and confirm with the
team before changing anything.

### Step 5 — Run the rest of the pipeline

Single command — DVC walks the DAG and skips stages whose inputs
haven't changed:

```bash
dvc repro
```

That runs:

| Stage | Script | What it produces |
|---|---|---|
| `convert` | `02_convert_labelme_to_yolo.py` | `data/labeled/yolo_txt/` |
| `merge` | `03_merge_and_version.py` | `data/datasets/v{NEW}/` |
| `train` | `04_train.py` | `models/v{NEW}/best.pt` + MLflow run |
| `evaluate` | `05_evaluate.py` | `runs/evaluate/v{NEW}.json` |

The evaluate stage is the gate. It exits 1 (and `dvc repro` fails) if:

```
new_mAP50 < old_mAP50 - threshold     # threshold default 0.01
```

### Step 6 — Inspect the result

```bash
cat runs/evaluate/v1.2.json
```

Look at three fields:

```json
{
  "decision": "promote",     // "promote" or "block"
  "delta_mAP50": 0.0123,     // new - old (positive = improvement)
  "threshold": 0.01
}
```

**`decision: promote`** — proceed to Step 7.

**`decision: block`** — do not promote. Open `docs/runbook.md`
("mAP dropped under threshold") and investigate. Common causes:

- A few mislabeled files in `auto_accepted/` snuck in. Re-review and
  move corrections to `reviewed/`.
- The autolabeler is overconfident → raise `CONF_HIGH` in
  `configs/autolabel.yaml` (and document the reason in the commit).
- New class imbalance vs prev version.

Never disable the gate to force a promotion.

You can also browse the MLflow UI for a richer view:

```bash
mlflow ui   # opens at http://localhost:5000
```

### Step 7 — Promote (locally)

If the gate passed and you want to do it by hand instead of via CI:

```bash
ln -sfn v1.2 models/current
git add params.yaml dvc.lock models/current
git commit -m "promote model v1.2"
git tag -a dataset-v1.2 -m "dataset v1.2"
git tag -a model-v1.2   -m "promoted model v1.2"
dvc push
git push origin main --follow-tags
```

The new `models/current/best.pt` is now what `01_autolabel.py` will use
on the *next* batch of incoming images. The cycle has closed.

---

## Running via CI instead

For a hands-off run, push your `data/raw/incoming` change (DVC-tracked)
or trigger the workflow manually:

1. GitHub → Actions → **retrain** → **Run workflow**.
2. Fill in `prev_version` and `new_version` (e.g. `v1.1`, `v1.2`).
3. The self-hosted GPU runner does Steps 5–7 automatically. On gate
   failure: no symlink flip, no tags, no `dvc push`. The previous model
   keeps serving. The eval JSON is uploaded as a build artifact for
   inspection.

The CI YAML is `/.github/workflows/retrain.yml` if you need to read
exactly what it does.

---

## Common operations

### Run a single stage by hand

```bash
python scripts/01_autolabel.py
python scripts/02_convert_labelme_to_yolo.py
python scripts/03_merge_and_version.py --prev v1.1 --new v1.2
python scripts/04_train.py --data data/datasets/v1.2/data.yaml --epochs 50
python scripts/05_evaluate.py --new models/v1.2/best.pt --old models/current/best.pt
```

Each script also accepts `--config` and a few stage-specific flags —
run with `--help` for the full list.

### Override training epochs without editing the config

```bash
python scripts/04_train.py --data data/datasets/v1.2/data.yaml --epochs 100
```

For one-off experiments only. Persistent changes go in
`configs/training.yaml`.

### Roll back a bad promotion

See `runbook.md` § "Rollback a bad promotion". Short version: flip the
`models/current` symlink at the previous version directory, commit,
push.

### Inspect history

```bash
# Versions ever promoted
git tag --list 'model-v*' --sort=-v:refname

# Reproduce any past dataset
git checkout dataset-v1.0
dvc checkout

# Compare metrics across versions
jq '.decision, .delta_mAP50' runs/evaluate/v*.json
```

---

## Configuration reference

All knobs live in `configs/`. Don't hardcode in scripts.

### `configs/autolabel.yaml` — autolabel + class spec

| Key | Default | What it controls |
|---|---|---|
| `CONF_HIGH` | 0.6 | Detections ≥ this are trusted; image goes to `auto_accepted/` only if all surviving detections clear this bar |
| `CONF_LOW` | 0.25 | Noise floor — detections below this are dropped before routing |
| `imgsz` | 640 | Inference image size |
| `device` | `0` | `0` for GPU 0, `"cpu"` for CPU |
| `iou` | 0.5 | NMS IoU threshold |
| `max_det` | 300 | Max detections per image |
| `pretrained_fallback` | `yolov11n.pt` | Cold-start weights when `models/current` is missing |
| `classes` | `[object]` | **Single source of truth for the class list.** Changing this is a MAJOR bump (Hard rule #5). Must match `data.yaml` of the current dataset. |
| `paths.*` | `data/...` | Where each bucket lives. Don't change without reason. |

### `configs/training.yaml` — training, evaluation, MLflow, paths

| Section | Notable keys |
|---|---|
| `training` | `epochs`, `batch`, `optimizer`, `lr0`, `patience`, `device`, `seed`, `pretrained_fallback` |
| `evaluation` | `threshold` (regression gate, default 0.01), `imgsz`, `device`, `holdout_data_yaml` |
| `mlflow` | `tracking_uri` (default `file:./mlruns`), `experiment_name` |
| `paths` | `models_dir`, `current_symlink`, `runs_dir`, `datasets_dir` |

If you raise the regression gate threshold, leave a comment in the
commit explaining why — see CLAUDE.md "Do NOT" rule on threshold
changes.

### `params.yaml` — per-cycle versions

```yaml
versions:
  prev: v1.1
  new:  v1.2
```

That's it. Edit before each cycle.

---

## Troubleshooting quick reference

| Symptom | First thing to check |
|---|---|
| `01_autolabel.py` exits with `Model not found` | `ls -l models/current` — if missing, the cold-start fallback should kick in; if it doesn't, your local script is stale |
| Every image lands in `needs_review/` | If on cold start, this is correct. Otherwise the model is unsure → check `CONF_HIGH`, recent data quality |
| `02_convert` raises `Label '...' is not in the configured class list` | A new class slipped into a labelme JSON. Either fix the JSON or trigger a MAJOR bump (read `data_policy.md` first) |
| `03_merge` raises `Class list changed but version is a minor bump` | You edited `configs/autolabel.yaml` `classes` without bumping MAJOR. Either revert the config or use `--new v(N+1).0 --allow-major` |
| `03_merge` raises `... already exists — refusing to overwrite` | `data/datasets/v{NEW}/` already exists from a previous attempt. Bump `new` to the next version, or delete the directory if you're sure |
| `04_train` logs `COLD START` | Expected on first ever run. **Unexpected** on any later cycle — investigate before promoting (check `models/current` symlink) |
| `05_evaluate` raises `Holdout data.yaml missing` | Step 3 of "One-time setup" wasn't completed |
| `05_evaluate` exit 1, `decision: block` | Regression detected. Read `runbook.md` § "mAP dropped under threshold" |
| `dvc pull` credentials error in CI | `runbook.md` § "CI fails at `dvc pull`" |

For incidents not on this list, see `runbook.md` or escalate.

---

## What you should NOT do

A short reminder — full list in CLAUDE.md.

- Don't add or remove a single file in `data/datasets/holdout_test/`.
- Don't train from scratch — `04_train.py` always fine-tunes from
  `models/current/best.pt` (or the cold-start fallback on first run).
- Don't merge images directly from `needs_review/` into training data.
- Don't add a new class without a MAJOR bump and re-labeling history.
- Don't commit `.pt` weights or images to git — those are DVC-tracked.
- Don't lower the regression threshold to "make CI pass."

When in doubt: stop, read the relevant Hard rule in CLAUDE.md, and ask.
