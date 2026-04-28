# Architecture

## TL;DR

A self-improving object-detection pipeline. A human drops images into
`data/raw/incoming/`; some time later a new `.pt` model is promoted to
production — or a regression is caught and the old model stays live.
The system is built around three containment rails:

1. **Label-drift rail:** every low-confidence prediction goes through a
   human before it can touch training data (`CONF_HIGH` gate +
   `needs_review/` bucket + whitelist-only merge).
2. **Metric-drift rail:** every candidate is measured against a
   **fixed** holdout; a drop past the threshold refuses promotion
   (`05_evaluate.py` exit 1 → CI never flips the symlink).
3. **Schema-drift rail:** the class list is immutable within a MAJOR
   version; changing it forces relabeling of history and a version
   break (`03_merge_and_version.py` enforces this in code).

Everything else — DVC, MLflow, YOLOv11, numbered scripts — is plumbing
chosen to serve those three rails.

## System overview

```
                                                ┌────────────────┐
                                                │   Reviewer     │
                                                │ (labelme UI)   │
                                                └───────┬────────┘
                                                  Gate  │ corrections
                                                        ▼
  ┌────────────┐   ┌────────────────────────┐   ┌─────────────────┐
  │ raw images │──▶│ 01 autolabel           │──▶│ auto_accepted/  │──┐
  └────────────┘   │  (current model)       │   │ needs_review/   │  │
                   └────────────────────────┘   └─────────────────┘  │
                                                                     │
  ┌──────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────┐   ┌──────────────────────────┐
│ 02 convert labelme   │──▶│ 03 merge + version        │
│    → YOLO txt        │   │    (v{N}.{M} snapshot)    │
└──────────────────────┘   └──────────┬────────────────┘
                                      │
                                      ▼
                           ┌───────────────────────┐
                           │ 04 train              │
                           │  (fine-tune from      │
                           │   models/current)     │
                           └──────────┬────────────┘
                                      │
                                      ▼
                           ┌───────────────────────┐
                           │ 05 evaluate           │
                           │  on fixed holdout     │
                           └──────────┬────────────┘
                                      │
                       pass ◀─────────┴────────▶ fail (exit 1)
                        │                          │
                        ▼                          ▼
               flip models/current         keep old model,
               + dataset/model tags        surface metrics,
               + dvc push + git push       page on-call
```

## End-to-end workflow (one retraining cycle)

1. **Ingest.** New images arrive under `data/raw/incoming/`. In
   production this is typically an S3 sync or an operator drop. Each
   arrival is the event that starts a cycle.
2. **Autolabel.** `01_autolabel.py` runs the current production model
   over every new image, emits a labelme JSON per image, and routes
   each image+JSON pair into **exactly one** of two buckets:
   - `auto_accepted/` — every detection has `conf ≥ CONF_HIGH`.
   - `needs_review/` — any detection is uncertain, or the model
     produced no detections at all.
3. **Human review.** A reviewer opens `needs_review/` in labelme, fixes
   labels, and moves corrected files to `reviewed/`. This is the only
   manual step per cycle and it is *gated* — the rest of the pipeline
   cannot proceed on an image until it leaves `needs_review/`.
4. **Convert.** `02_convert_labelme_to_yolo.py` transforms JSON →
   normalized YOLO `.txt` for every file in `auto_accepted/` and
   `reviewed/`, copying images alongside. The script **whitelists**
   those two buckets; `needs_review/` is unreachable by construction.
5. **Merge + version.** `03_merge_and_version.py` takes the prev
   dataset snapshot (`v{MAJOR}.{MINOR}`) plus the new pairs and writes
   the next snapshot — existing train/val assignments are preserved,
   new items are deterministically hashed into train/val, and a fresh
   `data.yaml` is emitted. The class-list invariant is enforced here.
6. **Train.** `04_train.py` fine-tunes from `models/current/best.pt`
   (NOT from scratch — ever) against the new snapshot's `data.yaml`.
   All hyperparams come from `configs/training.yaml`, all metrics and
   the resulting `best.pt` land in MLflow.
7. **Evaluate.** `05_evaluate.py` scores both the candidate and the
   current production model against `data/datasets/holdout_test/` and
   writes `runs/evaluate/v{NEW}.json`. If
   `new_mAP50 < old_mAP50 − THRESHOLD`, it exits 1.
8. **Promote.** Only if every earlier step succeeded, CI flips
   `models/current` to point at the new version dir, creates paired
   `dataset-v{NEW}` and `model-v{NEW}` tags, `dvc push` + `git push
   --follow-tags`. The next cycle's autolabeler is now this new model.

The user only touches step 3. Everything else is `dvc repro` +
`retrain.yml`.

## Data lifecycle: the life of one image

An incoming image moves through a strict state machine. The state is
encoded by which directory the file lives in — there is no database.

```
data/raw/incoming/foo.jpg
    │  (01_autolabel.py)
    ├──▶ data/labeled/labelme_json/auto_accepted/foo.{jpg,json}
    │        │  (02)
    │        └──▶ data/labeled/yolo_txt/foo.{jpg,txt}
    │                │  (03)
    │                └──▶ data/datasets/v{N}/{images,labels}/{train,val}/foo.*
    │
    └──▶ data/labeled/labelme_json/needs_review/foo.{jpg,json}
             │  (human fix + move)
             └──▶ data/labeled/labelme_json/reviewed/foo.{jpg,json}
                      │  (02)
                      └──▶ (same path as auto_accepted from here)

        data/raw/archive/foo.jpg      ◀── 01 moves the original here
```

Key invariants:

- **A file is never in two buckets at once.** `01_autolabel.py` moves
  (not copies) the raw image to `archive/`, then writes the
  labelme+image copy into exactly one of `auto_accepted/` /
  `needs_review/`.
- **`needs_review/` is terminal without a human.** No script reads from
  it. A file only leaves it by a human mv into `reviewed/`.
- **`archive/` is append-only.** It is the audit trail: "this image was
  ingested on this date at this version." Never clean it up in-place.
- **Training snapshots are append-only.** `v{N}.{M+1}` = `v{N}.{M}` ∪
  new, preserving old split assignments. Old snapshots are never
  modified, only superseded.

## Versioning scheme

`v{MAJOR}.{MINOR}` labels both datasets and models, in lockstep. CI
always produces a matching pair: `dataset-v1.2` and `model-v1.2`.

- **MINOR bump** = more data, same class list. This is the normal path;
  99% of cycles are MINOR bumps. Fine-tuning from the previous model
  checkpoint is safe across MINOR bumps.
- **MAJOR bump** = class list changed (added, removed, renamed, or
  reordered). Requires `--allow-major` AND relabeling of all historical
  images for the new class list. Fine-tuning across a MAJOR bump would
  silently corrupt the model's class head, so the runbook calls for a
  cold-start from `yolov11n.pt` in that case.

Why track in git tags (not just DVC): the tag pair gives a single
atomic lookup. `git checkout dataset-v1.2` + `dvc checkout` reproduces
the exact dataset that produced `model-v1.2`. This is the foundation
for every post-hoc investigation.

## Per-stage deep dives

### 01_autolabel.py

- **Input:** `data/raw/incoming/*.jpg|png|…`, current model, config.
- **Output:** labelme JSON + image copy in one of the two buckets;
  original moved to `archive/`.
- **Core decision:** for each image, inspect ALL detections. An image
  with even one detection below `CONF_HIGH` goes to `needs_review/`.
  An image with zero detections ALSO goes to `needs_review/` — a
  totally blank prediction is a strong drift signal.
- **Confidence thresholds:**
  - `CONF_LOW` (0.25) is the *noise floor*. Detections below this are
    dropped entirely before routing — we don't want to force a reviewer
    to delete a zoo of 0.1-confidence boxes.
  - `CONF_HIGH` (0.6) is the *trust threshold*. Only images where every
    surviving detection clears this bar bypass review.
- **Failure mode:** if the model is missing entirely, exit code 2
  (not 1). 2 = "bootstrap problem, seed manually"; 1 is reserved for
  the regression gate.

### 02_convert_labelme_to_yolo.py

- **Input:** `labelme_json/{auto_accepted,reviewed}/`, config.
- **Output:** flat `yolo_txt/` with paired `.jpg` + `.txt`.
- **Core decision:** the two allowed buckets are a **whitelist** in
  code (`ALLOWED_BUCKETS = ("auto_accepted", "reviewed")`), not a
  filter over all subdirs. A new bucket cannot be accidentally
  included by someone mkdir-ing a directory.
- **Bbox math:** labelme stores two arbitrary corner points per
  rectangle — we normalize to (min, max) before converting to YOLO's
  center-x/center-y/width/height in [0, 1]. Raw points outside the
  image or degenerate boxes raise loudly; we don't silently clip.
- **No partial success:** if any file fails, the script raises at the
  end after logging all failures. CI must not proceed on a
  half-converted set.

### 03_merge_and_version.py

- **Input:** prev dataset dir, `yolo_txt/`, config (class list),
  version strings.
- **Output:** new dataset dir with `data.yaml`, preserved old split,
  deterministically-split new samples.
- **Determinism:** the split function hashes `f"{seed}:{stem}"`. Same
  stem + same seed → same split forever. Adding more files later does
  not reshuffle existing assignments; an image stays in the split it
  was first placed in.
- **Class-list enforcement:** the validator has a full 2×2 truth table
  on (classes changed?) × (major bump?):

  | classes same | classes differ
  ---|---|---
  **minor bump** | ✓ proceed | ✗ error: major required
  **major bump** | ✗ error: bump reserved for classes | needs `--allow-major`

- **Refuses to overwrite.** If `v{NEW}/` already exists, hard fail. We
  won't clobber a versioned snapshot.

### 04_train.py

- **Input:** `data/datasets/v{NEW}/data.yaml`, training config.
- **Output:** `models/v{NEW}/best.pt`, MLflow run with params + metrics
  + artifact.
- **Core decision:** *never* train from scratch. Load
  `models/current/best.pt` if it exists; otherwise (first-ever run)
  fall back to pretrained `yolov11n.pt` with a loud `COLD START`
  warning. This is the second most important rail in the system —
  cold-starting silently on a random day would wipe out months of
  accumulated fine-tuning progress.
- **Does not flip `models/current`.** Promotion is a CI responsibility
  contingent on the evaluate gate. The script only writes
  `models/v{NEW}/best.pt` next to its MLflow run.

### 05_evaluate.py

- **Input:** new weights, old weights, config, holdout `data.yaml`.
- **Output:** `runs/evaluate/v{NEW}.json` with `{new, old,
  delta_mAP50, decision, threshold}`, plus an exit code.
- **Gate math:** promotion requires
  `new_mAP50 ≥ old_mAP50 − THRESHOLD`. The threshold is a *tolerance*,
  not an improvement requirement: we allow small noise-level drops
  (default 0.01 mAP50) but veto meaningful regression.
- **First-run special case:** if `--old` is absent, log a loud warning
  and pass — there's nothing to regress against. This only fires on
  the very first promotion; after that, every cycle has an old model.
- **Exit code is the CI gate.** 0 = promote, 1 = block. Nothing else
  is special-cased.

## DVC pipeline binding

`dvc.yaml` binds the five scripts into a DAG. Each stage declares its
`deps`, `outs`, and in the case of evaluate a `metrics` file.
`params.yaml` carries the `prev` / `new` version strings so the stage
commands interpolate (`${versions.new}`) without editing `dvc.yaml`.

Why DVC (instead of a plain shell script):

- **Incremental re-runs.** `dvc repro` skips stages whose inputs haven't
  changed. On CI, a config-only change re-runs train+evaluate but not
  the whole autolabel/convert path.
- **Content-addressed artifacts.** Every `outs:` path is hashed into
  DVC's remote; rolling back is `git checkout <tag> && dvc checkout`.
- **Metrics diffing.** `dvc metrics diff` across tags shows mAP50
  deltas between any two versions without hand-written tooling.

## CI: `retrain.yml`

Trigger: `workflow_dispatch` (manual, with prev/new version inputs) or
a push that touches `data/raw/incoming.dvc`. Runs on a self-hosted
`[self-hosted, gpu]` runner because training needs a GPU and we don't
want to lease one per minute.

Job shape:

1. Checkout + Python + deps.
2. `dvc pull` — fetch the prev dataset and current model from the
   remote.
3. Patch `params.yaml` with the user-provided `prev_version` /
   `new_version` (for the manual trigger).
4. `dvc repro` — runs the DAG end-to-end. The evaluate stage's exit
   code determines whether the job succeeds.
5. **Promote on success only:** flip `models/current`, commit
   `params.yaml`+`dvc.lock`+symlink, create paired
   `dataset-v{NEW}`/`model-v{NEW}` tags, `dvc push` + `git push
   --follow-tags`.
6. **Always:** upload `runs/evaluate/*.json` as a build artifact so a
   failed gate still leaves a trail.

Failure semantics:

- `dvc repro` fails → no promote step, no tags, no symlink flip. The
  previous model continues serving.
- `dvc push` fails after a successful promote → the tags and symlink
  exist in git but the weights aren't in the remote yet. Re-run
  manually; the tags are immutable but `dvc push` is idempotent.

## Storage + state summary

| Thing | Where | Tracked by
---|---|---
Source code, configs, `dvc.yaml`, `params.yaml` | git | git
Image payloads, labels, dataset snapshots | filesystem / S3 | DVC
Model checkpoints (`.pt`) | filesystem / S3 | DVC
MLflow runs (params, metrics, artifacts) | `./mlruns` | MLflow file backend
`models/current` | symlink in working tree | git (the symlink itself)
Versioned releases | `dataset-vX.Y` + `model-vX.Y` tags | git
Pipeline state (stage hashes) | `dvc.lock` | git

The rule of thumb: *if it's code or metadata, git. If it's a blob, DVC.
If it's a measurement, MLflow.*

## Observability

- **MLflow UI** (`mlflow ui`) shows per-run params, metrics, and the
  `best.pt` artifact. Cross-version comparison via the compare view.
- **`runs/evaluate/v{X}.json`** is the authoritative promotion record
  — `jq '.decision, .delta_mAP50' runs/evaluate/v*.json` gives a
  history.
- **Labelme queue depth.** `ls data/labeled/labelme_json/needs_review/
  | wc -l` is the best single health metric: if it grows unboundedly,
  either reviewers are behind or the model is drifting and `CONF_HIGH`
  needs inspection.

## Failure modes and what happens

| Failure | Where caught | Effect
---|---|---
New class appears in autolabel output | `02` label → class-id lookup | `ValueError` in convert; nothing merged
Reviewer uploads a polygon (not rectangle) | `02` shape-type check | `ValueError`; that file is the only failure
Someone adds files to `holdout_test/` | — (no code check yet) | Hard rule violation; caught at review/PR time
Class list edited without bump | `03` validator | `ValueError` at merge; no dataset produced
Model regresses past threshold | `05` | exit 1; CI blocks promote
Cold start on a non-first cycle | `04` logs `COLD START` warning | Visible in MLflow params (`cold_start: true`); investigate before accepting promote
`dvc push` credential issue in CI | CI step | Model already trained+evaluated; weights local, not remote. Runbook has recovery steps

## Non-goals (on purpose)

- **Real-time inference serving.** This repo trains and promotes
  `.pt`s. Serving is a separate concern.
- **Active learning / uncertainty sampling.** The `CONF_HIGH` gate is
  deliberately crude — one threshold, one human queue. We can add
  smarter sampling later, but only if the crude version shows concrete
  pain.
- **Multi-tenant / multi-task.** One repo, one model family, one class
  list (per MAJOR). Cloning the repo for a second model is the
  expected answer.
- **Docker packaging.** Per CLAUDE.md: production packaging comes
  later. Dev runs directly against the host Python environment.

---

# Decisions log

## Why DVC (not Git-LFS)

- Datasets easily reach tens of GB; Git-LFS charges per-GB and lacks pipeline/stage
  semantics.
- DVC `dvc.yaml` gives us declarative stages with deps/outs/metrics, so
  `dvc repro` skips unchanged stages — cheaper CI.
- Remote can be S3 in prod, a local directory in dev. Same commands.

## Why Ultralytics YOLOv11

- Batteries-included training + eval in one package; easy to fine-tune.
- Stable checkpoint format (`*.pt`) and a `model.val()` API that returns
  the mAP50 we gate on.
- Fine-tuning from a previous checkpoint is a one-liner, which matches
  Hard rule #2 exactly.

## Why MLflow (not W&B)

- File-backed tracking URI works out-of-the-box in CI with no external
  account; we can upgrade to a remote server later without code changes.
- Arbitrary artifact logging means we can persist `best.pt` next to its
  params and metrics.

## Why numbered scripts (01_…05_)

- The order IS the pipeline; the names are the documentation.
- Forces one responsibility per script. CLAUDE.md "Do NOT" #7 makes this
  a hard rule.

## Why a fixed holdout_test

- Cross-version comparisons of mAP50 are only meaningful on a stationary
  test set. If the test set drifts, we can't tell whether the model got
  better or the yardstick moved.
- Enforced by Hard rule #1 and by a separate `data.yaml` in
  `data/datasets/holdout_test/`.

## Why the needs_review gate

- Autolabel is only as good as the current model; a model confident in a
  wrong label is a drift accelerant.
- Routing anything below `CONF_HIGH` to a human queue keeps bad labels
  out of the next training set. `03_merge_and_version.py` is hard-wired
  to only read from `auto_accepted/` and `reviewed/` (whitelist, not
  blacklist) so a bug can't leak `needs_review/` content.

## Promotion flow

1. `05_evaluate.py` writes `runs/evaluate/v{X}.json` with `decision:
   promote|block`.
2. Exit code 1 means CI never reaches the promote step.
3. Promotion is a symlink flip on `models/current` + a dataset/model tag
   pair. Rolling back is another symlink flip — see `runbook.md`.
