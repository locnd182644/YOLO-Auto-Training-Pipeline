# Runbook

## Rollback a bad promotion

If a promoted model misbehaves in production:

```bash
# 1. Find the previous version
ls -l models/current                 # symlink → models/vX.Y
git tag --list 'model-v*' --sort=-v:refname | head

# 2. Point current back at the previous version
ln -sfn vA.B models/current          # e.g. vA.B = last known-good

# 3. Verify
ls -l models/current
python scripts/05_evaluate.py --new models/current/best.pt \
  --old models/<bad_version>/best.pt

# 4. Commit the symlink flip
git commit -am "rollback to model-vA.B"
git push
```

No `dvc push` needed for a rollback — the old weights are already in the
DVC remote.

## CI fails at `dvc pull`

Usually an S3 credential issue. On the self-hosted runner:

```bash
aws sts get-caller-identity     # confirm IAM identity
dvc remote list
dvc pull -v                     # verbose output shows which file failed
```

## mAP dropped under threshold

1. Check `runs/evaluate/v{X}.json` — look at `delta_mAP50` and per-class
   metrics if logged.
2. Inspect the new data added in v{X}:
   - Any images with label errors in `auto_accepted/` that should have
     gone to `needs_review/`? The routing is based on `CONF_HIGH` — if
     the autolabeler is overconfident, the threshold needs raising.
   - Class imbalance? Compare `labels/train/*.txt` counts vs prev
     version.
3. Possible fixes:
   - Re-review a sample of `auto_accepted/` and move genuine failures
     to `reviewed/` (after human correction).
   - Raise `CONF_HIGH` in `configs/autolabel.yaml` (document the reason
     in the commit message — see Hard rule: "Do not change thresholds
     without updating the config").
   - Train for more epochs.
4. Never disable the regression gate to force a promotion.

## Accidentally added images to `holdout_test/`

Stop. Do not commit. Hard rule #1:

```bash
git restore --staged data/datasets/holdout_test/
git checkout -- data/datasets/holdout_test/
dvc checkout data/datasets/holdout_test.dvc    # restore DVC-tracked state
```

If the change already made it to a DVC tag, open an incident — the
cross-version mAP comparisons before and after the change are no longer
directly comparable.

## First-ever run (no `models/current`)

1. Download `yolov11n.pt` into the repo root or adjust
   `training.pretrained_fallback`.
2. Hand-label a seed set of images with labelme into
   `data/labeled/labelme_json/reviewed/`.
3. Run `02_convert_labelme_to_yolo.py`, `03_merge_and_version.py --prev
   v0.0 --new v1.0` (prev is a sentinel; script tolerates missing prev
   dir), `04_train.py`.
4. Once `models/v1.0/best.pt` exists, `ln -sfn v1.0 models/current`.
5. Commit, `dvc push`, `git tag dataset-v1.0 model-v1.0`.
