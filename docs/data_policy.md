# Data policy

## Privacy

- No faces, license plates, or other PII may enter `data/raw/incoming/`
  without a documented legal basis. Default assumption: images are
  rejected.
- Per-image provenance is tracked via the DVC tag that first added the
  file. To honor a deletion request: find the tag, remove the file in a
  new dataset version, bump MINOR, retrain.

## Label spec

- Target class list is in `configs/autolabel.yaml` under `classes`. This
  list is the single source of truth.
- Labels are axis-aligned rectangles in labelme (`shape_type:
  rectangle`). Polygons and points are rejected by the converter.
- A bounding box must be fully inside the image (no negative coords, no
  extending past width/height). Labelme's "drag past edge" clipping is
  not applied here by design — we want to see when the labeler was
  imprecise.

## Acceptance criteria for new incoming images

- Format: JPEG, PNG, or BMP. No HEIC (convert upstream).
- Resolution: ≥ 320px on the short side. Below this, fine-tuning tends
  to overfit to low-res artifacts.
- Content: matches the domain the current model was trained on. If a
  new scene type or class appears, escalate — do not silently merge
  (Hard rule #5).
- Duplicate detection (future work): today we rely on reviewer
  judgment + DVC's content-addressed storage to flag re-added bytes.

## Class-list changes

A class-list change is a breaking change:

1. Re-label ALL historical images for the new class.
2. Bump MAJOR (`vN.M` → `v(N+1).0`).
3. Run `03_merge_and_version.py` with `--allow-major`.
4. Update `configs/autolabel.yaml` `classes` in the SAME commit.
5. Full retraining required — fine-tuning across class-list changes
   produces silent corruption.
