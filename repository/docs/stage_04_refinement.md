# Stage 4: Sparse refinement

Stage 4 consumes the sparse model created by Stage 3 and writes a refined sparse
model for dense reconstruction. The baseline is intentionally compact and robust:
COLMAP final bundle adjustment is mandatory; PixSfM remains an optional research
hook and is disabled by default.

## Inputs

- `data/sparse/site01/0/cameras.bin`
- `data/sparse/site01/0/images.bin`
- `data/sparse/site01/0/points3D.bin`
- `configs/site01.yaml`

## Outputs

- `data/sparse_refined/site01/0/cameras.bin`
- `data/sparse_refined/site01/0/images.bin`
- `data/sparse_refined/site01/0/points3D.bin`
- `data/sparse_refined/site01/command_history.json`
- `runs/<run_id>/reports/refinement_stats.json`
- `runs/<run_id>/logs/stage_04_refinement.log`

## Why this stage exists

The refined sparse model reduces residual camera and point uncertainty before
image undistortion and dense stereo. Better poses improve downstream dense
geometry, plane extraction, BIM registration, and deviation metrics.

## Robustness decisions

- No in-process PyCOLMAP binary parsing is used for Stage 4 statistics. COLMAP
  subprocesses convert/analyze the model and the Python code parses text outputs.
  This avoids Python-process segfaults when a COLMAP binary model is corrupt.
- `bundle_adjuster -h` is probed and optional BundleAdjustment flags are passed
  only if the installed COLMAP binary explicitly reports them. If help parsing
  fails, only required `--input_path` and `--output_path` are used.
- Principal point refinement is disabled by default because it is commonly
  ill-posed without strong calibration priors.
- The quality gate does not treat moderate point loss as failure. Bundle
  adjustment and filtering may remove bad points. The hard gate focuses on a
  minimum usable model, extreme point collapse, and reprojection-error increase
  when that metric is available.
- PixSfM is optional and disabled by default. Missing PixSfM never blocks the
  baseline.

## Optional research hooks

- `refinement.ba_rounds` supports multi-pass BA experiments, but the safe default
  is one pass.
- PixSfM can later be enabled as a controlled comparison after the operational
  baseline is stable.
- BIM/GCP anchoring is not implemented in Stage 4 baseline because it would
  introduce IFC/BIM dependencies before the BIM registration stage.

## Command

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 -m pipeline.stage_04_refinement.run_refinement `
  --config configs/site01.yaml `
  --force
```

## Acceptance criteria

- `STAGE_04_REFINEMENT_OK` is printed.
- `data/sparse_refined/site01/0/{cameras.bin,images.bin,points3D.bin}` exist and are non-empty.
- `refinement_stats.json` exists and reports a passing quality gate.
- `refinement_stats.json` reports `pycolmap_in_process_used: false`.
- Stage 5 can use `data/sparse_refined/site01/0` as input.


## Implementation note

COLMAP 4.x requires bundle_adjuster --output_path to be an existing directory. Stage 4 therefore creates a clean empty output directory before each BA round. Optional Ceres solver flags use the `BundleAdjustmentCeres.*` namespace when supported by `colmap bundle_adjuster -h`.
