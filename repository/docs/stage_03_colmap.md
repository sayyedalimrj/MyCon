# Stage 3 — COLMAP Sparse SfM

Stage 3 consumes the Stage 2 keyframes and manifest and produces the first sparse reconstruction product in the pipeline. It does not run dense stereo, DA3, BIM registration, or dashboards.

## Inputs

- `configs/site01.yaml`
- `data/frames/key/site01_manifest.csv`
- `data/frames/key/site01/*.jpg`

Only manifest rows with `keep_sparse=true` are staged into the COLMAP image set.

## Outputs

- `data/sfm/site01/images/*.jpg` staged SfM image set
- `data/sfm/site01/active_manifest.csv`
- `data/sfm/site01/image_list.txt`
- `data/sfm/site01/database.db`
- `data/sfm/site01/command_history.json`
- `data/sparse/site01/0/cameras.bin`
- `data/sparse/site01/0/images.bin`
- `data/sparse/site01/0/points3D.bin`
- `runs/<run_id>/reports/sparse_stats.json`
- `runs/<run_id>/logs/stage_03_colmap.log`

## Mainline

The main attempt uses COLMAP native `ALIKED_N16ROT` feature extraction and `ALIKED_LIGHTGLUE` sequential matching. This matches the thesis mainline for weak-texture construction videos.

## Conservative defaults in v2

This revision incorporates the accepted engineering review points without adding heavy optional dependencies:

- `colmap.aliked_max_num_features` defaults to `2048`, not `8192`, to reduce LightGlue memory pressure.
- `colmap.sift_max_num_features` defaults to `2048` for the same reason.
- Stage images are always copied. Symlink mode is ignored with a warning because Windows/WSL bind mounts and C++ tools can be fragile around symlinks.
- Sparse statistics avoid in-process `pycolmap.Reconstruction`. Counts are collected through COLMAP subprocesses (`model_converter`, `model_analyzer`) after binary file validation, preventing a corrupt model from segfaulting the Python process.
- Sequential overlap defaults to `15` with quadratic overlap enabled. Loop detection remains disabled by default because it may require a valid vocabulary tree and can introduce another deployment dependency. If a vocabulary tree is available, enable it explicitly with `colmap.sequential_loop_detection: true` and `colmap.sequential_vocab_tree_path`.

## Optional masks

Stage 3 does not run YOLO, SAM, or semantic segmentation. Those are optional research extensions and should not block the sparse baseline. However, if precomputed COLMAP-compatible masks already exist, Stage 3 can pass them to COLMAP:

```yaml
colmap:
  use_existing_masks: true
  mask_path: data/masks/site01
  require_masks: false
```

Masks must be named consistently with the staged images and use COLMAP's expected binary mask convention.

## Fallback

The configured fallback is `SIFT` + `SIFT_LIGHTGLUE`. All COLMAP commands run with `QT_QPA_PLATFORM=offscreen` so the SIFT path does not require an X11 display. An emergency `SIFT_BRUTEFORCE` fallback exists only if explicitly enabled with `colmap.allow_sift_bruteforce_emergency: true`.

## Commands

Apply config additions once:

```powershell
cd "\\wsl.localhost\Ubuntu-22.04\home\ali\projects\construction-progress-ai-bim"

docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/apply_stage_03_config.py --config configs/site01.yaml --force-update-report-path --force-safe-defaults
```

Run the smoke test:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/smoke_test_stage_03.py
```

Run Stage 3 on the real Stage 2 keyframes:

```powershell
docker compose -f docker\docker-compose.yml run --rm `
  -e http_proxy=http://host.docker.internal:10808 `
  -e https_proxy=http://host.docker.internal:10808 `
  -e HTTP_PROXY=http://host.docker.internal:10808 `
  -e HTTPS_PROXY=http://host.docker.internal:10808 `
  core python3 -m pipeline.stage_03_colmap.run_sparse `
  --config configs/site01.yaml `
  --force
```

Run tests:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  pytest -q tests/test_config.py tests/test_stage_01_ingest.py tests/test_stage_02_keyframes.py tests/test_stage_03_colmap.py
```

## Acceptance criteria

- `STAGE_03_COLMAP_OK` appears.
- `data/sfm/site01/database.db` exists and is non-empty.
- `data/sparse/site01/0/cameras.bin`, `images.bin`, and `points3D.bin` exist and are non-empty.
- `sparse_stats.json` reports a meaningful `registered_image_count` and `registered_ratio`.
- The log and command history are saved.
- Generated data under `data/sfm`, `data/sparse`, and `runs` remains out of Git.
