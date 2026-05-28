# Construction Progress AI+BIM Pipeline


## Current status snapshot

This repository is now a laptop-ready, server-handoff-ready framework for a staged construction progress AI/BIM pipeline.

Implemented or scaffolded stages:

- Stage 1/2: video ingest, frame quality, keyframes.
- Stage 3/4/5: COLMAP sparse/refinement/dense wrappers.
- Stage 6: DA3 assist scaffold / skip-safe depth completion hook.
- Stage 7: Open3D cleanup, plane extraction, meshing.
- Stage 7.5: VLM/visual QA scaffold.
- Stage 7.6: viewer export package.
- Stage 7.7: CAMS-GS/3DGS evidence package scaffold.
- Stage 8: BIM extraction/registration plus metric alignment, anchor validation, and visual-anchor server workflow.
- Stage 9: deterministic progress/deviation metrics and dashboard scaffold.
- Stage 10: evidence package and local/mock/Qwen-ready copilot.

Important limitations:

- Real Qwen/Qwen3-VL-8B-Thinking inference is server-only until the local model endpoint is installed and cached.
- DA3 and 3DGS/CAMS-GS real training/inference are not laptop-baseline requirements.
- Visual anchor picking requires real project video/images and real/project BIM; do not complete it on the laptop with demo data.
- Progress claims are only defensible when Stage 8 registration confidence and metric anchor residuals are acceptable.

Server handoff:

- See `docs/server_handoff_checklist.md`.
- See `scripts/server_readiness_check.py`.
- See `scripts/run_pipeline_plan.py`.


This repository is organized as a file-contract pipeline for thesis-grade construction progress monitoring:

`mobile video -> ingest/normalization -> adaptive keyframes -> COLMAP sparse/dense -> conditional DA3 -> Open3D cleanup -> IfcOpenShell BIM alignment -> deviation/progress reporting`

This bundle implements Phase 0 shared scaffolding, Stage 1 ingest/normalization, and Stage 2 adaptive keyframe selection. It does not implement COLMAP, DA3, BIM registration, deviation metrics, or dashboards.

## Working assumptions

- Docker working directory: `/workspace`
- Config root inside Docker: `/workspace`
- All paths are project-root-relative in `configs/site01.yaml`
- Docker Compose is run from Windows PowerShell
- Git status should be checked from WSL
- Do not modify or rebuild Docker for Stage 2 unless a test proves it is strictly necessary

## Stage 1 input

```text
data/raw/site01.mp4
```

## Stage 1 outputs

```text
data/normalized/site01_normalized.mp4
data/normalized/site01_metadata.json
data/normalized/site01_frame_quality.csv
runs/<run_id>/reports/stage_01_ingest_report.json
```

## Stage 2 inputs

```text
data/normalized/site01_normalized.mp4
data/normalized/site01_frame_quality.csv
```

## Stage 2 outputs

```text
data/frames/key/site01/*.jpg
data/frames/key/site01_manifest.csv
data/frames/key/site01_contact_sheet.jpg
runs/<run_id>/reports/keyframe_summary.json
```

## Stage 1 quality CSV

Required columns are preserved:

```text
frame_index
timestamp_sec
sharpness_laplacian
exposure_mean
exposure_std
exposure_jump
motion_score
duplicate_similarity
novelty_score
quality_score
reject_reason
```

Additional diagnostic columns may be present and are used by Stage 2 when available:

```text
histogram_similarity
feature_count
feature_density_score
adaptive_blur_threshold
rolling_shutter_score
jitter_score
sampling_method
warning_reason
scoring_width
scoring_height
```

## Stage 2 keyframe policy

Stage 2 selects useful frames rather than merely fewer frames. It reads Stage 1 scoring, rejects obviously bad rows, groups valid rows into stable subsequences, enforces `keyframes.min_time_gap_sec`, caps the first run with `keyframes.max_frames_first_run`, preserves temporal coverage, and writes an explainable manifest.

The baseline remains lightweight. It does not use COLMAP, DA3, SAM, YOLO, SIFT, or learned models. If too few frames survive strict gates, Stage 2 can use a controlled fallback that relaxes duplicate and aggregate rejection first while still rejecting dangerous blur, severe exposure jumps, and extreme motion. A final emergency fallback can select the best available rows only to avoid an empty manifest on severely degraded videos; the summary report flags this so the run can be reviewed before Stage 3.

Stage 2 validates frame-index bounds against the normalized video and checks timestamp/frame-index consistency by default. This catches accidental VFR/CFR drift before keyframes are extracted.

## PowerShell commands

Run from Windows PowerShell:

```powershell
cd "\\wsl.localhost\Ubuntu-22.04\home\ali\projects\construction-progress-ai-bim"
```

Stage 1:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 -m pipeline.stage_01_ingest.run_ingest `
  --config configs/site01.yaml
```

Stage 2:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 -m pipeline.stage_02_keyframes.select_keyframes `
  --config configs/site01.yaml `
  --force
```

Generic launcher:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/run_stage.py stage_02_keyframes `
  --config configs/site01.yaml `
  --force
```

Smoke tests:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/smoke_test_stage_01.py

docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/smoke_test_stage_02.py
```

Pytest:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  pytest -q tests/test_config.py tests/test_stage_01_ingest.py tests/test_stage_02_keyframes.py
```

## WSL commands

```bash
cd ~/projects/construction-progress-ai-bim
source .venv/bin/activate

python -m pipeline.stage_01_ingest.run_ingest --config configs/site01.yaml
python -m pipeline.stage_02_keyframes.select_keyframes --config configs/site01.yaml --force
python scripts/smoke_test_stage_01.py
python scripts/smoke_test_stage_02.py
pytest -q tests/test_config.py tests/test_stage_01_ingest.py tests/test_stage_02_keyframes.py
```

## Authoritative Git status

```powershell
wsl -d Ubuntu-22.04 -u ali -- bash -lc "cd ~/projects/construction-progress-ai-bim && git status --short"
```

## Acceptance criteria

Stage 1 is accepted when it creates the normalized video, metadata JSON, frame quality CSV, and report, and its smoke/pytest tests pass.

Stage 2 is accepted when:

1. `python3 -m pipeline.stage_02_keyframes.select_keyframes --config configs/site01.yaml --force` runs from Docker Compose.
2. Stage 2 reads only Stage 1 outputs and YAML config.
3. Keyframe JPGs are written under `data/frames/key/site01/`.
4. `data/frames/key/site01_manifest.csv` exists and contains the required manifest columns.
5. `data/frames/key/site01_contact_sheet.jpg` exists.
6. `runs/<run_id>/reports/keyframe_summary.json` exists.
7. `scripts/smoke_test_stage_02.py` prints `STAGE_02_SMOKE_OK`.
8. Pytest passes for config, Stage 1, and Stage 2.
9. Generated keyframes, manifests, contact sheets, and run reports remain ignored by Git unless deliberately added for a small fixture.

- Current project status: docs/current_project_status.md


## Server handoff ZIP

Create the official source handoff ZIP with:

```bash
python3 scripts/export_server_handoff_zip.py --output dist/construction-progress-ai-bim_server_handoff.zip
```

This exporter verifies that server-critical files such as requirements files and `env/server.env.example` are tracked and included.


## Handoff ZIP verification

After creating a server handoff ZIP, verify it before upload:

```bash
python3 scripts/export_server_handoff_zip.py --output dist/construction-progress-ai-bim_server_handoff.zip
python3 scripts/verify_server_handoff_zip.py dist/construction-progress-ai-bim_server_handoff.zip
```

The verifier fails if requirements files, mirrored requirements, or `env/server.env.example` are missing, or if generated runtime paths such as `data/`, `runs/`, `exports/`, or `model_cache/` are included.
