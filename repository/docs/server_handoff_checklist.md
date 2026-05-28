# Server Handoff Checklist

This document is the execution checklist for moving the laptop-prepared construction progress AI/BIM pipeline to the server.

## Current laptop status

- Branch: stage-07-5-vlm-qa
- Latest checkpoint includes:
  - Stage 4.5 CAMS-GS prepare skeleton
  - Stage 7 cleanup
  - Stage 7.5 VLM QA
  - Stage 7.6 viewer export
  - Stage 7.7 CAMS-GS evidence package
  - Stage 8 BIM registration
  - Stage 8 metric alignment
  - Stage 8 visual anchor workflow
  - Stage 9 progress metrics
  - Stage 10 copilot
  - Qwen/Qwen3-VL-8B-Thinking config
  - server model cache scripts
  - server readiness checker
  - pipeline dry-run plan

Validated laptop framework:

    62 passed

## Important rule

Do not complete visual anchor picking on the laptop.

The following require real server/project data:

- real video/images
- real COLMAP reconstruction
- real or project-specific BIM
- real visual anchor observations
- real Qwen VLM endpoint
- optional real 3DGS/CAMS-GS training

## Server step 0: clone and inspect

    git status --short
    git log --oneline -10

Expected:

- clean working tree
- latest commits include server readiness and pipeline plan checks

## Server step 1: configure environment

Copy or create server env file from:

    env/server.env.example

Expected model/profile:

- VLM HF model: Qwen/Qwen3-VL-8B-Thinking
- Ollama/local model name: qwen3-vl:8b-thinking
- local-only mode enabled
- project data remains local

## Server step 2: verify Docker/GPU

Run:

    docker compose -f docker/docker-compose.yml run --rm core python3 scripts/server_readiness_check.py

Expected on a proper GPU server:

- nvidia-smi available
- GPU runtime available
- Qwen profile configured
- missing project inputs are explicitly reported

On laptop, missing GPU is expected. On server, GPU missing is a blocking issue for real VLM and heavy reconstruction.

## Server step 3: prepare model cache

Status check only:

    docker compose -f docker/docker-compose.yml run --rm core python3 scripts/server_model_cache_status.py --config configs/site01.yaml

Prepare/download only on server:

    docker compose -f docker/docker-compose.yml run --rm core python3 scripts/server_prepare_model_cache.py --config configs/site01.yaml

Expected:

- model cache root exists
- Qwen/Qwen3-VL-8B-Thinking is cached or pull instructions are clear
- no cloud endpoint is used unless explicitly approved

## Server step 4: print pipeline plan

Dry-run only:

    docker compose -f docker/docker-compose.yml run --rm core python3 scripts/run_pipeline_plan.py --print-commands

Expected marker:

    PIPELINE_PLAN_OK

This command must not run heavy stages.

## Server step 5: real project inputs

Before real Stage 8/9/10, confirm these exist:

- real video/images
- real SfM/COLMAP model
- dense point cloud
- cleaned point cloud
- real/project BIM IFC
- metric anchors or visual anchor observations
- known distances / benchmark distances if available

## Server step 6: visual anchor workflow

Use this only after real images and BIM are available.

Prepare edge candidates:

    docker compose -f docker/docker-compose.yml run --rm core python3 scripts/detect_structural_edge_candidates.py

Prepare picking packet:

    docker compose -f docker/docker-compose.yml run --rm core python3 scripts/prepare_visual_anchor_picking_packet.py

Create and fill:

    data/bim/design/visual_anchor_observations.csv

Minimum requirement:

- at least 3 real anchors
- each anchor observed in at least 2 registered images
- preferably 3 to 5 observations per anchor

Validate:

    docker compose -f docker/docker-compose.yml run --rm core python3 scripts/validate_visual_anchor_observations.py --strict

Required:

    ready=true
    valid_anchors>=3

## Server step 7: metric alignment

Run after metric/visual anchors are ready:

    docker compose -f docker/docker-compose.yml run --rm core python3 -m pipeline.stage_08_bim_eval.run_metric_alignment --config configs/site01.yaml --force

Expected:

- not skipped_insufficient_anchors
- confidence should be medium/high for real usage
- report written to registration/metric alignment report path

## Server step 8: BIM registration

Run:

    docker compose -f docker/docker-compose.yml run --rm core python3 -m pipeline.stage_08_bim_eval.run_registration --config configs/site01.yaml --force

Required for real progress evidence:

- registration confidence must be defensible
- ICP fitness/RMSE must pass project thresholds
- synthetic demo BIM must not be used for real claim

## Server step 9: progress metrics

Run:

    docker compose -f docker/docker-compose.yml run --rm core python3 -m pipeline.stage_09_progress.run_progress --config configs/site01.yaml --force

Expected outputs:

- element metrics
- activity progress
- deviation summary
- coverage summary
- progress dashboard

If registration confidence is low, Stage 9 must mark results uncertain.

## Server step 10: copilot

Mock-safe test:

    docker compose -f docker/docker-compose.yml run --rm core python3 scripts/smoke_test_stage_10.py

Real ask after VLM and metrics are ready:

    docker compose -f docker/docker-compose.yml run --rm core python3 -m pipeline.stage_10_copilot.run_ask --config configs/site01.yaml --question "Based on the available BIM progress metrics, can this element be accepted?" --json

Expected:

- evidence package created
- selected element/activity traceability preserved when provided
- answer must not overclaim if registration confidence is low

## Optional server stage: 3DGS / CAMS-GS

Stage 4.5 currently prepares data only.

Real training is optional and server-only:

- Nerfstudio/Splatfacto
- CAMS-GS or another Gaussian Splatting method
- Potree/3D tiles if needed for point cloud client viewer

3DGS is for visualization/VR/client preview, not metric truth.

Metric truth remains:

- BIM alignment
- metric anchors
- known distances
- progress metrics

## Final server validation

Run lightweight framework first:

    docker compose -f docker/docker-compose.yml run --rm core pytest -q tests/test_server_readiness_and_pipeline_plan.py

Run full lightweight tests:

    docker compose -f docker/docker-compose.yml run --rm core pytest -q tests/test_stage_04_5_cams_gs.py tests/test_stage_07_cleanup.py tests/test_stage_07_5_vlm_qa.py tests/test_stage_07_6_viewer_export.py tests/test_stage_07_7_cams_gs_evidence.py tests/test_stage_08_bim_eval.py tests/test_stage_08_metric_alignment.py tests/test_stage_08_metric_initial_transform.py tests/test_stage_08_metric_anchor_validation.py tests/test_stage_08_visual_anchor_triangulation.py tests/test_stage_08_visual_anchor_observation_tools.py tests/test_stage_09_progress.py tests/test_stage_10_copilot.py tests/test_smoke_scripts_framework.py tests/test_server_readiness_and_pipeline_plan.py

Expected laptop baseline:

    62 passed

## Do not commit generated server/runtime artifacts

Do not commit:

- data/
- runs/
- exports/
- model_cache/
- models/
- ollama_models/
- hf_cache/
- .venv/

- Current project status: docs/current_project_status.md


## Handoff ZIP verification

After creating a server handoff ZIP, verify it before upload:

```bash
python3 scripts/export_server_handoff_zip.py --output dist/construction-progress-ai-bim_server_handoff.zip
python3 scripts/verify_server_handoff_zip.py dist/construction-progress-ai-bim_server_handoff.zip
```

The verifier fails if requirements files, mirrored requirements, or `env/server.env.example` are missing, or if generated runtime paths such as `data/`, `runs/`, `exports/`, or `model_cache/` are included.
