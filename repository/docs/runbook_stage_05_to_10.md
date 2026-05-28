# Full Pipeline Runbook: Stage 5 to Stage 10

This runbook documents the validated laptop-side continuation path for the construction progress AI/BIM pipeline.

## Current validated status

- Stage 5 dense preview was generated locally and bridged into the canonical dense path.
- Stage 6 DA3 correctly skipped because dense output was sufficient.
- Stage 7 cleanup passed tests, smoke, real-ish execution, and quality gate.
- Demo BIM assets were generated for Stage 8 and Stage 9 validation.
- Stage 8 BIM registration passed tests, smoke, and real-ish execution with the demo IFC.
- Stage 9 progress metrics were added and passed tests, smoke, and real-ish execution.
- Stage 10 copilot passed tests, smoke, real-ish ask, evidence rendering, confidence calibration, and selected element/activity traceability.

## Important interpretation rule

The current demo IFC is synthetic and is only for pipeline validation. It is not a real project BIM model.

Therefore, low Stage 8 registration confidence is expected when the demo IFC does not geometrically match the reconstructed real scan.

Stage 9 and Stage 10 must preserve this uncertainty. They must not claim real construction progress unless the scan is aligned to a real or project-specific BIM model with acceptable registration confidence.

## Validated chain

Stage 5 dense preview
Stage 6 DA3 assist or skip
Stage 7 cleanup
Stage 7.5 VLM/visual QA
Demo IFC generation
Stage 8 BIM registration
Stage 9 progress metrics
Stage 10 evidence-based copilot


## Stage 4.5: CAMS-GS / 3DGS Prepare

Stage 4.5 prepares optional CAMS-GS / 3D Gaussian Splatting assets for later real-time visualization, VR/client preview, and visual evidence rendering.

It is not a metric truth stage.

Metric truth remains:

- Stage 8 BIM registration
- Stage 9 progress metrics

Current MVP behavior:

- It selects source images from the SfM image directory.
- It writes a Nerfstudio/Splatfacto dataset stub.
- It writes a training manifest.
- It writes a training status file.
- It writes a report JSON.
- It does not execute training yet.
- If Nerfstudio tools are unavailable, the stage remains skip-safe.

Test:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_04_5_cams_gs.py

Smoke:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_04_5_cams_gs.py

Real-ish run:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_04_5_cams_gs.run_cams_gs_prepare `
      --config configs/site01.yaml `
      --force

Current validation result:

- Stage 4.5 pytest passed: 3 passed
- Stage 4.5 smoke passed
- Stage 4.5 real-ish run passed
- Current real-ish status: prepared
- Current real-ish image count: 250
- Current selected image count: 250
- Current Nerfstudio tools: not installed inside the core container

Expected outputs:

- data/cams_gs/site01/train_manifest.json
- data/cams_gs/site01/training_status.json
- data/cams_gs/site01/nerfstudio_dataset/selected_images.txt
- runs/2026-04-30_site01_baseline/reports/cams_gs_prepare_summary.json


## Stage 5: Dense preview bridge

Expected canonical outputs:

- data/dense/site01/fused.ply
- runs/2026-04-30_site01_baseline/reports/dense_summary.json

The laptop-side dense preview was bridged from:

- data/dense_preview/site01_cuda_local/

to:

- data/dense/site01/

## Stage 6: DA3 assist

Command:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_06_da3_assist.run_da3_assist `
      --config configs/site01.yaml `
      --force

Expected result for the current laptop run:

    STAGE_06_DA3_OK status=skipped_dense_sufficient

## Stage 7: Cleanup

Test:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_07_cleanup.py

Smoke:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_07.py

Real-ish run:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_07_cleanup.run_cleanup `
      --config configs/site01.yaml `
      --force

Expected outputs:

- data/clean/site01/cleaned_cloud.ply
- data/clean/site01/downsampled_cloud.ply
- data/clean/site01/mesh.ply
- data/clean/site01/planes.json
- runs/2026-04-30_site01_baseline/reports/cleanup_summary.json

Current laptop validation result:

- Stage 7 pytest passed
- Stage 7 smoke passed
- Stage 7 real-ish run passed
- Stage 7 quality gate passed after radius_m was tuned to 0.5


## Stage 7.5: VLM/visual QA

Stage 7.5 runs after Stage 7 cleanup and before Stage 8 BIM registration.

It creates deterministic visual/geometric QA evidence for the cleaned reconstruction output. It does not prove construction progress by itself.

Test:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_07_5_vlm_qa.py

Smoke:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_07_5_vlm_qa.py

Real-ish run:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_07_5_vlm_qa.run_vlm_qa `
      --config configs/site01.yaml `
      --force

Expected outputs:

- data/vlm_qa/site01/renders/clean_cloud_view.png
- data/vlm_qa/site01/renders/mesh_view.png
- data/vlm_qa/site01/renders/plane_overlay_view.png
- data/vlm_qa/site01/renders/qa_overview.png
- data/vlm_qa/site01/vlm_qa_evidence.json
- runs/2026-04-30_site01_baseline/reports/vlm_qa_summary.json

Current laptop validation result:

- Stage 7.5 pytest passed
- Stage 7.5 smoke passed
- Stage 7.5 real-ish run passed
- Stage 7.5 QA confidence was high for the current cleaned cloud
- cleaned_points=46697
- plane_count=3
- mesh_status=ok

Interpretation rule:

Stage 7.5 checks whether the cleaned reconstruction is visually/geometrically suitable to continue toward BIM registration. It must not claim construction progress by itself.

## Demo BIM asset generation

Command:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/generate_demo_bim_assets.py `
      --force

Expected generated outputs:

- data/bim/design/model.ifc
- data/bim/design/schedule.csv
- data/bim/design/element_activity_map.csv
- data/bim/design/metric_anchors.csv
- data/bim/design/known_distances.csv
- data/bim/design/reference_primitives.json
- data/bim/design/demo_bim_manifest.json

These files are generated runtime/demo assets and are ignored by Git.

The demo IFC contains:

- 1 slab
- 4 walls
- 4 columns
- 9 IFC elements total


## Stage 7.6: Viewer Export

Stage 7.6 packages the current reconstruction/BIM/progress artifacts into a portable viewer/export folder.

Current MVP behavior:

- It does not require PotreeConverter, PDAL, Entwine, py3dtiles, Open3D, or NumPy.
- It copies available artifacts into `exports/viewer/site01/artifacts/`.
- It writes `exports/viewer/site01/viewer_manifest.json`.
- It writes `exports/viewer/site01/index.html`.
- It records whether Potree/Cesium conversion tools are available.
- It is visualization/export only. Metric truth remains Stage 8 and Stage 9.

Test:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_07_6_viewer_export.py

Smoke:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_07_6_viewer_export.py

Real-ish run:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_07_6_viewer_export.run_viewer_export `
      --config configs/site01.yaml `
      --force

Current validation result:

- Stage 7.6 pytest passed: 3 passed
- Stage 7.6 smoke passed
- Stage 7.6 real-ish run passed
- Current real-ish artifact count: 16

Expected outputs:

- exports/viewer/site01/index.html
- exports/viewer/site01/viewer_manifest.json
- exports/viewer/site01/artifacts/



## Stage 7.7: CAMS-GS / 3DGS Evidence

Stage 7.7 packages Stage 4.5 CAMS-GS preparation outputs into a lightweight evidence/viewer package.

It is not a metric truth stage.

Current MVP behavior:

- Reads Stage 4.5 train manifest.
- Reads Stage 4.5 training status.
- Reads Stage 7.6 viewer export manifest when available.
- Writes a CAMS-GS evidence JSON.
- Writes a CAMS-GS viewer manifest.
- Writes a simple HTML evidence page.
- Reports `prepared_no_training` until Nerfstudio/Splatfacto or another 3DGS trainer is connected.

Test:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_07_7_cams_gs_evidence.py

Smoke:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_07_7_cams_gs_evidence.py

Real-ish run:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_07_7_cams_gs_evidence.run_cams_gs_evidence `
      --config configs/site01.yaml `
      --force

Expected current result:

- status: prepared_no_training
- readiness: prepared_stub_only
- is_metric_truth: false

Expected outputs:

- data/cams_gs/site01/evidence/cams_gs_evidence.json
- runs/2026-04-30_site01_baseline/reports/cams_gs_evidence_summary.json
- exports/cams_gs/site01/index.html
- exports/cams_gs/site01/cams_gs_viewer_manifest.json




## Stage 8.0: Metric Alignment

Stage 8.0 estimates a metric scan-to-BIM Sim3 transform from manual anchors and known distances.

It is lightweight and does not transform large point clouds yet.

Test:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_08_metric_alignment.py

CLI report-only run:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_08_bim_eval.run_metric_alignment `
      --config configs/site01.yaml `
      --force

Expected current behavior:

- If scan anchor coordinates are missing, status should be `skipped_insufficient_anchors`.
- After real anchor scan coordinates are filled, status should become `ok` or `alignment_warning`.
- This report will later feed Stage 8 registration.

Expected outputs:

- runs/2026-04-30_site01_baseline/reports/metric_alignment_report.json
- data/bim/aligned/site01/metric_transform_scan_to_bim.json later, after Stage 8 consumes the transform


## Stage 8: BIM registration

Test:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_08_bim_eval.py

Smoke:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_08.py

Real-ish run:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_08_bim_eval.run_registration `
      --config configs/site01.yaml `
      --force

Expected outputs:

- data/bim/aligned/site01/bim_reference.ply
- data/bim/aligned/site01/bim_reference_mesh.ply
- data/bim/aligned/site01/scan_aligned.ply
- data/bim/aligned/site01/transform_scan_to_bim.json
- data/bim/aligned/site01/bim_elements.jsonl
- runs/2026-04-30_site01_baseline/reports/registration_report.json

Important note:

The current demo run may have low ICP fitness because the demo IFC does not match the real reconstructed scan. This is acceptable for pipeline validation, but it is not acceptable as real progress evidence.

## Stage 9: Progress metrics

Test:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_09_progress.py

Smoke:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_09.py

Real-ish run:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_09_progress.run_progress `
      --config configs/site01.yaml `
      --force

Expected outputs:

- data/bim/metrics/site01/element_metrics.csv
- data/bim/metrics/site01/activity_progress.csv
- data/bim/metrics/site01/deviation_summary.json
- data/bim/metrics/site01/coverage_summary.json
- data/bim/metrics/site01/registration_quality.json
- data/bim/metrics/site01/deviation_map.ply
- runs/2026-04-30_site01_baseline/reports/progress_summary.json
- runs/2026-04-30_site01_baseline/reports/progress_dashboard.html

Interpretation rule:

If registration confidence is low, Stage 9 must still write deterministic pipeline artifacts, but it must mark element and activity status as uncertain. It must not claim real progress completion from weak registration.



## Stage 9 scientific interpretation layer

After Stage 9 writes deterministic metrics, run the lightweight interpretation post-processor:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/upgrade_stage09_progress_schema.py

Expected marker:

    STAGE_09_PROGRESS_INTERPRETATION_OK

This adds conservative fields such as `completion_state`, `visibility_confidence`, and `not_evidenced`.
It does not run heavy geometry and does not override Stage 8 registration confidence.

## Stage 10: Copilot

Test:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_10_copilot.py `
      tests/test_smoke_scripts_framework.py `
      tests/test_server_readiness_and_pipeline_plan.py `
      tests/test_stage_07_6_viewer_export.py

Smoke:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_10.py

Real-ish ask with a current element ID:

Because demo IFC GlobalIds may change whenever demo BIM assets are regenerated, first select a current element ID from Stage 9 metrics.

    $pick = @'
    import csv
    from pathlib import Path

    p = Path("data/bim/metrics/site01/element_metrics.csv")
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row.get("activity_id") == "A200":
            print(row["global_id"])
            break
    else:
        print(rows[0]["global_id"])
    '@

    $selectedElementId = ($pick | wsl -d Ubuntu-22.04 -u ali -- bash -lc 'cd /home/ali/projects/construction-progress-ai-bim && python3 -').Trim()
    $selectedActivityId = "A200"

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_10_copilot.run_ask `
      --config configs/site01.yaml `
      --question "Based on the available BIM progress metrics, can this element be accepted?" `
      --selected-element-id "$selectedElementId" `
      --selected-activity-id "$selectedActivityId" `
      --json

Expected behavior for the current demo run:

- confidence: low
- selected_element_id is preserved in the API response and evidence package
- selected_activity_id is preserved in the API response and evidence package
- the answer must not accept the element as complete when registration confidence is low

Expected evidence outputs:

- runs/2026-04-30_site01_baseline/copilot/evidence/latest_evidence_package.json
- runs/2026-04-30_site01_baseline/copilot/renders/*scan_view.png
- runs/2026-04-30_site01_baseline/copilot/renders/*bim_view.png
- runs/2026-04-30_site01_baseline/copilot/renders/*overlay_view.png
- runs/2026-04-30_site01_baseline/copilot/renders/*deviation_heatmap.png

## Final regression

Run:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q `
      tests/test_stage_07_5_vlm_qa.py `
      tests/test_stage_07_cleanup.py `
      tests/test_stage_08_bim_eval.py `
      tests/test_stage_09_progress.py `
      tests/test_stage_10_copilot.py

Expected:

    62 passed

## Smoke regression

Run:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_07.py

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_07_5_vlm_qa.py

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_08.py

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_09.py

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/smoke_test_stage_10.py

Expected:

- STAGE_04_5_SMOKE_OK
- STAGE_07_SMOKE_OK
- STAGE_07_5_SMOKE_OK
- STAGE_07_6_SMOKE_OK
- STAGE_07_7_SMOKE_OK
- STAGE_08_SMOKE_OK
- STAGE_09_SMOKE_OK
- STAGE_10_SMOKE_OK



## Lightweight smoke tests inside pytest

The smoke scripts are also wrapped by `tests/test_smoke_scripts_framework.py`.

This wrapper runs only lightweight synthetic/temp-directory smoke checks. It does not run heavy dense reconstruction, real point-cloud conversion, model download, or server-only processing.

Command:

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_smoke_scripts_framework.py

Expected:

    8 passed



- Server handoff checklist: docs/server_handoff_checklist.md

## Server readiness and dry-run pipeline plan

These commands are laptop-safe. They do not download models, train 3DGS, run dense reconstruction, or run heavy registration.

Server readiness check:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/server_readiness_check.py

Pipeline dry-run plan:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/run_pipeline_plan.py `
      --print-commands

Expected markers:

- SERVER_READINESS_CHECK_OK
- PIPELINE_PLAN_OK

The visual anchor picking workflow is server/project-data-only. It should not be completed on the laptop without real video/images and real/project BIM.

## Do not commit generated runtime artifacts

Ignored/generated artifacts:

- data/dense_preview/
- data/semantics/
- data/vlm_qa/
- data/bim/aligned/
- data/bim/metrics/
- data/bim/design/model.ifc
- data/bim/design/schedule.csv
- data/bim/design/element_activity_map.csv
- data/bim/design/metric_anchors.csv
- data/bim/design/known_distances.csv
- data/bim/design/reference_primitives.json
- data/bim/design/demo_bim_manifest.json
- runs/
- exports/
- .venv/





## Qwen VLM profile generation without downloads

Laptop-safe config generation:

    python3 scripts/apply_vlm_profile.py \
      --base configs/site01.yaml \
      --profile configs/local_qwen_vlm_profile.yaml \
      --output configs/site01_qwen_local.yaml \
      --force

Strict server config generation:

    python3 scripts/apply_vlm_profile.py \
      --base configs/site01.yaml \
      --profile configs/server_qwen_vlm_profile.yaml \
      --output configs/site01_qwen_server.yaml \
      --force

Connectivity check without downloads:

    python3 scripts/check_local_vlm_connection.py \
      --config configs/site01_qwen_local.yaml

Generated configs such as `configs/site01_qwen_local.yaml` and `configs/site01_qwen_server.yaml` should be treated as runtime/profile outputs unless we intentionally decide to version them.


## Server model cache and local VLM preparation

This step is laptop-safe by default.

Dry-run only:

    python3 scripts/server_prepare_model_cache.py \
      --config configs/site01.yaml \
      --provider both

Server execution with actual downloads:

    python3 scripts/server_prepare_model_cache.py \
      --config configs/site01.yaml \
      --provider both \
      --execute

Status check:

    python3 scripts/server_model_cache_status.py \
      --config configs/site01.yaml

Selected model contract:

- Ollama: `qwen3-vl:8b-thinking`
- Hugging Face: `Qwen/Qwen3-VL-8B-Thinking`
- Laptop provider remains `mock` until local/server VLM is available.
- Server provider target is `ollama_local`.
- All model downloads must be stored under persistent `model_cache/` paths.


## Next development targets

1. Connect Stage 7.5 to a real local Qwen/Ollama VLM provider.
2. Add offline local Qwen VLM profile for A5000 / RTX 3090 24GB.
3. Add metricization module using known distances, anchors, and primitive matching.
4. Replace demo IFC with real or project-specific BIM for scientific evaluation.
5. Run server-quality canonical dense reconstruction.
6. Re-run Stage 8 and Stage 9 with real metric alignment and defensible registration confidence.





## Visual anchor observation workflow

Visual anchors bridge BIM benchmark/control points to scan-space when direct scan coordinates are not available.

Prepare a manual observation template:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/prepare_visual_anchor_observations_template.py `
      --force

Validate observations before triangulation:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/validate_visual_anchor_observations.py

Strict server/project validation:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/validate_visual_anchor_observations.py `
      --strict

The observations file must contain at least three anchors, each observed in at least two registered COLMAP images.

## Metric anchor workflow

Metric anchors are physical/project benchmarks or geometric control points that exist in both BIM coordinates and scan coordinates.

Prepare a working CSV:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/prepare_metric_anchors_template.py `
      --source data/bim/design/metric_anchors.csv `
      --output data/bim/design/metric_anchors_working.csv `
      --force

Validate the current anchor files:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/validate_metric_anchors.py

Strict validation for server/project runs:

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 scripts/validate_metric_anchors.py `
      --strict

The current demo project is expected to be incomplete until `scan_x`, `scan_y`, and `scan_z` are filled for at least three anchors.

## Strict server readiness gate

Before spending GPU/server time, run:

```bash
docker compose -f docker/docker-compose.yml run --rm core \
  bash -lc "cd /workspace && PYTHONPATH=/workspace python3 scripts/server_readiness_strict_gate.py --config configs/site01.yaml"
```

On the real server, use strict mode:

```bash
docker compose -f docker/docker-compose.yml run --rm core \
  bash -lc "cd /workspace && PYTHONPATH=/workspace python3 scripts/server_readiness_strict_gate.py --config configs/site01.yaml --strict"
```

The strict gate requires project inputs, IFC, schedule, element map, a metricization route, Qwen/Qwen3-VL-8B-Thinking config, and GPU runtime visibility.

- Current project status: docs/current_project_status.md
