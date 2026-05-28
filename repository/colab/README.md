# MyCon — Colab Pipeline Helpers

This directory ports the existing MyCon pipeline (Stages 1–11) to a
modular Google Colab workflow with a Gradio UI. **No pipeline code was
rewritten** — everything in here calls the existing `scripts/run_stage.py`
launcher and the `pipeline.stage_*` runners.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker. |
| `log_capture.py` | Thread-safe ring buffer (powers the live log panel). |
| `cleanup.py` | `gc.collect()` + `torch.cuda.empty_cache()` + disk usage. |
| `environment.py` | apt + pip install in a Colab-safe order; environment validation. |
| `drive.py` | Mount Google Drive and create the persistent project tree. |
| `config_manager.py` | Clone `configs/site01.yaml`, apply Colab-safe overrides, write `<project_root>/configs/active.yaml`, validate via `pipeline.common.config.load_config`. |
| `stage_runner.py` | Stage catalog + subprocess driver with live log streaming + status checkpoint. |
| `artifacts.py` | Discover/list/zip the pipeline's output files for download. |
| `ui.py` | Gradio Blocks UI with 4 tabs (`Project & Inputs`, `Run Pipeline`, `Artifacts`, `Environment & Cleanup`). |
| `_build_notebook.py` | One-off generator for `MyCon_Colab_Pipeline.ipynb`. |

## Stage catalog

The catalog mirrors `scripts/run_pipeline_plan.py` and `scripts/run_stage.py`:

```
stage_01_ingest                 (light, mandatory) — video normalize + frame quality
stage_02_keyframes              (light, mandatory) — adaptive keyframe selection
stage_03_colmap                 (heavy)            — COLMAP sparse SfM
stage_04_refinement             (heavy)            — bundle adjustment
stage_04_5_cams_gs              (light)            — Nerfstudio dataset prepare (no train)
stage_05_dense                  (very heavy)       — PatchMatch + fusion (cap max_image_size!)
stage_06_da3_assist             (heavy, skip-safe) — DA3 depth assist (precomputed by default)
stage_07_cleanup                (heavy)            — Open3D cleanup, mesh, planes
stage_07_5_vlm_qa               (light, mock-safe) — pre-BIM VLM/visual QA
stage_07_6_viewer_export        (light)            — viewer artifact pack
stage_07_7_cams_gs_evidence     (light)            — optional 3DGS evidence (visual only)
stage_08_metric_alignment       (server-only)      — metric anchors required
stage_08_bim_registration       (heavy, server-only) — coarse + ICP scan-to-BIM
stage_09_progress               (server-only)      — progress metrics (needs Stage 8 quality)
stage_10_copilot                (light, mock-safe) — copilot ask (uses --question)
stage_11_schedule_variance      (light)            — schedule + variance + dashboard
```

The "Colab-safe default subset" the Gradio UI runs by default is:
`stage_01_ingest`, `stage_02_keyframes`, `stage_07_5_vlm_qa`,
`stage_07_6_viewer_export`, `stage_11_schedule_variance` — these are
the stages that succeed on a free Colab without any extra anchors,
without a real BIM, and without a real VLM endpoint.

## Where outputs live

`drive.setup_project_tree(run_id=...)` creates:

```
MyDrive/MyCon_Colab/projects/<run_id>/
    configs/active.yaml          # the effective config
    uploads/                     # raw user uploads (video, IFC, schedule)
    data/                        # pipeline data tree (frames, sfm, dense, ...)
    runs/<run_id>/reports/       # per-stage JSON summaries (canonical pipeline path)
    runs/<run_id>/logs/          # per-stage .log files mirrored from subprocess stdout
    exports/                     # zipped artifact bundles
```

The pipeline's `project.root` is set to that
`projects/<run_id>/` directory, so every Stage writes directly to
Drive — no extra sync step is needed and Colab disconnects do not lose
state.

## Memory & OOM safety

Between every stage `cleanup.free_memory()` runs:
- `gc.collect()`
- `torch.cuda.empty_cache()` and `ipc_collect()` (when torch is installed and CUDA is available)

The `COLAB_SAFE_OVERRIDES` block in `config_manager.py` also caps
`dense.max_image_size = 1024`, disables hard CUDA preflight failures,
mocks the VLM provider, and forces `cams_gs.execute_training = false`.

## Running outside Colab

The package degrades gracefully:
- `drive.mount_drive` returns `False` and falls back to `/content/MyCon_Colab`.
- `cleanup.free_memory` returns `cuda: torch_not_installed` when torch is missing.
- `environment.install_apt_packages` is skipped when `apt-get` is not on PATH.

This makes the helpers usable on a regular Linux box for unit testing.
