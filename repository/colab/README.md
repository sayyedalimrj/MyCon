# MyCon — Colab Pipeline Helpers

This directory ports the MyCon pipeline (Stages 1–11) to a **production-grade
Google Colab workflow** with a Gradio UI and a headless API. The pipeline
stage code is unchanged — everything here orchestrates the existing
`scripts/run_stage.py` launcher and the `pipeline.stage_*` runners — but the
orchestration is built for long, unattended runs on unstable Colab runtimes:

- **Drive-first persistence** — every artifact, report, log, checkpoint, cache
  and model lands on Google Drive. Fast local caches are mirrored to Drive by a
  resilient background daemon.
- **Checkpoint / resume** — progress is recorded after every stage; a disconnect
  is recovered by re-running with the same `RUN_ID`. Resume also works on a
  different device/Drive account because the whole project tree (config + state
  + artifacts) is portable.
- **Automated setup** — system packages, Python deps, COLMAP, ffmpeg, and an
  optional real local VLM (Ollama + Qwen-VL) are installed for you.
- **Reliability** — retries on apt/pip/Drive I/O, per-stage retries, atomic
  writes, health checks, structured logging.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker / public module list. |
| `log_capture.py` | Thread-safe ring buffer (powers the live log panel). |
| `cleanup.py` | `gc.collect()` + `torch.cuda.empty_cache()` + disk usage. |
| `sync.py` | **Resilient Drive sync**: atomic writes, retrying copies, `mirror_tree`, `DriveSyncManager` background daemon + remount recovery. |
| `checkpoint.py` | **Checkpoint/resume**: atomic `run_state.json` manifest, per-stage status, output-verified completion, portable resume planning. |
| `drive.py` | Mount Drive (retries + force-remount + write-probe health), create the persistent project tree, persistent `model_cache/` + `hf_cache/`, fast local scratch. |
| `models.py` | **Model/asset provisioning**: COLMAP/ffmpeg checks, Ollama install + Qwen-VL pull (cached on Drive), Hugging Face snapshots. |
| `environment.py` | apt + pip install (with retries) in a Colab-safe order; `detect_gpu()`; `bootstrap_environment()` one-call setup; validation. |
| `config_manager.py` | Clone `configs/site01.yaml`, apply an execution **profile** + programmatic + user overrides, write `active.yaml`, validate via `pipeline.common.config.load_config`. |
| `stage_runner.py` | Stage catalog + subprocess driver with live logs, **checkpoint/resume**, **per-stage retries**, canonical ordering, and Drive sync at stage boundaries. |
| `artifacts.py` | Discover/list/zip the pipeline's output files for download. |
| `ui.py` | Gradio Blocks UI (Project & Inputs / Run Pipeline / Artifacts / Environment, Models & Cleanup). |
| `_build_notebook.py` | One-off generator for `MyCon_Colab_Pipeline.ipynb`. |

## Execution profiles

`config_manager.PROFILES` bundles dotted-key overrides applied on top of
`configs/site01.yaml`:

| Profile | Intent |
| --- | --- |
| `colab_safe` | Bounded memory, mock VLM, precomputed DA3, no training. Always works on a free T4. |
| `colab_gpu` *(default)* | Full single-GPU run (real dense + cleanup, larger image budget). Real VLM if provisioned. |
| `production` | Server-grade settings (no artificial caps). Use an A100/L4 high-RAM runtime. |

Override precedence (low → high):
`base config` → `project rewrites` → `inputs` → `profile` → `override_dict`
(e.g. real-VLM wiring from `models.provision_vlm`) → `user_overrides_yaml`.

## Stage catalog & pipeline ordering

The catalog mirrors `scripts/run_pipeline_plan.py` / `scripts/run_stage.py` and
adds, per stage, the **output globs** used to verify resume-completion:

```
stage_01_ingest                 (light, mandatory) — video normalize + frame quality
stage_02_keyframes              (light, mandatory) — adaptive keyframe selection
stage_03_colmap                 (heavy)            — COLMAP sparse SfM
stage_04_refinement             (heavy)            — bundle adjustment
stage_04_5_cams_gs              (light)            — Nerfstudio dataset prepare (no train)
stage_05_dense                  (very heavy)       — PatchMatch + fusion (cap max_image_size!)
stage_06_da3_assist             (heavy, skip-safe) — DA3 depth assist (precomputed by default)
stage_07_cleanup                (heavy)            — Open3D cleanup, mesh, planes
stage_07_5_vlm_qa               (light)            — pre-BIM VLM/visual QA
stage_07_6_viewer_export        (light)            — viewer artifact pack
stage_07_7_cams_gs_evidence     (light)            — optional 3DGS evidence (visual only)
stage_08_metric_alignment       (server-only)      — metric anchors required
stage_08_bim_registration       (heavy, server-only) — coarse + ICP scan-to-BIM
stage_09_progress               (server-only)      — progress metrics (needs Stage 8 quality)
stage_10_copilot                (light)            — copilot ask (uses --question; not resumable)
stage_11_schedule_variance      (light)            — schedule + variance + dashboard
```

- `COLAB_SAFE_DEFAULT_KEYS` — the quick sanity subset
  (`stage_01`, `stage_02`, `stage_07_5`, `stage_07_6`, `stage_11`).
- `FULL_PIPELINE_KEYS` — the full end-to-end ordering. Stages without inputs
  (no BIM/anchors) exit 0 with a `skipped` marker, so running the whole list
  is safe; heavy SfM/dense stages need a GPU runtime.
- `order_stages()` reorders any hand-picked subset into canonical order.

## Where outputs live

`drive.setup_project_tree(run_id=...)` creates:

```
MyDrive/MyCon_Colab/projects/<run_id>/
    configs/active.yaml                  # the effective config
    uploads/                             # raw user uploads (video, IFC, schedule)
    data/                                # pipeline data tree (frames, sfm, dense, ...)
    runs/<run_id>/reports/run_state.json # checkpoint/resume manifest (atomic writes)
    runs/<run_id>/reports/*.json         # per-stage JSON summaries
    runs/<run_id>/logs/*.log             # per-stage subprocess logs (mirrored line-by-line)
    exports/                             # zipped artifact bundles
    model_cache/                         # persistent models (e.g. Ollama via OLLAMA_MODELS)
    hf_cache/                            # persistent Hugging Face cache (mirrored from local)
```

The pipeline's `project.root` is that `projects/<run_id>/` directory, so every
stage writes directly to Drive. Fast local caches under
`/content/mycon_scratch/<run_id>/` are mirrored to Drive by `DriveSyncManager`.

## Checkpoint / resume

`checkpoint.CheckpointManager` persists a `run_state.json` manifest after every
state transition. A stage is treated as **resumable-complete** only when its
recorded status is `ok`/`skipped` **and** its declared output globs still
resolve to real files on Drive (so a partially-synced/lost artifact forces a
re-run). `run_stages(..., resume=True)` skips complete stages and continues
from the first incomplete one. Because the manifest stores *relative* paths and
verifies artifacts on disk, resume is portable to another device or Drive
account that received a copy of `projects/<run_id>/`.

## Stage selection (start from any stage)

`run_stages(...)` runs whatever `spec_keys` you pass, in canonical order, and
with `resume=True` skips stages already complete on Drive. To **start from a
chosen stage** instead of stage 1, use `keys_from_stage("stage_05_dense")`
(returns that stage through the end) or the **Start from stage** dropdown on
Tab 2 of the UI. This does not restart from the beginning; earlier stages are
left untouched and any already-complete stages are skipped.

## Continuous Drive sync

The background `DriveSyncManager` flushes local caches to Drive **~every 60 s**
and at every stage boundary, so progress is saved continuously during a run,
not only at the end. All writes are atomic (temp file + `os.replace`).

## Real local VLM

`models.provision_vlm()` installs Ollama, starts the server (with
`OLLAMA_MODELS` pointed at Drive so models persist), pulls a Qwen-VL model, and
returns the config overrides that switch `copilot.vlm.provider` /
`vlm_qa.provider` to `ollama_local` at `http://127.0.0.1:11434/api/chat`. If
provisioning fails on a given runtime, the pipeline keeps the deterministic
mock provider and never blocks (`fallback_to_mock_when_unavailable: true`).

## Memory & OOM safety

`cleanup.free_memory()` runs after every stage (`gc.collect()` +
`torch.cuda.empty_cache()` + `ipc_collect()` when CUDA is available). The
`colab_safe` profile caps `dense.max_image_size`, disables hard CUDA preflight
failures, mocks the VLM, and forces `cams_gs.execute_training = false`.

## Running outside Colab (CI/laptop)

The package degrades gracefully and is unit-tested on a plain Linux box:

- `drive.mount_drive` returns `False` and falls back to `/content/MyCon_Colab`.
- `models.*` returns `ProvisionResult(ok=False, ...)` instead of raising.
- `cleanup.free_memory` reports `cuda: torch_not_installed` when torch is absent.
- `environment.install_apt_packages` is skipped when `apt-get` is not on PATH.

See `tests/test_colab_*.py` for the laptop-safe coverage (checkpoint, sync,
drive fallback, config profiles, stage planning/resume, model degradation).
