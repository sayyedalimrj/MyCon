# Colab Workflow — Running MyCon on Google Colab

This document is the user-facing companion to `MyCon_Colab_Pipeline.ipynb`
(at the repo root) and the helper package under `colab/`. It targets users who
want to run the pipeline against their own video on a Colab GPU, with outputs
that **survive disconnects** and runs that **resume automatically**.

## TL;DR

1. Open `MyCon_Colab_Pipeline.ipynb` in Colab.
2. Set **Runtime → GPU (T4 or better)** and **High-RAM** if available.
3. Run cells **0 → 4** in order. Cell 2 installs everything; cell 3 mounts Drive
   and starts the background sync daemon; cell 4 launches the Gradio UI.
4. In the Gradio UI:
   1. **1. Project & Inputs**: pick a **Run ID** and an **execution profile**,
      upload a video (and optional IFC + schedule), click **Save inputs**.
   2. **4. Environment, Models & Cleanup** (optional): click **Provision real
      local VLM** to install Ollama + Qwen-VL so Stages 7.5/10 use real vision.
   3. **2. Run Pipeline**: tick **Resume**, then click **Run FULL pipeline**
      (or **Run Colab-safe subset** for a quick sanity check).
   4. **3. Artifacts & Downloads**: refresh and build a zip bundle.
5. If Colab disconnects, re-run cells 1–4 with the **same Run ID** — finished
   stages are skipped and the run continues.

## Architecture

The notebook is thin: it bootstraps the environment, mounts Drive, sets up the
project tree, and launches `colab.ui.build_ui` (or the headless API). All real
work is dispatched as subprocesses to the existing pipeline runners.

```
   ┌──────────────────────────────┐
   │  MyCon_Colab_Pipeline.ipynb       │  (cells 0–8)
   └──────────────┬───────────────┘
                  │ uses
                  ▼
   ┌──────────────────────────────┐
   │  colab/ (this package)            │
   │   environment / drive / sync /    │
   │   checkpoint / models /           │
   │   config_manager / log_capture /  │
   │   stage_runner / artifacts /      │
   │   cleanup / ui                    │
   └──────────────┬───────────────┘
                  │ subprocess: scripts/run_stage.py  /  python -m pipeline.stage_*
                  ▼
   ┌──────────────────────────────┐
   │  pipeline.stage_*           │  (unchanged)
   └──────────────────────────────┘
```

## Execution profiles

Pick one on Tab 1 (or set `PROFILE` in the headless cells):

| Profile | Use when |
| --- | --- |
| `colab_safe` | First run / free T4. Bounded memory, mock VLM, no training; always succeeds. |
| `colab_gpu` *(default)* | Full single-GPU reconstruction (real dense + cleanup); real VLM if provisioned. |
| `production` | A100/L4 high-RAM or on-prem GPU; server-grade settings, no artificial caps. |

## Persistent project layout on Drive

For run id `<RID>`:

```
MyDrive/MyCon_Colab/projects/<RID>/
    configs/active.yaml                  ← effective config every stage reads
    uploads/                             ← raw uploads (video.mp4, model.ifc, schedule.csv)
    data/                                ← pipeline data tree (frames, sfm, dense, clean, bim, ...)
    runs/<RID>/reports/run_state.json    ← checkpoint/resume manifest
    runs/<RID>/reports/*.json            ← per-stage JSON summaries
    runs/<RID>/logs/*.log                ← per-stage subprocess logs (mirrored line-by-line)
    exports/                             ← zipped bundles built from the UI Tab 3
    model_cache/                         ← persistent models (Ollama via OLLAMA_MODELS)
    hf_cache/                            ← persistent Hugging Face cache (mirrored from local scratch)
```

Everything the pipeline writes is on Drive, so disconnects only cost wall-clock
time on the currently-running stage. Heavy caches are written to a fast local
scratch (`/content/mycon_scratch/<RID>/`) and mirrored to Drive every ~2 min by
`colab.sync.DriveSyncManager`, which also re-mounts Drive if the FUSE mount goes
stale after a reconnect.

## Checkpoint / resume

`colab.checkpoint.CheckpointManager` writes `run_state.json` atomically after
every stage transition (running → ok/failed/skipped). On resume, a stage is
skipped only when:

1. its recorded status is `ok` (or `skipped`), **and**
2. its declared output artifacts still exist on Drive.

So a manifest that says "ok" but whose artifacts were lost/partially-synced
forces a clean re-run of that stage. To resume:

- **Same machine after a disconnect:** re-run the notebook with the **same**
  `RUN_ID`.
- **Another device / Drive account:** copy or share
  `MyDrive/MyCon_Colab/projects/<RID>/` to the new Drive, set the same `RUN_ID`,
  and run. The config and manifest travel with the folder.

Per-stage **retries** (default 2 attempts with backoff) handle transient
failures (e.g. a flaky download) without manual intervention.

## Effective config — what we override

We always start from `configs/site01.yaml` (the canonical, validated config),
then apply, in order:

1. **Mandatory project-level rewrites** — `project.name/run_id/root`, report
   paths re-anchored to `runs/<RID>/...`.
2. **Inputs** — `inputs.video/ifc/schedule` (only when provided).
3. **Execution profile** — `colab_safe` / `colab_gpu` / `production`.
4. **Programmatic overrides** — e.g. the real-VLM wiring from
   `models.provision_vlm` (switches `copilot.vlm.provider` to `ollama_local`).
5. **Free-form user overrides** — a YAML mapping from the UI textbox.

The final config is written to `configs/active.yaml` and validated through
`pipeline.common.config.load_config()` before any stage runs.

## Automated dependency & model setup

- `environment.bootstrap_environment()` installs apt packages (ffmpeg, colmap,
  zstd, aria2, ...), pins `numpy<2`, installs `requirements-core` +
  `requirements-da3` + the UI deps (all with retries), and validates each layer.
- `models.provision_vlm()` installs Ollama, starts the server with
  `OLLAMA_MODELS` on Drive (so the model is reused after a reconnect), pulls a
  Qwen-VL model, and returns the config overrides for the real local VLM.
- `models.ensure_hf_model()` pre-downloads Hugging Face snapshots (resumable)
  into the Drive-backed cache.

## Stages

The Gradio "Run Colab-safe subset" button runs:
`stage_01_ingest → stage_02_keyframes → stage_07_5_vlm_qa →
stage_07_6_viewer_export → stage_11_schedule_variance`.

"Run FULL pipeline" runs the canonical end-to-end ordering (Stages 1 → 11).
Stages that lack inputs (no real BIM, no anchors) exit 0 with a `skipped`
marker, so the full run is safe; heavy SfM/dense stages need a GPU runtime. Any
hand-picked subset is auto-reordered into canonical order before running.

## Logs & progress

1. The **Live log** panel on Tab 2 (auto-refreshes every 3 s).
2. `runs/<RID>/logs/<stage>.log` on Drive — exact subprocess stdout/stderr.
3. `runs/<RID>/reports/run_state.json` — machine-readable per-stage status
   (also rendered in the Tab 2 "Stage status" table).

## Memory hygiene

`colab.cleanup.free_memory()` runs after every stage (`gc.collect()` +
`torch.cuda.empty_cache()` + `ipc_collect()`), and the Tab 4 buttons let you
free memory or force a Drive sync on demand.

## Known caveats

- **Free Colab disconnects after ~12 h.** Re-attach and resume; outputs are on
  Drive.
- **Stage 5 (dense) on a free T4** can OOM at large `dense.max_image_size`; use
  `colab_safe` or set `dense.max_image_size: 800` in the overrides textbox.
- **Stage 8b/9** need a real BIM IFC + matched scan; otherwise they exit 0 with
  a `skipped` marker.
- **Real VLM** requires the Ollama provision step; without it the deterministic
  mock answers (the pipeline never blocks on the VLM).

## Re-generating the notebook

The notebook is generated from `colab/_build_notebook.py` for deterministic
diffs:

```bash
python3 colab/_build_notebook.py
```
