# Colab Workflow — Running MyCon on Google Colab

This document is the user-facing companion to `MyCon_Colab_Pipeline.ipynb`
(at the repo root) and the helper package under `colab/`. It is aimed at
non-experts who want to run the pipeline against their own video without
deploying the full server stack.

## TL;DR

1. Open `MyCon_Colab_Pipeline.ipynb` in Colab.
2. Set **Runtime → GPU (T4 or better)** and **High-RAM** if available.
3. Run cells **0 → 4** in order. The Gradio UI will launch in cell 4.
4. In the Gradio UI:
   1. Tab **1. Project & Inputs**: pick a run id, mount Drive, upload a
      video (and optional IFC + schedule), then click **Save inputs &
      write effective config**.
   2. Tab **2. Run Pipeline**: click **Run Colab-safe default subset**
      to verify everything works end-to-end (Stages 1, 2, 7.5, 7.6, 11).
   3. Tab **3. Artifacts & Downloads**: refresh the artifact list and
      build a zip bundle for Drive.
5. After the safe run succeeds, you can opt into heavier stages
   (Stage 3 sparse, 5 dense, 7 cleanup, 8 BIM) one by one — each one is
   independent and can be re-run.

## Architecture

The notebook itself is intentionally thin: it just installs deps,
mounts Drive, sets up the project tree, and launches `colab.ui.build_ui`.
All real work is dispatched as subprocesses to the existing pipeline
runners.

```
   ┌──────────────────────────┐
   │  MyCon_Colab_Pipeline.ipynb  │  (cells 0–8)
   └──────────────┬───────────┘
                  │ uses
                  ▼
   ┌──────────────────────────┐
   │  colab/ (this package)        │
   │   environment / drive /       │
   │   config_manager / log /      │
   │   stage_runner / artifacts /  │
   │   cleanup / ui                │
   └──────────────┬───────────┘
                  │ subprocess: python -m pipeline.stage_*  /  scripts/run_stage.py
                  ▼
   ┌──────────────────────────┐
   │  pipeline.stage_*       │  (unchanged)
   └──────────────────────────┘
```

## Persistent project layout on Drive

For run id `<RID>` (default: `YYYY-MM-DD_HHMMSS_colab`):

```
MyDrive/MyCon_Colab/projects/<RID>/
    configs/active.yaml          ← the YAML config every stage reads
    uploads/                     ← raw uploads (video.mp4, model.ifc, schedule.csv)
    data/                        ← pipeline data tree (frames, sfm, dense, clean, bim, ...)
    runs/<RID>/reports/          ← per-stage JSON summaries (sparse_stats.json, dense_summary.json, ...)
    runs/<RID>/logs/             ← per-stage .log files (mirrored stdout/stderr)
    exports/                     ← zipped bundles built from the UI Tab 3
```

Everything the pipeline writes is on Drive, so disconnects only cost
you wall-clock time on the currently-running stage.

## Effective config — what we override

We always start from `configs/site01.yaml` (the canonical, fully
validated config) and then apply, in order:

1. **Mandatory project-level rewrites**
   - `project.name` ← run id
   - `project.run_id` ← run id
   - `project.root` ← `<project_root>` on Drive
   - `paths.*_report_json` etc. → re-anchored to `runs/<run_id>/...`
2. **Inputs** (only when the user provides them)
   - `inputs.video`, `inputs.ifc`, `inputs.schedule`
3. **Colab-safe defaults** (when the *Apply Colab-safe defaults*
   checkbox is on)
   - `dense.max_image_size: 1024`, `dense.fail_on_quality_gate: false`,
     `dense.require_cuda: false`
   - `da3.provider: precomputed`, `da3.fail_if_required_but_unavailable: false`
   - `vlm_qa.provider: mock`, `copilot.vlm.provider: mock`,
     `copilot.vlm.fallback_to_mock_when_unavailable: true`
   - `cams_gs.execute_training: false`
   - `cleanup.fail_on_quality_gate: false`,
     `bim.fail_on_low_registration_quality: false`
4. **Free-form user overrides** — a YAML mapping from the UI textbox.
   Deep-merged over the result of steps 1–3.

The final config is written to `configs/active.yaml` on Drive and
validated through `pipeline.common.config.load_config()` so a syntactic
or semantic error is reported before any stage actually runs.

## Stage catalog & safe defaults

See `colab/README.md` for the full table. The Gradio "Run Colab-safe
default subset" button runs:

`stage_01_ingest → stage_02_keyframes → stage_07_5_vlm_qa →
stage_07_6_viewer_export → stage_11_schedule_variance`.

These five stages have no external dependencies (no real BIM, no
metric anchors, no real VLM endpoint, no GPU pressure beyond the
ingest/keyframes layer) so they always succeed on free Colab.

For a richer run, add stages in this order:

- `stage_03_colmap` (sparse SfM, GPU recommended).
- `stage_04_refinement` (cheap once Stage 3 succeeded).
- `stage_05_dense` (cap `dense.max_image_size: 800` first time).
- `stage_07_cleanup` (Open3D cleanup + mesh + planes).
- `stage_08_metric_alignment` / `stage_08_bim_registration` /
  `stage_09_progress` — only when you have a real BIM and metric or
  visual anchors.
- `stage_10_copilot` — copilot ask. The UI exposes a `question` textbox.
- `stage_06_da3_assist` — only when you actually have DA3 depth maps.

## Logs

Three places will show progress:

1. The **Live log** panel on Tab 2 of the Gradio UI (auto-refreshes
   every 3 s via `gr.Timer`).
2. The notebook's stdout if you used the manual cells in Section 5.
3. `<project_root>/runs/<run_id>/logs/<stage_key>.log` on Drive — the
   exact subprocess stdout/stderr, mirrored line by line.

A `runs/<run_id>/reports/colab_run_status.json` checkpoint is written
after every stage so a session that disconnects mid-run can be inspected
("how far did we get?") without re-running.

## Memory hygiene

`colab.cleanup.free_memory()` runs after every stage:

```python
gc.collect()
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
```

The Tab 4 *Free GPU/CPU memory now* button calls the same function on
demand. Use it before launching Stage 5 / 7 / 8 if Tab 4's *Show disk
usage* shows residual GPU usage from a prior stage.

## Known caveats

- **Free Colab disconnects after ~12 h.** Re-run cells 0–4 in a fresh
  session and continue; outputs are on Drive.
- **`opencv-python-headless` vs `opencv-python`**: the requirements
  pin `opencv-python-headless` to keep installs minimal; it is enough
  for the Stage 1/2 frame quality and Stage 7 ops the pipeline uses.
- **Stage 5 (dense) on free T4** can OOM at the default
  `max_image_size: 1024`. Drop to `800` (or `640`) in the config
  overrides textbox.
- **Stage 8b/9** require a real BIM IFC + matched scan; on Colab they
  will often log `skipped_insufficient_anchors` and exit 0.
- **Stage 10 (VLM ask)** uses the **mock** provider by default. Real
  Qwen 3-VL needs Ollama or a vLLM endpoint reachable from the Colab
  runtime; this is documented in `docs/qwen_vlm_laptop_to_server_plan.md`.

## Re-generating the notebook

The notebook is generated from `colab/_build_notebook.py` for
deterministic diffs. To rebuild after editing a cell:

```bash
python3 colab/_build_notebook.py
```
