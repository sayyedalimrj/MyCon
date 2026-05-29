# Synthetic Floor — 7-Stage Construction Progress Dataset

A **complete, deterministic, reproducible** example dataset generator for
testing the MyCon construction-progress pipeline end-to-end without any
real site data.

## What This Is

This example creates a **single building floor** (18 m × 12 m) and
renders it at **seven progressive construction stages** — from bare
columns + slab all the way to a fully finished interior. Every stage
produces:

| Output | Format | Purpose |
|--------|--------|---------|
| BIM model | IFC4 (`.ifc`) | Scan-vs-BIM comparison |
| 3D mesh | OBJ + GLB + PLY | Visualization / Open3D |
| Video (smartphone sim) | H.264 MP4 | Pipeline Stage 1 input |
| Video (clean) | H.264 MP4 | Ablation / debugging |
| Depth maps | `.npz` | Dense reconstruction ground truth |
| Segmentation | `.npz` | Element-level labels |
| Camera path | JSON (4×4 matrices) | COLMAP / SfM ground truth |
| Element metrics | CSV | Stage 11 schedule variance |
| Metadata | JSON | Full provenance |

Plus dataset-level files:

* `schedule.csv` — synthetic 7-activity schedule (canonical format)
* `bim_schedule_mapping.csv` — element ↔ activity mapping
* `manifest.json` — SHA-256 indexed file registry

## The 7 Stages

| # | Name | What changes |
|---|------|-------------|
| 1 | Structural Skeleton | Slab + 12 columns only |
| 2 | Partial Exterior Walls | ~50% of exterior masonry |
| 3 | Full Exterior + Window Openings | All exterior walls; window cut-outs visible |
| 4 | Partial Interior Walls | ~50% of partitions; door openings |
| 5 | Full Walls + Partial Doors/Windows | All walls up; some glass + doors |
| 6 | Rough Finishing | Plaster, partial floors, ceiling |
| 7 | Fully Completed Interior | Painted, tiled, doors + baseboards done |

The geometry is **identical** across all stages. Only the completion
state and surface finish change.

## Quick Start

```bash
# From the repository root:
# 1. Install dependencies (if not already present)
pip install numpy Pillow PyYAML trimesh imageio imageio-ffmpeg ifcopenshell scipy

# 2. Run a fast smoke test (~2 min, 320×180 resolution, 1s clips)
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/scripts/run_generate.py --quick

# 3. Run the full default pipeline (~10-15 min, 640×360, 3s clips)
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/scripts/run_generate.py

# 4. Run at higher quality (much slower, ~60+ min)
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/scripts/run_generate.py \
        --width 1280 --height 720 --duration 6.0
```

All outputs land in `examples/synthetic_floor_7stage/output/`.

## Useful CLI Flags

| Flag | Effect |
|------|--------|
| `--preset {debug,balanced,hq}` | Quality preset (default: `balanced`) |
| `--quick` | Alias for `--preset debug` |
| `--stages 1 3 7` | Only generate specific stages |
| `--resume` | Skip stages whose artefacts already exist |
| `--force` | Delete prior outputs and rerun |
| `--strict-render` | Fail loudly on missing renders / videos |
| `--skip-render` | Skip rendering (BIM + mesh only) |
| `--skip-bim` | Skip IFC export |
| `--width N --height N` | Override resolution (wins over preset) |
| `--duration S` | Override per-stage clip length (wins over preset) |
| `--save-frames` | Also save per-frame PNGs |
| `--log-level DEBUG` | Verbose logging |

### Quality presets

| Preset | Resolution | Duration / stage | Notes |
|--------|------------|------------------|-------|
| `debug`    | 320×180   | 1.0 s | Smoke tests / CI; finishes in seconds |
| `balanced` | 640×360   | 4.0 s | Default; ~10–15 min full 7-stage run |
| `hq`       | 1280×720  | 6.0 s | Final figures; ~30+ min full run |

The GPU pipeline below uses the same preset names but tunes Blender Cycles
parameters (samples, frames, motion blur) instead of the CPU-only knobs.

### Resume / force / strict-render

Long pipeline runs sometimes die halfway through (Colab disconnects,
GPU OOM, etc.). Both runners support stage-level resume:

```bash
# First run dies somewhere in the middle ...
python3 .../run_generate.py --preset balanced
# ... Ctrl-C / crash ...

# Pick up where we left off; already-complete stages are skipped:
python3 .../run_generate.py --preset balanced --resume

# Force a single stage to be regenerated from scratch:
python3 .../run_generate.py --preset balanced --stages 5 --force
```

| Flag | Meaning |
|------|---------|
| `--resume` | Skip stages whose artefacts already exist on disk (alias for `--mode resume`). |
| `--force` | Delete prior outputs and rerun (alias for `--mode force`). |
| `--mode {run,resume,force,redo}` | Explicit selector. Default: `run`. |
| `--strict-render` | Fail loudly if a render or video step does not produce its expected outputs (no silent fallbacks). Use this for dataset releases. |

A successful stage drops a small `<output>/manifests/stage_NN.done`
(CPU) or `<output>/blender_renders/stage_NN/.done` (GPU) marker JSON
that records the preset, frame counts, and elapsed time.

### Running the smoke tests

```bash
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/tests/run_all.py
```

The suite covers the path helpers, presets, `metadata_exporter._entry`
(directories / `None` / missing files), the robust MP4 encoder
(mixed sizes + corrupt frames), the Blender 4.x compatibility shims,
and the resume/checkpoint logic. ~40 tests, finishes in <1 s.

## GPU / Colab pipeline (Blender 4.2 LTS)

If you have access to a CUDA GPU (or a free Colab T4 / paid A100), there
is a **second renderer** based on Blender + Cycles that produces much
more photorealistic frames — same scene, same 7 stages, same metadata
contract. It lives next to the CPU code:

```
examples/synthetic_floor_7stage/
├── src/synthetic_floor/blender_gpu_renderer.py    # runs INSIDE Blender
├── scripts/run_blender_gpu.py                     # host-side orchestrator
├── scripts/setup_colab_blender.sh                 # installs Blender 4.2
└── colab/
    ├── synthetic_floor_blender_gpu.ipynb          # one-click notebook
    └── README.md                                  # full GPU docs
```

Quick start (after opening the notebook in Colab with a GPU runtime):

```bash
bash examples/synthetic_floor_7stage/scripts/setup_colab_blender.sh
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/scripts/run_blender_gpu.py \
        --blender /content/blender/blender --preset debug
```

The GPU runner accepts `--preset {debug,balanced,hq}` with the same
override semantics as the CPU pipeline. Explicit flags such as
`--resolution 1920 1080` or `--samples 256` always win over the preset.

### Google Drive persistence + resume (Colab)

The GPU runner can mirror everything it produces to Google Drive so a Colab
disconnect never loses work, and a run can be resumed later (or from another
device / Drive account):

```bash
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/scripts/run_blender_gpu.py \
        --blender /content/blender/blender --preset balanced \
        --mount-drive \
        --drive-root /content/drive/MyDrive/MyCon_Colab/synthetic_floor_7stage/demo \
        --resume --strict-render
```

`--mount-drive` mounts Drive; outputs are pushed to `--drive-root/output/`
after every stage (plus a background sync); `--resume` pulls prior outputs
back and skips already-complete stages. A portable
`output/manifests/run_state_blender_gpu.json` records every stage's status.
`--strict-render` fails fast on a blank/over-exposed frame.

> **Note:** a previous version of the GPU renderer produced "nothing but
> light" frames because the trimesh GLB (authored Z-up) was rotated out of the
> camera frame by Blender's Y-up glTF import. The renderer now re-orients the
> mesh into the authored frame (`synthetic_floor/geometry_align.py`) and uses
> neutral exposure, so the floor, columns, walls and windows render correctly.

See [`colab/README.md`](colab/README.md) for full docs. The CPU
pipeline above is unaffected — Blender is **not** a dependency of the
default example.

## How to Use With the Main Pipeline

The generated outputs plug directly into the MyCon pipeline:

1. **Stage 1 input**: use `output/video/stage_NN.mp4` as `inputs.video`
2. **Stage 9 output**: use `output/manifests/stage_NN_element_metrics.csv`
3. **Stage 11 input**: use `output/manifests/schedule.csv` and
   `output/manifests/bim_schedule_mapping.csv`
4. **BIM reference**: use `output/bim/stage_NN.ifc`
5. **Camera ground truth**: use `output/camera/stage_NN_camera_path.json`

## How to Customize

### Change the layout
Edit `config/scene.yaml` → `floor.rooms`, `floor.doors`, `floor.windows`.

### Change textures / materials
Edit `src/synthetic_floor/materials.py`. Each material is a pure-Python
procedural texture generator. Replace with your own PNG textures by
loading images in `build_material_library()`.

### Change the camera path
Edit `src/synthetic_floor/camera_path.py` → `_stage_polyline()`. The
walk-through is defined as a simple polyline of 3D waypoints.

### Change smartphone realism
Edit `config/scene.yaml` → `camera.hand_jitter`, `camera.noise`,
`camera.exposure`, `camera.motion_blur`, `camera.rolling_shutter`.

## Architecture

```
examples/synthetic_floor_7stage/
├── config/
│   └── scene.yaml          # Master configuration (geometry, stages, camera, renderer)
├── scripts/
│   └── run_generate.py     # CLI entry point
├── src/synthetic_floor/
│   ├── __init__.py
│   ├── scene_spec.py       # YAML loader + typed dataclasses
│   ├── layout.py           # Deterministic geometry builder
│   ├── stage_controller.py # Per-stage element selection
│   ├── materials.py        # Procedural PBR texture library
│   ├── ifc_builder.py      # IFC4 export (IfcOpenShell)
│   ├── mesh_builder.py     # OBJ/GLB/PLY export (trimesh)
│   ├── camera_path.py      # Handheld walk-through planner
│   ├── renderer.py         # Pure-NumPy ray-cast renderer
│   ├── smartphone_sim.py   # Post-processing (noise, blur, exposure)
│   ├── video_exporter.py   # MP4 encoding (imageio + ffmpeg)
│   ├── metadata_exporter.py # JSON/CSV metadata + manifest
│   └── validate.py         # Sanity checks
├── output/                 # Generated outputs (gitignored)
└── README.md               # This file
```

## Dependencies

Only standard scientific Python + a few lightweight packages:

* `numpy` — array math + renderer
* `Pillow` — texture generation + image I/O
* `PyYAML` — config loading
* `trimesh` — mesh export (OBJ, GLB, PLY)
* `imageio` + `imageio-ffmpeg` — video encoding
* `ifcopenshell` — IFC4 BIM export
* `scipy` — optional (used by some texture generators)

**No GPU, no Blender, no OpenCV, no Open3D required.**

## Design Decisions

1. **Pure-Python software renderer** — works everywhere, no GPU needed.
   Trade-off: slower than Blender Cycles, but fully deterministic and
   zero external dependencies.

2. **Procedural textures** — no asset downloads needed. Every texture is
   generated from the random seed. Trade-off: less photorealistic than
   scanned PBR textures, but perfectly reproducible.

3. **Axis-aligned boxes** — simple geometry that still exercises the full
   pipeline. Trade-off: less architectural detail than real BIM, but
   fast to render and easy to validate.

4. **Fixed camera path** — same walk-through for every stage so
   frame-to-frame comparisons are meaningful.

## Limitations & Future Work

* The renderer uses a single-bounce Lambert model (no global
  illumination). For photorealism, swap the backend to Blender Cycles.
* Textures are procedural — replace with real PBR texture maps for
  higher fidelity.
* Geometry is axis-aligned boxes. A future version could use proper
  extruded profiles or import a real IFC model.
* The camera path is the same for all stages. A future version could
  vary it slightly per stage.
