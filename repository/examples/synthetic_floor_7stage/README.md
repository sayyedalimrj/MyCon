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

### Detailed example buildings (office / loft / warehouse)

Besides the original single room (`config/scene.yaml`), three richer,
**construction-logical** example buildings ship in `config/`. Select one with
`--config`:

| Config | Building | Final floor | Extra detail |
|--------|----------|-------------|--------------|
| `scene_office.yaml` | office room | **ceramic tile** | pad footings, concrete beams, exterior site, framed windows, 4 windows + 1 door |
| `scene_loft.yaml` | loft studio (3.6 m, denser grid) | **wood plank** | 7 large windows, exposed beams |
| `scene_warehouse.yaml` | warehouse bay (4.2 m) | **polished epoxy** | heavy beams, wide roller door, high clerestory windows |

All three add geometry the original lacked: **exterior ground/site** (seen
through the openings), **pad foundations**, **concrete beams** spanning the
columns, **visible window frames** (so finished windows read clearly, not as
empty holes), and a **distinct final floor finish** laid only in the last
stage.

Their seven stages follow a realistic sequence — and **doors and windows are
deliberately *not* present in the early stages**:

1. Earthwork & foundation (site + pad footings)
2. Ground slab + concrete columns
3. Concrete beams + roof/ceiling slab (frame topped out)
4. Masonry walls with rough openings (**no doors/windows yet**)
5. Services (overhead pipes) + rough plaster
6. **Doors + glazed windows + frames installed**, walls painted
7. Final floor finish (tile/wood/epoxy) + suspended lit ceiling

```bash
# Render the office example, save a downloadable .blend per stage, sync to Drive:
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/scripts/run_blender_gpu.py \
        --config examples/synthetic_floor_7stage/config/scene_office.yaml \
        --blender /content/blender/blender --preset balanced \
        --save-blend --mount-drive --resume
```

### Hyper-realistic camera operator (shared CPU + GPU)

`src/synthetic_floor/camera_path.py` is the single source of truth for camera
motion. It models a hand-held human operator rather than a dolly on rails:

- **Look-at is fully decoupled from translation.** The body walks a coverage
  path while the gaze independently scans left/right, periodically turns
  ~180 deg to inspect the area behind, and tilts up/down.
- **Full-coverage, collision-safe pathfinding.** A serpentine that covers the
  interior is smoothed (centripetal Catmull-Rom) and then strictly clamped
  inside the walls and pushed out of column footprints — no wall-clipping.
- **6-DOF verticality.** Scheduled "inspect floor" (crouch) and "inspect
  ceiling" (rise) maneuvers move the camera on Z with eased motion; the gaze
  pitches to match.
- **Physical inertia + micro-mechanics.** Gaze angles are low-pass filtered
  (a configurable time-constant) and breathing / footstep / high-frequency
  tremor ride on top.

The host generates this trajectory per stage (`camera_poses.json`) and the
Blender renderer keys the camera from it (`--camera-poses`), so the **exported
video matches the CPU dataset exactly**. Every parameter is read from
`camera.motion` in the YAML (see the table below) — nothing is hardcoded.

| `camera.motion` key | Meaning |
|---|---|
| `coverage_lane_spacing_m` | serpentine lane spacing (smaller = denser coverage) |
| `collision_margin_m` / `column_clearance_m` | keep-out distance from walls / columns |
| `path_smoothness` | 0 = polyline, 1 = very rounded |
| `scan_yaw_amplitude_deg` / `scan_period_sec` | horizontal look-around sweep |
| `turn_around_interval_sec` / `turn_around_duration_sec` | ~180 deg look-behind cadence |
| `gaze_inertia_tau_sec` | physical smoothing (inertia) of the gaze |
| `focus_distance_m` | look-at distance |
| `pitch_scan_amplitude_deg` / `pitch_scan_period_sec` | gentle up/down gaze drift |
| `vertical_inspect_enabled` | enable crouch/rise maneuvers |
| `crouch_height_m` / `rise_height_m` | floor-inspect / ceiling-inspect eye heights |
| `vertical_inspect_interval_sec` / `vertical_inspect_duration_sec` | crouch/rise cadence |
| `inspect_pitch_deg` | how far the gaze tilts during crouch/rise |

### Physically based materials

Raw concrete is now a physically plausible mid-dark grey (~0.35 albedo)
instead of the old near-white that clipped to white. `materials._build_concrete`
generates low-frequency tonal blotches plus high-frequency aggregate grain, and
the Blender presets (`FINISHING_PRESETS`) drive procedural **bump + roughness
variation** nodes so concrete/cinderblock/beams read as unpolished as-cast
surfaces. The over-bright paint/plaster were also tamed to avoid blow-out.

### Save the Blender project (.blend) for Windows/macOS

`--save-blend` makes the renderer write a self-contained `stage_NN.blend`
(procedural materials, packed data — **no external textures**) and the host
zips it to `<output>/blend/<project>_stage_NN.blend.zip`. Download it from the
notebook (cell 10) or from Drive, then open it in Blender 4.2+ on any OS and
press **F12** to re-render or inspect the scene.

### IFC for the main pipeline

Every stage of every example exports a valid **IFC4** file
(`<output>/bim/stage_NN.ifc`) with stable `GlobalId`s. The detailed examples
carry `IfcFooting`, `IfcColumn`, `IfcBeam`, `IfcWall`, `IfcWindow`, `IfcDoor`
and `IfcCovering` entities, so they feed the main MyCon BIM/registration
stages directly and can be diffed across stages by GUID for progress.

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
