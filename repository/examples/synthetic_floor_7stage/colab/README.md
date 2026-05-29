# Colab GPU Rendering — Synthetic Floor 7-Stage

One-click notebook for rendering realistic construction progress videos
using Blender Cycles + GPU (T4/A100) on Google Colab.

## Quick Start

1. Open the notebook in Colab:
   [`synthetic_floor_blender_gpu.ipynb`](synthetic_floor_blender_gpu.ipynb)

2. Set runtime to **GPU** (T4 or A100):
   *Runtime → Change runtime type → T4 GPU*

3. Run all cells.

## What the Notebook Does

| Cell | Action |
|------|--------|
| 1 | Verify GPU, **mount Google Drive**, clone repo, set a stable `RUN_NAME` |
| 2 | Install Blender 4.2 LTS |
| 3 | Install Python deps |
| 4 | Smoke render (stage 7, debug) **synced to Drive + resumable + blank-frame guard** |
| 5 | Preview a frame + brightness/coverage check |
| 6 | Full 7-stage render (balanced), resumable |
| 7 | Resume status from the portable `run_state_blender_gpu.json` |
| 8 | Inspect manifest + camera path |
| 9 | Download MP4 videos |
| 10 | Resilience / resume notes |

## Example buildings

Set `CONFIG` in cell 1 to pick which building to render:

| `CONFIG` | Building | Final floor |
|----------|----------|-------------|
| `config/scene.yaml` | original single room | raw slab |
| `config/scene_office.yaml` | office room | tile |
| `config/scene_loft.yaml` | loft studio | wood |
| `config/scene_warehouse.yaml` | warehouse bay | epoxy |

The three detailed examples add pad foundations, concrete beams, an exterior
site, framed windows and a distinct final floor, and follow a realistic
sequence where **doors and windows are installed only at stage 6** (after the
walls and rough plaster). Each stage also exports a valid **IFC4** file for the
main pipeline.

`--save-blend` writes a downloadable, self-contained `.blend` per stage (open
it in Blender on Windows/macOS); cell 10 lists the `.blend` zips.

## Google Drive persistence & resume

Pass `--mount-drive --drive-root <folder>` to `run_blender_gpu.py` (the notebook
does this for you). Then:

- Every stage's outputs are mirrored to
  `MyDrive/MyCon_Colab/synthetic_floor_7stage/<RUN_NAME>/output/` immediately
  after the stage finishes, and a background daemon flushes partial progress
  every ~2 minutes. A Colab crash costs at most the current stage.
- `--resume` pulls prior outputs back from Drive and skips stages that already
  completed (verified against their artefacts + `.done` markers).
- A single portable `run_state_blender_gpu.json` summarises every stage's
  status. Copy/share the run folder to another machine or Drive account, set
  the same `RUN_NAME`, and the run continues from where it stopped.
- A stale Drive FUSE mount (after a reconnect) is auto-detected and remounted.

```bash
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/scripts/run_blender_gpu.py \
        --blender /content/blender/blender --preset balanced \
        --mount-drive --drive-root /content/drive/MyDrive/MyCon_Colab/synthetic_floor_7stage/demo \
        --resume
```

## Quality Presets

| Preset | Resolution | Samples | Frames | Motion Blur | Use Case |
|--------|-----------|---------|--------|-------------|----------|
| `debug` | 480×270 | 32 | 30 | No | Smoke test |
| `balanced` | 960×540 | 96 | 120 | Yes | Development |
| `hq` | 1280×720 | 192 | 180 | Yes | Final figures |

## Custom Rendering

```bash
# 30-second video at balanced quality
--preset balanced --stages 7 --frames 900

# 60-second video at full HD (for paper/thesis)
--preset hq --stages 7 --frames 1800 --resolution 1920 1080 --samples 256

# Just stages 1 and 7 for comparison
--preset balanced --stages 1 7 --frames 900

# Resume after a crash (skip already-complete stages)
--preset balanced --resume
```

## Why earlier renders were "nothing but light" (and the fix)

Two compounding bugs made frames come out as a bright, empty scene with no
floor/columns/walls visible:

1. **Coordinate-frame mismatch (the main cause).** The geometry is authored
   **Z-up** and exported as GLB by trimesh, which writes the vertices
   verbatim. Blender's glTF importer assumes glTF's **Y-up** convention and
   rotates the mesh +90 deg about X on import. The room ended up rotated so its
   *height* landed on Blender's Y axis — far outside the hard-coded camera path,
   window light portals and ceiling lights. The camera saw almost only the sky.
2. **Over-exposure.** A `+1.2 EV` "interior boost" on top of a boosted sky blew
   out whatever little was visible to pure white.

The fix:

- `blender_gpu_renderer.align_to_author_frame()` re-orients the imported
  geometry back into the authored Z-up frame (computed from the elements
  sidecar bounding box; see `synthetic_floor/geometry_align.py`), so the
  camera, portals and lights line up with real geometry again.
- Exposure is back to neutral (`exposure=0.0`), sky strength `1.0`, sun `3.0`.
- A **blank-frame guard** (`--strict-render`) fails fast if a frame is almost
  entirely near-white or has no spatial structure.

The earlier interior-lighting work is retained: window openings are real gaps,
light portals guide Cycles, stage 7 has 6 ceiling area lights, and bounce
counts are high for proper indirect illumination.

## Output Structure

```
output/
├── blender_renders/
│   ├── stage_01/
│   │   ├── rgb/frame_0001.png ... frame_NNNN.png
│   │   ├── depth/frame_0001.exr ...
│   │   ├── seg/frame_0001.png ...
│   │   ├── camera_path.json
│   │   ├── blender_render.log
│   │   └── .done
│   ├── stage_02/ ...
│   └── stage_07/ ...
├── video/
│   ├── stage_01_blender.mp4
│   └── stage_07_blender.mp4
├── bim/
│   ├── stage_01.ifc ... stage_07.ifc
├── mesh/
│   ├── stage_01.glb ... stage_07.glb
│   ├── stage_01_elements.json ...
├── manifests/
│   ├── manifest_blender_gpu.json
│   ├── schedule.csv
│   └── bim_schedule_mapping.csv
└── 7_stages_overview.png
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No GPU" error | Runtime → Change runtime type → T4/A100 |
| Blender download fails | Re-run cell 2; Colab sometimes throttles wget |
| Bright/blank frames ("nothing but light") | Fixed: geometry is re-oriented to the authored Z-up frame and exposure is neutral. If you still see it, check `blender_render.log` for the `alignment:` line and run with `--strict-render` to fail fast. |
| Black/dark frames | Check `blender_render.log` — if `rgb=0` frames, the compositor failed |
| CUDA out of memory | Reduce `--resolution` or `--samples` |
| Very noisy output | Increase `--samples` to 192 or 256 |
| Lost work after a disconnect | Re-run with the same `RUN_NAME`; `--resume` restores from Drive |
