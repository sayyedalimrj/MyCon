# Blender GPU Renderer (Google Colab)

This is the **GPU-accelerated twin** of the CPU software renderer in
`src/synthetic_floor/renderer.py`. Same scene, same 7 stages, same
metadata contract — but rendered with **Blender 4.2 LTS + Cycles +
OptiX/CUDA + OpenImageDenoise** on a free Colab T4 (or paid A100).

## Why a separate folder?

The CPU pipeline must stay completely portable (no Blender required).
The Blender path requires a 200 MB binary download and a CUDA-capable
GPU, so the entry points live here under `colab/` and the renderer
script itself sits in `src/synthetic_floor/blender_gpu_renderer.py`.

Both paths share the **same**:

* scene specification (`config/scene.yaml`)
* layout / IFC / mesh / element-metrics / schedule modules
* manifest schema

The Blender renderer just produces additional outputs in
`output/blender_renders/stage_NN/{rgb,depth,seg,camera_path.json}` and
matching MP4s in `output/video/stage_NN_blender.mp4`.

## Quickest start

1. Open `synthetic_floor_blender_gpu.ipynb` in Colab.
2. Set runtime → GPU (T4 or A100).
3. Run all cells.

The notebook clones the repo, installs Blender, runs a 30-frame smoke
test on stage 7, then offers a full 7-stage 1280×720 render.

## Manual usage

```bash
# 1. Install Blender 4.2 LTS portable (idempotent)
bash examples/synthetic_floor_7stage/scripts/setup_colab_blender.sh

# 2. Smoke test (480x270, 30 frames, 32 samples)
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/scripts/run_blender_gpu.py \
        --blender /content/blender/blender \
        --stages 7 --quick

# 3. Full 7-stage render at 1280x720, 128 samples
PYTHONPATH=examples/synthetic_floor_7stage/src \
    python3 examples/synthetic_floor_7stage/scripts/run_blender_gpu.py \
        --blender /content/blender/blender \
        --resolution 1280 720 --samples 128 --device OPTIX
```

## What the GPU pipeline does

`run_blender_gpu.py` is the host-side orchestrator. For each requested
stage it:

1. Calls the existing CPU pipeline modules to write the per-stage
   `stage_NN.glb` mesh and a sidecar `stage_NN_elements.json` mapping
   each element_id to its category and finishing.
2. Spawns Blender as a subprocess:
   `blender -b --python blender_gpu_renderer.py -- <args>`.
3. Waits for Blender to write `rgb/`, `depth/`, `seg/` plus
   `camera_path.json`.
4. Encodes `rgb/frame_*.png` into a single H.264 MP4 with imageio +
   ffmpeg (already a repo dependency).
5. Writes `manifest_blender_gpu.json` for the run.

## How realism is achieved without external assets

* **HDRI / environment** — Blender's built-in **Sky Texture (Nishita)**
  is hooked into the World shader. It's a procedural physical sky that
  takes one sun-elevation angle and produces a believable open-sky
  background. **For early stages (1–3) where there's no ceiling yet,
  the script also hides the ceiling object**, so you actually see the
  sky from inside the slab.
* **Sun light** — a real Sun lamp is added with a 0.5° angular size.
  This gives crisp directional shadows on top of the sky's ambient
  contribution.
* **PBR materials** — every category (`raw_concrete`, `brick`,
  `painted`, `tile`, `glass`, `raw_wood`, …) gets a Principled BSDF
  whose Base Color is a procedural mix of a flat tint and a noise
  texture, plus a Voronoi-driven Bump for surface micro-variation.
  The element's `finishing` field (e.g. `painted_wood`) modulates
  roughness and tints, so late stages naturally look glossier and
  cleaner. **No external texture maps are downloaded.**
* **Camera** — a Bezier curve walks through the entrance, corridor,
  peeks into two rooms, and returns. The camera follows the path with
  a `Follow Path` constraint and looks at a slowly-jittered empty via
  `Track To`. Two layers of `Noise` F-curve modifiers per Euler axis
  give natural human sway (low frequency) plus micro-tremor (high
  frequency); a third set on `location` gives sub-cm hand wobble.
* **Smartphone feel** — Cycles motion blur is enabled, exposure uses
  Filmic + Medium Contrast, OpenImageDenoise cleans the rendered
  image, and the resulting MP4 is encoded at the same 30 fps the CPU
  pipeline uses, so the two videos are interchangeable.

## Outputs (per stage)

```
output/
└── blender_renders/
    └── stage_07/
        ├── rgb/frame_0001.png … frame_NNNN.png
        ├── depth/frame_0001.exr … frame_NNNN.exr   # 32-bit float, metres
        ├── seg/frame_0001.png  … frame_NNNN.png    # uint16 IndexOB
        ├── seg_palette.png                         # quick legend
        ├── camera_path.json                        # synthetic_floor_camera_path.v1
        ├── blender_render.log                      # renderer log
        ├── blender_stdout.log
        └── blender_stderr.log
output/video/stage_07_blender.mp4                   # MP4 encoded by host
output/manifests/manifest_blender_gpu.json          # global manifest
```

## Performance notes

| Setup | Resolution | Samples | Approx. time / 90-frame stage |
|-------|-----------|---------|-------------------------------|
| Colab T4 | 480×270 | 32 | ~30 s |
| Colab T4 | 1280×720 | 128 | ~2–4 min |
| Colab A100 | 1280×720 | 128 | ~30–60 s |
| CPU only | 1280×720 | 128 | many minutes |

Numbers are approximate; first run pays a one-time Blender cold start.

## Troubleshooting

* **`No GPU; will run on CPU`** — your Colab runtime isn't a GPU
  runtime. Switch via *Runtime → Change runtime type → T4 GPU*.
* **`Could not find the 'blender' executable`** — run
  `setup_colab_blender.sh` first, or pass `--blender /path/to/blender`.
* **Black renders** — early stages without a ceiling intentionally
  show a dark interior at night-like sun elevations. Try
  `--sun-elevation 38 --sun-azimuth 135`.
* **Out of memory** — drop `--samples` to 64 or render fewer frames at
  a time (the script accepts a `--stages` subset).
