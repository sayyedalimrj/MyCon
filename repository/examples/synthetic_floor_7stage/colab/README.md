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

| Cell | Action | Time |
|------|--------|------|
| 1 | Verify GPU + clone repo | 10s |
| 2 | Install Blender 4.2 LTS | 30-60s |
| 3 | Install Python deps | 15s |
| 4 | Quick smoke test (stage 7, debug preset) | 30-60s |
| 5 | Preview frames + brightness check | instant |
| 6 | Full 7-stage render (balanced preset) | 15-25 min (T4) |
| 7 | High quality single stage (30s video) | 10-20 min (T4) |
| 8 | Side-by-side 7-stage comparison | instant |
| 9 | Download MP4 videos | instant |
| 10 | Inspect manifest + camera path | instant |
| 11 | Download all outputs as ZIP | 10-30s |

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

## Why the Renders Are Bright (Not Black)

The old version produced black frames because:
1. Windows were solid boxes (light couldn't pass through)
2. No Light Portals (Cycles couldn't find the sky)
3. No interior lights (enclosed room = zero illumination)
4. Low bounce counts (max=4, light died quickly)

The new version fixes all four:
- Wall openings are **real gaps** in the geometry
- **Light Portals** at every window (guides importance sampling)
- **6 Area Lights** (80W warm white) for the finished ceiling
- **12 bounces** (diffuse=8, glossy=4, transmission=12)
- **Exposure=1.2** + Filmic for natural brightness

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
| Black/dark frames | Check `blender_render.log` — if `rgb=0` frames, the compositor failed |
| CUDA out of memory | Reduce `--resolution` or `--samples` |
| Very noisy output | Increase `--samples` to 192 or 256 |
