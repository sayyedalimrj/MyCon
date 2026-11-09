# Stage 4.5 — LongSplat thesis-only visual + geometry research branch

## Intent and scope

This is an **optional, research-only** implementation within `pipeline.stage_04_5_cams_gs`. It does not replace MyCON's canonical Stage 3–5, does not alter Recorder raw inputs, and **does not feed Stage 8/9 automatically**. It is designed to generate:

1. A trained 3DGS and standard `.ply` for **free-view navigation** in a splat viewer.
2. The reference LongSplat novel-view RGB/depth videos (`nvs_rgb.mp4`, `nvs_depth.mp4`) and per-frame image/depth/camera evidence.
3. A separate, clearly labeled depth-backprojected geometry *candidate* and optional metric registration with independently measured site/BIM anchors.
4. A held-out source-view appearance report, without claiming unseen viewpoints are universally correct.

The geometry and appearance branches must never be conflated. LongSplat's attractive PLY is a **Gaussian radiance representation, not a metrically verified surface scan**.

## Research-only licence

LongSplat: https://github.com/NVlabs/LongSplat; license restricts use to noncommercial research/evaluation. **Do not vendor LongSplat source into this repository.** Download its submodules and weights independently under the upstream terms. This MyCON adapter adds no NVIDIA source and is not a commercially licensed substitute. Upstream code requires CUDA extensions and its documented Python environment.

## Reproducible preparation

Run from the `repository/` directory with `python -m` imports and an **upright full-length MP4** exported once from the Recorder ZIP. Do not double-rotate, change the video hash, use the 10-second sample, or overwrite a previous experiment. The new project goes under `--workspace`, not under canonical Stage 3/5 checkpoint folders.

```bash
# Separate checkout and environment per the original instructions:
git clone --recursive https://github.com/NVlabs/LongSplat.git /research/LongSplat
# Provision upstream conda/PyTorch/CUDA and pretrained dependencies using
# https://github.com/NVlabs/LongSplat#installation

# Show *actual* commands and input identity first; no execution:
python -m pipeline.stage_04_5_cams_gs.longsplat_research \
  --phase plan --video /data/MyCON_96s_UPRIGHT.mp4 \
  --longsplat-root /research/LongSplat --workspace /data/mycon_research_3dgs

# Process the complete capture using ~10 selected fps (not 4), without double-downscaling:
python -m pipeline.stage_04_5_cams_gs.longsplat_research \
  --phase prepare --video /data/MyCON_96s_UPRIGHT.mp4 \
  --longsplat-root /research/LongSplat --workspace /data/mycon_research_3dgs \
  --accept-research-license

# Train in the upstream LongSplat CUDA environment (separate from R4):
python -m pipeline.stage_04_5_cams_gs.longsplat_research \
  --phase train --video /data/MyCON_96s_UPRIGHT.mp4 \
  --longsplat-root /research/LongSplat --workspace /data/mycon_research_3dgs \
  --python /research/longsplat_env/bin/python --accept-research-license

# Required for view synthesis playback and raw geometry evidence:
python -m pipeline.stage_04_5_cams_gs.longsplat_research \
  --phase render --video /data/MyCON_96s_UPRIGHT.mp4 \
  --longsplat-root /research/LongSplat --workspace /data/mycon_research_3dgs \
  --python /research/longsplat_env/bin/python --accept-research-license

# Standard 3D Gaussian PLY for freely changing viewpoint:
python -m pipeline.stage_04_5_cams_gs.longsplat_research \
  --phase export --video /data/MyCON_96s_UPRIGHT.mp4 \
  --longsplat-root /research/LongSplat --workspace /data/mycon_research_3dgs \
  --python /research/longsplat_env/bin/python --accept-research-license

# Extract colored geometry candidate from rendered depth and fitted cameras:
python -m pipeline.stage_04_5_cams_gs.longsplat_research \
  --phase geometry --video /data/MyCON_96s_UPRIGHT.mp4 \
  --longsplat-root /research/LongSplat --workspace /data/mycon_research_3dgs \
  --python /research/longsplat_env/bin/python --accept-research-license

# Quantify appearance quality at held-out captured viewpoints:
python -m pipeline.stage_04_5_cams_gs.longsplat_research \
  --phase audit --video /data/MyCON_96s_UPRIGHT.mp4 \
  --longsplat-root /research/LongSplat --workspace /data/mycon_research_3dgs \
  --accept-research-license
```

**Parameters must remain identical across phases**; the adapter refuses mismatched parameter receipts. `--width 512` and `--fps 10` follow the upstream recommended starting point; increasing width to 768 or using different sampling is a **new experiment**. `-r 1` in LongSplat avoids dividing an already-resized 512-pixel input by four. Before interpreting results, inspect actual room coverage, unobserved viewpoints and recorded poses. LongSplat cannot guarantee accurate surfaces for pure rotation/low parallax.

### Outputs under one immutable experiment directory

| Artifact | Meaning | Stage 8/9 allowed? |
| --- | --- | --- |
| `longsplat_model/converted_3dgs/point_cloud.ply` | Interactive novel-view Gaussian splat | **No** |
| `longsplat_model/train/ours_<N>/videos/nvs_rgb.mp4` | Smoothed-path novel-view video | No |
| `longsplat_model/train/ours_<N>/videos/nvs_depth.mp4` | Pseudocolored depth visualization | No |
| `longsplat_model/train/ours_<N>/depths/*.npy` | Rendered depth in **model units** | No |
| `longsplat_model/cameras_all_train.json` | Fitted camera extrinsics/intrinsics | No |
| `novel_view_qa.json` | Held-out source-view PSNR report | No |
| `geometry_candidate/depth_fused_arbitrary_units.ply` | Backprojected depth candidate; NOT splat PLY | **Only after independent validation** |
| `metric_review/depth_candidate_registered_m.ply` | 3D candidate aligned to site controls, subject to QA | **Stage 8 review only** |

Interactive visual free view can be explored with a compatible PLY 3DGS viewer such as Brush or a Three.js Gaussian viewer. The sample path `nvs_rgb.mp4` follows the camera trajectory and is **not evidence that every viewpoint anywhere in the room looks good**. Occluded walls/ceilings and surfaces never recorded cannot be inferred reliably.

## Metric gate and BIM registration

The `geometry` phase backprojects depth using the **exact LongSplat JSON pose convention** (`world2camera.R = camera.R.T`, `world2camera.T = camera.T`), and emits a PLY with explicit `ARBITRARY_SCALE_NOT_SURVEYED` metadata. The depth array is assumed to represent camera-Z distance and that convention must be checked before measuring geometry.

To create the optional metric registration, gather **six independent corresponding 3D site/BIM points** (at least four for fit and two held-out) in a CSV with columns:

```csv
id,source_x,source_y,source_z,target_x_m,target_y_m,target_z_m,split
anchor_1,0.0,0.0,0.0,1.0,1.0,1.0,fit
...
```

The `source_*` coordinates must be picked in the geometry candidate frame; the `target_*_m` coordinates must be independently measured/defined in the desired BIM coordinate system, **not picked from the same renderer**. A minimum held-out validation is mandatory. After obtaining valid anchors, run:

```bash
python -m pipeline.stage_04_5_cams_gs.metric_anchor_gate \
  --cloud /data/mycon_research_3dgs/<experiment>/geometry_candidate/depth_fused_arbitrary_units.ply \
  --anchors-csv /data/independent_site_anchors.csv \
  --output /data/mycon_research_3dgs/<experiment>/metric_review \
  --max-holdout-m 0.05
```

Even a passed sparse-anchor check **does not** declare Stage 8 registered or Stage 9 progress usable; verify the distribution of surface errors, occlusion/visibility, normals, wall/floor plane residuals and BIM coverage in Stage 8. Without survey controls, no direct comparable dimensions may be claimed for the thesis.

## Experimental thesis protocol

Compare the earlier successful short sample with the full 96-second capture on identical held-out source views and the **small room**, including turns upwards/downwards and side-to-side. Preserve the original image set, camera trajectory, all model logs and `novel_view_qa.json`. Evaluate full scene and room-specific held-out PSNR/SSIM/LPIPS where available, but also separately inspect geometry/pose drift and true BIM registration residuals. Report failures honestly; no perfect reconstruction of an arbitrary video is guaranteed.

## Operational limitations

- The external NVIDIA trainer has **no validated mid-training restart** through this adapter. Partial GPU outputs are preserved and never falsely resumed or silently erased.
- Python 3.10/CUDA/MASt3R versions should follow upstream; the MyCON R4 Colab environment is separate.
- There was no end-to-end NVIDIA CUDA training on the worksite video in the authoring environment. Offline tests check wiring and fail-closed behavior, **not the resulting reconstruction quality**.
- Do not merge this optional branch into an active R4 notebook ZIP without first reviewing version and checkpoint contracts; current GitHub `MyCon` stage 4.5 is not the same version as the independently packaged R4 Colab runtime.

References: [LongSplat](https://github.com/NVlabs/LongSplat), [LongSplat project demos](https://linjohnss.github.io/longsplat/), [COLMAP reconstruction](https://colmap.github.io/faq.html), [Nerfstudio Gaussian vs geometry](https://docs.nerf.studio/nerfology/methods/splat).