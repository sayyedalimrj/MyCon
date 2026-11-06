# Official NVIDIA LongSplat engine for MyCON thesis

This branch now contains the MyCON research adapter **and a bootstrapper for the real upstream NVIDIA implementation**, not a reimplementation. The NVIDIA source itself is checked out separately from `https://github.com/NVlabs/LongSplat.git` at the verified upstream revision `19750775a9d19f30aa05a8333c4c6c231b2d5f4a` with recursive submodules (MASt3R, Gaussian rasterizer, simple-knn, fused-ssim).

## Step 1 — Fetch NVIDIA's actual repository

```bash
python repository/scripts/mycon_fetch_nvidia_longsplat.py --dest /content/NVIDIA_LongSplat
python repository/scripts/mycon_fetch_nvidia_longsplat.py --dest /content/NVIDIA_LongSplat --verify-only
```

The installer preserves failed partial clones and never overwrites an existing directory, a Recorder capture, or a MyCON checkpoint. It verifies the NVIDIA license and all mandatory submodules. The official source is **not copied/relicensed into MyCON**; it is downloaded as the original research distribution.

## Step 2 — Install the official CUDA environment SEPARATELY from MyCON R4

Follow upstream installation precisely: https://github.com/NVlabs/LongSplat#installation.
The authors specify Python 3.10.13, CMake 3.14.0, PyTorch with compatible CUDA (example 12.1), `pip install -r requirements.txt`, three compiled CUDA extensions and optional CroCo RoPE kernels. Do not assume this works in the existing Python 3.12 MyCON R4 Colab environment or on Blackwell G4 without a compatibility check.

```bash
cd /content/NVIDIA_LongSplat
conda create -n longsplat python=3.10.13 cmake=3.14.0 -y
conda activate longsplat
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim
```

Actual CUDA installation/training has **not** been run by this branch.

## Step 3 — Use the existing Stage 4.5 MyCON bridge

Use the **full, already-upright 96-second MP4**, not the zipped Recorder bundle, and keep all original Drive state. Work from the MyCON `repository/` folder:

```bash
cd /path/to/MyCon/repository
python -m pipeline.stage_04_5_cams_gs.longsplat_research --phase plan \
  --video /data/MyCON_96s_UPRIGHT.mp4 --longsplat-root /content/NVIDIA_LongSplat \
  --workspace /data/mycon_thesis_longsplat

# Use the SAME parameters for every phase and the separate LongSplat environment.
for phase in prepare train render export geometry audit; do
  python -m pipeline.stage_04_5_cams_gs.longsplat_research --phase "$phase" \
    --video /data/MyCON_96s_UPRIGHT.mp4 \
    --longsplat-root /content/NVIDIA_LongSplat \
    --workspace /data/mycon_thesis_longsplat \
    --python /path/to/longsplat/env/bin/python \
    --accept-research-license
done
```

LongSplat upstream `train.py` optimizes unposed video and the trained 3DGS, `render.py` produces novel RGB/depth views, and `convert_3dgs.py` produces a viewer-compatible Gaussian PLY. An optional depth+pose geometry candidate and **separate independently measured** metric-anchor gate serve the BIM research comparison.

**Important limits:** (1) 3DGS novel view is not automatically correct far outside the observed viewpoints. (2) A Gaussian PLY is a radiance field, not surveyed point-cloud geometry. (3) Do not use it as a direct Stage 8/9 BIM comparison source. (4) The short-room rotation/low-parallax issue remains an experimental limitation, not an automatic-success case. (5) The bridge doesn't claim reliable mid-training resume; preserve partial outputs and investigate before rerunning.

**License:** Original [NVIDIA noncommercial research/evaluation terms](https://github.com/NVlabs/LongSplat/blob/main/LICENSE.md), plus each submodule's own license. Suitable for noncommercial thesis research subject to their terms; no commercial use is granted.
