# Reconstruction R — peer-reviewed evidence and pinned implementation review

Reviewed 2026-11-13 for MyCON's **multi-view construction-room reconstruction**, before independent site validation. No borrowed article metric is a MyCON measurement.

| Paper | DOI verified from publisher/bibliography | Civil relevance and caution |
|---|---|---|
| Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016 | https://doi.org/10.1109/CVPR.2016.445 | new camera intrinsics/poses and sparse points, **unscaled** until surveyed |
| Schönberger et al., *Pixelwise View Selection for Unstructured Multi-View Stereo*, ECCV 2016 | https://doi.org/10.1007/978-3-319-46487-9_31 | geometric depth/normal consistency, candidate wall/floor surfaces, low-texture/occlusion errors |
| Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG 2023 | https://doi.org/10.1145/3592433 | impressive NVS/appearance but Gaussian centers cannot certify physical corners |
| Huang et al., *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*, ACM SIGGRAPH 2024 | https://doi.org/10.1145/3641519.3657428 | oriented surfels, normal/depth regularization, mesh candidate; GPU/licensing constraints |

## Reviewed source pins and licence

- [COLMAP](https://github.com/colmap/colmap/tree/137f21680c79d3f2fffbb1d11deb367162aa6f83), commit \`137f21680c79d3f2fffbb1d11deb367162aa6f83\`; \`LICENSE\`: New BSD/3-clause, transitive components separately licensed. **Adopt** SfM/COLMAP new run, do not reuse historical poses.
- [gsplat](https://github.com/nerfstudio-project/gsplat/tree/512d366b67073d77ca099ede742683c165dfc23b), commit \`512d366b67073d77ca099ede742683c165dfc23b\`; Apache-2.0. Official \`examples/simple_trainer.py default\` expects COLMAP, PyTorch+CUDA. **Adapt** only with independent train-only imagery, explicit heavy training opt-in and frozen 6999 baseline; GPU peak VRAM not measured.
- [2DGS](https://github.com/hbb1/2d-gaussian-splatting/tree/f3e3b9fa67bbd1c75e05167ff37391d8dab2a678), commit \`f3e3b9fa67bbd1c75e05167ff37391d8dab2a678\`; Gaussian-Splatting research/noncommercial licence, NOT a permissive Apache/MIT licence. Python 3.8/PyTorch 2.0, custom CUDA rasterizer, Open3D. **Research-only alternative**; no restricted source copied into MyCON.

## Real executable notebook reviewed

Community [2D Gaussian Splatting Colab notebook](https://github.com/atakan-topaloglu/2d_gaussian_splatting_colab/blob/2308cca2a7b6fef2a6ae0c73a27e5d9adeef947e/2d_gaussian_splatting.ipynb), pinned commit \`2308cca2a7b6fef2a6ae0c73a27e5d9adeef947e\`. Notebook version 4; 14 cells (10 code, 4 markdown). Inspected commands: clone recursively, install CUDA extensions, mount Drive, point to existing COLMAP capture, \`python train.py -s $COLMAP_PATH\`, render. The notebook repository has no standalone licence, upstream 2DGS is restricted. **Smoke NOT_EXECUTED:** no NVIDIA CUDA, no local COLMAP, no required GPU package/Drive mount; not truthful to label reading cells a notebook run.

## Engineering decision and traceability

| Assertion | Scientific source | Code evidence / experiment | Interpretation |
|---|---|---|---|
| New frame requires its own camera pose | COLMAP SfM | selection-frame SHA+fresh COLMAP mapper/registrator proposed; offline synthetic tests | no historical pose reassignment |
| NVS PSNR alone is not a measured wall | 3DGS TOG | held-out SHA-paired raster scoring plan, not executed on real model | avoid false architectural geometry |
| Finite structural planes need multiview | COLMAP MVS and 2DGS | future reviewed dense COLMAP or 2DGS surface extraction | Gaussian centres are not measured as-built surfaces |
| Train/test leakage contaminates comparisons | independent grouped benchmark protocol | source/session SHA/split guards and isolated train-only GS staging | paper results require camera/site/GT holds |

**Search date/keywords:** 2026-11-13; IEEE/CVPR 2016 incremental SfM, ECCV MVS, ACM TOG 3DGS, ACM SIGGRAPH 2DGS, GitHub COLMAP gsplat 2DGS, Google Colab executable notebook, scene frame selection/blur/angle coverage. Rejected unsupported benchmark transfers from published papers and historical 6999 logs. CPU fixture success does not imply real-room accuracy.

**Tradeoffs:** COLMAP sparse+MVS needs robust parallax and often GPU dense step; gsplat CUDA improves appearance but is not a credible physical surface without GT; 2DGS improves geometric modeling but has noncommercial license and heavier build. No measured MyCON VRAM/runtime, all method comparisons remain open.
