# MyCON Universal HQ Reconstruction

[Open the notebook in Colab](https://colab.research.google.com/github/sayyedalimrj/MyCon/blob/research/mycon-work-consolidation-20261127/repository/reconstruction/notebooks/MyCON_Universal_HQ_Reconstruction.ipynb)

Choose a Drive path or upload, choose a quality profile, and run all cells. The default is Auto HQ. A folder may contain a set of still images. Separate multiple files or folders with a vertical bar. The engine verifies Recorder archives containing `arcore_recording.mp4` and the Recorder integrity manifest. Video rotation follows the container. Stills use their EXIF orientation. If metadata supplies neither, the report records that the encoded axes were retained. Intrinsics come from validated Recorder metadata when compatible with image dimensions. Otherwise SfM estimates relative intrinsics and radial distortion. The geometric units remain arbitrary.

## Operating the engine

```bash
cd repository
python3.12 -m venv .venv-mycon
source .venv-mycon/bin/activate
pip install -r requirements-universal-hq.txt pycolmap==4.2.1
python -m reconstruction.universal.runner /data/capture.mp4 \
  --profile "Auto HQ" --out /scratch/mycon-run \
  --checkpoint-mirror /persistent-storage/mycon-run
```

Linux requires Git, FFmpeg and ffprobe. Use `pycolmap-cuda12==4.2.1` in place of the CPU wheel in a CUDA runtime. The notebook attempts the GPU wheel when a GPU is visible, then selects calibrated CPU stereo if the wheel or the dense method cannot run. T4, L4 and A100 execution remain unverified. The notebook pins the exact source commit and uses private Drive storage when enabled.

| Decision | Active behavior |
| --- | --- |
| Keyframes | Native timestamps, texture and sharpness, optical-flow overlap, image displacement, apparent rotation, change in illumination, and possible cuts determine local selection. Profiles set spacing bounds, SIFT search budgets and dense resolution. |
| Registration | CPU SIFT and geometric matching, overlapping temporal neighborhoods and appearance retrieval for nonlocal candidates. Geometrically supported nonlocal links enter global bundle adjustment. Targeted matching recovers missing views where supported. Each disconnected model stays separate and the report identifies missing frames. |
| Depth on CPU | COLMAP undistortion and global poses, calibrated stereo rectification with per-pair disparity ranges, left/right checks, and agreement between two or more neighboring depth maps. Dense block checkpoints and spatial partitions limit memory. Low-texture or inconsistent areas can remain open. |
| Depth on GPU | Optional COLMAP PatchMatch with geometric fusion. If it raises a recoverable error, reduce resolution and try again, then use CPU stereo if necessary. This route has no GPU field result for this release. |
| Learned models | The existing DA3/DA3-Streaming and MapAnything work stays in the repository. The universal path uses no pretrained depth or historical cache. ALIKED/LightGlue and a universal Gaussian Splatting workflow were not benchmarked here. |
| Export | `Sparse.ply`, `Dense.ply` when available, the selected `Full.ply`, spatially sampled `Lite.ply`, and a colored point-mode `MyCON.glb`. The viewer uses WebGL2, allows mouse orbit, camera-frame selection, point size, section slider, linked model review and individual local spatial tiles. Mesh, texture, Gaussian splats and semantic layers are separate products. |
| Camera priors | The Recorder audit checks lag, angular-rate agreement and tracking discontinuities. It does not apply ARCore translations, GPS, IMU rotations or Recorder depth to the global solution. |

This release uses a global image graph and overlapping temporal neighbors. It has no standalone Sim(3) pose graph or independent scale/rotation drift measurement for unseen scenes. The separate DA3 historical fusion now rejects a camera update that would consume the allowed surface-seam budget. Plain walls, reflections, thin edges, rapid rotation and moving objects can still defeat visual registration or dense support. Coverage and apparent object motion reported by intake are proxies.

## Which profile to choose

| Profile | Use |
| --- | --- |
| Auto HQ | Default general run, with resource-dependent dense backend. |
| Fast Preview | Lower feature and dense budgets for a quick visual check. |
| High Quality | Higher feature and stereo resolution. |
| Ultra Quality | Largest configured image and feature budgets. The notebook still checks available resources. |
| Long Video | Bounded temporal neighborhoods with motion-adaptive spacing and conservative resolution. |
| Low VRAM | Lower image sizes and matching budgets. |
| Scientific Benchmark | Fixed Auto HQ parameters for comparison. Establish the held-out views and reference geometry separately before using external metrics. |

## Resume and inspect

Restart with the same inputs and profile. Pass the same local `--out` directory or an empty workspace with the same persistent mirror. Each completed stage has an immutable receipt with sizes and SHA256 digests. Camera databases are sealed before publication and mapping uses a separate SQLite copy. Dense CPU cache entries are checked individually. An interrupted stage stays under a `.working` or `.interrupted` directory. A completeness label on a stage means it emitted its documented artifacts. The quality report carries the geometry disposition independently. When a digest differs, preserve the directory and restore the known-good checkpoint or choose a fresh output path. Extract all release files before opening `Viewer.html`, including its `frames/` and `tiles/` neighbors.

Read `capture_analysis.json`, `Quality_Report.json`, and the execution log before interpreting coverage. Image reprojection assesses consistency of registered images. It does not establish accuracy or survey scale. The Fountain sensor-reference comparison below reports geometry distances and coverage for one scene. PSNR, SSIM and LPIPS remain unevaluated. The construction footage has no independent field reference in this release.

## Method references

- [COLMAP documentation](https://colmap.github.io/)
- [PyCOLMAP documentation](https://colmap.github.io/pycolmap/)
- [OpenCV StereoSGBM and calibrated rectification](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [MapAnything](https://github.com/facebookresearch/map-anything)
- [Open3D Fountain RGBD data](https://www.open3d.org/docs/release/python_api/open3d.data.SampleFountainRGBDImages.html)
- [Video Depth Anything sample videos](https://github.com/DepthAnything/Video-Depth-Anything)

## Observed evidence

| Input | Observed result |
| --- | --- |
| Real 10.2-second construction reference | Fresh Fast Preview registration of 50 selected views and a new calibrated stereo candidate. Historical DA3 yields a cleaner door and wall around 4 s in the common-view comparison. |
| Original 95.93-second Recorder archive | Generic intake verified the archive, used native PTS and selected 849 frames. The regular-grid gyro comparison reached about 0.319 at a -50 ms candidate offset, with two possible tracking discontinuities. The phone pose was withheld. |
| Original 95.93-second Recorder, dense replay | Stereo ran on the prior, visually matched 383-camera COLMAP solution using source RGB. Of 383 attempted reference views, 182 had multi-neighbor depth support. The complete cloud rendered some late surfaces that the old long model missed. It still left holes and degraded the shared early doorway against the short reference. Treat this route as a candidate. |
| Historical DA3 fusion | Unrestricted optimization increased seam P95 from 0.01156 to 0.02205 in unscaled reference coordinates. Limiting the update kept seam P95 around 0.01129, but improvement in reserved camera error missed the joint gate. No replacement hybrid was emitted. |
| Independent Open3D Fountain | A video encoded from the 33 original ordered RGB photos ran from media input through dense PLY, GLB and offline viewer on CPU. The run used neither the supplied depth images nor camera trajectory. A fresh workspace restored all stage hashes and reproduced the same full PLY. Browser review exercised mouse orbit, an original camera frame and a spatial tile. Geometry coverage is incomplete. |
| Independent native Rollercoaster MP4 | A separate native video failed the visual structure gate. No model was exported. |

The same-camera comparison uses a checked similarity from the prior SfM world to the historical short reference, fitted on early matching cameras and checked on reserved early cameras. This supports same-view inspection. It does not establish global accuracy or field-scale drift.


### Fountain RGB-D reference evaluation

After the independent RGB-only reconstruction, compare it to the public sensor-derived Fountain mesh and trajectory. The similarity is fitted from reference cameras and verified on six reserved reference poses. The position P95 was 0.0151 in reference coordinates and rotation P95 was 1.23°. Depth and camera references were excluded from reconstruction. Thresholds use the supplied reference coordinate units.

| Geometry | Accuracy fraction at 0.05 | Completeness at 0.05 | F1 at 0.05 | Symmetric mean nearest-vertex distance |
| --- | ---: | ---: | ---: | ---: |
| Sparse SfM | 0.733 | 0.857 | 0.790 | 0.0707 |
| Dense CPU stereo | 0.995 | 0.339 | 0.505 | 0.0735 |

Dense stereo has more accurate points in supported areas and misses large parts of the reference. The F1 and distance tables come from the complete supplied mesh vertices. They are independent of the source RGB geometry and are limited to this one benchmark scene.

```bash
python -m reconstruction.universal.fountain_benchmark /scratch/mycon-run/export \
  /data/SampleFountainRGBDImages.zip /scratch/Fountain_RGBD_Benchmark.json
```
