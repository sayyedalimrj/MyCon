# MyCON — Civil Geometry round 3: actual COLMAP pose vs depth-cloud verification
Date: 2026-11-14 | Owner: A | PR #42 | Stage 7 (read-only) | Status: **UNREGISTERED**

## Engineering acceptance and concrete inputs
These are actual project-derived captures, but *not* a field survey or a verified Depth Fusion ↔ COLMAP calibration. The source `depth_fused_candidate_nonmetric.ply` is immutable (104,161 XYZ; SHA256 `c1f706ec951a1120eb748122e54dd75cee5b3fac80135efcb5cdee844e1ef3ee`; the old text evidence mistakenly drops the initial `c`; use the full digest). It has an explicit depth receipt with `colmap_nonmetric=true`, `false_positive_geometry_possible=true`, and `narrow_baseline_and_unobserved_views_certified=false`. The 15 planes, 49 edges and 7 extrapolated corners are *candidate* geometry; **zero human/image-verified corners or walls**. Source outputs and checkpoints remain read-only.

The original COLMAP files on Google Drive have:
- `cameras.txt` SHA256 `4630c9deee24da80337e0d8c70d46077c395c2c0ae24e693343839b06e4e1b14` — SIMPLE_RADIAL 640×1138, f=843.1592474, cx=320, cy=569, k1=0.04312833;
- `images.txt` SHA256 `b5b6d85cc739ec29a853e9fa593dcc5b3ef2058e9c50a29636ac6593fa37cb94`, 192 COLMAP images with native **world-to-camera OpenCV** quaternions and translations;
- `points3D.txt` SHA256 `a735bdbcb15c58e5e9ccb0357f76ad7fbc969d6345e83084cd9fceba0330fbed`, 51,894 sparse tracks;
- `colmap_sparse_unscaled.ply` SHA256 `4b9a20b5642c1c3da8a9ed96e0585778f097358e87dd7561e4ef74db9b0df5c7`, 51,894 points, with **maximum nearest-point discrepancy only 9.57e-7 arbitrary units** against COLMAP points3D. This establishes sparse PLY↔COLMAP 3D track correspondence, not Depth Fusion correspondence.

Actual source photos (five in the original Drive `images` folder) match COLMAP registered filenames, decode at 640×1138, and have individually recorded SHA256 receipts. Reprojection of their own sparse 3D tracks uses **SIMPLE_RADIAL distortion k1**, not an uncorrected pinhole camera. 3D→2D internal model reprojection:

| Original photo | Matched 2D–3D tracks | Median px | p95 px |
|---|---:|---:|---:|
| frame_000096.jpg | 871 | 0.674 | 1.878 |
| frame_000120.jpg | 1719 | 0.780 | 2.137 |
| frame_000144.jpg | 957 | 0.762 | 2.445 |
| frame_000168.jpg | 2258 | 0.745 | 2.029 |
| frame_000192.jpg | 1033 | 0.713 | 1.988 |

These results establish **COLMAP camera/sparse model internal consistency only**. They do not establish camera ↔ source Depth Fusion correspondence. These source images lack independently verified *original capture-time* image SHA receipts, and the Depth Fusion receipt enumerates accepted view *pairs* 0–7 without an explicit verified mapping from depth pixel/frame IDs to the above COLMAP `frame_000*.jpg`, calibrated pose and image hash. Distorted camera intrinsic model must be preserved. Metric scale and gravity axis are unknown.

## Deliberately rejected registration claim
Use read-only KD-tree proximity from 30,000 randomly selected Depth Fusion points to the 51,894 COLMAP sparse points (all in arbitrary source units):

| Hypothesis | Median nearest sparse point | P95 | Fraction within 0.1 |
|---|---:|---:|---:|
| As-is identity | 0.3573 | 1.0915 | 0.1180 |
| Add +1 to X (negative control) | 0.2622 | 1.1380 | 0.1289 |
| Reflect X (negative control) | 0.7147 | 1.7966 | 0.0483 |

These are *unmatched* nearest-neighbor distances and can be influenced by sampling density, surface coverage and occlusion. The translated control even reduces the median distance. Therefore **neither identity nor any other transform can be certified**, and no metric precision or accuracy is inferred. The frame string `colmap_world_unverified` on the Depth Fusion Stage-7 manifests is *not* an alignment certificate. The new camera bundle reports independent camera frame `COLMAP_sfm_world_unscaled`; its registration state is `UNREGISTERED`. No depth feature is projected over the real photos. Current photographed COLMAP sparse dots are independent camera-track QA markers, not proof that one of the 15 detected depth planes is that photographed wall.

## Outputs and interface
Run ID `geom-civil-camera-20261114-real-depth-002` creates **only** a new isolated path `runs/<run_id>/geometry/`, never updates the source artifacts. `mycon.civil_camera_geometry_bridge.v1` registration_qa.json: source hashes, camera poses in w2c **and** inverse c2w, radial intrinsics, image SHA, per-view pixel reprojection statistics, sparse-v-depth proximity/control experiments, missing registration gates. `mycon.civil_geometry_viewer_handoff.v1` viewer_handoff.json: review photos, sparse model, old hashed finite geometry overlay pointers, unknown regions, strict `feature_photo_projection_allowed=false`, `metric_scale_verified=false`, registration `UNREGISTERED`; cross-workstream Viewer may display separate layers only.

`feature_evidence.json`: all 15+49+7 feature IDs, original geometry JSON hashes, finite support point indices & XYZ samples for planes/edges, local geometric residual/occupancy, `unknown` semantics, image candidate files and SHA **with image support 0** and manual decision pending. No fabricated support points are assigned to extrapolated corners. `unknown_regions.json` plus `unknown_point_indices.npy`: 57,850 unclassified original observations — NOT a single connected missing wall or a filled floor. `feature_review.csv`: engineer can record acceptance/rejection/unknown, but generated defaults remain pending.

`round3_shape3_photo_review.html`: previous interactive/clickable Shape3 in an iframe, plus original photographs in their own separate panel. Three `real_photo_shape3_review_0*.png` figures show real source photographs with *their own COLMAP sparse tracks* beside a separate Depth Fusion cloud view; the panes are conspicuously **UNREGISTERED**, and are **not** camera-coincident projections of extracted surfaces. Do not mistake them for spatially aligned multiview Shape3.

## Reproduce / test
From `repository/` on a machine with `numpy`, `scipy`, `trimesh`, `Pillow`, `matplotlib`, `pytest`:

```bash
PYTHONPATH=. pytest -q tests/test_research_geometry_structural_review.py tests/test_research_geometry_camera_bridge.py
PYTHONPATH=. python -m pipeline.stage_07_cleanup.camera_geometry_bridge \
 --project-root /isolated_project \
 --civil-manifest /isolated_project/runs/geom-civil-real-depth-20261113-final1/geometry/civil_geometry_manifest.json \
 --sparse-ply /isolated_project/inputs/colmap_sparse_unscaled.ply \
 --cameras-txt /isolated_project/inputs/colmap_text/cameras.txt \
 --images-txt /isolated_project/inputs/colmap_text/images.txt \
 --points3d-txt /isolated_project/inputs/colmap_text/points3D.txt \
 --photo /isolated_project/inputs/captured_images/frame_000096.jpg \
 --photo /isolated_project/inputs/captured_images/frame_000120.jpg \
 --photo /isolated_project/inputs/captured_images/frame_000144.jpg \
 --photo /isolated_project/inputs/captured_images/frame_000168.jpg \
 --photo /isolated_project/inputs/captured_images/frame_000192.jpg \
 --run-id geom-civil-camera-NEW_UNIQUE_ID
```

The algorithm rejects SHA tampering, malformed/non-unit COLMAP quaternion, wrong image dimensions/model, missing image file naming/registered poses, mismatched sparse PLY vs points3D, and duplicate RunID before writing. On local isolated CPU: **22 tests passed in 5.64 seconds, exit 0**, 14 preserved round-2 structural plus 8 new bridge tests; `python -m ... camera_geometry_bridge` real-source run exit 0. That test is *not* a GPU test, WebGL/browser interaction validation, field QA, or exact-commit GitHub CI result. See `round3_execution_evidence_20261114.json` for machine-readable details and SHA receipts.

## Additional scientific/code research (carry-forward)
Retain earlier 5 DOI primary methods (Schnabel RANSAC, Huang 2DGS, ScanNet++, screened Poisson, 3DGS-to-PC), pinned Open3D MIT and 2DGS research-only repositories, and official 2DGS notebook (not executed). Relevant additional papers:
- Schönberger & Frahm (2016), *Structure-from-Motion Revisited*, CVPR, https://doi.org/10.1109/CVPR.2016.445 : incremental SfM, pose/image registration; low reprojection is necessary for *internal model* reliability, not site measurement. Verifiable publisher/DOI.
- Schönberger, Zheng, Frahm & Pollefeys (2016), *Pixelwise View Selection for Unstructured Multi-View Stereo*, ECCV, https://doi.org/10.1007/978-3-319-46487-9_31 : multi-view depth fusion requires actual visibility/photometric consistency; cannot use 8 accepted pairs to infer photo-depth ID mapping.
- COLMAP official GitHub https://github.com/colmap/colmap at release tag **4.2.1 / commit `bd1fcf654d2dd8fefa1466999c190a246f83f4b9`**, root `LICENSE` **new BSD / 3-clause**, dependency licenses separately. This audit parses COLMAP text and mathematically inverts w2c into c2w; it does NOT assert that release v4.2.1 produced the historical MyCON capture (producer version unrecorded), did not run COLMAP reconstruction, and does not copy their source. Official executable reference https://colmap.github.io/format.html ; local executable adapter actually ran (command above). The 2DGS Colab remains not_executed because no CUDA training environment/permission.

## Pending G2 / G3 / G4
Need the exact Depth Fusion producer log listing *each* source image/depth pixel, camera/pose/intrinsics and hashes; an independently held-out set of cross-modal 2D↔3D correspondences with residuals and occlusion masks; actual field gravity/survey controls/scale; human-approved physical wall/floor/ceiling and corner ID+photo pairs. Never use unverified Gaussian centers, fill occluded openings or label unsupported physical construction quantities.
