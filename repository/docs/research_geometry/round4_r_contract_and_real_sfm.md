# MyCON A / Civil Geometry — Round 4 / Team R contract & same-SfM-camera geometry
Date: **2026-11-27** | PR: [#42](https://github.com/sayyedalimrj/MyCon/pull/42), branch `research/mycon-civil-geometry-20261113` -> Integration ONLY. **DRAFT, NO MERGE.**

## Engineering conclusion / scope separation

1. **New R real 35-second capture has no newly executed COLMAP registered model and no actual `camera_poses.json`**: R's own `round3_real_pilot_receipt.json` says COLMAP missing. Do NOT attach the historical R4 camera poses to that video. R's `camera_poses.v1` is an *interface contract* here, not a proven real-world common frame.
2. **Historical reference source:** original COLMAP `cameras.txt` + `images.txt` + `points3D.txt`, with 192 registered views and 51,894 sparse points, and five actual indexed photographs. They originate from the **same historical model**, so 3D sparse XYZ and the selected calibrated old cameras share its own arbitrary `colmap_sfm_world_nonmetric` coordinate frame. Internal 3D→2D reprojection on actual image tracks passed the diagnostic. This does **not** make source points a dense physical survey or verify global scale.
3. Original 104,161-point `depth_fused_candidate_nonmetric.ply` is derived from real captures but its direct mapping into the historical model is **UNREGISTERED**. Those points and the 15/49/7 previous hypotheses are retained in a **separate layer**; the new 10/13/0 counts are **not a revision of the same input cloud**. No invented ICP transform or merged coordinate frame.
4. **No field-confirmed walls, ceilings, floors, physical junctions or corners.** Orientation along gravity/height is unknown. All semantics/physical decisions remain `unknown`/`pending`. A convex boundary is a candidate finite support region, not complete built wall coverage; occlusions and missing structure remain unknown.

## Team R contract read from exact source

Inspected Team R branch at HEAD `639810de055bac7e5986c35731423fcee78c737e`:
- `reconstruction/workflow.py` `report_poses()` emits `mycon.reconstruction.camera_poses.v1`, `poses[]` with `view_id`, `image_relpath`, `image_sha256`, `capture_session_id`, `split` (train/test), `registered`, `camera`, `coordinate_frame='colmap_opencv_world_to_camera'`, `units='reconstruction_units_unscaled'`, `world_to_camera_4x4` and two source model SHA receipts.
- `mycon.reconstruction.dataset.v1` ties every image to hashed bytes, unique run/session, genuine train/test split. `mycon.reconstruction.registration_quality.v1` may show `registered_train/test` but its `registration_verified=false` is intentional pending independent survey.
- R pose `T_world_to_camera` must be explicitly inverted for Viewer C2W; it must **never** be renamed as C2W. R's old-camera reference is historical only; it belongs to a different captured video than fresh R pilot.
- A's optional R manifest gate checks matching schema, run, every R frame hash/session/split, local image SHA, proper rigid W2C matrix and registered COLMAP model `images.txt` SHA. With no R inputs, output is `R_OUTPUT_NOT_SUPPLIED`. Different model SHA returns `UNREGISTERED_DIFFERENT_R_COLMAP_MODEL`. Even an identical model SHA produces `SAME_COLMAP_IMAGES_HASH_PENDING_PER_VIEW_POSE_VERIFICATION`: **do not promote cross-team alignment merely by model hash**. This A module does not write R/V/Integration paths.

## Real executed pilot (old sparse observations, not synthetic)

**Source hashes:** `cameras.txt` `4630c9deee24da80337e0d8c70d46077c395c2c0ae24e693343839b06e4e1b14`; `images.txt` `b5b6d85cc739ec29a853e9fa593dcc5b3ef2058e9c50a29636ac6593fa37cb94`; `points3D.txt` `a735bdbcb15c58e5e9ccb0357f76ad7fbc969d6345e83084cd9fceba0330fbed`. Source originals were copied read-only to a separate input workspace; no `MyCon_R4` writes.

**Real extraction command** (run from `repository/`, paths are the separate isolated, read-only copied input path for this executed run; no root project modifications):

```bash
PYTHONPATH=. python -m pipeline.stage_07_cleanup.r_frame_geometry \
 --project-root /mnt/data/mycon_r4/sample_real \
 --cameras-txt /mnt/data/mycon_r4/sample_real/inputs/cameras.txt \
 --images-txt /mnt/data/mycon_r4/sample_real/inputs/images.txt \
 --points3d-txt /mnt/data/mycon_r4/sample_real/inputs/points3D.txt \
 --photo /mnt/data/mycon_r4/sample_real/inputs/frame_000096.jpg \
 --photo /mnt/data/mycon_r4/sample_real/inputs/frame_000120.jpg \
 --photo /mnt/data/mycon_r4/sample_real/inputs/frame_000144.jpg \
 --photo /mnt/data/mycon_r4/sample_real/inputs/frame_000168.jpg \
 --photo /mnt/data/mycon_r4/sample_real/inputs/frame_000192.jpg \
 --run-id geom-civil-r-frame-20261127-real-historical-cli02 \
 --threshold 0.07 --max-planes 9 --trials 90
```

**Actually executed exit 0** (Linux Python 3.13.5) — 51,894 original sparse points; stringent quality gate `COLMAP per-point error <= 2px` and `track length >= 4` retains **34,039**, rejecting **17,855**. 10 finite RANSAC candidates (one initial plane can split into multiple observed islands), 13 locally supported candidate lines, **zero trihedral candidates and zero physically confirmed corners**. 11,996 quality-gated points assigned to candidates; 22,043 remain explicitly unknown/unassigned. Five original photos were linked by their *actual historical SfM 2D observations and photo hashes*. The plotted 2D pixels are those **tracked source points**, not inferred walls. Some planes have tracks in only **one or two** sampled views. No independently checked physical edge, floor, ceiling, gravity, structural identity or metric size.

**Per-feature:** `planes.json`, `edges.json`, `corners.json` preserve finite geometry, normals, residuals, point support indices and original `POINT3D_ID` samples, source-photo view IDs/SHA and actual tracked pixels; calibrated confidence=null, manual review=pending. Source RGB reference PLY preserved separately from the labelled `planes_colored.ply`; no edits to historical COLMAP.

**Viewer:** `viewer_overlay.json`, self-contained clickable/rotatable `shape3_sfm_review.html`, and three `real_sfm_shape3_view_*.png` with original photographs + historical sparse tracked inlier marks. Labels in visual outputs state **INTERNAL SfM, NOT FIELD CONFIRMED**. Viewer can inspect and click candidate features but cannot depict absent surfaces as completed built walls. Frame `colmap_sfm_world_nonmetric`; `metric_scale_verified=false`; `r_run_shared_frame_verified=false`. Owner V must not blend separate Depth Fusion with the old model until an independently validated registration receipt exists.

## Actual real-source re-detection sensitivity (synthetic perturbations applied to real data)

Executed `python -m pipeline.stage_07_cleanup.r_frame_stability --points3d-txt /mnt/data/points3D.txt --output /mnt/data/mycon_r4/sample_real/repeatability_20261127.json --trials 90` with exit 0. The baseline 10 patches were **re-extracted** after modification, then matched by unsigned normal angle <=12° and >=20% recovered *visible* original support points. This is a conservative, heuristic repeatability diagnostic: not an independent ground-truth, not real reacquisition, not accuracy/precision/recall. Full per-patch matches are committed in `round4_repeatability_20261127.json`.

| Perturbation applied to real input | Matched old candidates / 10 | New candidate count |
|---|---:|---:|
| Half-density randomly retained | 5 | 8 |
| Remove XY spatial quadrant (simulated occlusion) | 4 | 10 |
| Add Gaussian coordinate noise sigma=0.025 arbitrary units | 7 | 10 |
| Fit threshold 0.85× | 6 | 9 |
| Fit threshold 1.15× | 6 | 10 |

**Interpretation:** unstable under density/visibility changes; this is **not** an accepted room layout. No numbers in metres or percent structural completeness may be inferred. Under observed wall holes, sloped walls and scaffolding/objects, a theoretical plane/corner can be materially misleading.

## Tests, safety, and handoff

- Exact locally tested A-owned source blob SHA1 (git blob): `r_frame_geometry.py` `4930284f431d078fed85c89a0afd3599282d117f`, `r_frame_stability.py` `714cd9b2d9b77e92c7b8b2047691845c7dbb904e`, `test_research_geometry_r_frame.py` `154f836b1611868eff1783be7429a9aab2eb9b86`. They were verified after pushing to the same PR branch.
- Local command `PYTHONPATH=. pytest -q tests/test_research_geometry_r_frame.py tests/test_research_geometry_camera_bridge.py tests/test_research_geometry_structural_review.py` → **39 passed / 0 failed / exit 0**. Eight initial synthetic safety tests plus one synthetic repeatability test are new, the previous 22 tests retained; no synthetic fixture is presented as original room evidence.
- GitHub Actions exact final HEAD **must be checked separately**; local pytest doesn't automatically certify all CI, GPU, actual R new run, browser/WebGL, multi-view independent survey or approved engineering semantics.
- Unique new `geom-civil-r-frame-*` output path, no force/delete/retrain, relative SHA256 manifest, explicit source units/frame and unknowns. No changes outside A ownership; PR stays DRAFT and **no merge**.
- Current gaps: fresh R run `camera_poses`/new SfM dense source, matching pose+image+pixel receipt, real Depth Fusion-to-historical source transform, independent heldout cross-modal alignment, gravity, surveyed dimensions/accuracy, manual signed engineer decision, geometry F1 and true physical corner/civil element validation.

## Method references (unchanged prior research, used here)

The existing A [`evidence_review.md`](evidence_review.md) contains >=3 peer-reviewed DOI articles (RANSAC Schnabel 2007 DOI `10.1111/j.1467-8659.2007.01016.x`; ScanNet++ ICCV2023 DOI `10.1109/ICCV51070.2023.00008`; 2DGS SIGGRAPH2024 DOI `10.1145/3641519.3657428`; Screened Poisson DOI `10.1145/2487228.2487237`), pinned Open3D (MIT) and 2DGS (noncommercial research/evaluation) code revisions, executable demos/notebook and explicit non-execution records. This round **did not run 2DGS GPU/Colab or claim new literature benchmarking**. R's real capture/pose contract was read from its code/receipt, not assumed from the papers.

### Dedicated CI discovery on PR #42 (post-report code adjustment)
The frozen Integration workflow discovers only `tests/test_research_geometry_structural_review.py` for owner A. Without editing coordinator workflow paths, owner A added a named re-export of all **nine** Round-4 tests into that existing A-owned module. Actual local commands on byte-matched source: `pytest -q tests/test_research_geometry_structural_review.py` → **31 passed / exit 0**; the three-module explicit suite → **48 passed / exit 0**, including nine duplicate imports (39 unique cases). Last A source/test commit before this doc update: `91f18956ded402e8538c8e2a09d1b8c60bc81d9f`; updated legacy specialized test Git blob `21bb3d7d0bd85fcb11c0c48d3699d01c8af2d9d1`. GitHub Actions results must be checked on actual final PR HEAD, not assumed from local results.
