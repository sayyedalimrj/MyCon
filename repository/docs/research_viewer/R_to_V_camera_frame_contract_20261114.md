# R → V camera / A → V civil geometry — read-only contract (2026-11-14)

**Implementation:** `repository/viewer_civil/producer_bridge.py`. **Source:** R draft PR #49 `mycon.reconstruction.dataset.v1` + `mycon.reconstruction.camera_poses.v1`; optional A draft PR #42 `mycon.civil_geometry_review.v1`. **Consumer:** `mycon.viewer_bundle.v1` with seven mandatory roles, optional point_cloud. This does NOT modify R, A, D, stage 8/9, or Colab.

## Operational usage

Run from `repository/` using a **new** V output directory only:

```bash
python -m viewer_civil.producer_bridge \
  --r-run-dir /readonly/R/runs/recon-RUN-001 \
  --out-dir /new/V/runs/viewer-real-RUN-001
python -m viewer_civil render \
  /new/V/runs/viewer-real-RUN-001 \
  /new/V/runs/viewer-real-RUN-001-preview
python -m pytest -q tests/test_research_viewer_bundle.py tests/test_research_viewer_producer_bridge.py
```

**Preconditions**: R run contains immutable `dataset_manifest.json`, `camera_poses.json` and source raster images. R camera rows must be `registered=true`, have original view/session/split/frame SHA matches, `coordinate_frame=colmap_opencv_world_to_camera`, and `units=reconstruction_units_unscaled`. Unregistered views are omitted from spatial camera markers and enumerated as excluded. **No COLMAP estimation/training** occurs here.

### SE(3) transformation

COLMAP OpenCV world-to-camera, column-vector homogeneous transform:

```
X_cam = R_wc * X_world + t_wc
T_cw  = inverse(T_wc) = [R_wc.T, -R_wc.T @ t_wc; 0,0,0,1]
X_world = T_cw * X_cam
```

The output uses row-major `camera_to_world`, with **camera +X right / +Y image down / +Z forward**, matching V frustum. Never re-label `T_wc` as `T_cw`; never flip axes without a documented transform. Reject reflections, scale, bad quaternions/non-rigid matrices, nonfinite values and inconsistent source hashes. Intrinsics support R's `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `OPENCV_FISHEYE` parameter vectors. Nonzero distortion is **preserved in metadata**, source image not undistorted; frustum corners computed from `fx,fy,cx,cy` are **approximate if distortion is nonzero**.

### Geometry co-registration: fail closed

A's reviewed `civil_geometry_manifest.json` and referenced `viewer_overlay.json` are SHA256 checked and source-native frame/unit identities checked. A source `colmap_world_unverified` is **not assumed** to be R's camera world frame. If `--a-manifest` is given **without** `--frame-bridge`, emit the seven Viewer roles with **empty planes/edges/corners/mesh** and state `A_camera_frame_link_missing; geometry withheld`. Camera/frusta and their original SHA-checked photos remain viewable, but no false room reconstruction is overlaid.

Optionally pass `--a-project-root /read-only/A/project --a-manifest runs/geom-civil-*/geometry/civil_geometry_manifest.json --frame-bridge audit/frame_bridge.v1.json`. The `frame_bridge` JSON schema `mycon.viewer.frame_bridge.v1` must include:
- exact `r_camera_poses_sha256`, `a_manifest_sha256`, and `a_coordinate_frame`;
- `r_pose_convention="colmap_opencv_world_to_camera"`;
- `geometry_to_r_colmap_world_4x4`: rigid, right-handed SE(3), and `same_reconstruction_unit_ratio=1`;
- `independent_review=true`, nonempty `reviewer_id`, `metric_scale_verified=false`;
- `independent_evidence: {path,sha256}` pointing to a readable, separately reviewed same-scene registration/identity record, all anchored to A project root.

Missing/changed proof or non-rigid transform **blocks before writing**. The proof is a **review convention, not a cryptographic signature**; human verification must be independently checked. Accepted A feature states remain `unknown`, coverage/confidence null (A fit-quality scores are preserved only as diagnostic source fields). A finite plane *boundary* must not be drawn as a filled measured wall. **No certified floors/walls/corners** and no construction progress values. A's actual real-derived depth cloud has 15 candidate planes, 49 edges, 7 extrapolated corners, **0 confirmed**; its coordinate frame has not been proven co-registered with R. This adapter DOES NOT assert co-registration.

## BIM metric and provenance gates

Result units `reconstruction_units` and metric flags **false**. Shape4 unregistered illustrative reference only if explicitly supplied by a future separately approved adapter; **numerical/colored deviation disabled** until D's independent survey, registration and holdout control receipts pass V's original strict gate. D `mycon.metric_bim_viewer_handoff.v1` is NOT equivalent to a V bundle and cannot by itself unlock metric coloring. UI unknown/occluded and not-evidenced are not “not built”.

Every V role contains schema version, safe path, SHA256, frame, units, run-id/provenance and quality; all source images are copied unchanged into NEW V output for an offline browser preview. Original R and A inputs are **read-only**. Output directory creation refuses an existing path, never `--force`.

## Executed evidence, not claimed field validation

- Local owner-only synthetic producer-shaped R fixture, `15 passed, 0 failed` on the local adapter (see session logs); **not real MyCON data**.
- GitHub exact commit `1690a0e66ae098a5274d8c20f10f4f1c8e5c9dfc`: dedicated CI [run 43516797441](https://github.com/sayyedalimrj/MyCon/actions/runs/43516797441) `28 passed, 3 skipped, 0 failed`; 3 skipped include optional browser or Plotly tests without deps.
- Headless local Chromium: actual *preserved standalone UI* + six **synthetic** R-style camera frames; camera #4 clicked, linked PNG decoded, layers toggled, metric blocked, 1440×900 screenshot + 3-second H.264 Canvas video. Source is synthetic even when screenshot capture is real.
- Local WebGL probe with software SwiftShader **could not create a WebGL context**; Canvas fallback worked. Thus **WebGL-on-GPU, exact GitHub-HEAD browser execution, and real site camera-photograph trial are NOT validated**. Current GitHub CSS/JS has been syntax/CPU checked, not GPU-certified.
- User Drive has historical COLMAP `cameras.txt`/`images.txt` and images, but the A real-derived depth-cloud coordinate registration is **not established**. No direct R v1 registered pose+per-frame SHA dataset for the exact real room was verified; no synthetic-to-real promotion.
- No independent surveyed metric control; no per-face deviation truth or quantitative thesis claim.

## Coordinator actions

Supply R's genuinely executed complete COLMAP run folder **with hash-verified source imagery**, and A's SHA-verified reviewed geometry only with a separately verified same-scene camera frame identity/SE(3) proof. Run `convert_r_run` read-only with a **unique** V run-id, open preview and click source cameras. Obtain full WebGL GPU browser screenshot/video and a separate engineer-signed Shape3 review. Coordinator alone merges and edits notebook cells.
