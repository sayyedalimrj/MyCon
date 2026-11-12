# MyCON: Reference-Agnostic Comparison-Ready Geometry — 2026-11-12

## Priority correction
The user explicitly **does not want work focused on 14021208 MAX or any particular reference model now**. The prior room-specific Issue #39 and contract are postponed to a later optional comparison phase. Current task: produce clean, structurally indexed reconstructed geometry that can be compared with CAD/IFC/BIM **later**. Do not import, crop, scale or fit the MAX model now. PRs A #38, B #37 and C #36 remain independently owned and unmerged.

## Required optional output schema: mycon.comparison_ready_geometry.v1

Input: completed reconstruction PLY/mesh, cameras/source images when available; preserve original captures, checkpoints, textured/colored geometry, camera poses and intermediate artifacts. Never destroy/smooth away small elements or synthetically fill unseen structures.

Under unique run-id prefix geom-ready-*, create:
- clean_cloud.ply and clean_cloud.npy, with documented source coordinate frame/units (including arbitrary/reconstruction units), cleaning configuration and provenance.
- planes.json: local finite structural planes, signed normalized coefficients, support sample IDs/counts, finite extent, residual and uncertainty/quality.
- edges.json: finite locally supported candidate segments from compatible intersecting planar patches or validated feature lines, plane IDs, endpoints, support extent and fit diagnostics.
- corners.json: 3D corner candidates from trihedral planes or structural line endpoints, feature IDs, coordinates, support/stability, source-image provenance when available, and state 'candidate_extrapolated', 'observed_supported' or 'unknown'. No infinite-plane extrapolation may be reported as a confirmed measured corner.
- comparison_ready.json: input/output SHA-256, source/model/camera hashes, version/seed/environment/parameters, density and removed-point ratios, quality metrics, failure and unknown reasons, full filename/coordinate/units map, optional geometry previews.
- optional RGB-preserving overlays, reviewable images and future robust registration feature set; design matching is not required.

Metric scaling and registration are *separate later gates*: until independent controls exist, metric_scale_verified=false and registration_verified=false. No IFC GlobalId is assigned to a generic point/line/corner and no construction percentage follows from it. Keep line/corner hypotheses distinct from independently confirmed features.

## Scientific and engineering requirements
1. Reuse existing Stage 7 Open3D cleanup, normals, blind/guided RANSAC planes, meshing and Stage 8 registration quality modules rather than replacing them blindly.
2. Conservative cleaning and adaptive voxelization with before/after morphology/edge preservation; report point loss and why it is safe. Preserve original RGB/textures as references.
3. Extract locally supported structural planes, patches, edges, corners; protect slanted/non-Manhattan walls, curved features and partial captures; detect corner degeneracy and gaps.
4. Calibrated image projection and multi-view repeatability if camera intrinsics/extrinsics exist; otherwise mark visual confirmation unverified. Report precision/repeatability against independent GT if supplied, not invented accuracy.
5. Prepare transforms and stable correspondence features for future SE(3)/Sim(3) CAD/BIM match, but do NOT fit any reference model now.
6. Every run is deterministic and non-destructive, CLI+unit+regression tested with exact exit codes, logs, versions, hashes and failed/not-executed cases. No training from scratch without permission.

## Parallel ownership and coordination
- **A / PR #38:** productionize clean reconstruction and feature outputs after Stage 7, preserving source checkpoints and camera pose/color data. A owns pipeline stages 1–8. Full real PLY smoke and corner QA when source available.
- **B / PR #37:** optional consumer of feature package for audit-only geometry. Without verified metric scale and IFC mapping, physical_progress_estimate must stay unknown unless separately approved measured quantity evidence exists. Do not touch A files.
- **C / PR #36:** test structural feature accuracy, repeatability, noise/occlusion/density/non-Manhattan stress, schema and runtime, using independent synthetic truth and future real surveyed GT; do not change A/B code.

**Coordinator CPU pilot:** MyCON_Comparison_Ready_Geometry_Pilot.zip created in the user conversation; six synthetic local pytest cases passed. This standalone source is **not yet committed to GitHub**; owner A should review/port its algorithm or a stronger replacement into its own files. It gives intersection candidates, NOT field-certified corners, full texture preservation, or final publication-grade results.

## Release criteria
- Full relevant pytest and CLI executed against real checked-out MyCON.
- Real captured reconstruction processed with QA and human-reviewable plane/edge/corner visuals plus source camera evidence.
- Multi-scale, cross-view and non-Manhattan tests; reject false corners/planes with reportable uncertainty.
- Stable additive outputs consumed independently by B/C; all old interfaces preserved and merged by coordinator only.
- No metric/BIM/progress claims until separate calibration and survey/IFC quality gates.

Reference algorithms: https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html ; https://www.open3d.org/html/tutorial/geometry/pointcloud_outlier_removal.html ; https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html
