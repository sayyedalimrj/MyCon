# Geometry A — Parallel Research Handoff (2026-11-12)

## Ownership, commits and review protocol

- Owner/issue: **A / #33** (`https://github.com/sayyedalimrj/MyCon/issues/33`).
- Branch: `research/mycon-geometry-20261112`.
- PR target: `research/mycon-integration-20261112`; never `main`.
- Contract base commit: `f4fc2b4aad73e8516e031129780e56069e42257e`; observed integration/geometry branch fork at `33f1b68b2dc099db97d9e9101f56dd5ce2100d3f`.
- **Last implementation-and-test commit before this handoff:** `2b851318a287ca6be4206ed8661ce35f28ffe946`. This is an exact commit SHA; the handoff commit and final PR head are subsequent to it.
- Changes limited to A-owned Stage 4.5, Stage 8, new geometry tests/docs, and additive geometry requirements. No changes to Stage 9, Stage 11, validation C, common shared code, root requirements, or `main`.

## Files added/modified

- Modified: `repository/pipeline/stage_04_5_cams_gs/run_cams_gs_prepare.py` (explicit resume opt-in, no overwrites of prior Stage 4.5 metadata, no destructive force on nonempty CAMS-GS output).
- Added: `repository/pipeline/stage_04_5_cams_gs/research_training.py` (real Nerfstudio `ns-train` *resume-only* subprocess, isolated dataset/run/output, optional camera-path render).
- Added: `repository/pipeline/stage_04_5_cams_gs/research_gsplat_render.py` (opt-in actual gsplat checkpoint `weights_only=True` tensor rasterization to RGB/depth in a new geom-run, NO training).
- Added: `repository/pipeline/stage_04_5_cams_gs/research_gsplat_audit.py` (read-only existing canonical gsplat checkpoint/PLY/log/video audit).
- Added: `repository/pipeline/stage_04_5_cams_gs/research_backends.py` (optional VGGT COLMAP-export route, external reviewed DUSt3R adapter interface).
- Added: `repository/pipeline/stage_08_bim_eval/research_frames.py` (gravity rotation, Sim3 validation, BIM coordinates and meter-scale diagnostics).
- Added: `repository/pipeline/stage_08_bim_eval/research_metrics.py` (bidirectional nearest-neighbor, Chamfer, F-score, held-out anchor and gravity diagnostics).
- Added: `repository/tests/test_research_geometry_training.py`, `test_research_geometry_gsplat_render.py`, `test_research_geometry_gsplat_audit.py`, `test_research_geometry_metrics.py`, `test_research_geometry_frames.py` (19 synthetic, lightweight pytest function definitions; **not executed in-repo here**).
- Added: `repository/docs/research_geometry/README.md` (CLI, guards, scientific semantics, reproducibility and risk) and `repository/requirements-research-geometry.txt` (additive only).
- Added: this handoff.

## New interfaces and backward compatibility

- Opt-in Stage 4.5 keys: `cams_gs.research_resume_dir`, `cams_gs.research_run_id` (must be `geom-*`), `cams_gs.research_resume_approved` (boolean), optional `cams_gs.research_camera_path_json`, together with original `cams_gs.execute_training`.
- Stage 4.5 writes the same legacy JSON and selected-image-list paths (no keys removed). New optional `training.geometry_resume_evidence` field and explicit blocked/failed states. Historical manifest files cannot be overwritten in opt-in mode.
- New isolated output artifacts: `runs/geom-*/geometry/training_evidence.json`; `runs/geom-*/geometry/gsplat_render/` with RGB PNG, unscaled depth `.npy`, report JSON; optional Nerfstudio rendered MP4; `runs/geom-*/geometry/existing_gsplat_audit.json`; `runs/geom-*/geometry/backend_*/backend_evidence.json`; new standalone geometry metric report.
- Stage 8 frozen downstream files preserved: `data/bim/aligned/<project>/scan_aligned.ply`, `bim_reference.ply`, `bim_elements.jsonl`, `runs/<run_id>/reports/registration_report.json`. No Stage 9 `global_id` semantics/units or CSV changes.
- `research_metrics.compare_clouds` signatures use *explicitly pre-aligned meter* Nx3 clouds and return directed means/quantiles, precision/recall, F-score and Chamfer. No claim of GT based on model vs scan self-comparison.
- `research_frames.validate_scan_to_bim_transform` rejects reflecting, sheared, nonuniform/invalid transforms; `gravity_to_bim_rotation` returns a proper 3x3 matrix, without writing outputs.
- `research_gsplat_render.validate_camera_path` requires `colmap_opencv_world_to_camera` and explicit per-frame 4x4 + intrinsics; camera holdout and metric scale remain **unverified**.

## Real historical evidence inspected (not reproduced or independently verified)

- Google Drive historical Nov 6 log `finish_canonical_model.log` reports canonical recovery of `resume_step_006999.pt`, `ckpt_6999_rank0.pt`, and `point_cloud_6999.ply` from the same splat state; Drive file metadata reports **348,744,578 bytes** for that PLY. Source video/PLY was NOT directly read or rendered in this task.
- Nov 6 `final_canonical_eval.log`: **192 COLMAP images, one camera; reported 1,477,725 Gaussian primitives; PSNR 20.736 dB, SSIM 0.6176, LPIPS 0.522**; trajectory render progress **181/181** and saved `traj_6999.mp4`. The log is untrusted for establishing held-out split: **evaluation denominator, frame overlap, evaluation settings, input hashes and independent test views unavailable**. Thus these metrics are *reported historical metrics*, not fresh results.
- Prior Nvidia LongSplat log shows **769 reused frames** with T4 (15,360 MiB) and model command; completion not verified; official MASt3R weights `SHA256 e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2` was reported verified by earlier weights preflight with inference NOT executed. This SHA is not a hash of scan/3DGS input data.
- Historical Nov 1/2 Drive `pipeline_state.json` (7,206 B, schema `mycon.r4.reality_pipeline_state.v1`) records three stages: Stage 1 `REUSED`, Stage 2 `REUSED`, Stage 3 `REFUSED`. This is a different run from the Nov 6 gsplat log and must never be conflated.
- Existing `diagnostics/important_outputs_manifest.txt` enumerates prior Stage 8 PLYs/reports, but this is a historical file-list snapshot only, not verification of their contents.

## Tests / quantitative evidence / environment

- **Actual executed here:** 3 **independent CPU-only numerical sanity checks** in a Python 3.13.5 local scratch environment, NumPy 2.3.5 / SciPy 1.17.0: identical 4-point clouds -> Chamfer L1 = **0 m** and F-score **1**; two-point scan shifted +0.20 m -> Chamfer L1 = **0.4 m**, F-score **0** at 0.05 m; 90-degree gravity rotation -> rotation determinant **1** and mapped up vector aligned. All **3/3 passed**. These were external formula sanity checks, **not execution of the GitHub modules**, not measurements on real site data.
- **Not executed:** checked-in pytest suite (source tree could not be cloned into local runner, outbound GitHub DNS unavailable); all 19 added pytest test functions; Stage 1–8 pre-existing lightweight regression; GPU Nerfstudio/gsplat/VGGT/DUSt3R; live camera/PLY physical geometry verification and real site registration. `open3d` also absent in local scratch runtime.
- **Fresh real-world geometry metrics (Accuracy, Completeness, Chamfer, F-score, Registration RMSE): `not_executed`, sample count 0, no estimated confidence interval.** Do not substitute synthetic sanity values as site metrics.
- New code compiled, import-loaded and run against actual repo via CI: **not yet verified**. New unit tests require a Python environment with at least pytest, NumPy and SciPy; actual Stage 8 run needs Open3D/IFC prerequisites.
- No long training from scratch was initiated, no prior Checkpoint/PLY/Recorder changed, no common directories shared with B/C used.

## Input hashes and artifact locations

- Verified historic MASt3R SHA-256 reported above; official COLMAP input, gsplat checkpoint, PLY and video SHA-256 **unknown/not computed** in this execution. The new research auditors optionally calculate hashes on the machine holding originals.
- Historical canonical gsplat files and logs are under the user's existing `MyCon_R4/research_colmap_gsplat/.../gsplat_result/canonical/` Drive workspace. The repo contains no current actual rendered files.
- This code creates zero new GPU outputs until explicitly invoked; `geom-*` run IDs are proposed output namespaces, not claims of completed runs.

## Integration blockers / handoff requirements for coordinator

1. **Run actual lightweight pytest plus full Stage 3/4.5/6/8 regression** in a checked-out repo; fix reported failures before merge. The new test suite was *defined*, not run in this GitHub-only environment.
2. Provision compatible `gsplat` + PyTorch/CUDA, check tensor shapes and camera conventions for the existing canonical `ckpt_6999_rank0.pt`, then run `research_gsplat_render --execute` in a NEW `geom-*` workspace and inspect PNG/depth. This renderer is implemented but GPU-unvalidated.
3. Official gsplat `.pt` cannot be used for Nerfstudio `--load-dir`; treat these backends separately. The direct `ns-train` resume path needs an existing Nerfstudio `step-*.ckpt` with valid COLMAP model.
4. Any claim of new viewpoints requires capture/session-disjoint test poses and reference images; run quality PSNR/SSIM/LPIPS on held-out views with denominator and uncertainty. Historical 181-frame trajectory is only a reported rendered path.
5. For BIM-grade metric geometry use verified GCP/control-distance scale, gravity evidence and independent anchors, with IFC georeferencing and axis handedness checked. Stage 8 must pass its real IFC/scan quality gates before Stage 9 consumes evidence.
6. Verify actual python CLI compatibility for `ns-train` COLMAP dataparser and `gsplat` rasterization with installed package versions; no claim this code has been exercised on a CUDA node.
7. C owns independent scientific benchmarking; A's point sampled diagnostics are not C's benchmark or construction completion metrics.

**Overall status:** implemented and versioned on A branch; independent CPU math sanity 3/3 successful; new repo pytest + GPU/real-case execution **not verified**. PR should remain **Draft/unmerged** until coordinator approves the blockers.

---

# Model-agnostic geometry readiness — final A extension (2026-11-12)

**Governing Integration ref:** `repository/docs/integration/model_agnostic_geometry_readiness_contract_20261112.md`. The latest PR #38 coordinator correction supersedes the earlier room-specific MAX/IFC proposal: **NO target reference, alignment, MAX/BIM/IFC import or new metric claim was used here.** This is an A-owned, create-only Stage-7 companion output. Previous Stage-8 A→B IFC-specific optional interface is retained and left untouched.

## Actual new commits / changed paths

- `repository/pipeline/stage_07_cleanup/comparison_ready.py` — Python CPU implementation of `mycon.comparison_ready_geometry.v1`; committed initially `30bf0075473ba7ad3743494aa8732a933809641b`, corrected `6c80c84aaaf89030bb6d383ce3ed18936f68970c`. Actual local module SHA-256: `571186cd1ae78db22fe8573cd455250e1058197d92a5ff467fbb596d59b45ecb`.
- `repository/pipeline/stage_07_cleanup/run_cleanup.py` — optional config gate and independent create-only companion processing after legacy Stage 7; non-destructive opt-in guards and fail-closed failure handling. Last implementation commit `2ec5567fd0eb01ad3ecfa3a292eef9b254dad6d1`.
- `repository/tests/test_research_geometry_comparison_ready.py` — 12 isolated deterministic CPU tests; commit `0d82c76076983ca549c3d7b070ddc7933122e83a`.
- `repository/docs/research_geometry/comparison_ready.md` — schema, CLI, input units, QA limitations, room synthetic reference and safety. Commit `5896b8a7768d20c012f581503b3fcf87c15c8bd6`.
- `repository/docs/research_geometry/generate_comparison_ready_fixture.py` — explicitly labeled **synthetic** reproducibility script; commit `b651d1ccc78f5b456346e156a13a6319a92e4eb1`.
- `repository/docs/research_geometry/comparison_ready_synthetic_qa_20261112.json` — machine-readable executed synthetic findings/hashes; commit `b352b9de829ba1dfea8828c741f7c6616809a99e`.
- `repository/requirements-research-geometry.txt` — additive `trimesh` and optional `matplotlib` dependencies; commit `23f2796a9bf3b739219b0f293bae4395fd6eb748`.
- This file updated in the current extension commit; **actual final PR SHA must be taken from PR #38 after this write**, not confused with the preceding file-specific commits.
- **No changes made to B/C files, Stage 9/11, common code, integration branch, main, original video, checkpoint, camera/calibration or PLY. PR remains Draft and unmerged.**

## Versioned output contract (new optional additive interface)

Independent run ID **must** match `geom-ready-*`. Under `runs/<run_id>/geometry/`, produce `clean_cloud.ply` (original vertex RGB preserved if present), `clean_cloud.npy` (float64 XYZ), `normal_samples.npz`, `planes.json`, `edges.json`, `corners.json`, `comparison_ready.json`, `xy_geometry_preview.png`, `xyz_geometry_preview.png`. Every file is SHA-256 recorded in manifest and never overwritten.

- Manifest: `schema_version=mycon.comparison_ready_geometry.v1`, unique run ID, source SHA-256/path, optional source camera-spec SHA, source frame, units (or explicit unknown), seed, versions, cleaning counters, thresholds, counts and nonmetric status.
- Planes: normalized signed SVD-refitted [n_x,n_y,n_z,d], observed finite 3D/UV convex-hull patch, point indices, P95 fitting residual, deterministic run-local ID and signature.
- Edges: pairwise non-parallel line intersections restricted to *both* local sets of observed points and their overlapping finite intervals. Unsupported/short segments omitted rather than invented.
- Corners: nondegenerate trihedral solutions, three plane and optional three edge IDs, local support and finite-patch validation, repeatability across two point-density levels. Status `observed_supported`, `candidate_extrapolated`, or `unknown`. No `observed_supported` promotion when multiscale consistency fails.
- Normal QA: up to 10,000 sampled PCA normals, local curvature quantiles, sign/orientation explicitly unverified. Operator-supplied gravity may classify surfaces as horizontal/vertical/sloped *candidates*; floor versus ceiling remains ambiguous absent semantic/height evidence.
- Image reprojection: only user-provided verified calibrated COLMAP/OpenCV cameras in **same frame**; returns in-FOV candidates with occlusion unverified, or unknown/invalid. No camera/visibility ground truth fabricated.
- RGB, checkpoint, mesh, depth, texture, camera and source input files stay read-only; the original source path/hash is referenced for later recovery.
- `metric_scale_verified=false` and `registration_verified=false` remain **hard-coded**. This is not geometrically certified as-built measurement; no IFC `global_id`, physical progress or reference-fit is inferred.

## Actual executed commands and results

**CPU environment:** Linux x86_64, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2, trimesh 4.11.1, matplotlib 3.10.8; Open3D not installed, CUDA/GPU trainer not invoked.

1. **Syntax/import on local A package copy:** `python -m py_compile .../comparison_ready.py` **exit 0**. Imports and execution of this new module were exercised locally. This is NOT full repository clone/CI.
2. **Thirteen synthetic tests:** `cd /mnt/data/mycon_ready_work/repository && PYTHONPATH=. pytest -q tests/test_research_geometry_comparison_ready.py` **exit 0, 13 passed, 0 failed, approximately 2.11 s**. Tests cover finite 3-plane/corner, color-preserving conservative cleaning, rotated and non-Manhattan geometry, missing corner support, parallel degeneracy, FOV projection uncertainty, actual binary PLY/NPY and hashed manifest generation, create-only immutability, arbitrary units, repeatability, density perturbation and occlusion. One isolated-layout `PytestUnknownMarkWarning` because local skeleton did not contain real repository pytest marker configuration (actual repo registers `lightweight`).
3. **Full synthetic CLI end-to-end:** `cd /mnt/data/mycon_ready_work/repository && PYTHONPATH=. python -m pipeline.stage_07_cleanup.comparison_ready --input-ply /mnt/data/mycon_ready_work/local_run/synthetic_three_planes.ply --project-root /mnt/data/mycon_ready_work/local_run --run-id geom-ready-qa-20261112 --threshold .02 --min-plane-points 90 --trials 350`: **exit 0**. Input **2352**, clean **2352**, removed **0**, observed finite planes **3**, candidate supported edges **3**, observed-supported trihedral corner **1**. The source is **entirely synthetic** with added Gaussian noise std=0.001 *reconstruction units*.
4. QA: 3 planar P95 residuals approx `0.002047/0.002031/0.001980` reconstruction units; candidate corner offset from known synthetic origin ~`0.000158` source units; multiscale drift ~`0.0001484` source units. These are **not metres**, and NOT field accuracy, precision, false-positive rate or publication metrics. Preview images inspected in 3D locally.
5. Synthetic input SHA-256: `509bde00a3509600a4ef3ecd1a20b7fea7834ca99dea23e0263f51512e7fac63`. Output hashes are saved in `comparison_ready_synthetic_qa_20261112.json`; the source and eight output files plus code/tests are in the downloadable local inspection ZIP.

### Later enhancement: Stage7 existing plane reuse and extended QA

- `comparison_ready.py` now accepts `--stage7-planes-json` and consumes Stage7's original `planes[].cloud_path` and `plane_id` records; nearest-source matching with a bounded geometric tolerance, deterministic point ownership and no repeated assignment; invalid supplied Stage7 metadata fails closed. No new Stage7 RANSAC is run when prior planes are explicitly provided. New module hash SHA-256 `571186cd1ae78db22fe8573cd455250e1058197d92a5ff467fbb596d59b45ecb`; source commit `939caccfa92207a51380ccf6e1300e4ab186d826`.
- `run_cleanup.py` opt-in hook now always forwards the existing Stage7 `paths["planes_json"]`; commit `bd605dbb784cc4d05c197f80dfaf3f83bc47c150`. Legacy flag remains default false.
- New test verifies Stage7 reuse produces disjoint point assignments and source JSON SHA-256. **Full local new-module test suite: 13/13 pass, pytest exit 0, approx 2.11 s**, compile exit 0; this is still an isolated A package test, not a real Git checkout or Open3D integration.
- Real end-to-end **synthetic** Stage7-planes reuse: `--stage7-planes-json` with deterministic three-plane input, exit 0, 3 prior planes, 1 finite line candidate, 1 trihedral *candidate* but **0 observed-supported corners**, because locally observed junction support from prior Stage7 segmentation was inadequate; fail-closed as intended. Stage7 planes JSON SHA-256 `5ca0b9beaf1749d79ccc1d7a1b8fafa8a7c9df843ee11f91ef47064c273a367d`. All eight outputs passed SHA verification.
- Further CLI smoke: `python -m ... --help` exit 0; second invocation using same preexisting geom-ready ID exit **2**, no overwrite. 8/8 generated output SHA-256 checks pass. Synthetic output `clean_cloud.ply`, `clean_cloud.npy`, planes/edges/corners JSON, normal diagnostics and 2D/3D previews supplied in local zipped review package.
- Review ZIP SHA-256 `73e1f5d36a3b230cbce6e8a902d771b4fb624060e18bdf803175068a902ecab5`; this ZIP is an attached user artifact, **not a GitHub-tracked real-capture output**.
- **All real-world acceptance gates still blocked** by original >256 MiB files, absent CUDA and surveyed metric control / Open3D full checkout. No actual construction corners/scale/error claimed.

## Field-data, scientific and release blockers (do not label 'passed')

- **Real project PLY pipeline: blocked_missing_input.** The original `point_cloud_6999.ply` and `ckpt_6999_rank0.pt` each exceed the connected Drive 256 MiB download limit (~349 MB). This revision therefore did **not** execute full real-collected-cloud RANSAC/surface/corner extraction. Execute `comparison_ready` next to the source on its original Drive-mounted Colab/workstation for read-only processing.
- **GPU gsplat replay, 3DGS-to-PC, SuGaR and 2DGS: not_executed.** Gaussian centres cannot be assumed to lie on engineering surfaces. The existing historical 181-frame video confirms prior render provenance only, not metric geom. No matching real geometry outputs or CUDA environment here.
- **Full Git checkout Stage 3–8 regressions: blocked_network_and_dependencies.** Outbound GitHub DNS resolution fails in local container; only the authenticated GitHub connector can read/write the repo, and Open3D is absent. Do **not** label the 13 local tests as full-checkout regression or GPU validation. Stage-7 opt-in integration was source-integrated and reviewed but its full Open3D pipeline smoke was not executed here.
- **Scientific external GT:** no independent surveyed dimensions, capture-session-separated test images, ICP survey anchors, room/full building correspondences, or ground-truth corners available. Thus real false-corner rate, registration RMSE, surface accuracy and scale remain `not_executed` / `unknown`.
- **Known algorithmic limitations:** RANSAC is sensitive to threshold, finite convex hulls can bridge occluded holes, short/curved features need further verification, unoriented normals cannot imply actual wall/floor/ceiling identity, and multi-scale consistency at only two densities is diagnostic not independent GT.
- **Merge blockers:** coordinator must run project-source checkout pytest + Stage3–8 smoke, source-side real PLY extraction + visual human QA + split/camera provenance checks, preserve B/C contract and verify hashes before approving PR #38. No automatic merge or release promotion.
