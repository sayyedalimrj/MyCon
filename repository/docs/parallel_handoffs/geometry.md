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