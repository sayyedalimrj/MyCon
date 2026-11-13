# Reconstruction R — coordinator handoff (Round 2)

**PR:** https://github.com/sayyedalimrj/MyCon/pull/49 (Draft, unmerged). **Base:** `research/mycon-integration-20261112`, read at `c81962332cb4220f52eeb95acfcda61e48d39d24`. **Full source milestone commit:** `ce60096dbec12d674df3b299a7ec08f3aa4ef2cf`; latest authoritative source SHA is the PR HEAD (this file's commit cannot self-cite its own SHA). Owner approval: `repository/docs/integration/parallel_6_streams_approval_20261113.md`, Issue #48.

## Changed files — R-owned only

- `repository/reconstruction/{__init__.py,workflow.py,coverage.py,preflight.py,gs_experiment.py,visual_review.py,safety_gates.py}`.
- `repository/tests/{test_research_reconstruction_workflow.py,test_research_reconstruction_round2.py,test_research_reconstruction_safety_gates.py}`.
- `repository/docs/research_reconstruction/{README.md,round2_operations.md,evidence_review.md}`.
- `repository/docs/parallel_handoffs/reconstruction.md`.
- `repository/requirements-research-reconstruction.txt`.

Original 10 + nine additional tests are **native on this PR** and their source Git blob SHA has been compared to the local tested package. Six extra native tests guard the external schema. No Stage1–8, Colab, shared config, A/B/C-owned path or main has been edited. No run has modified videos, source PLY or legacy gsplat/COLMAP.

## Data contracts and integration

- `mycon.reconstruction.dataset.v1`: selected frames tied to `view_id`, `capture_session_id`, `image_sha256`, run-relative image, split, timestamp, blur/exposure evidence and source-video hash.
- `mycon.reconstruction.camera_poses.v1`: per-frame `registered` state, intrinsics, source SHA and `colmap_opencv_world_to_camera` extrinsics, explicit unscaled coordinates.
- `mycon.reconstruction.registration_quality.v1`, `mycon.reconstruction.coverage.v1`, `mycon.reconstruction.gs_train_dataset.v1`, `mycon.reconstruction.heldout_scores.v1`, `mycon.reconstruction.visual_review.v1`, `mycon.reconstruction.baseline_comparison.v1`, `mycon.reconstruction.artifact_receipt.v1`.
- Geometry/Viewer consumption is strictly read-only; no physical surface or confirmed corner from raw Gaussian centers. `metric_scale_verified=false`, no IFC GlobalId or Stage9 progress inference.

## Actually executed (synthetic CPU only)

Local source package (not GitHub Actions): `cd repository && PYTHONPATH=. pytest -q tests/test_research_reconstruction_*.py` → **19 passed**, exit 0; `python -m compileall -q` → exit 0; CLI `--help` → exit 0. Original 10 tests preserved, nine additional tests cover angular coverage, SHA-pose mismatch, Colab/runtime preflight, conditional side-by-side MP4 and GS train-only staging.

Synthetic test recipe creates two short AVI captures, 35 selected train + 35 selected test, mock COLMAP text registers 35 train and 2 test. There was **no COLMAP executable, camera bundle adjustment, real GS, physical measurement or actual GPU training**. Synthetic manifests SHA-256 from one local fixture: `7dff74715e7b9d1b1b310b889c1830fe3b4aea5b76401955649937342e295110`; camera pose SHA-256 `e7fc2261331b7fcf27dad3beec5400ffa84cb9d3fa622e89c0c728ef0f1f81c6`. These hashes certify only that fixture, not field accuracy.

GitHub Actions: the preliminary six `test_research_reconstruction_safety_gates.py` tests passed on commit `89e2dd795e34fda00f810cd85bc5f3a1f436b6c4` in workflow run `43500483397`; the source workflow/19 tests were pushed **after** that run, so do not count the initial result as exact-HEAD validation. **Exact-HEAD re-run evidence must be read from PR checks.**

Local environment: Python 3.13.5, torch 2.10.0+cpu; CUDA unavailable, COLMAP/gsplat/LPIPS unavailable. 2DGS public Colab notebook inspected at pinned revision but NOT executed due to missing GPU, CUDA extension and Drive mount. Literature, code commit/license and method trade-offs are in `evidence_review.md`.

## Unexecuted / blockers — no invented field metrics

1. Real original room source frames not processed in this runner; source-video, selected views and actual new COLMAP training/test pose alignment blocked.
2. No reviewed real COLMAP MVS fused surface cloud, wall/floor/ceiling/door/junction and small-detail quality, or independent survey scale; physical surface quality **unknown**.
3. No fresh opt-in GPU Gaussian training or verified resumption, no new checkpoint/PLY, true trajectory/renderer, real per-view PSNR/SSIM/LPIPS or real multi-view video.
4. No validated same-test-image/same-camera comparison with prior 6999. Historical 6999 log metrics remain `historical_only_comparable=false`; no quality improvement claim.
5. Separate coordinator-owned Colab integration not touched. Workstream run IDs, Drive folders, snapshot hashing, CUDA/driver/gsplat compatibility and interruption recovery must be approved and tested on capable hardware.

**Decision:** G0 implemented; G1 local synthetic tested and branch CI pending check; G2 real room, G3 metrology, G4 thesis-grade matched baseline and G5 full integration remain `not_executed`. Coordinator reviews and merges into Integration only; never merge directly to `main`.
