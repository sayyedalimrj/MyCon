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


---
# Round 3 handoff — actual room pilot, 2026-11-14

**Code/notebook/evidence commit SHA:** `1bda163a272d6f02105b1e0922cc3cc46e6dc5f4` (follows `750fce3` and `7e532a3`). This handoff-document update is a follow-on documentation commit; refer to the PR HEAD for its actual SHA. Base Integration remains the user-provided `research/mycon-integration-20261112`; no merge or force push.

**Changed R-owned paths:**
- NEW `repository/reconstruction/notebooks/round3_colab_real_room_safe.ipynb` (22 valid JSON cells, 11 Python code cells; Google Colab runtime execution **NOT_EXECUTED** locally)
- NEW `repository/reconstruction/historical_camera_audit.py` (historical camera W2C→C2W receipt, no new pose reuse)
- NEW `repository/reconstruction/checkpoint_safety.py` (future-new-run checkpoint copy-only, explicit approval gate; local negative guard exercised)
- NEW `repository/docs/research_reconstruction/round3_real_capture_pilot.md` and `round3_real_pilot_receipt.json`
- UPDATED this `repository/docs/parallel_handoffs/reconstruction.md` only. No edits to A/B/C/V/D or coordinator-owned `colab/**`, Integration or `main`.

## Actual real-source execution on local CPU (not synthetic)
- Original accessible R4 video `20260816_090005.mp4`: 123427847 bytes, SHA256 `b622c340cef772b5f14456889ca59ecbc5720652506e0ff70d0d98a714984424`, ~34.92sec, H264 1920x1080, 2096 frames. Video read-only throughout.
- **New Run ID `recon-real-room-20261114T0020Z`** (local independent output, not written under MyCon_R4). Executed existing CLI `workflow select` (exit 0): sampled 70, selected Train 42 and temporal-heldout Test 10, rejected 18 (coverage cap 12, perceptual near-duplicate 3, split embargo 3). One-capture exploratory split; not independent site GT. `dataset_manifest.json` SHA256 `b31e720a06d7234d2b416cba5181b069057b798440d10c590a03b4e15bbe9f54`.
- `python -m reconstruction.coverage --run-dir ...` **exit 0**; 34.489235 sec selected span, largest temporal gap 1.499532 sec, angular/physical surface visibility **UNKNOWN** until new camera poses exist.
- `python -m reconstruction.workflow colmap --run-dir ... --execute-colmap` **exit 2**: `COLMAP not installed: colmap`. No fresh sparse model, actual new intrinsics/poses, or independent 6999 paired NVS. The absence of registration is **NOT** counted as a failed geometry match; execution never started.
- `plan-gs`, `compare`, `preflight` each exit **0** but inert/not-training. `comparison_6999.comparable=false`. New GPU training, checkpoint/PLY, real NVS renders and PSNR/SSIM/LPIPS **NOT_EXECUTED**.
- Source GPU/COLMAP preflight: PyTorch 2.10.0+cpu; CUDA false, COLMAP absent; no existing checkpoint modified.
- Existing Round-2 local dedicated suite `PYTHONPATH=. pytest -q tests/test_research_reconstruction_*.py` **19 passed (exit 0)**, `python -m compileall -q reconstruction` **exit 0**. This local checkout was the available Round-2 source snapshot + new R-only helper, **not** a full exact latest PR checkout; dedicated GitHub Actions must ground final HEAD separately. New negative checkpoint permission guard tested locally; historical audit on local real old TXT also executed separately (see below).

## Actual historical R4 96-second reference (different from the 35-second new video)
- Old source original recording SHA from previous frozen verified R4 receipt `b6f8e01a258e0f1dcaa98b536fa835099e1b43f872a5fd2acd1018693f078518`; derived upright ~96.074sec `full_video_upright.mp4` SHA from existing R4 cache receipt `2f58a3df2bfe955cb5935d748cf58b47cc1928fc12d6b0ea7bc5610aac5bff6a`, 651883372 bytes. Direct authenticated download **blocked HTTP 413 (268435456-byte cap)**. Source stayed read-only.
- Old model `images.txt` independently downloaded+SHA-verified: `b5b6d85cc739ec29a853e9fa593dcc5b3ef2058e9c50a29636ac6593fa37cb94`; **192 historical registered images**, SIMPLE_RADIAL intrinsics 640x1138, source W2C quaternion translation. 5 real images from same historical scene individually downloaded and checked against `frames_receipt.json` by SHA256. Computed historical C2W matrix inverse for all old 192, **never attached to new Run images**.
- Optical direction nearest-camera median **3.522°** after correcting an earlier method error in a local report (all-pairs median `81.932°` was incorrect for nearest-neighbor); correction saved create-only. Camera direction diagnostic does not certify structural surface visibility/metric truth. Old 192/192 registration and mean reprojection 0.7951px remain **historical receipt values**, not new executed COLMAP.
- Old 6999 test-image split not independently proven. No valid 6999-vs-new numeric comparison. No scale certification, IFC, wall/ceiling/corner precision GT.

## Reproducibility and handoff requests
- Full local SHA-audited evidence bundle (source clips EXCLUDED) `MyCON_Reconstruction_R3_Real_Evidence_20261114.zip`: **1494361 bytes; SHA256 `66de4d94800d5e53575f228d7a974803434b2848867e7b725850e72c043387ae`**, delivered in contributor conversation. Includes real dataset_manifest, frame_selection, actual error output, old image-SHA/pose audit, corrected orientation diagnostic and contact sheets. Do not interpret its old camera reference as a new `camera_poses.json`.
- **Safe Colab owner-only notebook** uses `MyCon_R4` as read-only source and writes snapshots under separate MyDrive root `MyCON_Reconstruction_R3_Runs/<run_id>` (create-only + SHA verification). COLMAP install/execute require explicit flags. Heavy training absent; a future NEW-run checkpoint archival helper refuses operation unless separately authorized, refuses 6999/recovery paths and overwrites.
- **G2 remains incomplete**: request owner access to actual Google Colab runtime / mounted 651.9MB verified source plus COLMAP installation, run new Train mapper before Test localization, record camera JSON/pose/intrinsics/coverage and failure logs, then Viewer adapter with explicit C2W inversion and SHA/capture identity. A historical old pose should never be silently assigned to a new frame. No GS retraining without explicit NEW authorization.
