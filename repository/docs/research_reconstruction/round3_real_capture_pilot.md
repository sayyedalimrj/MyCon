# MyCON Reconstruction — R3 real-capture execution evidence (2026-11-14)

## Verdict
**Partial real-data operational pilot; new COLMAP/GPU NOT executed.** No heavy GS training, old/source overwrite, model restart, new checkpoint/PLY, metric-scale certification or 6999 quality improvement claim. PR #49 remains Draft.

## Two different real captures — never attach old poses to newly sampled images

**A. Locally retrieved real room video, exploratory independent run**
- Google Drive source: `MyCon_R4/20260816_090005.mp4` (read-only), 123,427,847 bytes, **SHA256 `b622c340cef772b5f14456889ca59ecbc5720652506e0ff70d0d98a714984424`**.
- Verified locally with ffprobe: H.264 1920×1080, 2096 frames, ~60.019 fps, 34.922422 s. Real content: blockwork, doorway/opening, steel/concrete interfaces, partial floor/roof.
- Create-only local Run ID **`recon-real-room-20261114T0020Z`**, frame selection via native `reconstruction.workflow select`: 70 candidates sampled @2fps, 42 Train, 10 Test, 18 rejected (12 capacity, 3 near-duplicate, 3 split embargo). No blur/exposure/clipping threshold failures at these selected thresholds. Train/Test is a **single-session exploratory temporal block**, not a site-disjoint benchmark.
- `dataset_manifest.json` SHA256 **`b31e720a06d7234d2b416cba5181b069057b798440d10c590a03b4e15bbe9f54`**. New run material in separate local workspace; no old R4 output touched.
- Run `reconstruction.coverage`: selected span 34.489235 s, maximum selected timestamp gap 1.499532 s. Registered new cameras **0 because registration NOT executed**, not a COLMAP registration failure rate. New image intrinsics/poses and view-angle/surface coverage are UNKNOWN.
- Actual `reconstruction.workflow colmap --execute-colmap` attempted: **exit 2, `COLMAP not installed: colmap`**, before creation of `colmap/` and before any train/test-camera registration. `reconstruction.preflight`: PyTorch CPU, CUDA unavailable, no `colmap` executable. No new `camera_poses.json`, new SfM or GS training dataset can legitimately be produced.
- `gs_training_plan.json` generated only as **inert training plan**; no training or checkpoint writing. `comparison_6999.json` `comparable=false`, `paired_test_views=0`.
- Visually reviewed true frames: the camera repeatedly observes blockwork at close range; floor and overhead coverage is intermittent, doorway/window and columns partially visible, repetitive textures may weaken feature matching. These are **review observations**, not confirmed surface visibility, reconstruction error or calibrated occlusion metrics.

**B. Historical 96-second R4 reconstruction — read-only reference, NOT dataset A**
- Actual original recorder member `arcore_recording.mp4`: reported 436,645,888 bytes, SHA256 `b6f8e01a258e0f1dcaa98b536fa835099e1b43f872a5fd2acd1018693f078518` (verified by existing R4 CPU receipt, not re-downloaded in this environment).
- Derived `MyCon_R4/input_cache/worksite_v480_934414c04ed4ef8a/full_video_upright.mp4`: **651,883,372 bytes**, SHA256 **`2f58a3df2bfe955cb5935d748cf58b47cc1928fc12d6b0ea7bc5610aac5bff6a`** from existing R4 cache receipt; 96.0736s. **Authenticated connector HTTP 413** at 268,435,456-byte cap; entire media NOT retrieved here.
- Existing historical `images.txt` 41,577,031 bytes downloaded and **SHA independently checked `b5b6d85cc739ec29a853e9fa593dcc5b3ef2058e9c50a29636ac6593fa37cb94`**, matching previous immutable COLMAP receipt. Camera: `SIMPLE_RADIAL`, 640×1138 pixels, focal 843.159247...; existing COLMAP 192/192 registered, 51,894 sparse points, mean reprojection error 0.7951 px (these numeric properties from **historical reports only**).
- **Five real old-model images** (frames 96,120,144,168,192) downloaded separately and SHA-verified against old `frames_receipt.json`. Historical pose inversion was checked for all 192 frames; the corrected median nearest optical direction = **3.522°**, purely a camera-direction diagnostic. An earlier transient report mistakenly measured the median of *all pairs* (81.932°); a **create-only correction receipt** was preserved. No old pose is transferred to dataset A.
- The old 192-image set has no independently validated 6999 Train/Test partition. Therefore reported PSNR 20.736, SSIM 0.6176, LPIPS 0.522 remain historical log claims and cannot be compared fairly with this new Run.

## Actual local commands and status
- `python -m reconstruction.workflow select ... --train-video /mnt/data/20260816_090005.mp4 --train-session room20260816 --sample-hz 2 --max-frames 55 --min-blur 25 --min-mean 25 --max-mean 235 --max-clip .42 --embargo-sec 1.0` → **exit 0; 42/10 train/test**.
- `python -m reconstruction.coverage --run-dir ...` → **exit 0**, angle/structure unknown pending fresh COLMAP.
- `python -m reconstruction.workflow colmap --run-dir ... --execute-colmap` → **exit 2**, `COLMAP not installed`.
- `python -m reconstruction.workflow plan-gs --run-dir ...`, `compare --run-dir ...`, `preflight --run-dir ...` → **exit 0** (no training or scoring).
- `PYTHONPATH=. pytest -q tests/test_research_reconstruction_*.py` on local previous Round-2 source → **19 passed, exit 0**; **not** a newly run PR-HEAD CI and not proof of COLMAP performance.

## Civil engineering interpretation and integration needs
- Historic camera look direction and registration cannot certify physically captured floor, ceiling, wall, doorway or supported 3D corners; geometry must be independently audited with calibrated same-frame 3D evidence.
- The two original-video hashes differ: no direct camera/pose/image reuse. Historic 192 frame image receipts can inform a **historical-only** Viewer reference using explicit W2C→C2W inversion and SHA; **do not use it as the new run's camera_poses**.
- For the required G2 complete case, run the attached R-owned Colab notebook on the exact 96s Drive media (SHA preflight), install COLMAP explicitly and create brand-new train-only sparse and test registration under unique Run ID; inspect pose/registration evidence and coverage. Supervisor retains freeze. GPU heavy training requires **separate user authorization**; not triggered by this pilot.
- Field error/metric scale, new real cloud/PLY/checkpoint, matched 6999 Test renders/PSNR/SSIM/LPIPS and publication conclusions: **NOT_EXECUTED/UNKNOWN**.
