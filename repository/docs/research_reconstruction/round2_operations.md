# Reconstruction R — guarded operational workflow (Round 2)

Run from `repository` root after installing CPU extras via `pip install -r requirements-research-reconstruction.txt`. Do **not** write to `MyCon_R4` or historical output paths. Use a unique Drive directory and `recon-*` run id each time. No `--force` exists.

```bash
python -m reconstruction.preflight
python -m reconstruction.workflow select --run-root /NEW/R_WORKSPACE/runs --run-id recon-room-unique-0001 --train-video /READONLY/source_capture_A.mp4 --train-session capture_A --test-video /READONLY/independent_capture_B.mp4 --test-session capture_B --sample-hz 2 --max-frames 180
python -m reconstruction.workflow colmap --run-dir /NEW/R_WORKSPACE/runs/recon-room-unique-0001                 # dry inspect
python -m reconstruction.workflow colmap --run-dir /NEW/R_WORKSPACE/runs/recon-room-unique-0001 --execute-colmap # CPU SfM; NEW RUN ONLY
python -m reconstruction.coverage --run-dir /NEW/R_WORKSPACE/runs/recon-room-unique-0001
python -m reconstruction.gs_experiment --run-dir /NEW/R_WORKSPACE/runs/recon-room-unique-0001
python -m reconstruction.workflow plan-gs --run-dir /NEW/R_WORKSPACE/runs/recon-room-unique-0001
```

`select`: blur/exposure/highlight/shadow clipping, near-duplicate dHash, temporal spread with per-session hash, `frame_selection.csv` and `dataset_manifest.json`. For a *single* session, temporal block+embargo is an **exploratory** holdout, not an independent-site benchmark. Calibrated per-image intrinsics and poses become valid only if new COLMAP registers that image; otherwise explicitly `registered=false`.

`colmap`: create-only train-first feature database, train-only SfM, recorded SHA256 of frozen train sparse, **then** add test features, exhaustive matching and `image_registrator` against frozen train. Never bundle-adjust with test RGB, never relabel old poses. COLMAP version is detected by `-h` CPU flags; if unsupported, fail closed. Test registration may not succeed in textureless/weak-overlap cases. Outputs: `colmap/sparse/*`, `camera_poses.json`, `registration_quality.json`, `coverage_report.json`, hashes and log. Run with real binaries and review registration percentages/missing view IDs before downstream work.

`gs_experiment`: copies only *train* PNGs to `gs_train_dataset/images` (no test images), plus a hashed copy of fresh COLMAP train sparse to `sparse/0`; prints an inert upstream gsplat command template. **No training occurs**. Explicit opt-in, independent GPU environment and authorization to run a *fresh new-model* trainer are still required. Historical 6999 or recovery states must never be passed to fresh COLMAP scene without verified compatible scene transform. The gsplat example trainer's own `test_every` is not the MyCON independent test set. Automatic resumption after interruption is **not implemented**; preserve partial runs and logs and require a verified backend-specific restore plan before continued optimization. No silent train-from-zero.

After a genuinely executed and independently rendered held-out test set, provide `mycon.reconstruction.render_manifest.v1` with **verified new checkpoint path and SHA256**, `camera_poses_sha256`, run id, and an array per test `view_id`, `image_sha256`, `render_relpath`, `render_sha256`, with render paths anchored to the manifest directory. Then:

```bash
python -m reconstruction.workflow score --run-dir /NEW/R_WORKSPACE/runs/recon-room-unique-0001 --render-manifest /NEW/R_WORKSPACE/RENDERS/render_manifest.json --compute-lpips
python -m reconstruction.visual_review --run-dir /NEW/R_WORKSPACE/runs/recon-room-unique-0001 --render-manifest /NEW/R_WORKSPACE/RENDERS/render_manifest.json --fps 4
python -m reconstruction.workflow compare --run-dir /NEW/R_WORKSPACE/runs/recon-room-unique-0001 --baseline-per-view /READONLY/historical_6999_validated_paired_test.csv
python -m reconstruction.workflow receipt --run-dir /NEW/R_WORKSPACE/runs/recon-room-unique-0001
```

Outputs: `per_view_metrics.csv`, `heldout_scores.json`, `visual_review/comparison_*.png`, `visual_review/heldout_multiview_comparison.mp4`, `comparison_6999.json`, `artifact_receipt.json`. Baseline comparison **remains NOT scientifically comparable** without independently certified baseline camera/scene/split lineage. Real render evaluation is impossible without matching fresh test camera poses and checkpoint.

**Viewer/Geometry interface:** viewer consumes `camera_poses.json`, tested image/renders, SHA receipts and `visual_review_manifest.json` as *unscaled appearance*; Civil Geometry may consume independent COLMAP MVS `fused.ply` (or reviewed depth-fusion), never the Gaussian centers as measured construction surfaces. No Stage8/9/11 progress inference, no meter units, and no IFC IDs until separate independent survey/registration.

**Colab GPU installation outline (lead-owned global notebook; DO NOT edit Colab files here):** use isolated runtime, check `nvidia-smi` and `torch.cuda.is_available()`, install pinned official gsplat/PyTorch-compatible CUDA wheels, validate COLMAP binary, stage only new train-only dataset, upload outputs in R-specific Drive folder, audit SHA and independent test renders. 2DGS optional separate research environment uses Python 3.8 / PyTorch 2.0 / CUDA extension and non-commercial license; do not blend with the historical active R4 notebook. CUDA peak VRAM / wall clock / notebook execution **not measured** in this task.
