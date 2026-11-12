# Geometry research A — execution, evidence and cross-stage compatibility

**Owner:** A / Issue #33. **Branch:** `research/mycon-geometry-20261112`. **PR base:** `research/mycon-integration-20261112`. **Run IDs:** `geom-*`.

This is an additive research implementation, not a claim that a real site is now reconstructed. Existing Stage 8/9/11 file schemas are unchanged. Source captures, previous COLMAP data, .pt/.ckpt checkpoints and historical artifacts are READ-ONLY.

## 1. Actual state of the inherited pipeline

- Stage 3 performs COLMAP SfM, Stage 4/5 refine and densify it; Stage 6 aligns DA3 **to COLMAP depth**, which is not by itself a certified meter scale.
- Legacy Stage 4.5, despite its name CAMS-GS, writes a Nerfstudio dataset **stub** and was not an executing trainer. The optional research resume adapter has been wired only behind explicit config gates; the default path remains preparation-only.
- Stage 7.7 reads Stage 4.5 metadata but does not establish genuine withheld-camera visual synthesis.
- Stage 8 already has optional metric anchors, known distances, Sim(3), FPFH/ICP and a rigid-by-default safe scale strategy. Do not apply bbox-derived scale to partial structures.
- A point-cloud PLY, a valid ckpt and a rendered trajectory are **different scientific claims**; all require distinct checks.

## 2. Legacy Stage 4.5 safe execution contract

Default behavior is `cams_gs.execute_training: false` and only a preparation manifest is produced. Existing output contracts stay at their configured paths (`train_manifest.json`, `training_status.json`, stage report and selected image list). The opt-in adapter understands the following **additive** keys under `cams_gs`:

```yaml
cams_gs:
  execute_training: true
  research_run_id: geom-site01-resume-001
  research_resume_approved: true
  research_resume_dir: /read-only/existing-nerfstudio-checkpoints
  research_camera_path_json: /read-only/test-camera-path.json # optional
  max_num_iterations: 7000
```

The opt-in path refuses an explicit `--force`, absent `ns-train`, missing COLMAP model, missing Nerfstudio `step-*.ckpt`, or existing `runs/geom-*/geometry` target. **Do not use an official gsplat `ckpt_6999_rank0.pt` as a Nerfstudio checkpoint**. The `--force` protection also refuses to delete an existing nonempty Stage 4.5 directory in the default prepare path. Select a fresh isolated destination instead.

The real adapter is directly available in inspect mode:

```bash
python -m pipeline.stage_04_5_cams_gs.research_training \
  --project-root /workspace --run-id geom-ns-inspect-001 \
  --images /workspace/data/sfm/site01/images \
  --sparse /workspace/data/sparse_refined/site01/0 \
  --resume-dir /read-only/nerfstudio/outputs/.../nerfstudio_models
```

The additional `--execute` flag **only resumes an existing checkpoint** and runs a real `ns-train splatfacto` subprocess using an isolated copied COLMAP data workspace. It does not launch from zero. Optional `--camera-path /path/cameras.json` executes `ns-render camera-path` only after a reported checkpoint exists. Actual rendered-video presence is not automatically labeled a scientifically held-out view; overlap, train/test session grouping and reproducibility must be independently audited. New records: `runs/geom-*/geometry/training_evidence.json`, `ns_train.log`, optional `ns_render.log` and `novel_camera_path_render.mp4`. Original artifacts never deleted/rewritten.

### Critical distinction: existing official gsplat runs

A separate **read-only** auditor exists for a legacy `gsplat`-format `.pt`, point-cloud PLY, evaluation log and optional video, which are NOT directly compatible with `ns-train --load-dir`:

```bash
python -m pipeline.stage_04_5_cams_gs.research_gsplat_audit \
  --project-root /workspace --run-id geom-existing-gsplat-001 \
  --checkpoint /read-only/canonical/ckpts/ckpt_6999_rank0.pt \
  --ply /read-only/canonical/ply/point_cloud_6999.ply \
  --eval-log /read-only/final_canonical_eval.log \
  --video /read-only/canonical/videos/traj_6999.mp4
```

By default this does **not write anything**. `--write` creates a new immutable `runs/geom-*/geometry/existing_gsplat_audit.json`; optional `--hash-large-inputs` hashes large original files without changing them. No untrusted Torch pickle is deserialized. The audit distinguishes file presence, log-reported metrics, output video presence, unverified checkpoint content and unverified withheld camera quality. This is not a gsplat training/resume implementation.

### Historical existing evidence (2026-11-06, read from saved Drive logs, NOT replayed here)

- `finish_canonical_model.log` reports recovery of `resume_step_006999.pt` and canonical export of `ckpt_6999_rank0.pt` and `point_cloud_6999.ply` from one splat state.
- The separately inspected PLY Drive folder lists `point_cloud_6999.ply` at **348,744,578 bytes**. Bytes and geometry have not been downloaded or independently inspected.
- `final_canonical_eval.log` reports COLMAP data **192 images / one camera**, a renderer with **1,477,725 Gaussians**, evaluation log numbers **PSNR 20.736 dB, SSIM 0.6176, LPIPS 0.522**; a **181/181 frame** trajectory and saved `traj_6999.mp4`.
- Important unknowns: number of held-out eval views, train/test capture-session separation, evaluation protocol, true dimensional scale, video playback/quality, actual checkpoint tensor validation and compatibility of saved training code.
- These values are **reported by historical logs only**, not newly measured results; they are not independent confirmation of NVS or BIM-grade geometry.
- Another prior NVIDIA LongSplat run used verified 769 source frames on a Tesla T4 according to its log; invocation alone does not establish completion. MASt3R checkpoint SHA-256 was recorded in weights preflight, with model inference explicitly *not executed*.

### Existing canonical gsplat model: actual RGB + depth rendering (opt-in)

The additive `research_gsplat_render.py` uses the canonical `{'step': int, 'splats': {'means', 'scales', 'quats', 'opacities', 'sh0', 'shN'}}` tensor layout. On GPU execution it uses `torch.load(..., weights_only=True)` to read the original checkpoint without training or modifying it; `gsplat.rendering.rasterization` produces fresh RGB PNG and unscaled depth NPY outputs in a new `runs/geom-*/geometry/gsplat_render/` directory. Inspect-only mode requires no Torch, GPU, or pickle loading.

**Camera JSON contract:**
```json
{
  "frame_convention": "colmap_opencv_world_to_camera",
  "cameras": [{"view_id":"non_training_session_candidate","is_training_image":false,
    "width":1280,"height":720,"fx":900.0,"fy":900.0,"cx":640.0,"cy":360.0,
    "world_to_camera_4x4":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]
}
```

Identity camera above is an **interface fixture, not a real camera estimate**. Supply COLMAP-calibrated or independently specified pose/intrinsics in the *same coordinate frame* as the checkpoint. From `repository/`:
```bash
python -m pipeline.stage_04_5_cams_gs.research_gsplat_render \
  --project-root /workspace --run-id geom-gsplat-render-001 \
  --checkpoint /read-only/canonical/ckpts/ckpt_6999_rank0.pt \
  --cameras-json /read-only/validated_new_camera_path.json
# Add --execute only after confirming a compatible CUDA+gsplat environment.
```

Outputs are candidate rendered views, **NOT** an authenticated held-out NVS benchmark: `is_training_image` is an operator-provided label. Depth is in unverified reconstruction units and must never be passed into Stage 8 as metric truth before measured scale and independent registration. This path was not GPU executed in this handoff.
## 3. Optional alternative SfM backends

`research_backends.py` keeps VGGt and DUSt3R isolated from production Stage 3 and C's benchmark workspace. It neither downloads model weights nor starts GPU compute in default inspect mode.

```bash
python -m pipeline.stage_04_5_cams_gs.research_backends \
  --backend vggt --backend-root /read-only/vggt --images /read-only/images \
  --project-root /workspace --run-id geom-vggt-001
```

`--execute` copies image inputs into a brand-new `runs/geom-vggt-001/geometry/backend_vggt/scene/images` and invokes VGGT's upstream `demo_colmap.py --scene_dir`. `--use-ba` is optional. Validate output `sparse/{cameras,images,points3D}.bin` (or `sparse/0`) before any follow-on use. DUSt3R uses `--backend dust3r --dust3r-adapter /reviewed/colmap_export.py`; no upstream native export is assumed. Required adapter convention: `--scene-dir <directory>`. Neither route is **metric** or **gravity-aligned** by default. GPU integration and COLMAP format quality have not been verified in this environment.

## 4. Meter scale, gravity, Sim3, registration and BIM

Stage 8 owns metric reference with IFC/BIM-world coordinate basis and meters. The existing registration pipeline outputs:

- `data/bim/aligned/<project>/scan_aligned.ply`
- `data/bim/aligned/<project>/bim_reference.ply`
- `data/bim/aligned/<project>/bim_elements.jsonl`
- `runs/<run_id>/reports/registration_report.json`

No existing column, JSON field or identity key was removed or renamed. `research_frames.py` offers pure, opt-in gravity-to-BIM proper rotation, nonreflecting uniform Sim3 validation (reject shear, invalid homogeneous row, reflections, unwanted scale), and point transforms. These helpers do NOT silently change Stage 8's alignment. An accelerometer/IMU up vector or anchored structural vertical is still necessary to certify gravity. A validated Sim3 matrix **alone cannot verify** actual meters; require measured control lengths/anchors and independent residuals. Beware IFC local placements, site georeference, global axis conventions, residuals on **independent** anchors and degenerate planar geometry. Protect Stage 9 consumers from using geometry lacking certified metric scale.

## 5. Scientific geometry metrics (research diagnostics only)

`research_metrics.py` computes deterministic sampled point-to-point metrics on **explicitly asserted common BIM meter coordinates**:

- accuracy = directed mean/median/P95 nearest-neighbor distance **scan -> BIM** in meters
- completeness-distance = directed mean/median/P95 **BIM -> scan** in meters
- precision and recall at distance threshold `tau_m`, their harmonic `F-score`, both dimensionless
- symmetric Chamfer L1 **sum** of directed means (m); Chamfer L2 **sum** of squared directed means (m²)
- independent-anchor registration RMSE/median/max (m), if separately provided
- gravity error (degrees), only with operator-supplied up vectors

```bash
python -m pipeline.stage_08_bim_eval.research_metrics \
  --scan-aligned /workspace/data/bim/aligned/site01/scan_aligned.ply \
  --bim-reference /workspace/data/bim/aligned/site01/bim_reference.ply \
  --output /workspace/runs/geom-site01-metrics-001/geometry/metrics.json \
  --threshold-m 0.05 --max-points 100000 --seed 42 \
  --assert-bim-world-meters
```

Optionally `--anchors-json /measured/independent_anchors.json` with `{"scan_aligned_m":[[x,y,z],...], "bim_m":[[x,y,z],...]}`; and `--gravity-scan x y z --gravity-bim x y z`. CSV Stage 9 columns are **unchanged** and not generated here. These point-cloud estimates are sensitive to sampling density, partial visibility, inconsistent scans, BIM LoD, noisy synthetic fallback and missing reference surfaces. Completeness in this report is **NOT a construction completion percentage**. Report dataset/site/capture split and denominators. Ground-truth assessment is Section C's independently owned benchmark; do not contaminate its holdout captures.

## 6. Work status and known blockers

- Implemented: opt-in Nerfstudio resume adapter; existing gsplat file/log auditor; optional VGGT/external DUSt3R wrappers; meter-safe geometric diagnostic functions; tests.
- CPU-only mathematical unit-test scenarios are synthetic, NOT real-site results.
- Historical logs show an actual gsplat render, but its outputs are not hosted as GitHub tracked code/artifacts and were not executed in this patch.
- Not run here: GPU `ns-train`, gsplat restore/eval, VGGT inference, DUSt3R inference, actual Stage 8 registration on site data, independently validated novel-view rendering or BIM geometry accuracy.
- No claim of GPU validation or ground-truth scientific metrics. The coordinator must verify full dependency installation, existing lightweight regression suite, frame units, original asset hashes, split segregation, and baseline vs variants before approving integration.
