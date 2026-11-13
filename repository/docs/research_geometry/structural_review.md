# MyCON — finite civil geometry review (Owner A)

`mycon.civil_geometry_review.v1` is a **non-destructive, opt-in read-only consumer** of the existing `mycon.comparison_ready_geometry.v1` Stage-7 reconstruction run. It expects the actual Stage-7 / COLMAP / DA3 **dense or reconstructed physical point cloud** already processed into `clean_cloud.npy` + `clean_cloud.ply`, planes, edges and corners. **Raw Gaussian centres alone are not an acceptable civil-surface input.** No CAD, IFC, BIM or 3ds Max model is necessary.

## Commands

First, run the existing production Stage-7 `comparison_ready` CLI to produce `runs/geom-ready-<source-id>/geometry/comparison_ready.json` using the real dense / mesh-sampled reconstruction and, when available, verified camera metadata. Then, from `repository/`, run:

```bash
python -m pipeline.stage_07_cleanup.structural_review \
  --project-root /workspace/MyCON \
  --base-manifest /workspace/MyCON/runs/geom-ready-<source-id>/geometry/comparison_ready.json \
  --run-id geom-civil-20261113-siteA-001 \
  --input-kind real_reconstruction --seed 17
```

The software **checks every input SHA-256 before creating any output**. It refuses paths outside project root, duplicate run IDs, unsupported schema and a base manifest asserting certified metric scale. Every run is create-only; use a new `geom-civil-*` run ID for any rerun. No `--force` exists. Input PLY, checkpoints, videos, COLMAP and camera files are never edited, deleted or trained from scratch.

## Outputs

All under `runs/geom-civil-*/geometry/`:

- `clean_cloud.ply`, `clean_cloud.npy`: exact byte-for-byte copies of the previously hashed clean cloud, never a modified original.
- `planes.json`, `edges.json`, `corners.json`: old feature IDs and local support retained; explicit schema/frame/units/source SHA, calibrated-confidence null, and manual-review pending. Corner candidate becomes `unknown` when required local/multiscale support is absent. `field_confirmed` is always false. Gravity/camera cannot be inferred from PLY; semantic floor/ceiling/wall labels stay unknown.
- `planes_colored.ply`: **separate** review-only colored cloud; gray=unassigned, plane palette=source plane membership. Does not replace source RGB.
- `geometry_qa.json`: input kind, unknowns, point denominators, source-unit residual/support diagnostics, limited fixed-inlier plane refit sensitivity to half density, additive noise and spatial occlusion. **Not a detection recall, accuracy estimate, or statistical confidence interval.**
- `viewer_overlay.json` and self-contained offline `geometry_review.html`: local rotatable/clickable Shape 3 overlay of observed plane boundaries, finite edge segments, corner states and a bounded sample of the cloud, without externally hosted JS. Source per-sample indices remain in full JSON rather than bloating viewer files.
- `shape3_view_01.png`, `shape3_view_02.png`, `shape3_view_03.png`: three rendered source-cloud views. These become *real-reconstruction images only if source is actually a real reconstruction*. Synthetic fixture images are conspicuously labeled `SYNTHETIC`.
- `components.csv`: one row per feature, support count, source-unit residual where applicable, unknown calibration status and review state. `manual_review_template.csv`: external operator decisions without altering the generated evidence.
- `civil_geometry_manifest.json`: all produced file paths relative to project root and SHA-256, source run, source manifest hash, coordinate frame, units, software versions, seed and nonmetric flags. Avoid self-hashing the manifest.

## Scientific interpretation / limitations

`metric_scale_verified=false` and `registration_verified=false` are immutable gates. Plane hulls are **outer mathematical bounds of observed points, not solid filled scans**; the occupancy grid is explicitly diagnostic, can have gaps and cannot be promoted to a complete measured patch. Physical surface confidence, image-level occlusion, floor/ceiling identity, element identity and progress remain unknown. The source Stage-7 edge detector limits edges to locally jointly supported finite intervals; this companion does not extrapolate missing edges. A corner marked `observed_supported` is still only a **locally supported candidate**, not a surveyed or human-confirmed physical corner.

The fixed-support stress checks quantify *fit sensitivity* (absolute normal angle under ± sign, source-unit residual) only. They are not full re-detection under independently perturbed reconstruction nor a held-out precision/recall study. Full noise/density/occlusion detection and paper-style confidence calibration belong to an independent C benchmark after A supplies real reconstructed data and independently validated GT.

## Current evidence

Local isolated prototype, Linux CPU/Python 3.11, `PYTHONPATH=. pytest -q tests/test_research_geometry_structural_review.py`: 7 synthetic tests passed, 0 failed, exit 0. The tests instantiate hashed synthetic source files, validate review outputs / SHA, unknown state, source tampering, unique run guard, gravity gate and sensitivity denominators. This is **not a full GitHub checkout/CI or real-site test**. GPU, CUDA/gsplat, large COLMAP dense site files, independently surveyed scale/camera/GT, and published accuracy remain `not_executed` / blocked.