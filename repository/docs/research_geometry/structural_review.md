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


## Round 2 — 2026-11-13 | real-source nonmetric geometry candidates

### Research obligation
See [evidence_review.md](evidence_review.md): five peer-reviewed DOI-verified methods, pinned official Open3D/2DGS revisions and license boundaries, official runnable examples, explicit no-GPU/not-executed blockers, engineering claim-to-evidence table.

### Safe direct input when Stage7 clean cloud is unavailable
New **opt-in candidate adapter**, `pipeline.stage_07_cleanup.research_dense_candidates`, uses only a **non-Gaussian PLY** obtained from actual dense COLMAP or depth-fused camera reconstruction (or independently sampled mesh), preserving XYZ+RGB, its SHA-256, arbitrary source units and unknown frame. Research RANSAC is not a substitute for verified segmentation. It breaks disconnected planar occupancy regions; candidate intersections require finite nearby points on *both* patches. Trihedral corners require local three-plane/edge evidence and remain `candidate_extrapolated` without verified image observations and surveyed independent controls. No semantic floor/wall/ceiling or metric dimensions are invented.

Run locally from `repository/`:

```bash
python -m pipeline.stage_07_cleanup.research_dense_candidates \
  --input-ply /drive/read_only/depth_fused_candidate_nonmetric.ply \
  --source-receipt /drive/read_only/depth_fusion.json \
  --source-kind depth_fused_candidate_nonmetric \
  --project-root /separate/mycon_civil_workstream \
  --run-id geom-ready-real-depth-local-unique \
  --threshold 0.09 --min-points 220 --max-planes 9 --trials 100 --max-sample 14000

python -m pipeline.stage_07_cleanup.structural_review \
  --project-root /separate/mycon_civil_workstream \
  --base-manifest /separate/mycon_civil_workstream/runs/geom-ready-real-depth-local-unique/geometry/comparison_ready.json \
  --run-id geom-civil-real-depth-local-unique --input-kind real_reconstruction
```

These numeric thresholds are a **site-specific exploratory example in source reconstruction units**; they must not be interpreted as 9 cm. Existing runs are never overwritten; changing parameters requires another Run ID. `research_dense_candidates` explicitly refuses `gaussian_centers` and unsupported source kinds. A caller cannot force `real_reconstruction` on a synthetic or unknown `comparison_ready` manifest.

### Optional source-image evidence (not available for the current real run)
`--camera-index /project/path/camera_index.json` may be added to `structural_review`. Input schema `mycon.camera_evidence_index.v1` requires a JSON object with `coordinate_frame`, `units` **identical to the base**, and `cameras[]` entries each containing: `view_id`, `image_path` (inside project root), `image_sha256`, `width`, `height`, `frame_convention='colmap_opencv_world_to_camera'`, `calibration_verified=true`, `world_to_camera_4x4` rigid pose, and calibrated intrinsics `fx,fy,cx,cy`. The image must decode correctly as PNG/JPEG/WEBP and match declared width/height. Any file SHA, path or pose mismatch blocks **before** an output folder is created. Feature-centroid/edge-midpoint/corner projections can then include `supporting_images` and click-through localized source photographs, but status stays `projected_in_fov_occlusion_unverified` until audited visibility/human review. The offline HTML viewer marks projections with a red cross; no calibration/camera/image exists for the actual Nov 6 depth fusion trace, so current real overlays remain empty/unknown.

### Actual real-derived source result and caveat
Read-only Google Drive `depth_fused_candidate_nonmetric.ply` was obtained from earlier `10_confidence_depth_fusion` processing. SHA256 `f86c1f706ec951a1120eb748122e54dd75cee5b3fac80135efcb5cdee844e1ef3ee`; 104,161 point vertices. Its receipt explicitly says `colmap_nonmetric=true`, `false_positive_geometry_possible=true`, and the narrow-baseline/unobserved-view certification is false. This **does not** qualify as surveyed dense GT or a faithful scan of complete room structure.

Real-derived exploratory run identified **15** finite plane hypotheses, **49** locally supported candidate edge segments, **7** trihedral extrapolation candidates, **0** verified/observed-supported corners, and **57,850** unassigned points. Three Shape3 viewpoints, colored PLY, clickable Canvas viewer, `components.csv`, `stability_report.csv`, `manual_review_template.csv` and a SHA-verified `civil_geometry_manifest.json` were produced. Their source is real-derived but their **physical interpretations have not passed engineer review**; the preview is visibly fragmented and cannot be interpreted as a trustworthy floor/wall layout. All semantics unknown, field confidence uncalibrated, scale false. For exact SHA ledger and test/CLI receipts see [real_depth_execution_evidence_20261113.json](real_depth_execution_evidence_20261113.json).

### Independent limitations and correct next action
1. Missing true production-quality dense/mesh with calibrated time-matched camera image indices and intrinsics; do **not** upgrade visual plausibility to evidence.
2. Same-feature refit stability (density/noise/occlusion) is a *sensitivity proxy*; not an independent re-detection sensitivity or calibrated confidence interval. Side-to-side camera-angle repeatability on actual data is **not_executed**.
3. Region occupancy grid/convex hull can still bridge unobserved internal gaps; the colored Shape3 is a diagnostic overlay only; unknowns remain unknown.
4. Require signed human acceptance/rejection with original image crop hashes and independent surveyed controls before making construction geometry and quantity/progress claims.
5. Consumer team V owns WebGL; A owns only the additive `viewer_overlay.json`/HTML sample and proposed `mycon.civil_geometry_review.v1` source schema; no V/Integration code edited.
