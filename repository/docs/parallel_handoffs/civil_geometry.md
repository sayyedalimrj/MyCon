# A / Civil Geometry — independent handoff (2026-11-13)

## Ownership / target

Project: `sayyedalimrj/MyCon`, topic branch `research/mycon-civil-geometry-20261113`, proposed PR target `research/mycon-integration-20261112` (never `main`). The earlier A/B/C PRs were already merged to Integration. **No attempt to rewrite their history, merge, rebase, or update main**. The unit of work is new A-owned `stage_07_cleanup/structural_review.py`, corresponding `test_research_geometry_*.py`, `docs/research_geometry/structural_review.md`, and this file; no B/C/common code touched.

## Immutable source and provenance

Inputs are a Stage-7 comparison-ready run's existing SHA-256-validated clean cloud and finite plane/edge/corner records, which are **read only**. Source `MyCon_R4` video, checkpoints `ckpt_6999_rank0.pt`, `resume_step_006999.pt`, historical COLMAP and the original reconstruction remain unaltered. No from-scratch training, forced writes or destructive processing. Independent `geom-civil-*` run IDs and create-only output paths.

## Code, actual tests and evidence

Implementation source and synthetic tests: see GitHub branch / PR and `repository/docs/research_geometry/structural_review.md`. Local standalone CPU test command: `cd repository && PYTHONPATH=. pytest -q tests/test_research_geometry_structural_review.py`: **7 passed, 0 failed, exit 0**; Python version and SHA256 of the final source should be recorded from the actual final checked-in commit. This does **not** assert complete repository checkout compatibility, GPU validation, real field geometry, or independently measured accuracy.

Inputs: artificial trihedral planes used solely in an isolated offline fixture (no train/test leakage inference). Outputs: generated `clean_cloud.ply/.npy`, support and confidence-gated `planes/edges/corners.json`, `planes_colored.ply`, `geometry_qa.json`, `viewer_overlay.json`, interactive `geometry_review.html`, 3 Shape3 PNGs, `components.csv`, `manual_review_template.csv`, and `civil_geometry_manifest.json` with SHA256. A sample synthetic artifact bundle is available separately in the user session; no source room/frame/site data used.

## Open integration and scientific blockers

- Full real checked-out repo tests and Stage3–8 regression: **not executed in this offline prototype**.
- Real video/COLMAP dense reconstruction, camera poses/source-frame overlay: **blocked_missing_accessible_real_reconstruction**. Historical large source files must be processed on the source-side Drive/Colab/server; never substitute Gaussian PLY centres as a civil surface.
- Gravity up / floor-vs-ceiling, metric scale, real construction corners, uncertainty calibration, surveyed GT and photo-based visibility: **unverified**, therefore labels and physical progress stay unknown.
- Re-detection under noise, occlusion, densities, GT-annotated corner precision/recall: **not executed in this companion**, and should be audited by independent validation C. The current noise/occlusion tests only refit fixed existing planar support.
- Viewer approval and study/report acceptance require a human-review CSV plus real source-image evidence and independent survey. No fabricated Accuracy/Completeness/F1 numbers or geometry superiority claim.

## Downstream interface to coordinator

Additive, versioned, read-only `mycon.civil_geometry_review.v1`; no existing Stage 8/9/11 fields changed. Consumers must validate SHA256, schema version, path, input-kind, units, coordinate frame, and explicitly false `metric_scale_verified` before use. They must not translate `observed_supported` into construction completion or into field-certified dimensions. Cross-team consumer changes require coordinator interface PR; none were attempted here.