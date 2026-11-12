# MyCON Round 2 — Cross-Workstream Interfaces & Release Gates (2026-11-12)

**Authority:** Coordinator on integration branch. This document extends, and does not replace, `repository/docs/parallel_research_contract_20261112.md`. A/B/C must read this from **integration** before further commits. Workstream branches must not push to integration or main; no unauthorized rebase/merge. Current PRs: A #38, B #37, C #36.

## Non-negotiable scientific distinctions

1. **Geometry reconstructed/rendered** is not necessarily metrically scaled or a usable as-built surface.
2. **Observed geometry agreement** is not **construction completion**. No 0% from occlusion/no observations.
3. **Measured quantities / approved BOQ weights** are distinct from geometric coverage and classification acceptance.
4. **Partial-identification bounds** are not 95% confidence intervals; bootstrap CIs are not automatically valid when points/frames are dependent.
5. No held-out, ground-truth or real-world method-superiority claims absent verified files, sample counts, splits and independent measurements.
6. Existing Stage8/9/11 CSV/JSON consumers are kept backward-compatible. New research fields are additive and versioned.

## A → B geometry-evidence interface (optional, opt-in, versioned)

A owns producer; B owns reader; C owns independent validator. Preserve all current Stage 8 outputs. A proposes canonical output manifest:

`runs/<run_id>/reports/bim_geometry_evidence.v1.json`

Required manifest keys:
- `schema_version: "mycon.bim_geometry_evidence.v1"`, `run_id`, `project_id`, `created_at_utc`.
- `coordinate_frame: "bim_world"`, `units: "m"`, `ifc_sha256`, `scan_sha256`, `registration_report_sha256`, `registration_quality_gate`, `scan_to_bim_4x4`, `metric_scale_verified: bool` and independent-control evidence reference.
- `surface_samples`: project-relative file path and SHA-256; owner-produced, immutable, with per-sample XYZ metres, optional normal, stable sample ID, element `global_id`, IFC surface/triangle ID, geometry version, explicit unknown/unassigned IDs and sampling policy. Use NPZ/Parquet or JSONL plus a schema file; document and validate every field. **Do not infer element ownership using AABBs as confirmed identity.**
- `camera_poses`: project-relative file path and SHA-256; per image `view_id`, `capture_session_id`, `image_sha256`, pixel width/height, calibrated intrinsics, 4×4 extrinsics with explicit world-to-camera/camera-to-world convention, covariance or QA, and units.
- `element_visibility`: project-relative file path and SHA-256; records by (`global_id`, `view_id`) or by (`global_id`, `capture_session_id`) with `state` in `visible | occluded | out_of_fov | ambiguous | unknown`, visible/eligible surface sample or area counts, method `raycast_zbuffer | audited_manual`, and QA provenance. A visibility quantity is **null/unknown** if camera pose, occlusion model or registration is insufficient; never set 0 merely because no scan points match.
- `provenance`: code commit, config hash, seed, hardware/software versions, dataset/capture and independent-scaling/control IDs.

**Fail-closed consumption rule:** B may compute visibility-based element observability only when A manifest validates, the shared IFC `global_id`, coordinate frame, camera provenance, scale and registration gate pass. Otherwise `observability_ratio=null`, `geometry_evidence_state=unknown_*`. This does not prohibit independently approved quantity evidence, but never relabels geometry as quantity measurement.

A must commit compact synthetic producer fixture in its owned tests/docs; B must commit consumer contract tests in B-owned paths. C supplies read-only validation of both. For format ambiguity, file an interface proposal to coordinator; no cross-owner file edit.

## B → C progress interface

- Stage9 additive per-element CSV retains `global_id`, `status`, `coverage` legacy and has `physical_progress_estimate` (nullable fraction 0–1), `observability_ratio` (nullable fraction 0–1), `evidence_confidence` (nullable calibrated scalar 0–1), `physical_progress_status`, `geometry_evidence_state`, approved quantity fields, unit/source/reviewer/dates and evidence provenance.
- Scientific Stage11 JSON: `schema_version="mycon.scientific_progress.v2"`; per-activity `physical_progress_estimate` nullable percentage 0–100; `physical_progress_lower_bound_pct`, `physical_progress_upper_bound_pct` (deterministic bounds, not statistical CI); approved baseline ID/as-of data date, `schedule_variance_bounds_pct` and reason codes; project roll-up only with approved compatible weights.
- C must **score categorical state and numeric physical fraction separately**. A categorical unknown cannot drop independently measured numerical fraction when evaluating numeric MAE. Conversely numeric physical fraction must not be imputed from geometry coverage or a classification label.
- C must use same independent IFC/BOQ/survey ground truth, stable `global_id`, same building/session partitions, paired same-sample comparisons, explicit denominators, missing output counts, and scientifically appropriate uncertainty.

## Completion criteria / coordinator release gates

G0 — **Safety & diff:** A/B/C own only their paths, no main changes, no rewrite of original video, scans, existing .pt/.ckpt/PLY, no destructive force; all artifacts have unique run IDs and SHA-256.
G1 — **Runnable source:** import/compile tests + full relevant lightweight pytest and CLI examples actually executed on the checked-out integrated code. Record exact commands, exit codes, dependency versions, logs, and failures. CI or externally captured run logs are acceptable evidence; source-only test definitions are not.
G2 — **Geometry validity:** real existing gsplat checkpoint restored/read-only and calibrated RGB/depth rendered if GPU exists; independent train/test camera validation; Stage8 real scan-vs-IFC registration with calibrated scale, units, axes and independent held-out control-anchor error. A produced versioned surface/visibility artifact, B consumed it without mismatched IDs.
G3 — **Physical progress validity:** one complete sample end-to-end with as-planned IFC + as-built capture + independently approved measured quantities/BOQ and schedule baseline, with explicit unknown and occluded elements, under validated registration.
G4 — **Scientific validity:** C independent real GT; grouped held-out site/building/session splits, declared baselines and ablations, paired evaluation, metrics with denominators, uncertainty/failure analysis, reproducibility; paper tables from actual measured artifacts only.
G5 — **Integrated delivery:** merge A then B then C into integration only after reviewed diff and tests; compatibility + end-to-end smoke runs, snapshot hashes, short reproducibility recipe, limitations and paper-draft tables. Do not promote to main without authorizing final review.

**Release status:** `implemented`, `unit_tested`, `gpu_validated`, `real_case_validated`, `publication_ready` are separate milestones. Never call a milestone complete on the basis of another. Missing real measurements/permission means `blocked_missing_input` or `not_executed`, not `passed`.
