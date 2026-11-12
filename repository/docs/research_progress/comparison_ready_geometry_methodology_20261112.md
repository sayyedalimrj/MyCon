# MyCON B — Reference-agnostic geometry and scientific progress (2026-11-12)

**Contract:** repository/docs/integration/model_agnostic_geometry_readiness_contract_20261112.md (Integration branch). **Scope:** no 14021208/MAX/IFC reference was imported, fitted, cropped, or scaled. No A/C source edits. Owner B, PR #37, unmerged.

## A → B: geometry diagnostics only

The optional producer manifest is **mycon.comparison_ready_geometry.v1** with unique run ID prefix geom-ready-. It contains project/run ID, UTC creation time, coordinate frame, declared units (m, mm, cm, reconstruction_unit, arbitrary), source/model/config hashes, code version, seed, software provenance, removed-point ratio and quality data. Each project-relative file is described by path and SHA-256:
- clean_cloud.ply and clean_cloud.npy: original reconstructed XYZ/RGB-preserving review and numeric cloud; Python NPY loader rejects pickle/object arrays.
- planes.json: stable plane_id, signed unit normal, offset, finite local extent, support-sample IDs/count, residual, state and QA gate.
- edges.json: stable edge_id, linked plane IDs, finite endpoints, source support, state and QA gate.
- corners.json: stable corner_id, 3D XYZ, intersecting plane IDs, support and source-image IDs, QA and state.
- optional cameras.json: view_id, capture/session information, source-image SHA and verified camera geometry.

Each feature state is exactly **observed_supported**, **candidate_extrapolated**, or **unknown**. The B reader retains original producer state separately from audit_state. Degenerate trihedral intersections and unsupported observed corners are downgraded to unknown; no infinite-plane extrapolation is treated as a confirmed measured corner.

Reader: repository/pipeline/stage_09_progress/comparison_ready_adapter.py. Its load_comparison_ready_geometry() checks manifest/project provenance, safe paths, size limits, SHA-256, finite numeric arrays, normalized planes, IDs, local geometry, cameras, support and corner degeneracy. A diagnostic bundle with units=arbitrary and metric_scale_verified=false can be **valid for structural review** while BIM matching and physical completion remain **blocked/unknown**. summary().physical_progress_estimate is always null.

Standalone CLI for IFC-free evidence inspection:
    cd repository
    F=configs/research_progress/comparison_ready_fixture
    python -m pipeline.stage_09_progress.run_comparison_ready_diagnostics --manifest "$F/comparison_ready.json" --project-root . --expected-project-id mycon-synthetic-only --out-json /tmp/geom-ready-unique-output.json

## Independent optional IFC association

A generic plane, edge, corner or scan point has **no IFC GlobalId** by default. The B-side separate future link schema **mycon.geometry_feature_ifc_mapping.v1** requires:
- comparison-ready manifest SHA-256, independently supplied IFC and registration-report SHA-256, approved metric control reference, quality-gated registration, an explicit 4x4 reconstruction-to-IFC transform and independent reviewer ID;
- for non-metric source units, positive independent metric_scale_to_m and independent_scale_qa_ref;
- exact feature_id/type → ifc_global_id, observed-supported feature state, non-duplicated ownership, an audited association method (independent_survey_correspondence or audited_ifc_surface_match), quality_gate=pass, and association_qa_ref;
- audited camera view plus visibility method/ref before any visible/occluded/out_of_fov label can be used; otherwise visibility is unknown;
- finite deviation and tolerance (metres), valid association and confirmed visible state for geometric conforming/nonconforming labels.

**Insufficient scale, registration, identifiers, feature support or optical QA → unknown.** Nonconformance is NOT nonconstruction, and occlusion/absence is NOT 0% physical progress. B never calculates observed surface ratio from feature counts.

## Stage 9 → Stage 11 → C

Stage9 scientific CSV CLI adds optional --comparison-ready-manifest, --comparison-ready-project-id, --comparison-feature-ifc-json and --expected-comparison-link-sha256. Stage9 main pipeline adds opt-in progress.comparison_ready_manifest_json and optional trusted feature-to-IFC mapping config. The old optional mycon.bim_geometry_evidence.v1 interface remains independent.

New additive CSV fields: comparison_ready_schema, comparison_ready_manifest_sha256, comparison_ready_valid, comparison_coordinate_units, comparison_coordinate_frame, comparison_metric_scale_verified, comparison_registration_verified, comparison_ifc_quality_gate, comparison_feature_ids, geometric_agreement_state, comparison_visibility_state, comparison_evidence_reason_codes. No prior Stage9/11 keys are removed.

Stage11 mycon.scientific_progress.v2 carries these observations and evidence-presence flags independently of measured quantity and approved weights. **Physical progress = verified, dated executed_quantity / planned_quantity**, with approved BOQ/cost aggregation. Missing records/weights keep point-estimate null and may yield deterministic partial-identification bounds; these are NOT 95% CIs. The approved dated schedule baseline controls variance; missing approvals block schedule decisions. Structural features never contribute to executed quantity or to BOQ weight.

For C: score independent feature/geometry state separately from physical quantity, observability, evidence calibration and schedule. Use stable IDs, units, hashes, missing counts, denominators and separately validated reference truth. Never impute percentages from corner count, edge fit, PLY density or area coverage.

## Synthetic A→B→C fixture and executed test evidence

repository/configs/research_progress/comparison_ready_fixture includes binary NumPy cloud, PLY, hash-checked plane/edge/corner/camera JSON, example manifest, and an **actual executed** synthetic diagnostics JSON. There are 4 finite plane candidates, 1 supported edge, 1 extrapolated corner and 1 **false degenerate corner** that fails audit. Units arbitrary, metric_scale_verified=false, registration_verified=false, physical_progress_estimate=null; BIM association blocked. The example is **not field data**.

Local execution: 21 new pytest cases passed (exit 0), compileall exit 0, audit-only CLI exit 0, invalid manifest and duplicate output both exit 2. See repository/docs/research_progress/comparison_ready_execution_log_20261112.md for exact commands, versions and hashes.

## Open release gates

- No authentic A-produced comparison-ready manifest in A PR #38 yet. B fixture is an **interface proposal** to synchronize with A/C, not a producer implementation.
- No genuine site capture with independent surveyed geometry, metric scale, matched IFC, approved signed BOQ quantities or dated baseline supplied for this new route.
- The local runner has no full GitHub checkout/network; full original Stage9/11 regression and Open3D tests have **not executed** in this round. Prior Round2 tests are documented independently. No real field accuracy, statistical calibration, or publication readiness is claimed.
- Producer QA flags and syntactically valid hashes do not by themselves prove external auditor approval. Independent source verification is mandatory before decision support.
- Legacy v1 acceptance-score/Wilson outputs stay for compatibility but are scientifically unsuitable as physical progress estimators. Use scientific v2.
