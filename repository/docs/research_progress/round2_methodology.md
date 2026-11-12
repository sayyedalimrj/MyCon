# MyCON B — Round 2: scientific progress and evidence interface

**Milestones:** Implemented; 27 local synthetic contract tests passed; no real-site or publication validation. This document describes the *scientific v2* estimator. The v1 acceptance-score average and its Wilson interval are deprecated for physical progress.

## Four independent quantities

| Quantity | Meaning | Unit | When null? |
|---|---|---|---|
| element \`physical_progress_estimate\` | Approved executed quantity / approved planned quantity, never estimated from matched scan points | fraction 0..1 | No signed and dated measurement of consistent units |
| activity/project \`physical_progress_estimate\` | Approved BOQ/cost-weighted executed work | percent 0..100 | Any mapped quantity or required weight missing |
| \`observability_ratio\` | Union of visible IFC surface-sample IDs / union of eligible QA-audited sample IDs over calibrated views | fraction 0..1 | Missing/invalid Stage8 manifest, camera, registration or view support |
| \`evidence_confidence\` | Independently calibrated quality *score*, not probability of completion | scalar 0..1 | No independent calibration reference |
| \`geometry_conformance_state\` | conforming/nonconforming/unknown based on QA-passed metric deviation and tolerance | categorical | No validated geometry assessment |

An occluded eligible surface can have **observability=0** while *construction progress remains unknown*. Out-of-FOV/unknown views have null observability, not zero. A high scan-to-BIM coverage score is neither an executed quantity nor a proven IFC identity. AABB overlap exclusion is conservative candidate screening only.

## Required quantity and BOQ provenance

Each Stage 9 element record must keep \`global_id\`, \`planned_quantity\`, \`executed_quantity\`, \`quantity_unit\`, optional \`planned_quantity_unit\` and \`executed_quantity_unit\`, \`quantity_verified=true\`, and \`quantity_source\` in \`approved_measurement|signed_site_log|survey_qa\`. Units allowed: \`m|m2|m3|count|kg|tonne|lump_sum\`; planned and executed units must be identical. In addition, **research v2** requires \`quantity_record_id\`, \`quantity_approval_id\`, \`quantity_approved_by\`, \`quantity_approved_at_utc\`, \`quantity_as_of_utc\`, \`quantity_document_sha256\`. All timestamps UTC offset-aware. Approval may not postdate the data date; as-of snapshot must match the data-date day. Missing/invalid source, unit, range, date, signer or SHA-looking hash blocks physical progress.

BOQ mapping: \`activity_id\`, \`ifc_global_id\`, finite positive \`weight\`, comparable \`weight_basis=approved_boq|approved_cost|approved_quantity\`, \`weight_approval_id\`, \`weight_approved_by\`, \`weight_approved_at_utc\`, \`weight_document_sha256\`. A hash or a signer label does not verify an actual signature: real-site validation requires separately checking the underlying signed documents.

For verified element \`e\`, \`p_e = executed_quantity_e / planned_quantity_e\`. For an activity with approved comparable mapping weights, \`P_a = 100 × Σ(w_e p_e)/Σw_e\`. If any \`p_e\` is unknown, the *point estimate is null*. Under the explicit assumption \`0 ≤ p_e ≤ 1\` and valid weights, report conditional set-identification bounds:

\`L_a = 100 × Σ_known(w_e p_e)/Σ_all w_e\`

\`U_a = 100 × (Σ_known(w_e p_e) + Σ_unknown w_e)/Σ_all w_e\`

These are deterministic bounds, **NOT Wilson/95% confidence intervals**. They do not account for errors in independently approved quantities. Wilson binomial-score intervals are not valid around a weighted average of ordinal acceptance scores or correlated scan points.

## Time-phased schedule baseline and aggregation

A baseline row requires \`activity_id\`, \`baseline_id\`, \`baseline_source=approved_baseline\`, \`approved_planned_percent\`, \`planned_as_of_utc\`, \`baseline_approved_at_utc\`, \`baseline_approved_by\`, and \`baseline_document_sha256\`. Exactly one approved baseline must match the UTC data date. Neither \`schedule.percent_complete\` (planner-reported actual) nor linear date interpolation is an approved S-curve. Both remain separate, non-authoritative fields.

\`variance_pp = approved actual percent − approved baseline percent\`. With user-specified \`band_pp\` (default 5), \`behind\` if variance is below \`-band\`, \`ahead\` if above \`+band\`, \`on_schedule\` otherwise. If the point actual or approved planned percent is missing, classify \`unknown_evidence\` or \`unknown_schedule_baseline\` and block the schedule-decision gate; variance may have deterministic bounds \`[L−planned,U−planned]\` without a point estimate. Project and WBS grouping additionally require independently approved, comparable \`activity_weight\`, \`activity_weight_basis\`, \`activity_weight_approval_id\`, \`activity_weight_approved_by\`, \`activity_weight_approved_at_utc\`, \`activity_weight_document_sha256\`. Never weight by raw IFC object count. Current WBS aggregation groups **exact codes**, not unverified descendant hierarchies.

## Stage 8 → Stage 9 contract (read-only, opt-in)

Expected manifest: \`runs/<run_id>/reports/bim_geometry_evidence.v1.json\`; schema \`mycon.bim_geometry_evidence.v1\`. Fields: project/run/date, \`coordinate_frame=bim_world\`, \`units=m\`, \`scan_to_bim_4x4\`, independent-control/metric-scale confirmation, passed registration gate, IFC/scan/registration SHA256 and code/config/seed/dataset provenance.

Three project-relative, SHA256-checked JSONL sidecars:
1. \`surface_samples\`: unique \`sample_id\`, stable \`global_id\`, \`xyz_m\`, \`triangle_id\`/\`surface_id\`, \`geometry_version\`, \`sampling_policy\` and explicit unassigned semantics;
2. \`camera_poses\`: unique \`view_id\`, capture session ID, image SHA256, calibrated \`intrinsics\` (fx/fy/cx/cy), dimensions, \`world_to_camera_4x4\`, pose QA reference;
3. \`element_visibility\`: \`global_id\`, valid view ID, \`state=visible|occluded|out_of_fov|ambiguous|unknown\`, \`eligible_sample_ids\`, \`visible_sample_ids\`, \`method=raycast_zbuffer|audited_manual\` and QA provenance; optional QA-passed \`geometry_assessment\` with max deviation/tolerance in metres.

B validates IDs, sample ownership, duplicate IDs, per-view unions, dimensions, transforms, path safety, hashes, segmentation failure, registration and scale. It **also requires independent expected IFC/scan/registration hashes and expected project ID** to accept the manifest. Failed inputs => geometry fields \`unknown_*\`; independently audited work quantities remain valid. One view leaves \`geometry_quality_gate=limited_view_support_or_unassessed_geometry\` even when an auditable visibility ratio exists. A remains producer owner; B does not modify Stage 8.

## Reproduce from repository root (fixture is synthetic-only)

\`\`\`bash
F=configs/research_progress/round2_contract_fixture
OUT=/tmp/progress-round2-demo
mkdir -p "$OUT"
python -m pipeline.stage_09_progress.enrich_scientific_evidence \
 --element-metrics-csv "$F/elements_stage9.csv" \
 --geometry-manifest "$F/bim_geometry_evidence.v1.json" \
 --project-root . \
 --out-element-csv "$OUT/scientific_elements.csv" \
 --out-summary-json "$OUT/stage9.json" \
 --data-date-utc 2026-01-16 \
 --expected-ifc-sha256 1111111111111111111111111111111111111111111111111111111111111111 \
 --expected-scan-sha256 2222222222222222222222222222222222222222222222222222222222222222 \
 --expected-registration-sha256 3333333333333333333333333333333333333333333333333333333333333333 \
 --expected-project-id mycon-round2-synthetic
python -m pipeline.stage_11_schedule_variance.run_scientific_progress \
 --schedule-csv "$F/schedule.csv" \
 --mapping-csv "$F/mapping.csv" \
 --element-metrics-csv "$OUT/scientific_elements.csv" \
 --out-json "$OUT/scientific_progress.v2.json" \
 --data-date-utc 2026-01-16 \
 --research-run-id progress-round2-demo
python -m pytest -q tests/test_round2_progress_contract.py
\`\`\`

Use a **fresh output path / progress- run ID** for subsequent executions (exclusive create). Expected synthetic output: IFC E1 work fraction 0.8, IFC E2 null, approved weights 3:1, activity/project/WBS bounds 60..85%, point actual null, planned 50%, variance bounds +10..+35 pp, E2 occluded but not unbuilt, weighted observability 0.375, \`unknown_evidence\` schedule status, schedule decision gate blocked. All SYNTH approval IDs/hashes are placeholders, not evidence of real construction.

## B → C benchmark

Use \`mycon.scientific_progress.v2\` as the authoritative schema. Evaluate categorical \`geometry_evidence_state\` **separately** from numeric \`physical_progress_estimate\`, \`observability_ratio\` and calibrated quality scores. Match exact IFC \`global_id\`, \`activity_id\`, \`wbs_code\`, capture sessions and input hashes; keep denominator/missing counts. Do not drop numeric independently approved values when geometry is unknown, nor impute numeric progress from classifications. Evaluate ablations separately for camera dropout, occlusion, failed segmentation, registration misalignment, invalid BOQ weights and missing audit. Group held-out splits by site/building/session and require independent signed quantity/BOQ/schedule truth.

**Unmet release gates:** No actual A-produced artifact or independent GT/control source, no real-site signed BOQ/quantities, no camera calibration accuracy verification or held-out-site scientific evaluation. The synthetic contract tests prove software logic, not scientific superiority or paper readiness.
