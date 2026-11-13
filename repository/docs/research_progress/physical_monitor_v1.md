# MyCON B — Physical monitor Stage 09→11

Schema: mycon.physical_monitor.v1. Scope: only B-owned Stage 09/11 modules, fixtures, tests and handoff. Do not modify MyCon_R4, video, COLMAP, .pt checkpoints, original BIM/IFC, source captures, or A/C ownership paths. Outputs use a unique progress-* directory; no overwrite.

## Scientific meanings

Per IFC global_id, numeric physical_progress_estimate is a fraction [0,1] from an independently approved, dated, unit-compatible executed_quantity / planned_quantity with reviewer and document SHA256. The previous geometry coverage, AABB proximity, old status-score and Wilson CI are NOT physical progress estimates.

- built: signed full planned quantity; does not imply certified workmanship.
- in_progress: signed partial positive quantity.
- not_built: independently signed zero quantity AND separate documented absence_verified=true, absence_evidence_ref, absence_document_sha256 and absence_approved_by.
- unknown: no valid signed quantity, duplicate identity, inconclusive zero or other missing proof. not_evidenced, occluded, out_of_fov, low registration and no scan hits NEVER mean not_built.

Independent numeric zero may coexist with categorical unknown and must not be discarded in numeric evaluation. Quantity/observability/calibrated confidence are distinct metrics.

Geometry via mycon.bim_geometry_evidence.v1 is optional and must pass independent IFC+scan+registration hashes, verified metric scale (m), bim_world frame, camera/visibility QA and IFC identity. Only then do output rows carry audit-only source-image SHA refs and surface sample IDs. Generic plane/edge/corner hypotheses are diagnostics, not field quantities or IFC identity.

Activity and WBS rollups reuse mycon.scientific_progress.v2 with approved BOQ/project weights and an approved dated baseline. Missing quantities imply deterministic partial-identification bounds, never statistical 95% CIs.

## Five output files

1. element_status.csv: mycon.element_status.v1, IFC global_id, construction_status, activity IDs/WBS, quantity gate, reason codes, occlusion, independently validated image and surface sample references.
2. element_metrics.csv: original Stage9 metrics plus additive scientific fields and construction_status, retaining old field names.
3. activity_progress.csv: mycon.activity_progress.v1, weighted approved percent [0,100], identification bounds, schedule plan and variance.
4. schedule_variance.json: mycon.physical_monitor.schedule_variance.v1, approved baseline, variance, interval and gate.
5. progress_dashboard.json: mycon.physical_monitor.dashboard.v1, WBS rollup, reasons, metric/registration gates, full scientific_progress.v2, optional GT metrics and provenance.

Every output has common run ID/data-date. Input artifacts have project-relative paths and SHA256. Dashboard includes output SHA256 for first four files; a file cannot self-embed its hash, so its SHA256 is computed externally.

## Synthetic-only fixture command (from repository root)

    python -m pipeline.stage_11_schedule_variance.run_physical_monitor \
      --element-metrics-csv configs/research_progress/round2_contract_fixture/elements_stage9.csv \
      --mapping-csv configs/research_progress/round2_contract_fixture/mapping.csv \
      --schedule-csv configs/research_progress/round2_contract_fixture/schedule.csv \
      --project-root . --data-date-utc 2026-01-16T00:00:00Z \
      --output-root runs/b-progress-monitor --research-run-id progress-monitor-synthetic-01

Expected for synthetic fixture, NOT claimed executed: 2 elements, states in_progress and unknown, quantity-based project point estimate null, partial-identification bounds [60%, 85%], planned 50%, schedule-decision quality gate blocked. Real A-produced manifest additionally requires --geometry-manifest, --expected-ifc-sha256, --expected-scan-sha256, --expected-registration-sha256, --expected-project-id. Invalid Stage8 gates disable geometric evidence, never independently signed quantities.

## Optional independently audited truth / expert

--ground-truth-csv with columns global_id,ground_truth_status,ground_truth_fraction,split,site_id,capture_session_id,reviewer,verified_at_utc,document_sha256. Emits per-class precision/recall/F1 (built/not_built/in_progress), macro F1 and physical fraction MAE, with explicit denominators, missing/unknown predictions and abstentions. Unknown predictions are false negatives; numeric physical fraction is scored separately. Missing independent GT means metric=null, not zero.

--expert-progress-csv with activity_id,expert_progress_pct,split,site_id,capture_session_id,reviewer,verified_at_utc,document_sha256. Paired comparison with existing approved activity percentages and MAE in percentage points, only when both fields exist. Never impute expert reports into predictions.

Test and train must not share site_id. Reviewer/date/doc SHA fields are required, but approval truth cannot be established solely by textual assertions; independent verification is needed.

## Lightweight test commands

    cd repository
    python -m compileall -q pipeline/stage_09_progress pipeline/stage_11_schedule_variance
    python -m pytest -q tests/test_research_progress_monitor.py tests/test_research_progress_scientific.py tests/test_round2_progress_contract.py tests/test_stage_11_schedule_variance.py

No physical construction or held-out accuracy claims without real as-planned IFC, capture, independently verified registration/metric scale, signed BOQ/measures, exact schedule baseline and independently held-out source truth.


## Round-2 engineering implementation (2026-11-14)

- Construction status (built / not_built / in_progress / unknown) is INDEPENDENT of optical observation status (observed / not_observable / unknown, with occluded vs out_of_fov as distinct causes). A certified executed quantity can remain known when a camera cannot see an installed element.
- Numeric zero requires dated approved quantity. Categorical not_built requires a separate absence_verified=true, absence_evidence_ref, absence_document_sha256, absence_approved_by and absence_approved_at_utc not later than the data date. Legacy not_evidenced and zero scan points cannot create not_built.
- Optional --ifc-index-manifest consumes a read-only mycon.ifc_identity_index.v1 with source-relative immutable IFC file hash, real 22-character IFC compressed GlobalIds/classes, source-to-metre conversion, approved index provenance and IfcOpenShell parser verification. Matching index JSON and IFC SHA by themselves do NOT certify identity. Missing parser, mismatched class/units or absent IFC blocks ifc_identity_quality_gate and forces ifc_linked_construction_status=unknown without discarding independently approved measured fractions.
- All five legacy named output artifacts remain unchanged. progress_dashboard.json now embeds engineering_audit_schema_version=mycon.progress_engineering_audit.v1 and observability_report, approved_quantity_report, wbs_boq_weight_report, uncertain_elements_report, manager_summary. These report camera image SHA refs, partial bounds, WBS/project weights, nullable activity percent and date-approved baseline without producing invented quantity/geometry.
- Real operator protocol (not a field trial): docs/research_progress/4d_bim_field_case_protocol.md with real IFC/MEP/occluded/unobserved cases and independent BOQ, survey, camera, schedule and held-out expert source requirements.

## Actual CPU and synthetic evidence, source commit c073a7cc43c4b46ee3c845400372a7808d5c36f2

Local code archive from authentic GitHub Actions was extracted; all six relevant B source/test files have git blob SHA matching the PR branch exactly. Environment Python 3.13.5, pytest 9.0.2, CPU. Executed commands and exit results:

- pytest -q tests/test_research_progress_monitor.py tests/test_research_progress_scientific.py => 36 passed, exit 0.
- pytest -q tests/test_round2_progress_contract.py tests/test_stage_11_schedule_variance.py tests/test_stage_09_progress_interpretation.py tests/test_research_progress_geometry.py => 68 passed, exit 0.
- python -m compileall -q pipeline/stage_09_progress pipeline/stage_11_schedule_variance => exit 0.
- Synthetic Stage9-to-11 CLI with authored synthetic A manifest, expected synthetic IFC/registration hashes, approved sample quantity/BOQ/baseline => exit 0, five files produced; statuses 1 in_progress and 1 unknown; optical states 1 observed and 1 not_observable; physical point estimate null, partial-identification bounds 60–85%, planned 50%, decision blocked. No independent ground truth: categorical Precision/Recall/F1 and numeric MAE null. SHA output/input receipts are recorded in the B handoff.

GitHub Actions on same B code commit (pull_request merge checkout with Git-blob-identical B files): specialized CPU https://github.com/sayyedalimrj/MyCon/actions/runs/43507962773 => 24 passed; Integration CPU https://github.com/sayyedalimrj/MyCon/actions/runs/43507962877 => 143 passed, 1 skipped. Both exit 0 on Python 3.11. Official Open3D Jupyter global registration notebook was inspected but NOT EXECUTED locally because open3d is missing and external GitHub DNS unavailable. MyCON synthetic CLI execution is distinct from external notebook.

G2–G5 real field and publication gates remain blocked: no real as-built/IFC site case, heldout independent surveyed dimension/quantity/BOQ, real expert GT, or site precision/recall/F1 progress MAE. Field release is never declared passed automatically.
