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
