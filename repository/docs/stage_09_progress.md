# Stage 9 Progress Metrics

Stage 9 consumes the aligned scan/BIM outputs from Stage 8 and produces deterministic progress artifacts for Stage 10.

## Inputs

- `data/bim/aligned/site01/scan_aligned.ply`
- `data/bim/aligned/site01/bim_reference.ply`
- `data/bim/aligned/site01/bim_elements.jsonl`
- `runs/<run_id>/reports/registration_report.json`
- `data/bim/design/schedule.csv`
- `data/bim/design/element_activity_map.csv`

## Outputs

- `data/bim/metrics/site01/element_metrics.csv`
- `data/bim/metrics/site01/activity_progress.csv`
- `data/bim/metrics/site01/deviation_summary.json`
- `data/bim/metrics/site01/coverage_summary.json`
- `data/bim/metrics/site01/registration_quality.json`
- `data/bim/metrics/site01/deviation_map.ply`
- `runs/<run_id>/reports/progress_summary.json`
- `runs/<run_id>/reports/progress_dashboard.html`

## Important interpretation rule

If Stage 8 registration confidence is low, Stage 9 still writes deterministic files for pipeline validation, but it marks element and activity confidence as low. This prevents synthetic/demo BIM runs from being interpreted as real project progress.

## Visibility-aware interpretation

Stage 9 now separates metric completion from visibility-aware interpretation.

Important rule:

- Low coverage with unknown visibility is `not_evidenced`.
- Low coverage in a visible area is `not_observed_in_visible_area`.
- Low registration remains `uncertain_low_registration`.
- Stage 9 does not claim definitive `not_built` from absence alone.

Additional output fields:

- `visibility_confidence`
- `visibility_status`
- `visibility_evidence_status`
- `construction_state_interpretation`
- `visibility_decision_risks`
