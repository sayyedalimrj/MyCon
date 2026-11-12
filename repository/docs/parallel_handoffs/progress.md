# MyCON B — Progress & Schedule Research Handoff
Date: 2026-11-12
Owner: B / Issue #34
Head: `research/mycon-progress-20261112`
Target: `research/mycon-integration-20261112` (PR only; **do not merge**)
Base: `33f1b68b2dc099db97d9e9101f56dd5ce2100d3f`
Last code/fixture commit before this handoff: `669f657474d47b1601bbb4d44406c13c8f80ea24`.
The PR head includes this handoff commit after that SHA.

## Ownership and files changed (all B-owned)

- `repository/pipeline/stage_09_progress/point_assignment.py` — new; exclusive element AABB attribution, ambiguous unassigned, element-local NN distances.
- `repository/pipeline/stage_09_progress/evidence_metrics.py` — new; independently sourced physical quantity, observability, and quality.
- `repository/pipeline/stage_09_progress/run_progress.py` — exclusive scan/BIM point attribution, additive CSV fields, ambiguity counts and summary.
- `repository/pipeline/stage_09_progress/progress_interpretation.py` — stops treating legacy geometric coverage as independently audited visibility.
- `repository/pipeline/common/bim_schedule_mapping.py` — additive optional `weight_basis`; invalid explicitly approved weights no longer silently default to 1.
- `repository/pipeline/stage_11_schedule_variance/scientific_progress.py` — new; verifiable quantity+BOQ rollup, unknown, partial identification bounds, approved baseline and schedule variance.
- `repository/pipeline/stage_11_schedule_variance/run_schedule_variance.py` — additive nested v2 payload and independent isolated v2 JSON with `progress-` run IDs. Labels legacy v1 acceptance-score and Wilson limitations, preserves original v1 JSON structure/fields.
- `repository/tests/test_stage_09_progress_interpretation.py` — adjusted for audited visibility and unknown visibility regression.
- `repository/tests/test_research_progress_geometry.py` — new geometry, overlap/occlusion/registration/quantity unit tests.
- `repository/tests/test_research_progress_scientific.py` — new weighted progress, unknown, bounds, approved schedule, CLI versioning unit tests.
- `repository/docs/research_progress/methodology.md` — exact semantics, assumptions, review gates, Stage 8 interface request and scientific references.
- `repository/configs/research_progress/synthetic_sample/schedule.csv`
- `repository/configs/research_progress/synthetic_sample/mapping.csv`
- `repository/configs/research_progress/synthetic_sample/elements.csv`
- `repository/configs/research_progress/synthetic_sample/expected_fields.json`
- `repository/docs/parallel_handoffs/progress.md` — this report.

No files in `repository/pipeline/stage_08_bim_eval`, Stage 1–8, Colab, C benchmarks, GUI, default requirements, `main` or integration branch were changed by B.

## New API, semantics and versioning

- Stage 9 element CSV **adds** `physical_progress_estimate` (fraction [0,1], empty if unknown),
  `observability_ratio` ([0,1], audited visible-design-surface fraction, empty if unknown),
  `evidence_confidence` ([0,1], only with calibrated provenance, empty if unknown),
  `physical_progress_status`, `geometry_evidence_state`, `scientific_metrics_schema`,
  `assigned_scan_point_count`, `ambiguous_scan_point_count`,
  `ambiguous_bim_point_count`, `scan_candidate_point_count`,
  and preserved quantity/audit provenance columns.
- Stage 11 standalone `mycon.scientific_progress.v2` JSON is the **authoritative scientific** estimator.
  Nested `scientific_progress_v2` is appended to all v1 JSON outputs. Legacy
  `actual_percent_complete` continues to be a **deprecated status proxy** for frozen
  dashboards, explicitly labeled `legacy_progress_validity`; its old Wilson
  intervals do **not** quantify weighted fractional progress.
- Valid element physical fraction requires `planned_quantity>0`,
  `0<=executed_quantity<=planned_quantity`, `quantity_verified=true`,
  an approved provenance and a nonempty shared `quantity_unit`.
- Activity percent requires all mapped elements' verified quantity and positive
  weights with a uniform approved `weight_basis`.
  If some quantities are missing and the weights are valid, return **no point
  estimate** but deterministic [lower, upper] *identification bounds*. These
  are not CIs and no 95% coverage is claimed.
- Actual-vs-planned comparisons require approved
  `approved_planned_percent`/`baseline_source`/`baseline_id`.
  Schedule `percent_complete` and linear time interpolation are not silently
  promoted into an approved baseline.
- Run IDs for v2 use `progress-*`. Example:
  `progress-synthetic-demo`; default runner creates a fresh ID.
  v2 standalone JSON does not overwrite an existing research run.

## Test and evaluation evidence — report actual status, not inferred success

- **Independent mathematical smoke check: PASS**, executed in the available local
  Python runtime on a synthetic two-element example:
  element E1 approved ratio 8/10 with weight 3; E2 unknown with weight 1.
  Expected/checked activity point estimate `null`, physical bounds
  `[60,85]%`, approved planned 50%, variance bounds `[+10,+35]`
  percentage points. A floating point `60.00000000000001` is accepted
  with tolerance, not exact equality.
- **Repository pytest suite: NOT EXECUTED**. New isolated pytest test files
  and patched existing Stage 9 test were committed, but this execution
  environment cannot fetch the repository into its Python container (GitHub
  DNS/network unavailable). GitHub connector can inspect/commit code but does
  not execute Python. No false claims of passing pytest or CI.
- **Stage 9 Open3D end-to-end geometry:** NOT EXECUTED; aligned real scan,
  IFC, GPU/registration evidence and local repo environment not available.
- **Stage 11 CLI as committed:** NOT EXECUTED; proposed invocation in methodology.
- **Real-site evaluation:** NOT EXECUTED; no approved site quantity, camera
  visibility audit, registration QA or real baseline attached to this branch.
- **Scientific calibration / held-out site benchmark:** NOT EXECUTED; no
  independent reference/ground-truth and no independent test split.
- **Hardware/seed:** local arithmetic smoke: CPU-only, Python runtime,
  deterministic arithmetic (no RNG); benchmark GPU/hardware/seed not applicable.
- **Sample counts:** synthetic fixture 1 activity, 2 mapped IFC IDs, 1
  signed/verified synthetic quantity, 1 intentionally unknown quantity.
  No measured real-world accuracy, recall, MAE, ECE, latency or runtime claims.

Suggested commands to execute in a checkout rooted at `repository/`:

```bash
python -m pytest -q -m lightweight tests/test_research_progress_geometry.py tests/test_research_progress_scientific.py tests/test_stage_09_progress_interpretation.py tests/test_stage_11_schedule_variance.py
python -m pytest -q tests/test_stage_09_progress.py tests/test_stage_09_sampling_helpers.py
python -m pipeline.stage_11_schedule_variance.run_schedule_variance \
 --schedule-csv configs/research_progress/synthetic_sample/schedule.csv \
 --mapping-csv configs/research_progress/synthetic_sample/mapping.csv \
 --element-metrics-csv configs/research_progress/synthetic_sample/elements.csv \
 --activity-progress-json /tmp/mycon-progress-B/activity.v1.json \
 --schedule-variance-json /tmp/mycon-progress-B/schedule.v1.json \
 --dashboard-summary-json /tmp/mycon-progress-B/dashboard.v1.json \
 --data-date-utc 2026-01-16 \
 --research-run-id progress-synthetic-demo
```

## Synthetic input hashes (SHA-256 of exact UTF-8 repo fixture bytes)

- `schedule.csv`: `9d4af8b6f94e451fe606e1473a39cc83b33eb801f7c668ab6e028a7cfb9fadb3`
- `mapping.csv`: `8dee6d5cdea54a08112f2d9fdcb23301f0d3174e1b3a74c575b05827f59a7589`
- `elements.csv`: `c92520f71fa200696c9fd099d93e2454b6216eb79e803a90c5991f0d13d9c861`

Generated output paths on executing the sample:
`/tmp/mycon-progress-B/activity.v1.json`,
`/tmp/mycon-progress-B/schedule.v1.json`,
`/tmp/mycon-progress-B/dashboard.v1.json`,
`/tmp/mycon-progress-B/progress-synthetic-demo/scientific_progress.v2.json`.
They are **expected paths**, not claimed to exist or have been produced.

## Ownership-safe interface request to owner A / lead (integration blocker)

Stage 8 currently publishes globally aligned PLY plus coarse
`bim_elements.jsonl` AABBs. Exact attribution/occlusion estimation is
impossible for overlapping elements using AABBs alone. Owner A should
propose an optional backward-compatible Stage 8 export:

1. IFC `global_id` for **each** BIM surface sample/triangle, with
   unique deterministic sample IDs and geometry version/hash;
2. synchronized calibrated camera intrinsics/extrinsics, registered poses,
   per-view depth/z-buffer or ray visibility metadata, capture session ID;
3. confidence/quality of registration, same frame and metre units, explicit
   unassigned/occluded/ambiguous semantics.

The B branch intentionally does not modify Stage 8. Lead should review
the contract before merging A and B.

## Integration / limitations

- Run light tests and synthetic CLI before merge. Verify all changed files
  match B ownership; do not merge automatically.
- AABB exclusivity removes duplicate evidence but can reduce coverage when
  expanded boxes overlap. This is a conservative **fail-closed** measure,
  not an element identity solver.
- V1 remains scientifically unsafe for physical-progress research; consumers
  must explicitly use v2. A future coordinated GUI/API migration requires
  lead approval, outside B ownership.
- Wilson intervals remain in legacy geometry/acceptance fields for
  compatibility; **not** used for v2 weighted physical progress.
- A source string indicating `quantity_verified` or
  `heldout_calibration` is only a data contract, not proof the site
  signed or the calibration was independently validated.
