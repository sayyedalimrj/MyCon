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


---

# Round 2 addendum — B / Scientific progress (2026-11-12)

This section **supersedes earlier test-status statements for the new Round-2 module tests** but does not retroactively assert that any old Open3D or full repository tests ran. Authoritative contract read from integration: \`repository/docs/parallel_research_round2_contract_20261112.md\` (integration commit \`14f0fcdfa78a95bc3b990a2253a9b27ee8aeacbb\`); coordinator's two PR #37 comments reviewed. **Do not merge or rebase B yourself.**

## Delivered Round-2 code (B ownership only)

- \`pipeline/stage_09_progress/geometry_evidence_adapter.py\`: optional read-only producer-manifest adapter; project-relative SHA-256 sidecars, metric units/frame, independent IFC/scan/registration hash matching, per-sample IFC GlobalId, camera calibration/poses, capture session, audited visibility, segmentation and registration fail-closed. AABB overlap is **never** promoted to authenticated IFC element attribution.
- \`pipeline/stage_09_progress/enrich_scientific_evidence.py\`: independently runnable Stage9 scientific CSV adapter, no Open3D dependency, preserves signed quantities when geometry is blocked.
- \`pipeline/stage_09_progress/evidence_metrics.py\`: provenance-aware quantity, geometry, observability and calibrated quality as **separate** fields. Strict scientific v2 rejects missing date, approval ID, approver, document hash and inconsistent execution/planned units.
- \`pipeline/stage_09_progress/run_progress.py\`: opt-in new manifest via \`progress.geometry_evidence_manifest_json\`, plus trusted SHA/project configuration and dated audit fields; leaves original Stage8 and existing Stage9 columns intact.
- \`pipeline/common/bim_schedule_mapping.py\`: additive BOQ weight approval metadata (original constructors/defaults retained).
- \`pipeline/stage_11_schedule_variance/scientific_progress.py\`: strict BOQ/quantity aggregation, partial-identification bounds, exact as-of approved baseline, WBS and project weights, separate schedule-decision Quality Gates, all per-element audit metadata.
- \`pipeline/stage_11_schedule_variance/run_scientific_progress.py\`: stand-alone primary scientific v2 JSON CLI, input SHA-256, isolated progress-prefixed run ID.
- \`pipeline/stage_11_schedule_variance/run_schedule_variance.py\`: legacy v1 preserved, explicitly logged as **legacy acceptance proxy**, optional distinct \`--baseline-csv\`, nested v2 unchanged schema family.
- Tests: \`tests/test_round2_progress_contract.py\` (28 local passes); old B-owned scientific tests adjusted for stronger provenance requirements.
- For C: \`configs/research_progress/round2_contract_fixture/\` includes immutable synthetic manifest and JSONL sidecars, IFC IDs, schedule/mapping/Stage9 CSV, \`expected.json\`, complete \`scientific_progress.v2.synthetic.json\`; provenance/units/unknown reason codes included.
- Docs: \`docs/research_progress/round2_methodology.md\`; \`docs/research_progress/round2_execution_log_20261112.md\` for commands, dependency versions, exit codes, fixture SHA-256, caveats.

All paths above are prefixed with \`repository/\` in GitHub.

## Executed verification (do not misrepresent)

- \`python -m compileall -q pipeline/stage_09_progress pipeline/stage_11_schedule_variance\`: **exit 0** (present local B modules).
- \`python -m pytest -q -ra tests/test_round2_progress_contract.py\`: **exit 0, 28 passed**, one warning about \`pytest.mark.lightweight\` registration in partial mirror, ~0.10s.
- Stage9 standalone manifest+quantity adapter CLI: **exit 0**, produced 2-element CSV, \`geometry_gate=pass\` for the **synthetic** manifest. This is schema validation, not independent capture validation.
- Stage11 scientific v2 CLI: **exit 0**; synthetic progress unknown, [60,85]% bounds, observability 0.375, approved synthetic planned 50%, variance bounds [+10,+35] pp, \`quality_gate=blocked\`. E2 occluded but has unknown physical execution; no synthetic physical point estimate fabricated.
- The local test tree contains exact Git-blob-verified B source and fixture bytes; **the common mapping/schedule loaders are compatible local mirrors, not original GitHub checkout bytes**. Full original Stage9/11 regression suite, GPU/Open3D geometry, and **real-world end-to-end** were **not executed**. No CI run was available. These remain mandatory coordinator release blockers.
- Python 3.13.5, pytest 9.0.2, NumPy 2.3.5, SciPy 1.17.0, Linux x86_64; no GPU/Open3D tested.

## A→B and B→C interface / unresolved inputs

**Owner A action:** publish canonical \`runs/<run_id>/reports/bim_geometry_evidence.v1.json\` and associated authentic hash-checked element surfaces, registered camera poses and audited visibility. Producer must document metric scale/held-out control error, registration residual and independently verified IFC and scan hashes. **Owner B never changed Stage8.** The current A branch was checked, but no actual v1 manifest or comparable calibrated real survey files were available for direct consumption. The B reader and synthetic contract fixture are ready independently.

**Owner C action:** read the B v2 schema/fixture; score IFC categorical geometry state **separately** from physical fraction and observability, preserve numeric verified fractions even if geometry unknown, report missing counts/denominators and held-out building/session splits, and do not equate identification bounds with CIs. B did not modify C tests/benchmark code.

**Coordinator acceptance gate:** run full relevant lightweight pytest in a **real checked-out GitHub tree**, run Open3D Stage 9 tests, exercise real Stage8 producer-to-B reader with original as-built capture + approved BOQ/quantity/schedule and external independent ground truth. Then review scientific and dashboard consumers; PR #37 remains Draft, unmerged. No valid publication/real-case result is claimed from B software synthetic fixture.

## Unique run IDs and status

- Round-2 local execution ID: \`progress-round2-27\`.
- Schema: \`mycon.bim_geometry_evidence.v1\` (A→B) and \`mycon.scientific_progress.v2\` (B→C).
- \`implemented: true\`, \`unit_tested_local_synthetic: true\`, \`full_repo_tested: false\`, \`gpu_validated: false\`, \`real_case_validated: false\`, \`publication_ready: false\`.
