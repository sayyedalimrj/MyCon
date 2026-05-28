# Reproducibility Guide

This document is the bit-for-bit reproducibility reference for the
**Construction Progress AI+BIM** project. It is targeted at:

- **Thesis defenders** who need to demonstrate "I can re-run the
  numbers from scratch" on a laptop in front of a committee;
- **Q1 reviewers** who want to verify the headline metrics claimed in
  a paper match an artefact published with the submission;
- **Future maintainers** who need to know which inputs determine which
  outputs, what the locked schemas are, and where the random seeds live.

The companion documents are:

- [`docs/end_to_end_finishing_plan.md`](end_to_end_finishing_plan.md) — architectural target;
- [`docs/literature_q1_2024_2026.md`](literature_q1_2024_2026.md) — Q1 literature positioning;
- [`docs/schedule_format.md`](schedule_format.md), [`docs/hitl_workflow.md`](hitl_workflow.md), [`docs/calibration_workflow.md`](calibration_workflow.md) — per-component workflows.

## 1. The reproducibility tripod

We rely on three orthogonal mechanisms; together they make every
artefact in `runs/<run_id>/` traceable end-to-end:

1. **Locked schema versions.** Every JSON output carries a stable
   `schema_version` string (e.g. `dashboard_summary.v1`,
   `calibration_report.v1`, `hitl_correction.v1`,
   `method_comparison.v1`). A reviewer who reads only the JSON can
   tell which exporter produced it and which contract it is supposed
   to satisfy.
2. **Provenance envelopes.** Phase 1 introduced a uniform
   `ArtifactEnvelope` with `config_hash`, `git_sha`, `git_dirty`,
   `seeds`, `inputs[]`, `environment`, and `generated_at_unix`. Every
   stage's report carries one (see
   [`pipeline.common.provenance`](../pipeline/common/provenance.py)).
   Phase 5 added per-artefact `_provenance` blocks for Stage 11 and
   the calibration / HITL endpoints, recording input file paths +
   SHA-256 hashes + byte sizes.
3. **Determinism.** `pipeline.common.determinism.seed_everything(seed)`
   seeds NumPy and (when imported) PyTorch / Open3D random number
   generators. Every stage records the seed it ran with so a re-run
   with the same seed produces identical numbers.

## 2. Verifying a published run

If a paper or thesis ships a `runs/<run_id>/` directory:

```bash
# 1. Re-derive the SHA-256 of every artefact.
python3 - <<'PY'
import hashlib, json
from pathlib import Path

run_dir = Path("runs/<run_id>")
summary = json.loads((run_dir / "reports" / "walkthrough_summary.json").read_text())
for tag, entry in summary["files"].items():
    p = Path(entry["path"])
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    ok = "OK" if digest == entry["sha256"] else "MISMATCH"
    print(f"{ok} {tag} {entry['sha256']} {p.name}")
PY
```

If every file reports `OK`, the directory is a literal copy of the
published outputs. A `MISMATCH` means the file was edited after the
walkthrough_summary was written.

```bash
# 2. Re-run the walkthrough from the committed inputs and compare.
python3 scripts/run_end_to_end_walkthrough.py \
    --output-dir /tmp/repro_check/ \
    --inputs-dir examples/end_to_end/inputs/ \
    --data-date-utc 2026-04-16

# 3. Diff the dashboard summary against the published one.
diff <(jq -S . runs/<run_id>/dashboard_summary.json) \
     <(jq -S . /tmp/repro_check/dashboard_summary.json)
```

The dashboard summary's `kpi` block is fully deterministic given the
three input CSVs and the `data_date_utc`; the only field that legitimately
varies between runs is `data_date_utc`/`generated_at_utc`. The
walkthrough test (`tests/test_end_to_end_walkthrough.py`) asserts the
canonical-data-date invariants in CI.

## 3. The synthetic walkthrough as the canonical fixture

[`examples/end_to_end/inputs/`](../examples/end_to_end) is a tiny but
realistic site committed to the repo:

- 2-activity schedule (Foundations + Floor 2 walls);
- 7 BIM↔schedule mapping rows;
- 7 element rows mixing `likely_completed` / `partially_observed` /
  `not_evidenced`;
- 6 HITL corrections (3 confirms, 3 overrules) so the calibration step
  has real signal.

Determinism guarantees:

- Stage 11 with `--data-date-utc 2026-04-16` always reports 1 activity
  on schedule and 1 behind, with `n_unknown_evidence = 0`.
- Calibration: `n_samples = 6`; ECE / MCE / Brier / smooth-ECE are
  closed-form on the 6 records and bit-stable to floating-point
  precision.
- Grounding-guard demo: 3 answers cover 3 documented failure modes
  (well-grounded, hallucinated numeric, unsupported named entity).

The `walkthrough_summary.json` records SHA-256 of every output, so a
post-hoc comparison is one `diff` away.

## 4. Random seeds and the determinism module

For stages that involve any stochastic component
(NumPy bootstrap intervals, PyTorch model evaluation, Open3D RANSAC
seeds), the recommended pattern is:

```python
from pipeline.common.determinism import seed_everything
seed_everything(42)
```

The function records the seed in the run's provenance envelope so
downstream consumers see exactly which seed was used. Phase 4 modules
that consume seeds — bootstrap intervals in
[`pipeline.stage_09_progress.uncertainty`](../pipeline/stage_09_progress/uncertainty.py),
multi-view fusion conflict accumulation in
[`pipeline.stage_09_progress.multiview_fusion`](../pipeline/stage_09_progress/multiview_fusion.py)
— are themselves deterministic given the seed and the input.

## 5. Schema-version drift policy

Schema versions are deliberately conservative. **Any** change to the
shape of a published JSON requires the schema_version to bump
(`v1` → `v2`); old consumers should keep working unchanged. Conventions:

- A new optional field at the top of an existing schema is a backward-
  compatible addition; bump the **patch** convention informally
  (e.g. document it in `phase_X_summary.md`) but keep the same
  `schema_version` string. Existing readers ignore the new field.
- A required-field rename, a removed field, or a changed numeric
  scale **must** bump `schema_version` to `vN+1` and ship a parallel
  reader for the previous version.

The Phase 4 finishing-layer schemas
(`schedule.v1`, `bim_schedule_mapping.v1`, `activity_progress.v1`,
`schedule_variance.v1`, `dashboard_summary.v1`, `hitl_correction.v1`,
`calibration_report.v1`, `grounding_guard.v1`,
`method_comparison.v1`) are locked. The Phase 5 wrappers
(`hitl_submit_response.v1`, `hitl_list_response.v1`,
`calibration_run_response.v1`, `walkthrough_summary.v1`,
`grounding_guard_demo.v1`, `schedule_import_summary.v1`) inherit the
same policy.

## 6. Method × metric × CI tables for paper drafts

Phase 5 added [`pipeline.common.method_comparison`](../pipeline/common/method_comparison.py)
for paper-side reproducibility. To produce a thesis-grade LaTeX table
from a `method_comparison.v1` JSON:

```bash
python3 scripts/render_method_comparison_latex.py \
    --input  runs/<run_id>/method_comparison.json \
    --output paper/figures/method_comparison.tex \
    --label  tab:method_comparison \
    --also-ascii paper/figures/method_comparison.txt
```

The exporter is deterministic: same JSON → byte-for-byte identical
`.tex` and `.txt` output. Per-metric `decimals` overrides let you
render F-score at 1 decimal and ECE at 3 decimals in the same table
without touching the input.

## 7. CI gates

Three CI gates worth wiring in (each documented in the relevant
workflow doc):

```bash
# Calibration regression gate (returns exit 2 if exceeded):
python3 scripts/run_calibration_report.py \
    --input    runs/<run_id>/reports/hitl_corrections.jsonl \
    --out-json runs/<run_id>/reports/calibration_report.json \
    --max-ece  0.15 \
    --max-brier 0.25

# Lightweight test set (every Phase 1 / 4 / 5 module):
pytest -m lightweight --ignore=tests/test_service_api.py --ignore=tests/test_service_websocket.py

# GUI lint + tests + production build:
( cd gui && npm run lint && npm test && npm run build )
```

## 8. Common pitfalls when re-running

- **Time-zone confusion.** Schedule `planned_start_iso` /
  `planned_finish_iso` columns must be ISO-8601. Naive datetimes are
  treated as UTC; do **not** silently re-emit them as local time after
  editing in Excel (Excel will helpfully convert them, ruining
  `planned_percent_complete_at(when)`).
- **Mixing schema versions across runs.** A `runs/<run_id>/` directory
  produced by an old commit may carry stale schemas. Always check the
  `schema_version` of every JSON before comparing across runs.
- **Random seeds passed via shell vs config.** Stages that take a seed
  via CLI (`--seed`) record it in the provenance block; stages that
  read it from YAML record it in the config hash. Same seed via a
  different path → same output.
- **Floating-point reproducibility across hardware.** NumPy operations
  on x86_64 vs ARM64 can differ in the last few bits of the mantissa.
  Stage 9 / Stage 11 metrics are bit-stable on x86_64 + NumPy ≥ 1.26;
  for ARM64 verification, use the SHA-256 *of the rounded JSON*
  (`jq -c .` followed by `sha256sum`).

## 9. Where to look when something doesn't reproduce

| Symptom | First thing to check |
|---|---|
| Different KPI numbers from the same inputs | `walkthrough_summary.json` SHA-256 hashes; then per-input `_provenance.input_sha256` |
| Different ECE on rerun | The HITL log changed (someone appended a record); inspect the JSONL with a diff tool |
| Different LaTeX table | The input `method_comparison.v1` JSON changed; check `generated_at_utc` |
| Dashboard shows stale numbers | The React Query cache holds the old report; click the **Replay** button on the ReliabilityCard or hard-refresh |
| Schedule variance flips between runs | Activity-id mismatches; run [`pipeline.common.bim_schedule_mapping.validate_mapping`](../pipeline/common/bim_schedule_mapping.py) and inspect `activities_in_mapping_not_in_schedule` |
