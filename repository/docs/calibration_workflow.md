# Calibration & Reliability — Workflow

This document is the reviewer-facing reference for the calibration
report described in `docs/literature_q1_2024_2026.md` (§4.1 and §5).
The report is the answer to a question every Q1 reviewer asks:

> *Are the confidences your system reports actually trustworthy?*

If your pipeline marks an element as `confidence: high`, the reviewer
expects it to be correct most of the time (e.g. ≥ 85 %), and similarly
for `medium` and `low`. The calibration report measures whether that
expectation is met.

## 1. Inputs

The calibration script consumes a JSON Lines file. Each line is an
object with at least:

```json
{"confidence": "high",   "correct": true}
{"confidence": 0.65,     "correct": false}
{"confidence": "medium", "correct": true}
```

`confidence` may be:

- one of the discrete labels supported by
  `pipeline.common.calibration.DEFAULT_CONFIDENCE_LABEL_PROBABILITIES`
  (`high` / `medium` / `low_to_medium` / `low` / `unverified`);
- or any numeric value in `[0, 1]` (clamped if out of range).

`correct` may be:

- a Python `bool` / 0 / 1;
- a string `"true"` / `"false"` / `"yes"` / `"no"` / `"accepted"` /
  `"rejected"` (case-insensitive).

Anything else is dropped silently from the calibration set so a single
malformed row does not abort the report.

The natural source of these records is the HITL corrections log
(`docs/hitl_workflow.md`) replayed through `build_calibration_records`.
The script also reads any other JSONL with the same schema, so you can
build calibration sets out of validation campaigns, expert audits, or
ground-truth-labelled fixtures.

## 2. Running the report

```bash
python3 scripts/run_calibration_report.py \
    --input    runs/<run_id>/reports/hitl_corrections.jsonl \
    --out-json runs/<run_id>/reports/calibration_report.json \
    --n-bins   10 \
    --strategy equal_mass
```

Optional thresholds for CI gating:

```bash
python3 scripts/run_calibration_report.py ... --max-ece 0.15 --max-brier 0.25
```

Exit code 2 is reserved for *threshold violation*; the report is still
written even on violation, so CI can both fail and surface the metrics.

## 3. What the report contains

```jsonc
{
  "schema_version": "calibration_report.v1",
  "n_samples": 73,
  "binning_strategy": "equal_mass",
  "n_bins": 10,
  "label_probability_mapping": {
    "high": 0.85, "medium": 0.65, "low_to_medium": 0.55,
    "low": 0.30, "unverified": 0.5, ...
  },
  "metrics": {
    "expected_calibration_error": 0.07,
    "maximum_calibration_error":  0.18,
    "brier_score":                0.16,
    "smooth_ece":                 0.06
  },
  "reliability_table": [
    { "bin_index": 0, "lower_edge": 0.0, "upper_edge": 0.30,
      "count": 11, "mean_confidence": 0.30, "empirical_accuracy": 0.27,
      "gap": 0.03 },
    ...
  ],
  "notes": [
    "ECE/MCE per Naeini et al., AAAI 2015.",
    "Brier score per Brier, MWR 1950.",
    "Smooth ECE per Blasiok and Nakkiran, ICLR 2024 ...",
    "Equal-mass binning per Roelofs et al., AISTATS 2022 ..."
  ]
}
```

The report is **self-contained**: it records its own input mapping and
binning strategy so a reviewer can reproduce the numbers exactly without
knowing what CLI flags were used.

## 4. How to read the metrics

| Metric | Range | Lower-is-better? | Reading |
|---|---|---|---|
| `expected_calibration_error` | `[0, 1]` | yes | Average gap between mean confidence and empirical accuracy across bins. ≤ 0.05 = well calibrated; ≤ 0.15 = mildly miscalibrated; > 0.15 = miscalibrated. |
| `maximum_calibration_error` | `[0, 1]` | yes | Worst-bin gap. Useful as a worst-case safety summary. |
| `brier_score` | `[0, 1]` | yes | Mean squared error of probability vs binary outcome (Brier 1950). Lower = better. Brier is *strictly proper*: it cannot be gamed by relabelling. |
| `smooth_ece` | `[0, 1]` | yes | Kernel-smoothed ECE (Blasiok & Nakkiran ICLR 2024). Less sensitive to bin-edge placement; preferred when `n_samples` is small. |

The dashboard `Reliability` card (`gui/src/components/ReliabilityCard.tsx`)
renders these four numbers and a tone (good / warn / bad) based on the
ECE thresholds above.

## 5. Equal-mass vs equal-width binning

We default to **equal-mass** (quantile) bins because empty bins inflate
the variance of the binned ECE estimator. Roelofs et al. (AISTATS 2022)
recommends equal-mass for the same reason. If you need to compare your
ECE against a paper that uses equal-width bins (the original Naeini
2015 default), pass `--strategy equal_width`.

Calibration is invariant to monotone relabellings, so the *rank* of two
methods is the same under either strategy. Only the *absolute* ECE
value differs.

## 6. Operating-point sensitivity

`DEFAULT_CONFIDENCE_LABEL_PROBABILITIES` maps the discrete labels
(`high`, `medium`, `low`, …) to numeric probabilities. The defaults are
chosen as the *operating-point midpoints* implied by the Stage 9
decision policy (`pipeline/common/progress_decision_policy.py`):

- accept threshold 0.65 → `medium = 0.65`
- midpoint of `[0.65, 1.0]` → `high = 0.85`
- midpoint of `[0, 0.65]`, biased downward → `low = 0.30`

If your team's operating points differ (e.g. you accept at 0.70), call
`calibration_report(records, label_probability_mapping={...})` with
your own mapping. The report records *both* the mapping and the metrics
in-band so the report is self-describing.

## 7. Recommended cadence

- **Per run.** Append the run's HITL corrections to the corrections
  JSONL and run the calibration script. Commit the result alongside
  the rest of `runs/<run_id>/reports/`.
- **Quarterly recalibration.** Replay the *project-wide* corrections
  log and inspect ECE drift. If ECE has worsened beyond your CI
  threshold, that's your signal to retune the decision policy.
- **Pre-publication.** Generate a final calibration report on the
  consolidated corrections set; cite ECE / Brier / smooth-ECE in the
  paper's reliability section alongside the bidirectional accuracy /
  completeness / F-score @ τ already produced by Stage 9.

## 8. Reading the reliability table

Each row of `reliability_table` is one bin:

- `count` — how many predictions fell in this bin.
- `mean_confidence` — average reported confidence in this bin.
- `empirical_accuracy` — fraction of those predictions that were
  actually correct.
- `gap` — `|mean_confidence - empirical_accuracy|`. ECE is the
  count-weighted average of these gaps.

The dashboard `ReliabilityCard` does not yet render the per-bin chart;
that's a Phase 5 polish item. Today the JSON is the source of truth.

## 9. Common pitfalls

- **Per-bin sample size too low.** With `n_bins=10` and `n_samples=20`
  most bins will be sparse. Start with `n_bins=5` and grow the bin
  count as the corrections set grows.
- **Mixing target kinds.** Per-element acceptance and per-activity
  completion have different operating points. Use
  `build_calibration_records(replay, target_kinds=["element_acceptance"])`
  so the report is meaningful.
- **Treating Brier alone as calibration.** Brier conflates calibration
  with refinement (resolution + sharpness). Use ECE / smooth-ECE for
  the calibration question; use Brier as the proper-scoring-rule
  benchmark.
