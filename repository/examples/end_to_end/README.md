# End-to-End Walkthrough Fixture

This directory contains a tiny, fully synthetic site that exercises
**every Phase 4 module** in one shot:

- `pipeline/common/schedule_io.py` (canonical schedule CSV)
- `pipeline/common/bim_schedule_mapping.py` (BIM ↔ schedule mapping)
- `pipeline/stage_11_schedule_variance/` (per-activity rollup, variance,
  dashboard summary)
- `pipeline/common/hitl.py` (reviewer corrections store + replay)
- `pipeline/common/calibration.py` (ECE / Brier / smooth-ECE report)
- `pipeline/common/method_comparison.py` (method × metric × CI tables)
- `pipeline/stage_10_copilot/grounding_guard.py` (VLM claim verification)

The fixture is *deterministic and dependency-free* — no Open3D, no
OpenCV, no IfcOpenShell, no live VLM. It runs on a laptop in under a
second and is committed to the repo so the paper's reproducibility
section can point at it directly.

## Files

| File | Role |
|---|---|
| `inputs/schedule.csv` | Two-activity schedule in canonical form (`schedule.v1`). |
| `inputs/bim_schedule_mapping.csv` | Maps each scheduled activity to a handful of synthetic IFC GlobalIds. |
| `inputs/element_metrics.csv` | Stage-9-shaped per-element status rows (mix of `likely_completed`, `partially_observed`, `not_evidenced`). |
| `inputs/hitl_corrections.jsonl` | Six reviewer corrections in `hitl_correction.v1` form (3 confirm, 3 overrule) so the calibration step has real signal. |
| `expected/dashboard_summary_data_date_2026_04_16.json` | Golden output of Stage 11 at `2026-04-16T00:00:00Z`; used for byte-for-byte regression checks. |

## Running it

From the repo root:

```bash
python3 scripts/run_end_to_end_walkthrough.py \
    --output-dir runs/example_walkthrough/ \
    --data-date-utc 2026-04-16
```

The script:

1. Runs Stage 11 against the three input CSVs and writes
   `activity_progress.json`, `schedule_variance.json`, and
   `dashboard_summary.json` under `--output-dir/`.
2. Replays the HITL log into a `calibration_report.v1` JSON.
3. Runs the VLM grounding guard against a small set of
   pre-canned answers to demonstrate the verification result shape.
4. Writes a single `walkthrough_summary.json` linking every output by
   filename and SHA-256.

The smoke test in `tests/test_end_to_end_walkthrough.py` runs the same
script and asserts shape-and-version invariants on every output.

## What the schedule represents

```
A0001  Foundations           2026-03-01 → 2026-04-01   (mapped: 2 elements)
A0432  Floor 2 Zone B walls  2026-04-01 → 2026-05-01   (mapped: 5 elements)
```

At the canonical data date (`2026-04-16`):

- A0001 is past its planned finish and the elements are reported as
  `likely_completed` → on schedule, high confidence.
- A0432 is half-way through but only one of five mapped walls is
  reported as `likely_completed` → behind schedule, medium confidence.

That asymmetry is intentional: it lets the dashboard show one row of
each colour, and gives the calibration step real corrections to work
with.

## Reusing the fixture

The fixture is the recommended starting point for:

- thesis defence reproducibility runs;
- CI regression suites for end-to-end output stability;
- pedagogical walk-throughs of how the canonical CSVs and Stage 11 fit
  together;
- benchmarking new methods against the existing baselines via
  `pipeline/common/method_comparison.py`.

If you want to grow it (e.g. five activities, fifty elements), copy
this directory and edit the CSVs — every script and test in the repo
takes the four input file paths via CLI args, so nothing is hard-coded
to this particular site.
