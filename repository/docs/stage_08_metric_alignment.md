# Stage 8 Metric Alignment

Stage 8 Metric Alignment estimates a metric scan-to-BIM transform before BIM registration.

## Purpose

The reconstruction may be scale-ambiguous or weakly aligned. This module uses:

- manual anchor pairs
- known distances
- later primitive matching

to estimate a defensible Sim3 transform:

- uniform scale
- rotation
- translation

## Important rule

Axis-wise/non-uniform scale can be reported as a diagnostic later, but it must not silently become metric truth.

## Current module

Current implementation:

- reads `metric_anchors.csv`
- reads `known_distances.csv`
- estimates known-distance scale
- estimates Sim3 from anchor pairs with Umeyama alignment
- reports per-anchor residuals
- writes `metric_alignment_report.json`

## Lightweight test

    docker compose -f docker\docker-compose.yml run --rm core `
      pytest -q tests/test_stage_08_metric_alignment.py

## Lightweight CLI

This does not transform the point cloud. It only reads CSV anchors/distances and writes a JSON report.

    docker compose -f docker\docker-compose.yml run --rm core `
      python3 -m pipeline.stage_08_bim_eval.run_metric_alignment `
      --config configs/site01.yaml `
      --force

## Expected current real-project behavior

The current demo `metric_anchors.csv` has BIM anchor coordinates but empty scan coordinates.

Therefore the CLI should report insufficient anchors until scan anchor coordinates are provided.

## Future connection

Stage 8 BIM registration should later consume:

- `metric_transform_scan_to_bim.json`
- `metric_alignment_report.json`

before running ICP refinement.


## Stage 8 registration integration

The coarse registration step now checks `metric_alignment.report_json`.

If the report is usable:

- `status` is `ok` or `alignment_warning`
- `can_feed_stage8` is true
- `quality_gate.passed` is true
- a Sim3 transform is present

then Stage 8 uses the metric Sim3 as the initial transform.

If not, Stage 8 falls back to the previous coarse registration path.

This keeps the laptop/demo workflow safe while allowing server/project runs to use benchmark anchors when available.

## Quality hardening

Stage 8 metric alignment now includes deterministic quality checks:

- minimum registration anchor count
- degenerate or collinear anchor rejection
- uniform similarity transform only for metric alignment
- leave-one-out residual helper for anchor sensitivity
- deterministic RANSAC-style outlier rejection helper
- metric alignment report quality-gate enrichment

If the quality gate fails, downstream Stage 9/10 must treat progress acceptance as low-confidence or uncertain.
