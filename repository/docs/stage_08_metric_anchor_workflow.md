# Stage 8 Metric Anchor Workflow

This document defines the practical benchmark/control-point workflow used to make the scan metric and align it to BIM.

## Concept

Each anchor is a physical or geometric point that exists in both:

- BIM coordinates, in meters
- scan/point-cloud coordinates

The metric alignment stage estimates a Sim3 transform:

```text
BIM_point = scale * R * Scan_point + t
```

## Required anchor columns

```csv
anchor_id,description,bim_x_m,bim_y_m,bim_z_m,scan_x,scan_y,scan_z,use_for_scale,use_for_registration
```

At least three anchors with both BIM and scan coordinates are required for registration.

## Prepare a working template

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/prepare_metric_anchors_template.py `
  --source data/bim/design/metric_anchors.csv `
  --output data/bim/design/metric_anchors_working.csv `
  --force
```

## Validate anchors

Non-strict validation:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/validate_metric_anchors.py
```

Strict validation:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/validate_metric_anchors.py `
  --strict
```

The strict command exits non-zero if fewer than three registration anchors are complete.

## Important rule

Anchor alignment is metric evidence. It is more important than visual similarity.

If anchor residuals are poor, Stage 8 and Stage 9 must remain low-confidence even if the visual overlay looks plausible.
