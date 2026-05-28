# Stage 8 — BIM Extraction and Scan-to-BIM Registration

Stage 8 converts the design IFC model into explicit Open3D geometry, aligns the cleaned scan into BIM coordinates, and writes the file contracts needed by Stage 9 deviation and progress metrics.

The implementation follows the mainline rule that IFC is parsed with IfcOpenShell and registration is done with Open3D after explicit geometry is available. This stage is intentionally robust for partial captures: the scan may be an indoor room, an open exterior area, a slab zone, a facade, or a site with no detected walls.

## Inputs

- `data/clean/site01/cleaned_cloud.ply` or another configured scan candidate
- `data/bim/design/model.ifc`
- optional `data/clean/site01/planes.json` from Stage 7, for future semantic-prior workflows
- optional `data/bim/design/schedule.csv`, if schedule-aware filtering is enabled

## Outputs

- `data/bim/aligned/site01/bim_reference.ply`
- `data/bim/aligned/site01/bim_reference_mesh.ply`
- `data/bim/aligned/site01/scan_aligned.ply`
- `data/bim/aligned/site01/transform_scan_to_bim.json`
- `data/bim/aligned/site01/bim_elements.jsonl`
- `runs/<run_id>/reports/registration_report.json`

## Critical safety decisions in v2

### No bounding-box scale by default

The default `initial_scale_strategy` is `fixed_1`. Stage 8 does **not** estimate scan scale from BIM and scan bounding boxes. A mobile-video scan may cover one room, one corridor, a floor edge, or an open exterior area while the IFC may cover a whole building. Global bbox ratios can therefore create catastrophic scale drift. Use `known_initial_scale` only when it comes from a measured control distance, GCPs, or a metric scale already established by Stage 6/DA3 alignment.

Available strategies:

- `fixed_1`: use metric scale as-is; safest default.
- `known_scale`: apply `known_initial_scale` from a trusted source.
- `bbox_unsafe`: explicitly enable legacy bbox scale estimation for controlled experiments only.

### Staged ICP

The default ICP sequence is:

1. `point_to_point`
2. `point_to_plane`

This prevents point-to-plane ICP from collapsing when coarse rotation/translation is still rough or normals are not yet reliable. If the scan is an open site, contains no walls, or has incomplete planes, the point-to-point stage still provides a safe fallback.

### Optional visible-shell BIM filtering

`visible_shell_filter_enabled` is off by default. Enable it for exterior/open-site scans when hidden internal IFC surfaces attract ICP incorrectly. It uses Open3D hidden point removal from multiple virtual viewpoints and keeps the unfiltered BIM sample if the filter would remove too much geometry.

### Optional schedule-aware BIM filtering

`bim.schedule_filter_enabled` is off by default. If enabled, Stage 8 can remove not-yet-planned IFC elements before registration using a schedule CSV with columns such as `global_id`, `ifc_global_id`, `status`, or `planned_start_day`. Missing or unmatched schedule rows are kept by default to avoid breaking partial schedules.

## Commands

Patch config:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/apply_stage_08_config.py `
  --config configs/site01.yaml `
  --force-update-report-path `
  --force-safe-defaults
```

Smoke test:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/smoke_test_stage_08.py
```

Pytest:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  pytest -q tests/test_stage_08_bim_eval.py
```

Run real Stage 8:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 -m pipeline.stage_08_bim_eval.run_registration `
  --config configs/site01.yaml `
  --force
```

## Acceptance criteria

- IFC opens through IfcOpenShell or smoke test uses explicit synthetic fallback.
- `bim_reference.ply`, `bim_reference_mesh.ply`, and `scan_aligned.ply` are non-empty.
- `transform_scan_to_bim.json` contains a 4x4 scan-to-BIM transform.
- `registration_report.json` records scan/BIM counts, coarse method, ICP method, RMSE, fitness, schedule filter status, visible-shell filter status, and quality warnings.
- Low registration quality is reported as warnings unless `bim.fail_on_low_registration_quality=true`.

## Handoff to Stage 9

Stage 9 should consume `scan_aligned.ply`, `bim_reference.ply`, `bim_elements.jsonl`, and `transform_scan_to_bim.json` to compute per-element coverage, deviation, confidence, and progress metrics.
