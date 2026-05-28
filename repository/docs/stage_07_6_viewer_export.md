# Stage 7.6 Viewer Export

Stage 7.6 creates a portable visualization/export package for client review and engineering inspection.

It is intentionally skip-safe. The MVP does not require PotreeConverter, PDAL, Entwine, py3dtiles, Open3D, or NumPy.

## Purpose

- Package cleaned point cloud, mesh, plane records, BIM alignment outputs, deviation map, and dashboard.
- Write a stable `viewer_manifest.json`.
- Write a lightweight `index.html` artifact portal.
- Prepare a future path for Potree, Cesium/3D Tiles, and optional CAMS-GS/3DGS visualization.

## Current MVP Outputs

- `exports/viewer/site01/index.html`
- `exports/viewer/site01/viewer_manifest.json`
- `exports/viewer/site01/artifacts/...`

## Interpretation Rule

Stage 7.6 is for visualization and delivery. It is not a metric truth stage.

Metric truth remains Stage 8 registration and Stage 9 progress metrics.

## Command

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 -m pipeline.stage_07_6_viewer_export.run_viewer_export `
  --config configs/site01.yaml `
  --force
```
