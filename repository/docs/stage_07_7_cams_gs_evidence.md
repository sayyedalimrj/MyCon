# Stage 7.7 CAMS-GS / 3DGS Evidence

Stage 7.7 packages optional CAMS-GS / 3DGS preparation and future training outputs into a viewer/evidence artifact.

It is not a metric truth stage.

## Current MVP

The current implementation reads Stage 4.5 outputs and writes:

- `data/cams_gs/site01/evidence/cams_gs_evidence.json`
- `runs/2026-04-30_site01_baseline/reports/cams_gs_evidence_summary.json`
- `exports/cams_gs/site01/index.html`
- `exports/cams_gs/site01/cams_gs_viewer_manifest.json`

If Stage 4.5 prepared the dataset but no training has run yet, Stage 7.7 reports:

- `status=prepared_no_training`

This is expected until Nerfstudio/Splatfacto or another 3DGS trainer is connected.

## Command

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 -m pipeline.stage_07_7_cams_gs_evidence.run_cams_gs_evidence `
  --config configs/site01.yaml `
  --force
```
