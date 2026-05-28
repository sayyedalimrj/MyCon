# Stage 4.5 CAMS-GS / 3DGS Prepare

Stage 4.5 prepares optional CAMS-GS / 3D Gaussian Splatting assets for later real-time visualization, VR/client preview, and VLM evidence rendering.

It is not a metric truth stage.

Metric truth remains:

- Stage 8 BIM registration
- Stage 9 progress metrics
## Current MVP

The current implementation is skip-safe and preparation-only.

It writes:

- `data/cams_gs/site01/train_manifest.json`
- `data/cams_gs/site01/training_status.json`
- `runs/2026-04-30_site01_baseline/reports/cams_gs_prepare_summary.json`
- `data/cams_gs/site01/nerfstudio_dataset/selected_images.txt`

It does not run training yet.

## Future training adapter

The planned production path is to connect this stage to Nerfstudio/Splatfacto or another 3DGS backend.

## Command

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 -m pipeline.stage_04_5_cams_gs.run_cams_gs_prepare `
  --config configs/site01.yaml `
  --force
```
