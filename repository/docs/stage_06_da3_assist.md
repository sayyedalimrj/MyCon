# Stage 6: Conditional DA3 Assistance

Stage 6 is conditional. It does not replace the COLMAP dense baseline. It only activates when Stage 5 dense coverage is weak according to `dense_summary.json` and the thresholds in `configs/site01.yaml`.

## Inputs

- `runs/<run_id>/reports/dense_summary.json`
- `data/sparse_refined/site01/0` or a pre-exported text model in `da3.sparse_text_dir`
- `data/sfm/site01/images`
- Optional precomputed DA3 depth maps in `da3.depth_input_dir`

## Outputs

- `data/da3/site01/decision.json`
- `data/da3/site01/depth_manifest.csv`
- `data/da3/site01/alignment_manifest.csv`
- `data/da3/site01/fusion_plan.json`
- `data/da3/site01/da3_assisted_points.ply`
- `runs/<run_id>/reports/da3_summary.json`

## Provider contract

The baseline provider is `precomputed`; it reads `.npy`, `.npz`, `.png`, `.tif`, `.tiff`, or `.pfm` depth maps whose stem matches the source image stem. External DA3 inference should be connected through:

```yaml
da3:
  provider: external_command
  external_command: "python3 tools/run_da3.py --image-dir {image_dir} --output-dir {output_dir}"
```

Model weights and raw DA3 exports must not be committed to Git.

## Geometry policy

Stage 6 aligns DA3 depth maps using **scale-only RANSAC**:

```text
Z_colmap = scale * Z_da3
shift = 0
```

The shift term is intentionally locked to zero. A global additive shift in depth changes the backprojected X/Y coordinates differently at different pixel locations under the pinhole model, which can bend planar construction geometry. RANSAC and optional depth bucketing make the scale estimate robust to sparse outliers, reflections, transient objects, and uneven anchor distributions.

## Fusion policy

Aligned depths are fused into a DA3-assisted point cloud with:

- binary PLY output by default, using Open3D when available and a binary fallback writer otherwise;
- edge-aware flying-pixel removal using depth gradients before backprojection;
- bounded reservoir-style downsampling to avoid unbounded RAM growth;
- optional `da3.fusion_bounding_box` for future BIM/frustum culling.

## Commands

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/apply_stage_06_config.py `
  --config configs/site01.yaml `
  --force-update-report-path `
  --force-safe-defaults
```

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/smoke_test_stage_06.py
```

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 -m pipeline.stage_06_da3_assist.run_da3_assist `
  --config configs/site01.yaml `
  --force
```
