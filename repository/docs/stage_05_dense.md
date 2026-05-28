# Stage 5: COLMAP Dense Stereo Baseline

Stage 5 consumes the refined sparse reconstruction from Stage 4 and produces the
first dense geometry backbone for later cleanup, DA3 gating, and BIM comparison.
It remains a COLMAP dense-stereo baseline stage:

1. `image_undistorter`
2. `patch_match_stereo`
3. `stereo_fusion`

## CUDA requirement

COLMAP `patch_match_stereo` requires a CUDA-enabled COLMAP build. Stage 5 now
performs a fail-fast CUDA preflight before it deletes or creates the dense
workspace. If `colmap help` reports `without CUDA`, Stage 5 stops with a clear
environment message instead of allowing COLMAP to abort inside PatchMatch.

On a GPU server, run with the optional Compose override:

```powershell
docker compose -f docker\docker-compose.yml -f docker\docker-compose.gpu.yml run --rm core nvidia-smi
```

Then run Stage 5:

```powershell
docker compose -f docker\docker-compose.yml -f docker\docker-compose.gpu.yml run --rm core `
  python3 -m pipeline.stage_05_dense.run_dense `
  --config configs/site01.yaml `
  --force
```

The override only exposes NVIDIA devices to the container. The image still must
contain a CUDA-enabled COLMAP binary.

## Inputs

- `configs/site01.yaml`
- `data/sparse_refined/site01/0/{cameras.bin,images.bin,points3D.bin}`
- `data/sfm/site01/images/*`

## Outputs

- `data/dense/site01/fused.ply`
- `data/dense/site01/command_history.json`
- `runs/<run_id>/reports/dense_summary.json`
- `runs/<run_id>/logs/stage_05_dense.log`

## Adaptive GPU profile

Stage 5 probes visible GPU memory through `nvidia-smi` when available, then
falls back to `torch.cuda.get_device_properties()` if PyTorch is already
installed in the image. With `dense.adaptive_gpu_profile: true`, it adjusts
image-size, source-view count, and cache defaults at runtime while leaving the
YAML file unchanged. The runtime choices are written to `dense_summary.json`
under `dense_runtime_profile` and `weak_texture_preset`.

Approximate memory profiles before image-count caps:

- 24GB-class GPUs such as RTX 3090 Ti / RTX A5000: up to `max_image_size=2200`.
- 16GB-class GPUs: up to `max_image_size=1800`.
- 10-12GB GPUs: up to `max_image_size=1500`.
- 6-8GB GPUs: up to `max_image_size=1200`.

PatchMatch memory also depends on the density of the image graph. Therefore,
when `dense.adaptive_image_count_caps: true`, large registered image sets cap
aggressive high-VRAM profiles. For example, image sets with 50+ registered
views are capped to at most 1600 px, 120+ views to at most 1500 px, and 200+
views to at most 1400 px. These caps are intentionally conservative for long
construction walkthrough videos.

These are conservative construction-site defaults. You can disable runtime
auto-tuning with:

```yaml
dense:
  adaptive_gpu_profile: false
```

## Safety decisions in v4

- The dense workspace is only deleted if it is under `data/dense/` or already
  contains `.dense_workspace_lock`. This prevents an accidental `--force` from
  deleting Stage 3/4 products such as `data/sfm`, `data/sparse`, or
  `data/sparse_refined`.
- The weak-texture preset keeps `patch_window_radius: 5` by default to control
  PatchMatch memory pressure.
- Geometry consistency is retained, while consistency costs are relaxed to keep
  construction-site weak texture from over-filtering.
- The quality gate uses fused point density, `fused_vertex_count / input_images`,
  rather than treating depth-map ratio as a hard geometric quality metric.
- Dynamic semantic masks and fusion bounding boxes remain optional hooks. They
  are passed only if the installed COLMAP command help explicitly supports the
  relevant options.
- COLMAP command streaming uses UTF-8 replacement decoding to avoid hiding the
  real failure behind Python `UnicodeDecodeError` on native crashes.
- GPU auto-tuning no longer depends exclusively on `nvidia-smi`; it uses an
  optional PyTorch CUDA fallback when available.
- High-end GPU profiles are capped by registered image count to reduce OOM risk
  on dense image graphs.

## Acceptance criteria

- `fused.ply` exists and has at least `dense.quality_min_fused_points` vertices.
- `points_per_input_image >= dense.quality_min_fused_points_per_image`.
- `quality_gate.passed == true` in `dense_summary.json`.
- `dense_runtime_profile.cuda_build_detected != false`.
- `command_history.json` records all COLMAP commands.

Stage 6 may use `dense_summary.json` to decide whether DA3 assistance is needed.
