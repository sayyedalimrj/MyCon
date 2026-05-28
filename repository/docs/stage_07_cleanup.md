# Stage 7: Open3D Cleanup, Meshing, and Plane Extraction

Stage 7 converts the dense or DA3-assisted point cloud into a cleaner architectural geometry package for BIM registration.

## Inputs

- `data/da3/site01/da3_assisted_points.ply` when Stage 6 produced an assisted cloud.
- `data/dense/site01/fused.ply` as the COLMAP dense fallback.
- Optional semantic context:
  - `data/semantics/site01/yolo_detections.jsonl`
  - `data/semantics/site01/vlm_scene_report.json`

## Outputs

- `data/clean/site01/downsampled_cloud.ply`
- `data/clean/site01/cleaned_cloud.ply`
- `data/clean/site01/mesh.ply`
- `data/clean/site01/planes.json`
- `data/clean/site01/plane_clouds/*.ply`
- `data/semantics/site01/yolo_summary.json`
- `data/semantics/site01/vlm_summary.json`
- `runs/<run_id>/reports/cleanup_summary.json`

## Key engineering choices

1. **Voxel before point-count caps**

   The stage never randomly throws away the raw dense cloud before voxelization. Open3D voxel downsampling groups points into a regular grid and produces one point per occupied voxel, which keeps spatial density more uniform. Random subsampling is used only as a final safety cap after voxel downsampling if a cloud is still too large.

2. **Dynamic voxel sizing**

   If the first voxel pass still leaves too many points, the voxel size is increased iteratively until the point count is near `cleanup.max_processing_points` or the iteration cap is reached.

3. **Normal-prior semantic RANSAC**

   Plane extraction is no longer blind by default. The stage first extracts horizontal candidates whose normals are close to the configured up axis, then extracts wall candidates whose normals are roughly perpendicular to the up axis, and only then optionally runs residual RANSAC. This reduces diagonal false planes and produces floor/wall/ceiling priors that are better suited for BIM alignment.

4. **Normal orientation is opt-in**

   `orient_normals_consistent_tangent_plane` is disabled by default for open indoor scenes. Open3D notes that normal estimation may not be consistently oriented and that tangent-plane orientation propagates orientation through a graph; this can be helpful for object scans but is risky in open architectural interiors. The stage keeps it available through `cleanup.normal_orientation_strategy` when needed.

5. **Actionable YOLO/VLM hooks**

   Stage 7 still does not run heavy YOLO/VLM models or store weights in Git. It consumes their output contracts. When YOLO reports transient objects, an optional conservative HSV filter can remove high-visibility safety-color points with a strict removal-ratio cap. This is a weak but useful action hook until full 2D-to-3D semantic projection is added on the server.

6. **Mesh defaults for architecture**

   Ball pivoting is the safer default for open architectural point clouds because Poisson meshing can close open rooms when normals are unreliable. Poisson remains available, and optional plane-distance trimming can be enabled when an architectural planar mesh is desired.

## Run

```bash
python3 -m pipeline.stage_07_cleanup.run_cleanup --config configs/site01.yaml --force
```

With Docker Compose:

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 -m pipeline.stage_07_cleanup.run_cleanup `
  --config configs/site01.yaml `
  --force
```

## Acceptance criteria

- Smoke test passes.
- Unit tests pass.
- `cleaned_cloud.ply` and `planes.json` are written for valid input clouds.
- Stage 7 fails fast if neither Stage 5 nor Stage 6 produced a valid input point cloud.
- Reports include cleanup counts, plane records, mesh status, quality-gate status, and semantic summaries.
