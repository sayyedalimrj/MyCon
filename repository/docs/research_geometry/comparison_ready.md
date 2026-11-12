# Comparison-ready geometry after Stage 7

**Schema:** `mycon.comparison_ready_geometry.v1` (opt-in)  
**Owner:** A / PR #38  
**Implementation:** `pipeline.stage_07_cleanup.comparison_ready`  
**Scope:** reference-agnostic geometry in source reconstruction coordinates. Do **not** match MAX/IFC/BIM, infer construction progress, or certify metric scale.

## Processing stages

1. Start from the **existing** Stage 7 cleaned PLY when available: its Open3D cleanup, color filtering, normal estimation, RANSAC planes and optional meshing already execute as part of the baseline. The independent companion tool never rewrites them.
2. Optionally remove isolated noise via conservative kNN median-absolute-deviation diagnostics (suppresses points only with a large nearest-neighbour gap); optional voxel medoid *selects an actual original input point* and its RGB instead of inventing an averaged position.
3. Compute sampled unoriented local PCA normals and curvature QA, saved in `normal_samples.npz`. The sample covers up to 10,000 points; it is not a validated camera-oriented normal map.
4. **Prefer reuse of Stage-7 guided RANSAC segmentation** when `--stage7-planes-json` is given (and always in the opt-in Stage7 hook): associate each existing plane-cloud PLY to nearest original clean points within a bounded tolerance and refuse duplicated point ownership. Without that input, use deterministic research RANSAC. Robustly refit with SVD, and represent them as **finite convex hull patches of observed points**. Convex hulls may bridge internal occlusion gaps; this is disclosed and must not be interpreted as measured uninterrupted surface.
5. Fit local candidate intersections of **nonparallel** patch pairs. Clip line segments to the overlap of nearby *observed support points on both planes*. Parallel/poorly supported lines are rejected.
6. Solve nondegenerate trihedral plane intersections. Store `observed_supported` **only** with all three local observed-support sets, finite-patch inclusion, supported pairwise lines, and multiscale repeatability; otherwise `candidate_extrapolated` or `unknown`. Do not interpret mathematical extrapolations as measured corners.
7. Optionally project corners with user-supplied calibrated COLMAP/OpenCV world-to-camera intrinsics/extrinsics. In-FOV is reported `in_fov_occlusion_unverified`, **never visible** without independent occlusion validation.

No Gaussian-centre positions are automatically declared faithful physical surface measurements. Real COLMAP-vs-GS-to-PC / 2DGS / SuGaR comparisons have NOT been run: available official gsplat checkpoint and PLY exceed the 256 MiB remote download limit, and independent metric ground truth is absent.

## Standalone CLI: recommended on historical Stage-7 outputs

From the checkout's `repository/` folder:

```bash
python -m pipeline.stage_07_cleanup.comparison_ready \
  --input-ply /workspace/data/clean/site01/cleaned_cloud.ply \
  --project-root /workspace \
  --run-id geom-ready-site01-20261112-001 \
  --threshold 0.03 \
  --min-plane-points 80 \
  --trials 350 \
  --source-frame colmap_reconstruction_world \
  --source-units unknown_reconstruction_units
```

For an existing Stage-7 run, add `--stage7-planes-json /workspace/data/clean/site01/planes.json` (use the actual Stage-7 `planes_json` path from its report). The importer hashes that JSON, loads its existing `cloud_path` PLYs, matches each point back to the clean cloud within tolerance and prevents assignment to multiple planes. If the Stage7 plane cloud is absent or unsupported, it fails explicitly rather than silently claiming Stage7 reuse. The Stage7 config hook passes its actual `paths["planes_json"]` automatically. Standalone research RANSAC remains a fallback **only when no Stage7 planes JSON was provided**.

**Important:** `--threshold` and `--voxel` are expressed in **input reconstruction units**, *not metres* unless independently scaled. Tune the threshold to the cloud's measured NN spacing and known image/depth precision; the command's 0.03 is an interface example, not a universal physical tolerance. To preview no images: `--no-preview`. `--outlier-mad 0` and `--voxel 0` preserve Stage-7-cleaned points by default. For an uncleaned PLY, `--outlier-mad 5 --voxel <carefully chosen source-unit size>` are opt-in and report losses. `--max-feature-points 125000` caps RANSAC work **without removing points** from the exported clean cloud. `--gravity-up x y z` classifies surface orientation only when the operator provides a credible up-vector; floor vs ceiling remains ambiguous.

## Optional additive Stage-7 configuration

The legacy Stage 7 pipeline is unchanged unless the flag below is explicitly enabled. If enabled, it refuses destructive `--force` and refuses to overwrite any existing Stage 7 outputs (use the standalone CLI for those).

```yaml
cleanup:
  comparison_ready_enabled: true
  comparison_ready_run_id: geom-ready-site01-first-001
  comparison_ready_threshold: 0.03
  comparison_ready_min_plane_points: 80
  comparison_ready_trials: 350
  comparison_ready_source_frame: unknown_reconstruction_frame
  comparison_ready_source_units: unknown_reconstruction_units
  # comparison_ready_camera_json: /read-only/calibrated_colmap_cameras.json
```

Run `python -m pipeline.stage_07_cleanup.run_cleanup --config configs/site01.yaml`. An optional `comparison_ready` section is added to Stage 7 report; the previous report fields, outputs, and downstream BIM/progress contracts are not renamed or removed. Failed/blocked feature production is explicitly recorded.

## Output files under `runs/geom-ready-*/geometry/`

| File | Contract |
| --- | --- |
| `clean_cloud.ply` | Binary LE XYZ, with original RGB if input PLY contains vertex colors; source PLY kept read-only |
| `clean_cloud.npy` | NumPy float64 shape Nx3 in original reconstruction coordinates |
| `normal_samples.npz` | Deterministic sample indices and unoriented unit PCA normals, not all-points certification |
| `planes.json` | `planes[]`: `plane_id`, signed normalized `coefficients=[nx,ny,nz,d]`, SVD residuals, finite boundary XYZ/UV, support indices and geometric signature |
| `edges.json` | `edges[]`: candidate finite `endpoints_xyz`, parent `plane_ids`, both local support sets, length in input units |
| `corners.json` | `corners[]`: `xyz`, plane/edge IDs, condition, observed support, multi-scale drift, `state`, optional image projection |
| `xy_geometry_preview.png`, `xyz_geometry_preview.png` | Human-review plots, subject to optional matplotlib |
| `comparison_ready.json` | Manifest with `schema_version`, hashes, run/source/frame/unit fields, cleaning counters, extraction params, provenance, QA and filenames |

All new products are **create-only**. If a `geom-ready-*` target exists, processing stops with a nonzero exit code; no previous output is removed, replaced or merged. When camera JSON exists, a SHA-256 is recorded; images, original camera files, textures, Depth and checkpoints are *referenced and preserved*, not silently rewritten or falsely included in the geometry package.

**IDs:** `pl-###`, `ed-###`, `co-###` are deterministic within a fixed input/config/seed and are not persistent BIM/IFC GlobalIds. The per-plane `stable_geometry_signature` is a rounded, geometry-derived diagnostic, not guaranteed invariant under noise/density/coordinate-frame changes. Match across runs must be verified geometrically.

**Quality:** `metric_scale_verified=false`, `registration_verified=false` are hard-coded, even if a text field says “metres”; independent surveyed dimensions/control and BIM reference matching are separate later tasks. False-positive/negative corner rates require independent ground truth. Unobserved surfaces are not filled.

## Actual CPU synthetic QA (2026-11-12)

- Command: `PYTHONPATH=. pytest -q tests/test_research_geometry_comparison_ready.py` on local A package layout: **13 passed, exit 0**, around 2 s; pytest emitted one `PytestUnknownMarkWarning` because the isolated local copy did not include the repository's registered `lightweight` marker.
- Command: `PYTHONPATH=. python -m pipeline.stage_07_cleanup.comparison_ready --input-ply /mnt/data/mycon_ready_work/local_run/synthetic_three_planes.ply --project-root /mnt/data/mycon_ready_work/local_run --run-id geom-ready-qa-20261112 --threshold .02 --min-plane-points 90 --trials 350`: **exit 0**, clean cloud **2352/2352** (0 removal), **3 finite planes, 3 supported candidate edges, 1 locally supported multiscale corner**; P95 plane residuals approx 0.00198–0.00205 *synthetic reconstruction units*, multiscale corner drift ~0.000148 units. No real-site benchmark or metric quantities.
- Source synthetic cloud SHA-256: `509bde00a3509600a4ef3ecd1a20b7fea7834ca99dea23e0263f51512e7fac63`. Dataset is a deliberately constructed three-plane 2×2 pseudo-room with Gaussian perturbation `std=0.001`; it is NOT a real project capture.
- **Stage7 reuse synthetic CLI**: `python -m pipeline.stage_07_cleanup.comparison_ready --input-ply .../local_run_stage7/synthetic_three_planes.ply --project-root .../local_run_stage7 --run-id geom-ready-stage7-reuse-20261112 --threshold .02 --min-plane-points 90 --trials 350 --stage7-planes-json .../local_run_stage7/stage7_planes.json` → **exit 0; 3 reused source planes, 1 local edge candidate, 1 extrapolated/unknown corner candidate, 0 observed-supported corners**, with 8 generated output hashes verified. This is the intended fail-closed behavior when local Stage7 segmentation leaves insufficient visible junction support. Stage7 planes input SHA-256: `5ca0b9beaf1749d79ccc1d7a1b8fafa8a7c9df843ee11f91ef47064c273a367d`.
- **CLI guard QA:** `--help` exited 0; requesting a second run with an existing `geom-ready-qa-20261112` ID exited **2**, with explicit refusal to replace outputs; 8/8 existing output hashes validated after generation.
- **Updated new module SHA-256:** `571186cd1ae78db22fe8573cd455250e1058197d92a5ff467fbb596d59b45ecb`. `PYTHONPATH=. pytest -q ...`: **13 passed**, zero failures (local isolated runner, not full repo regression).
- CPU: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Linux x86_64. No CUDA or heavy training invoked.
- Whole-repo checkout and Stage 3–8 pytest **not executed here**: outbound GitHub DNS blocked in the local container, and Open3D missing. The new code was fetched/committed through the authenticated GitHub connector and the exact new module/test files were exercised in a local package layout. GitHub Actions workflows did not exist on this branch at inspection. Do not claim full-repository regression.

## Real-case blocked

- The Drive connection rejects the existing canonical `ckpt_6999_rank0.pt` and `point_cloud_6999.ply` (~349 MB each) at its 256 MiB download limit. Source-side, read-only invocation is required. No RTX/CUDA/gsplat runtime is available here, and no calibrated held-out images or independent metric controls were found for scientific scoring.
- The historical `traj_6999.mp4` can be decoded and its SHA-256 matched its canonical evaluation report, but visual appearance alone is not reliable geometric accuracy. Do not convert 3DGS Gaussian centres to geometry without an independently evaluated extraction baseline.
- No true project `clean_cloud.ply`, `planes.json`, `edges.json`, `corners.json` has been produced or field QA-certified. Use a new `geom-ready-*` run **alongside the original PLY on Drive/Colab/Workstation**, never download/rewrite the original.

## Future comparison

The feature package can feed later robust SE(3)/Sim(3) registration against IFC/Revit/3ds Max/OBJ/CAD scans. That stage needs explicit calibrated scaling, coordinate-frame conventions, correspondences, independent held-out ground truth and quality gates; the current package performs **no such matching**.
