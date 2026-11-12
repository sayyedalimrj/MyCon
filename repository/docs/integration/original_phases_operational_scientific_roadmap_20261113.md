# MyCON — original-phase completion plan and science/production acceptance gates
Updated 2026-11-13 — continuation of Recorder → R4 Bridge → MyCON → BIM → Progress → Schedule

## Decision
Continue the original research + practical workflow, **not** an isolated paper reproduction.
Primary reconstruction: video/Recorder → COLMAP/DA3 → geometry plus separate 3DGS/LongSplat visual track.
Reference-specific 14021208 MAX fitting remains postponed; a *generic* BIM-ready interface may be built, validated and retained.
The application must produce useful reviewed engineering outputs (registered elements, verified quantities, schedule), and the research must be reproducible without presenting synthetic QA as real-site accuracy.

## Common acceptance levels
- **L0 implementation**: module and versioned schema exist, compile and isolated tests.
- **L1 integration**: real checked-out repository A→B→C contract, deterministic CLI, SHA provenance, regression.
- **L2 real-data operational**: original capture/PLY/cameras processed with human-reviewed preview, no data loss, bounded runtime and documented failure/unknown results.
- **L3 metrologically credible**: independent survey GT, camera/image and BIM registration verification, held-out control points and no test leakage.
- **L4 scientific comparative**: same scenes/splits/seeds/units across baselines and proposed method; statistical uncertainty, ablations, runtime, failures and data provenance, tables and paper claims.

## Original phases and current gates

| Original phase | Deliverable that makes the tool useful | Scientific comparison and minimum evidence | Current state |
|---|---|---|---|
| 0. Capture / Recorder / R4 Bridge | Valid video/images, timestamps, cameras/IMU/pose metadata, reproducible manifest, no destructive edits | Input QC, repeated frame counts, calibration/coverage baselines, train/held-out split | Implementation/historical captures; L2 integrated run not shown |
| 1. COLMAP / sparse reconstruction | Coherent source camera trajectories, sparse scene and track provenance | Registration coverage, reprojection errors, held-out visibility, compare feature matching methods | Implementation/historical COLMAP exists; final site rerun L2 not proven |
| 2. Dense/DA3 metric-ready geometry | Density controlled cloud + real source image/ray/depth lineage; distinct unscaled vs metric | Depth/geometry comparisons on shared frames; scale and distance control; failure modes | Code path exists, GPU/real DA3 unverified |
| 3. 3DGS / LongSplat novel views | **Separately preserved** visual 3DGS PLY/checkpoint and novel-view renders, plus separately measurable geometry | 8fps/full-video input audit, frozen gsplat step 6999 baseline; hold-out PSNR/SSIM/LPIPS, COLMAP/LongSplat fair render comparison | Historical gsplat checkpoint/PLY exists; LongSplat and held-out comparison not executed |
| 4. Stage7 cleaned comparison-ready geometry | RGB cloud, finite supported planes, finite edge islands, candidate vs supported corners, hashed manifest and preview | A/B compare over exact same raw PLY, point retention, p95 distances, independent GT Chamfer/F-score and corner F1, noise/occlusion/density ablation | L1 passes; real L2/L3 pending; no proven real-site improvement |
| 5. Stage8 metric/IFC registration | Documented metric scale from independent survey and gravity/camera QA, valid IFC mappings and residuals | Held-out survey anchors, TRE/Chamfer and registration failure gate; no geometry-only completion claims | Interface/tests; L3 field evidence absent; special MAX comparison suspended |
| 6. Stage9 physical progress/visibility | Verified executed quantities, element IDs and evidence with explicit unknown/occlusion, project-ready element result | Independent measured BOQ activity GT; precision/recall, quantity MAE, calibration, abstention and partial-identification bounds | L1 evidence-safe logic; real physical quantities absent |
| 7. Stage10 grounded VLM | Inspectable image/crop/time/3D feature references, evidence-grounded QA answers; unsupported claims rejected | VLM hallucination/abstention, cross-view agreement, human review on heldout images | Software grounder exists; real local VLM evaluation missing; Stage7.5 remains mock-only |
| 8. Stage11 schedule and dashboard | BOQ-weighted, approved baseline plan vs verified actual, readable visual report and export | Activity/project error and schedule variance bounds, explicit Unknown and data lineage | L1 scientific v2 schema; real approved baseline not tested |
| 9. C independent benchmark and article | Reproducible manifests, frozen GT, tables/figures, ablations and failure studies | Multi-scene holdout, pairwise baselines, uncertainty and exact versioned artifacts | Synthetic tool QA only; no defensible field accuracy/publication result |

## Rule for "cleaner than before"
Do **not** claim "after is better" from fewer points, smooth-looking surfaces, or more planes/corners alone.
Compare **same input and coordinate frame** with frozen old and new output, documenting original source and hashes; use `scripts/compare_geometry_versions.py`.
Without independent ground truth, report only within-threshold proximity and retention proxies + counts; quality conclusion stays `not_determined_without_independent_ground_truth`.
With synthetic GT, evaluate algorithm correctness but no field validation; with surveyed independent GT, external review must verify unit/frame/registration and geometry labels. For actual corners report TP/FP/FN on reviewed GT instead of assigning quality from visual plausibility.

Example, from repository root (units and radius must match *your* reconstruction):
```bash
python -m scripts.compare_geometry_versions \
  --source /path/source_cloud.ply --before /path/old_cloud.ply \
  --after /path/runs/geom-ready-example/geometry/clean_cloud.ply \
  --frame source_reconstruction --units reconstruction_units \
  --threshold 0.03 --max-points 20000 \
  --out /path/new_create_only_review.json
```

Scientific variant additionally requires `--ground-truth /path/independent_gt.ply --gt-kind independent_survey --gt-provenance survey_reference`; still cannot self-certify ground truth or physical scale.

## Immediate work order
1. Freeze and hash legacy 3DGS baseline, associated videos and camera poses; obtain a **separate immutable existing** real 3DGS-/COLMAP-derived cloud under Drive-compatible access (source-side execution acceptable). Never restart long training without evidence.
2. Run both old and new cleanup on **the same real physical geometry cloud**, produce side-by-side 2D/3D previews and local edge/corner human audit. Compare retention/coverage and errors using operator-approved thresholds. Mark metric quality unknown unless independent GT exists.
3. Validate camera calibration and two-density/multiview feature consistency and non-Manhattan/occluded regions, then confirm any measured corner with independent sources.
4. Bring actual IFC revision, real-world survey controls and approved quantities/baseline only at the gated BIM/progress stages; no particular MAX building dependency.
5. Run full reproducible baselines/ablations/paper packaging *after* real data and L3 gates; label unsatisfied runs `not_executed`.

## Interpretation of current test results
CPU and lightweight regressions certify that named code paths, fixtures and negative guards work in CI on tested platforms; they **do not** prove that the recovered geometry is cleaner on any actual MyCON building, that GS surfaces are metric, or that physical progress is measured.
