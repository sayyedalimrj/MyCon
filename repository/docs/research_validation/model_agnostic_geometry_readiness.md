# Model-agnostic geometry readiness — C scientific protocol (2026-11-12)

**Authority:** Integration `repository/docs/integration/model_agnostic_geometry_readiness_contract_20261112.md` and coordinator PR #36 priority-correction comment, 2026-11-12. **No MAX, room-specific CAD, BIM, IFC registration, or constructed-progress percentage is evaluated in this phase.** C owns only benchmark/tests/config/docs and reads A output without mutation.

## Research questions / conditional contributions

**RQ1 / H1:** Does A's conservative cleanup retain independently annotated real structural points while rejecting added nuisance/outlier points more effectively than raw COLMAP and fixed Open3D statistical/radius filters on the *same* frozen capture? Null: paired site-level improvement <= 0.

**RQ2 / H2:** Does locally supported finite-plane/edge/corner extraction improve precision/recall/F1, corner localization and explicit unknown handling over matched finite RANSAC/region-growing baselines, including non-Manhattan/sloped structures? Null: paired improvement <= 0. Mathematical intersections of infinite planes are hypotheses, **not observed physical corners**.

**RQ3:** How do point density, noise, outliers, occlusion, missing walls, sloped/near-parallel planes, gaps, camera angle/view count, RANSAC threshold, voxel resolution and seeds affect error and failure/unknown frequency?

**RQ4:** What is the tradeoff between GS novel-view PSNR/SSIM/LPIPS and *separately* measured reconstruction Chamfer/F-score and optional later BIM registration? Image similarity does not certify metric geometry.

**RQ5:** Are VLM claims grounded to reviewer-labelled frame ROI and feature identity, with low hallucination and safe abstention for unsupported lengths/progress? No VLM output is ever GT.

## Read-only A interface and independent GT

The `mycon.comparison_ready_geometry.v1` input manifest supplies `clean_cloud` (PLY), `planes`, `edges`, `corners`, optional normals, and SHA256 per file. It declares `units`, `coordinate_frame`, run/input/model/camera/config provenance and input point count. The adapter verifies paths/hashes and refuses self-certified metric scale or registration without independent evidence. The independent `mycon.structural_gt.v1` contains annotated structure/outlier source points, bounded planar patches (normal/offset/centroid/finite support extent), finite feature edges, observed vs occluded true corners and independently paired normal sample IDs. Coordinates must have a matching frame and declared units, otherwise scoring fails. For real experiments, GT must be frozen, surveyed, independently signed and SHA-verified; A's reconstruction is not GT.

A finite plane needs positive local support, bounded extent and matching normal/centroid/offset. An observed edge needs finite endpoints plus support and at least two source plane IDs (or validated feature-line evidence). `observed_supported` corners must have support_count and feature_ids; `candidate_extrapolated`/`unknown` are abstentions, not measured true positives. Invisible GT corners cannot create false negatives on an observability-conditioned denominator. Explicitly report unsupported features and degeneracies; preserve slanted/non-orthogonal surfaces.

## Metrics and denominators

- **Cleaning:** structural-point preservation (# true structural GT points within tau of cleaned cloud / # true structural GT points); isolated-outlier removal (# annotated nuisance GT points with no cleaned neighbor within tau / # isolated nuisance GT points); ambiguous nuisance close to real surfaces excluded with separate count; removal fraction from actual original input count. Point loss is *not* inherently improvement, especially at thin edges.
- **Geometry:** bidirectional accuracy/completeness/F-score across preregistered tolerances plus symmetric Chamfer L1/L2; directed sample counts, reference density, uncertainty and units. **Meters only when independently metric verified.** If arbitrary SfM units, use source-unit labels and prohibit comparison to centimetre thresholds.
- **Normals:** unsigned orientation error (degrees), signed flip rate, stable common sample IDs; no unverified NN normal correspondence.
- **Features:** one-to-one TP/FP/FN, precision/recall/F1 and localization MAE for finite planes, edges and observed corners; separate n GT visible, n GT unobservable, n extrapolated/abstained and unknown rate. Undefined denominators are `null`, not automatically perfect accuracy.
- **Cost/safety:** per-case wall time, measured memory, hardware/seed, source/GT hashes, input/test split, attempted/failed/missing output denominator. No silent removal of OOM cases from failure rates.
- **VLM:** per-frame normalized bbox IoU against independently reviewer-annotated ROI, matching feature ID/type, supported claim rate, asserted-case hallucination, all-claim abstention and real cross-view agreement. Numeric dimensions/progress require independently reviewed measurement ID, same value/unit/source; a plausible statement alone is unsupported.

## Preregistered stress design and statistics

See `configs/research_validation/model_agnostic_stress.v1.json`: **23 controlled scenarios**, seeds 11/23/42/67/101, density 25/50/100%, low/high spatial noise, low/high outlier rate, 25/60% occlusion, 15/35-degree slopes, missing or extrapolated corners, near-parallel finite planes, 3/12 frames, 70-degree oblique cameras, RANSAC and voxel variants. Run unchanged COLMAP, conventional Open3D cleanup/RANSAC, A feature package, and optional existing real GS-extracted geometry on **exact same** test images/GT, where available. Never retrain from zero or alter captures/checkpoints. Camera repeatability requires actual calibrated intrinsic/extrinsic and source-session provenance; mark `not_executed` without these.

Split entire site, building and session across train/validation/test without hash or near-duplicate leakage. Freeze masks and thresholds before test. Different seeds or pixels within a site are **not independent** experimental sites. Prefer paired site-level effect sizes with 2,000 seeded site-cluster resamples only when independent site count supports uncertainty statements; single-site case is exploratory, with no generalized CI or superiority claim. Report lighting, density, element type, capture baseline, sloped geometry, occlusion, scanner GT accuracy and adjudication disagreement. Include failures and missing denominators, and distinguish implementation tests from scientific method performance.

## Reproduction and publication status

From repository root:

```bash
python -m pytest -q tests/test_research_validation_metrics.py tests/test_research_validation_round2_contract.py tests/test_research_validation_model_agnostic.py
python -m benchmarks.mycon_2026.runner --help
python -m benchmarks.mycon_2026.runner --manifest /frozen/data/manifest.json --output-root /separate/bench-runs --run-id bench-model-free-001 --include-fixtures
```

Manifest GT key `structural_gt_json`, corresponding method key `comparison_ready_manifest_json`; optional `grounding_gt_json` and `vlm_claims_json`. **Never use `--include-fixtures` for publication**; it tests the tool and excludes synthetic values from `scientific_results.json`. Existing NVS scorer remains separate. Export tables/figures only from verified real manifest results; absent real cases remain `not_executed` with empty scientific table.

**Execution tiers:** Source modules and synthetic regression definitions added. Full genuine checkout + integrated pytest, real A producer artifact, site GT, COLMAP/GS paired data, calibration and GPU/field verification remain **not_executed** until evidence is supplied and executed. Real-case performance, generalized corner accuracy and publication readiness **cannot be claimed yet**. Literature (Duran et al.) is a methodological reference, not a numeric cross-dataset baseline.
