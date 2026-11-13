# MyCON — independent scientific evaluation: methods and parameters

**Status:** evaluation protocol and synthetic software validation, **not** a field-study result. Version `mycon.thesis_evidence.v1`; Owner C only. No A/B/Stage8/9/11 source files are modified, and no training, checkpoint restart, registration optimization against test GT, or reference BIM/MAX fitting is done.

## Immutable test protocol and prerequisites

1. Acquire consented RGB sequences and **independently surveyed or independently manually reviewed** ground truth with dated reviewer, immutable files, SHA-256, sampling uncertainty, sensor calibration and explicit coordinate frame. For metric geometry, independent survey must verify scale, axes, registration and units in metres; otherwise all metric geometry scores are `not_evaluated`.
2. Freeze splits **by site first**, then building/capture session; never mix a site, building, session or capture hash across `train`, `validation`, `test`. The test set stays untouched throughout model selection, registration fitting, cleaning threshold tuning and hyperparameter search. Store separate fit-control and **held-out** survey anchors. Do not use the same image/GT for fitting and evaluation.
3. Compare historic baseline versus new reconstruction on *exactly the same* cases, cameras, masks, image dimensions, independent GT and control survey. Lock tolerances and exclusion rules before results. Report all attempted/failed/unavailable cases, not just successes.
4. Metric admission requires a verified real held-out case, reviewer/date, immutable independent-GT file and capture file, matching SHA-256. A compatible producer manifest may be validated read-only; it is **not** independent GT itself. Synthetic cases are exclusively for software testing.

## Definitions

- **NVS**: PSNR = \(-10\log_{10} \mathrm{MSE}\) for 8-bit sRGB arrays converted to RGB [0,1]; SSIM = skimage channel-wise SSIM with `data_range=1` and the library's default window; both are per held-out view. Infinite PSNR for bit-identical images has no finite numeric summary (`not_evaluated` in current exporter). LPIPS is **not evaluated** until a pinned, audited LPIPS network and preprocessing are explicitly supplied. PSNR/SSIM/LPIPS do **not** imply geometric validity.
- **Camera coverage**: registered/attempted camera count, **not** surface visibility coverage. Reprojection RMSE in pixels = sqrt(mean((u_pred-u_gt)^2 + (v_pred-v_gt)^2)) on separately verified registered correspondences. Registration-fit and held-out control points must be distinguished in upstream provenance; current camera CSV is descriptive until independent correspondence audit.
- **Geometry**: Accuracy is mean nearest-neighbour distance prediction→independent GT; Completeness GT→prediction. Symmetric Chamfer-L1 = (mean pred→GT + mean GT→pred)/2; distinct from squared Chamfer and percentile Hausdorff. At fixed tolerance \(\tau\), P = fraction predicted points within \(\tau\) of GT; R = fraction independent GT within \(\tau\) of prediction; F1 = 2PR/(P+R). Baseline preregistration thresholds: 0.01, 0.02, 0.05, 0.10 m. Threshold on each case must be supplied *before* evaluation and applied equally to paired methods. GT surface density and original geometry sample count must be reported separately.
- **Point-to-plane**: mean |(p−q)·n_q| for nearest independent GT point q with reviewed normal n_q. Not computed if there are no valid normals. No surface is silently metrically calibrated from reconstruction units.
- **Corners**: only `observed_supported` predicted corners count; extrapolated candidate/unknown never become observed measurements. One-to-one optimal matching within \(\tau\) gives precision, recall and matched corner-location mean absolute distance. Unmatched predicted/GT features must remain in failure denominators. With zero matched corners, localization MAE is null.
- **Planes**: angular discrepancy is arccos(|n_pred·n_gt|) for independently reviewed, finite-support, stable-ID plane pairs; unreviewed and infinite-plane extrapolation are excluded from numeric plane angle. Plane detection TP/FP/FN and planar coverage require a separate independent reviewed region association, not simply nearest infinite-plane equations.
- **CI/ablations**: descriptive site-mean aggregate and percentile bootstrap (2,000 draws, unit=independent site, fixed seed) only when at least 2 independent sites; `not_evaluated` otherwise. Cluster bootstrap with 2 sites is extremely unstable; scientific publication requires more multi-site evidence and sensitivity analyses. Paired ablation delta = candidate − reference on identical case, capture checksum, GT checksum, units and metric. A delta is not a significance test. Hold out a separate final evaluation set for every selected variant.
- **Performance**: wall clock, peak GPU VRAM, host peak RSS and serialized file size require measured telemetry bound to run ID, hardware and artifact hashes. No fabricated benchmark numbers; an absent telemetry field stays null.
- **Physical progress**: no construction percentage derived from geometry coverage or image score. Actual executed/planned quantity and approved BOQ weight with independent GT must be assessed by the existing, separately owned B adapter. Stage9 fractions [0,1] are not Stage11 percentages [0,100]; deterministic bounds are not CIs.

## Ablation protocol

Pre-register columns for `old_new`, `frame_count`, `resolution`, `cleaning`, `algorithm`, `seed`, with fixed capture sessions, camera splits and independent GT. Run the same test case under each valid variant; record GPU model, driver/CUDA, package versions, elapsed time, VRAM/RSS, source hashes, failure reason and output bytes. Use five fixed seeds for variance; site-level CI still requires multiple independent sites. Never silently omit failures. Any absent variant = `not_evaluated`, not 0.

## Publication and limitations

The `experiment_registry.json` and five CSVs are outputs of the fail-closed exporter. The schematic `methodology_flow.*` is **not** a scientific performance plot. Actual boxplots are generated **only** from qualified independently reviewed real held-out cases, and must undergo researcher signoff. No real-world reconstruction, metric geometric accuracy, as-built completion, PSNR superiority, cross-site generalization, runtime/VRAM advantage or publication readiness can be claimed from the included no-data run. For a rigorous thesis, obtain external survey, blinded review, site-disjoint test sites, camera calibration, checked baselines and telemetry, and execute the entire existing integrated regression on the full repository checkout.

## CLI

Run from the `repository/` directory, with optional dependencies numpy/scipy/Pillow/scikit-image/matplotlib and pytest for tests:

`python -m pytest -q tests/test_research_validation_evidence_export.py`

`python -m benchmarks.mycon_2026.evidence_export --manifest configs/research_validation/no_real_data_manifest.v1.json --output-root <unique_results_parent>`

The exporter creates `output_root/run_id` exclusively and refuses to overwrite it. The manifest's parent is the read-only root for all hashed paths. It never launches upstream training or edits original inputs.