# MyCON C — Round-2 independent evaluation protocol (preregistered design)

Date: 2026-11-12. Contract authority: repository/docs/parallel_research_round2_contract_20261112.md on Integration. This protocol is a plan, **not a measured result**. Execution status: synthetic unit tests only; publication readiness is blocked by independent reference material.

## Estimands and research questions

**RQ1 / H1:** On the exact same held-out sites, registered, metrically scaled multi-view MyCON geometry improves the preregistered F-score at 0.05 m over unmodified COLMAP SfM/MVS. Null: paired mean site-level delta <= 0. A superiority claim requires complete paired real-site results and multiplicity control.

**RQ2 / H2:** Gravity, independent metric anchors, depth priors and robust BIM registration reduce TRE and scale drift relative to corresponding isolated ablations. Null: paired site-level TRE reduction <= 0. Geometry quality before registration, registration quality after alignment, and BIM element recognition are different outcomes.

**RQ3 / H3:** Independent approved-quantity progress estimates improve physical-progress MAE (percentage points) against a preregistered baseline; unknown label handling reduces unsafe false-negative conclusions without hiding missing predictions. Null: paired reduction <= 0. No prediction is equated with zero progress.

**RQ4 / H4:** Numerically calibrated confidence offers improved selective-risk/coverage compared with uncalibrated evidence scores; evaluate Brier score and ECE on fixed test splits. This is separate from geometric observability and quantity-based completion.

**RQ5:** What are the time/VRAM/RSS and failure-rate trade-offs by method, capture density, site complexity, weather/illumination, occlusion and element class? These comparisons are descriptive unless an inferential design is preregistered.

The primary outcomes are reconstruction F-score at 0.05 m, held-out registration TRE RMSE (m), and physical-progress MAE (percentage points). Secondary outcomes include Chamfer L1 and L2, directional accuracy/completeness at 0.01/0.02/0.05/0.10 m, orientation error, IFC-ID precision/recall, activity/project/schedule MAE, selective macro-F1, unknown/abstention rates, ECE, Brier, NVS PSNR/SSIM/LPIPS (image quality ONLY), runtime and memory. Report every sample denominator before aggregating.

## Independent Ground Truth SOP and traceability

1. Freeze immutable site, building, session, image and model identity; device calibration, time, weather, light level, viewpoint count, occlusion class, point density and sensor specifications. Hash original captures with SHA-256. Obtain permissions/consent for images; avoid publishing identifiable workers or private construction plans.
2. A qualified external surveyor independently measures the same-date as-built surfaces via registered terrestrial LiDAR or total station and provides instrument model, calibration certificate, control network, covariance/accuracy, coordinate reference, date, source files and signed review. The predicted cloud and registration-fitted points are **not** the geometry GT.
3. Reserve at least four spatially distributed surveyed check/control points where feasible. Explicitly partition **registration-fit anchors** and untouched **held-out TRE anchors** before alignment. Never score TRE on anchors used to optimize the transform. For fewer usable anchors, record a failed/insufficient-controls case rather than a precise accuracy estimate.
4. Approved IFC inventory: stable GlobalId, IFC class, geometry revision/LoD and construction-date change order. An independent reviewer reconciles as-designed vs verified as-built element identity. AABB inclusion or geometric nearest neighbor is not confirmed element ownership.
5. Physical GT: executed and planned measured quantities in the same unit, approved BOQ/cost/quantity weights, signed source/reviewer/date, physical work acceptance rules and independently reviewed activity mapping. Use two reviewers blinded to predictions and adjudicate disagreements. Record weighted kappa or measurement inter-rater error when appropriate. Photographic coverage is NOT executed quantity.
6. Schedule GT: baseline_id, explicit approval, approved planned percentage at the same project data date, documented progress-to-activity mapping, approved project-level cross-activity weights and change control. Legacy Stage11 status-average / Wilson CI is not progress truth.
7. Freeze and hash all GT sheets before test inference. An independent custodian signs the version. Record exclusions and missing ground-truth reasons. If genuine GT is missing, task status = not_executed.

### Quantities, frames and thresholds

- Geometry and anchors: metres, exact scanner/world frame and original/estimated transforms; determine axis conventions, proper rotation, gravity and scale before scoring.
- Stage9 physical_progress_estimate: **fraction [0,1]**, only where independently approved quantities are supplied; observability_ratio is independently audited fraction [0,1]; evidence_confidence must be calibrated [0,1].
- Stage11 scientific_progress.v2 activity/project physical_progress_estimate: **percentage [0,100]**. Schedule variance and MAE use **percentage points**. Null is null, never zero. Stage11 deterministic lower/upper progress bounds are *partial identification*, not a confidence interval.
- Geometric tolerance must be selected from engineering LoD requirements and reference survey uncertainty before test outcomes are inspected. Report sensor uncertainty, surface resampling density, and sensitivity at 1, 2, 5, 10 cm. Do not tune thresholds on held-out results.
- NVS: paired held-out calibrated RGB images, same view/image shape and color space, disjoint from training, PSNR in dB, SSIM dimensionless, LPIPS only when a pinned audited evaluator is available. NVS image scores do not establish 3D metric accuracy.

## Split and leakage control

Outer split on **entire site**, then building and acquisition session; never move sessions of one building across partitions. Freeze cross-site device repeats, exact hashes, perceptual near-duplicates, time adjacency and shared BIM revisions. Maintain train/validation/test assignments and immutable source hashes in manifest. Suggested 60/20/20 by site only when enough sites exist. With few sites use nested leave-one-site-out validation; do not rerun final test during model selection. Image counts, resolution, overlap, capture baseline and GT masks must be equal for paired methods.

## Fair baseline / ablation design

See configs/research_validation/benchmark_matrix.v2.json. Required primary COLMAP baseline uses same images/masks and survey controls. Optional VGGT and DUSt3R, explicit Gaussian-splat geometry extraction, GS held-out rendered views, and depth/gravity/metric/ICP hybrids are included only if output artifacts actually exist; an unscaled splat PLY is not a validated metric surface. A and B produce variants on their owned branches; C never retrains or overwrites checkpoint/captures.

Preregister five seeds [11,23,42,67,101] per eligible algorithm and separate output directories. Evaluate ablations without depth, GS, gravity, independent metric alignment, ICP, bidirectional multiview, observability gate and confidence calibration; explain when a variant is a diagnostic-only unsafe control and must not inform schedule decisions. Preserve identical sites, reference and quantity source for every paired comparison. No comparisons based solely on Duran's ZFD/LSAA published numeric errors.

## Error analysis and statistics

- Report classification confusion matrix, class support, abstentions, missing outputs, and exact category denominator. Independently score numeric executed fraction even if the categorical prediction or GT label is unknown. Missing numeric GT and missing numeric prediction remain distinct.
- Report progress MAE using verified pairs only; include weight coverage and deterministic upper/lower bounds on unscored quantities. Do not report zero error when all estimates abstain.
- Report per-class stratified error by lighting, occlusion, viewpoint count, construction trade/element class, scene scale, camera quality and point density. Keep all non-comparable scenes and exclusions in the failure ledger.
- For confidence, report bin denominators, accuracy-vs-confidence reliability and selective risk-vs-coverage on valid classification labels only. Do not treat a self-reported confidence without held-out calibration as probability.
- Bootstrap **whole sites**, not individual pixels/points. Use deterministic 2,000 draws; each site contributes its case mean; report cluster count and 95% percentile interval only with >=2 independent sites. Multi-site pilot confidence is still unstable with few sites; formal publication should use a power analysis, sensitivity to resampling, and multiplicity-adjusted paired hypotheses. CIs of differences are paired by identical sample IDs and GT hashes.
- Report all attempted, failed, missing and not_executed runs. Failure denominator is attempted runs; execution coverage denominator is planned eligible runs. An OOM, missing checkpoint or failed registration cannot silently disappear from mean-error denominators.
- Independent reference error and registration/checkpoint provenance are reported alongside all metrics. Correlated capture sessions within one site are not independent observations.

## Reproduction, audit and release ladder

Working directory: repository/.
Install optional dependencies from requirements-research-validation.txt. Run isolated pytest tests/test_research_validation_metrics.py and tests/test_research_validation_round2_contract.py, then exact CLI from scripts/research_validation_run.py with --manifest, --output-root outside input dataset, --run-id bench-..., and optionally --include-fixtures for TOOL TESTS ONLY. Use scripts/research_validation_preflight.py for optional A/B versioned artifacts. The runner writes input_hashes, cases with per-task status/exit codes, failure summary, metadata (source manifest hash, seed, runtime, environment), scientific results, paired comparisons, tables and templates. Render figures only using python -m benchmarks.mycon_2026.figures --result-dir <bench-run>.

Milestones are **implemented**, **unit_tested**, **gpu_validated**, **real_case_validated**, and **publication_ready**. None implies the next. Publication requires independent signed survey/quantity/schedule GT, held-out partitions and hashes, actual A/B output artifacts, pinned model/checkpoint/commit and hardware, complete paired baselines/ablations, non-trivial site count, real raw per-case results, error/failure analysis and manuscript review. In the absence of these inputs, all scientific results remain **not_executed**.

## Duran et al. comparison boundary

Duran et al., *Automated facade solar retrofits using feed-forward 3D reconstruction and generative image editing*, Automation in Construction 193 (January 2027), article 107294, DOI: 10.1016/j.autcon.2026.107294; public implementation https://github.com/ycdrn/image2bipv. Their feed-forward geometry (DUSt3R/VGGT), VLM metric scaling, gravity/orthorectification and controlled generation are relevant methodological baselines. Their ZFD and LSAA facade-aspect-ratio/dimension/floor-count results measure different datasets and different estimands, so are not direct MyCON reconstruction, BIM-registration, physical-progress, or schedule accuracy comparisons. A direct comparison would require implementing both on precisely the same independent MyCON site/test subsets and outcomes. No superiority claim is permitted from published cross-dataset numbers.
