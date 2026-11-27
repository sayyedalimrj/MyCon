# MyCON C / preregistered independent scientific benchmark v3

**Version:** `mycon.validation.protocol.v3`; **date:** 2026-11-27; **owner:** C; **target:** Draft PR #46 → `research/mycon-integration-20261112`; no merge by C.

## Engineering questions and scope

- **RQ1:** Does a genuinely held-out novel view from a fixed scene improve under an opt-in new reconstruction compared with the immutable checkpoint 6999, on the **same frozen reference image, camera, mask, resolution and image split**? Metrics: per-view PSNR↑ / SSIM↑ / pinned LPIPS↓; failed/ineligible views included in denominators.
- **RQ2:** Do structurally meaningful finite surfaces, edges and `observed_supported` corners improve, including atypical/sloped planes, after conservative cleaning, **without inventing unseen geometry**? Metrics: field-independent structural annotations, feature precision/recall/F1 and positional errors only where independent geometry truth is available.
- **RQ3:** Is any observed improvement robust to training count, scene density, resolution, cleaning and seeds? Ablations: matched controls, same frozen test; site-disjoint external replication required for generalization.
- **RQ4 (conditional):** If independently surveyed as-built geometry + controls exist, are dimension, scale, registration and point-to-plane errors within engineering tolerances? This is currently **not_evaluated**.
- **RQ5 (conditional):** If signed IFC GlobalId, actual executed quantities and approved BOQ/WBS/schedule baseline exist, can progress be evaluated without interpreting occlusion as unfinished? This is currently **not_evaluated**.

The immutable source artifacts (`MyCon_R4`, source videos, original COLMAP, `ckpt_6999_rank0.pt`, `resume_step_006999.pt`) are read-only. This protocol never authorizes fresh training, overwrite, `--force`, or modifications outside C-owned paths. Inputs from R/A/V/D/B are **consumed** by SHA-verified versioned adapters; failing contracts are escalated to the coordinator, not silently patched.

## Gating and scientific status taxonomy

1. `not_executed`: no actual run, producer outputs unavailable, or runtime absent.
2. `not_evaluated`: attempted check but scientific prerequisites absent/incompatible: unavailable GT, unverified scale or missing matched held-out predictions.
3. `failed`: execution/receipt contract or reproducibility error; report attempted denominator and error provenance.
4. `diagnostic_real_internal_only`: source is genuinely real, but the metric is evaluated against training/self-consistency rather than independent GT; never export as field-valid accuracy or paper comparison result.
5. `evaluated_real_heldout`: exact independent locked test, SHA-checked GT/capture and method provenance; eligible for descriptives only; cross-site CIs/generality require multiple independent sites.
6. `synthetic_unit_only`: code behavior in fabricated scenes; **separate directory and CSV tables** from real diagnostics and real heldout.

Use `scientific_status` and `data_kind` on **every** row, not just header. `not_evaluated` is not zero. Missing view/prediction yields a row and denominator penalty, not silent removal. A base model trained on the same images cannot be evaluated on those images as an independent test, even when the file is labeled `test`.

## Frozen common test set (before predictions)

- Independent custodian freezes `mycon.validation.frozen_test_set.v1` with site/building/session identifiers, capture timestamp and source SHA256, exact **test view IDs**, camera model/intrinsics/extrinsics convention, color space and undistortion, resolution, mask policy, camera receipt SHA256, frame, units and exclusions. Sign split + roster before training or result selection.
- Outer partition by **site**, then keep all buildings, rooms, continuous capture sessions, adjacent frames and repeated walkthroughs of that site together in one partition. Near-duplicate image SHA/perceptual duplicates and GT-leaking COLMAP covisibility must be inspected. For one site, internal held-out views do **not** justify site-level generalization or CI.
- Existing 6999 training membership needs independent evidence. If train list is missing or used all original captures, overlapping views are excluded from held-out comparison. Do not select test based on observed PSNR.
- Pair by `(site_id,building_id,capture_session_id,view_id,reference_rgb_sha256,camera_receipt_sha256,mask_sha256,color_space,independent_gt_sha256)`; only complete matched pairs contribute to paired mean. Report all missing/unmatched views in `paired_view_audit.csv`. Same held-out images for 6999/new, 192/expanded, COLMAP and any other eligible method. R/A may vary input training count; test remains untouched.
- Camera coverage denominator is the frozen eligible reference camera roster; count reconstructed and correctly registered cameras separately. Reprojection on COLMAP bundle-adjustment 3D points is explicitly **internal fitting diagnostic**, never independent holdout camera accuracy.

## Baselines and fair contrasts

| Study | Reference | Candidate | Must be identical | Current gate |
|---|---|---|---|---|
| E01 | frozen GS6999 original checkpoint | new opt-in R render | same camera, raw RGB, test view, mask/color, output pixel grid; versions/hardware | no matching real held-out render |
| E02 | 192 training frames | >192 frames from same acquisition plan | fixed test roster, method family, comparable budget; disclose changing input data | no independently frozen tests or expanded outputs |
| E03 | upstream geometry before cleanup | A cleaned finite cloud | source SHA, original density domain, frame/units, masks | real source exists; independent structural GT missing |
| E04 | COLMAP SfM/suitable MVS surface | verified GS mesh/surface | same split and metric frame, sample strategy, density/coverage | COLMAP sparse cannot stand in for dense GT |
| E05 | blind/unguided plane extraction | gravity/depth-aware | same raw cloud and finite patch GT; no Manhattan assumption | no independently reviewed plane GT |
| E06 | Stage9 legacy status | B scientific progress v2 | same GlobalIds, signed quantity/BOQ and as-of date | no quantity GT, scale/registration |

A COLMAP sparse cloud is a **baseline for camera/structure-of-motion diagnostics**, not a reference as-built metric survey. Gaussian centers are **not** physical measured surfaces. No arbitrary Sim3 may be optimized using held-out survey anchors.

## Metrics and definitions

**Images (valid masked pixels only, per reference view):** PSNR(dB) = 10log10(MAX²/MSE), MAX=1 after a documented RGB transfer function; MSE=0 means +∞ (display separately, exclude from finite-mean unless preregistered cap). SSIM is computed with explicit library, window and color channel policy; LPIPS requires exact pinned library, network/weights SHA256, normalization and CPU/GPU, otherwise `not_evaluated`. Do not automatically resize predictions to hide a geometry/projection mismatch. Report worst 10% PSNR/SSIM, worst 10% high-LPIPS, valid pixel counts, missed views, per-view p5/p50/p95. Identical images and masks are mandatory.

**Geometry (independently surveyed as-built surface after held-out registration):** use nearest-neighbor distance of prediction→GT for `accuracy_distance_m` and GT→prediction for `completeness_distance_m`; report median/p90/p95 as well as mean. Also provide ETH3D-style threshold **accuracy fraction** `A_tau=#valid(pred nearest GT ≤τ)/#eligible pred` and **completeness fraction** `C_tau=#valid(GT nearest pred ≤τ)/#eligible GT`, and `F1_tau=2 A_tau C_tau/(A_tau+C_tau)` (with explicit both-zero handling), at preregistered τ=0.01/0.02/0.05/0.10 m **only once sensor LoA supports them**. Distinguish distances [m] from threshold fractions [%]. Chamfer-L1 = average of two directional mean distances, with a stated `mean`/`sum` convention. Point-to-plane uses surveyed normal orientations, valid support and normal confidence. Mask unsupported/occluded regions as unknown; report eligible surface area, structural class and sampling/voxel method; do not score unobserved hypothetical regions as measured.

**Structural features:** finite area-weighted plane TP/FP/FN with patch IoU, angular error modulo normal sign *if* signed normal convention is unknown, signed flip error separately when normal direction defined; lines matched by finite segment overlap and endpoint/angle tolerance; corners only independently labeled `observed_supported`, separately score `candidate_extrapolated` and `unknown`. Each feature matching is 1:1 with a preregistered assignment rule, no infinite-plane intersection as a field-verified corner. Report slopes/non-Manhattan and clipped/occluded structures separately.

**Registration:** freeze fit-survey anchors vs held-out check anchors. Report holdout TRE p50/p95/RMSE, scale bias, orientation, location/axes, covariance and observed construction tolerances. `metric_scale_verified` and `registration_verified` must be separately true; any disagreement or missing independent signed controls → metric geometry `not_evaluated`.

**Physical progress:** Ground Truth = signed executed quantity / signed baseline planned quantity using compatible units, date, GlobalId and approved mapping. Evaluate categorical unknown/not-evidenced separately from a independently supported numeric fraction [0,1] and Stage11 activity percentage [0,100]. Unknown/occluded/out-of-FOV do not map to 0%. Deterministic progress bounds are not statistical confidence intervals. Report MAE, coverage, abstention and failure denominators for independent measured quantities only.

**Runtime:** exact wall-clock including/excluding startup, peak RAM, peak VRAM if available, GPU/CUDA, energy if measured and output bytes. If not instrumented, `not_evaluated` not 0. Report hardware and environment separately for cold/warm runs.

## Ablation, negative controls and statistical analysis

- Prespecify 5 independent seeds for stochastic methods; change **one factor per contrast** (training count {192, expanded}, resolution, voxel/cleaning threshold, depth/gravity, RANSAC, camera calibration, confidence/abstention); save all seeds and failed attempts. Source scenes must not be leaked to held-out site selection.
- Stress tests: noise, variable density, occlusion, tilted walls, partial/missing ceiling, thin columns, curved/non-Manhattan members, out-of-bounds inputs and wrong units; synthetic oracle stress data explicitly separate from field-test evidence.
- Negative controls: mismatched camera receipt, wrong source image SHA, duplicate frames across train/test, intentionally unregistered depth cloud, scaled-by-100 length units, unsupported corner extrapolation, missing BOQ/IFC; **must fail closed**.
- Evaluate paired per-view differences only when full frozen test roster pairs exist. Aggregate within site; site is the independent statistical unit. For >=5 independent test sites, estimate paired site-bootstrap CI (2,000+ resamples) for a prespecified primary endpoint; for 2–4 sites label exploratory, for one site do not report inferential CI. Never bootstrap correlated points or per-view frames as if they were independent sites. Include effect distribution, median, standard deviation, candidate-minus-reference direction and n independent sites. Multiple comparisons addressed via Holm within preregistered endpoint family where applicable; report all failure/dropout rates.
- For a single real site, report only descriptive case results and limitations, never generalized site CI or categorical superiority.

## Evidence acceptance / thesis release

Every defensible thesis claim must resolve in `claim_evidence_matrix.csv`: `claim_id → registered experiment_id → methods and versioned run IDs → hash-checked prediction + independent GT → locked test split → metric row + denominator → figure/table → literature DOI and code revision → reviewer/QA signature`. Auto-generated paper figures and result tables **filter on evaluated_real_heldout**, never synthetic or internal-only diagnostics. Contrast statuses remain `not_evaluated` when any link missing.

Minimum outputs: experiment_registry.json, per_view_metrics.csv, geometry_metrics.csv, ablation_results.csv, confidence_intervals.csv, failure_cases.csv, source_asset_hashes.csv, source-specific reproducibility_manifest.json, thesis claim ledger and `Thesis_Evidence` immutable zip. Preserve original data and maintain per-workstream Drive folders and unique `validation-c-*` run IDs.

**References:** ScanNet++ ICCV 2023 https://doi.org/10.1109/ICCV51070.2023.00008 ; 3DGS TOG 2023 https://doi.org/10.1145/3592433 ; 2DGS SIGGRAPH 2024 https://doi.org/10.1145/3641519.3657428 ; ETH3D CVPR 2017 evaluator https://github.com/ETH3D/multi-view-evaluation ; official ScanNet++ NVS code https://github.com/scannetpp/scannetpp/blob/21ba1dc724e0d09c6a1f72b0d63944f5363eb675/eval/nvs.py . These sources justify method design, **not direct quantitative comparability to MyCON scenes**. See `evidence_review.md` for DOI, license, code pins and limitations.
