# Draft article structure — MyCON independent scientific evaluation (English)

**Working title:** An Evidence-Calibrated, Metric-Scale 3D-to-BIM Framework for Construction Progress Assessment: A Preregistered Evaluation Protocol.

**Manuscript state:** METHOD DESIGN ONLY / RESULTS NOT_EXECUTED. No accuracy, superiority, generalized reliability or deployment claim is currently defensible. All numerical results tables and figures must be generated from independently measured real cases and immutable benchmark manifests, never from synthetic fixtures.

## Proposed contributions (to be demonstrated, not assumed)

1. A read-only, reproducible and versioned assessment pipeline connecting geometric 3D reconstruction, independent metric survey control, BIM element identities and construction quantity measurements.
2. Explicit distinction between geometric agreement, visibility, categorical evidence acceptance, verified executed physical quantity and authorized schedule baselines, preventing unsupported 0%-progress assignments from occlusion.
3. Separate estimands for classification and numeric progress. Numeric verified quantity is scored even when an image-only state classifier abstains; no numeric fraction is inferred from geometric coverage or the class label.
4. Fair same-site comparisons with classical COLMAP and optional DUSt3R/VGGT/3DGS/metric-alignment methods, seeded ablations and site-clustered uncertainty; NVS image fidelity is not interpreted as as-built geometry accuracy.
5. A reproducible failure and resource-cost audit including missing prediction denominators, unknown/abstention, input hashes, execution provenance and selective confidence calibration.

## Suggested article organization

### 1. Introduction and problem formulation
Construction monitoring needs evidence linking observed imagery to measured work quantities and approved project baselines. Typical scan-to-BIM coverage and nearest-neighbor agreement are not themselves proof that a physical element is complete. State the challenge of scale, registration, partial visibility, data leakage, uncertainty and highly imbalanced construction progress labels.

### 2. Related work
Group (a) classical SfM/MVS (COLMAP), (b) feed-forward dense 3D DUSt3R/VGGT, (c) GS rendering and extracted surfaces, (d) scan-to-BIM/IFC integration, (e) observable-vs-built progress evidence, (f) quantity/BOQ-weighted 4D BIM and schedule variance, (g) selective prediction and uncertainty calibration.

**Reference boundary:** Duran, Waibel, Bickel, Armeni and Schlueter, *Automated facade solar retrofits using feed-forward 3D reconstruction and generative image editing*, Automation in Construction 193 (2027), 107294, DOI 10.1016/j.autcon.2026.107294, code https://github.com/ycdrn/image2bipv. Relevant methodological elements include feed-forward scene geometry, gravity and metric scale, orthorectification and independent held-out evaluation. Their Zurich Facades/LSAA outcomes measure facade aspect ratio, size or floor count, **not** MyCON's held-out as-built reconstruction, IFC mapping, progress or schedule. No direct numerical ranking across those datasets.

### 3. Proposed architecture and estimands
Introduce (i) input capture/feature reconstruction, (ii) global metric scaling and gravity, (iii) optional GS and novel-view image quality, (iv) BIM registration and IFC element mapping, (v) explicitly observed/occluded/unknown evidence, (vi) independently approved quantity progress, (vii) approved schedule variance. Present a notation table:
- P_i in [0,1] = verified executed_i / approved planned_i (not scan surface coverage)
- O_i in [0,1] = audited observable design surface / eligible surface
- C_i in [0,1] = independently calibrated classification correctness probability
- A_j and P_project in [0,100] = approved-weighted activity/project completion percentages
- V_j in percentage points = A_j - authorized planned completion at date t
- Unknown quantities imply deterministic partial-identification bounds, NOT statistical intervals.

### 4. Methods and Ground Truth
Independent same-date total station or TLS survey for GT geometry and held-out TRE control anchors, approved IFC GlobalId inventory with change history, signed measured quantities and BOQ weighting, baseline ID and approval dates, blinded dual reviewer annotation and adjudication. Report sensor accuracy and engineering tolerance (metres), per-site GT counts and sampling masks. Distinguish registration fitted anchors from withheld evaluation anchors. Train/validation/test splits remain site-disjoint with building, session, exact hash and near-duplicate safeguards.

### 5. Experimental setup and preregistration
Fix capture count, image resolution, check-point, frame, seed, GPU class, software and data hashes. Require paired samples for every baseline. Compare COLMAP, available DUSt3R/VGGT, metric geometry extracted from GS and combined Depth/Gravity/Sim3/ICP variants. Record ablation variants at seeds [11,23,42,67,101]. Prohibit forced training restart or modification of protected A/B files. Declare primary geometry F-score at 5 cm, held-out TRE RMSE (m), and verified physical-progress MAE (percentage points) before inspection. NVS PSNR/SSIM/LPIPS is a separate photometric endpoint.

### 6. Evaluation and statistical analysis
Directed accuracy/completeness and symmetric Chamfer L1/L2; F-score in four metric tolerances; held-out registration TRE, scale ratio and orientation; IFC GlobalId class/identity; class confusion/F1 plus distinct missing/abstention denominators; numeric physical progress versus independent quantity; activity/project/schedule percentage-point errors; Brier/ECE and selective risk with valid confidence. Report runtime and peak memory and failures/attempts/missing; 2,000 seeded bootstrap resamples of whole sites for 95% descriptive intervals and paired method deltas on shared sites only.

### 7. Results — tables generated from real inputs only
- Table 1: Site/building/session split, GT instrument uncertainty, lighting/occlusion and exclusions. **not_executed**
- Table 2: Reconstruction (Chamfer/F-score vs tolerance), held-out TRE, scale/orientation and IFC identity. **not_executed**
- Table 3: Classification F1, unknown, observability and numeric element/activity/project/schedule error. **not_executed**
- Table 4: Paired baselines and associated site-cluster CI, effect sizes, failures, runtime and VRAM. **not_executed**
- Table 5: Registered five-seed ablations with exact GT and method version. **not_executed**
- Figure 1: Geometry and TRE metric ranges; Figure 2: physical/schedule error; Figure 3: reliability and selective risk; Figure 4: failure rates; Figure 5: NVS fidelity. All **not_executed** until verified real test.

### 8. Discussion — questions, not unsupported conclusions
Discuss when geometric agreement transfers to robust BIM alignment, when independent quantity is possible under occlusion, how uncertainty and abstention trade off, false positives/negatives by element and site condition, whether improved geometry changes downstream progress, and engineering implications. Distinguish statistically descriptive differences from supported causal ablation interpretation; include limitations of survey uncertainty and calibration.

### 9. Threats to validity and limitations
Small independent-site counts, nonuniform data collection, hidden duplicate/scene overlap, annotation disagreements, IFC modeling LoD/class ambiguity, moving materials and scaffolding, time mismatch between survey and image capture, domain shift, training-data contamination in foundation models, checkpoint selection, mixed hardware profiles, and no certified progress measurement when approved quantities are absent. Calibration confidence and deterministic bounds measure different forms of uncertainty.

### 10. Reproducibility and data/code availability
Commit SHA, model/checkpoint hashes, GPU/CPU/RAM, software versions, seed, exact command/exit code, immutable captures/GT with restricted access as needed, evaluation output JSON/CSV/LaTeX, hashes and structured failure ledger. Source paths confined to C-owned benchmarks/docs; merge is coordinator controlled.

## Draft publication gating statement

At the protocol stage, the independent benchmark is implemented and subjected only to synthetic tool validation. Independent real-site as-built surveys, held-out camera imagery, quantity records and approved schedule baselines are not yet linked to signed evaluation manifests. Consequently the paper must not include quantitative real-case performance conclusions. All result placeholders are explicitly **not_executed**, pending documented, independently reviewable experimental runs.
