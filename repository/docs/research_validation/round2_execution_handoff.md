# Owner C — Round 2 scientific validation handoff (2026-11-14)

Branch: `research/mycon-validation-evidence-20261113`; Draft PR #46 into `research/mycon-integration-20261112` only; Issue #48; original PR/code preserved and extended. Approved six-workstream contract read from Integration. No A/B/R/V/D/GUI/Colab or shared dependency file edited. No source video, `MyCon_R4`, checkpoint, historical COLMAP, GT, or training artifact modified; no force/rebase/merge/training.

## Changed files (all C-owned)

- `repository/docs/research_validation/evidence_review.md`: 6 DOI-verified peer-reviewed sources; 3 pinned real GitHub repositories/licenses/compatibility; official ScanNet++ executable NVS eval and ETH3D benchmark inspected but not executed; source-to-claim table and rejected options.
- `repository/benchmarks/mycon_2026/paired_views.py`: new strict per-view read-only comparison. Verifies entire preregistered test roster, identical source capture/independent GT, camera/intrinsics receipt SHA, RGB reference SHA, mask and color protocol; records missing views. Case metric only complete when all expected views match. Per-site paired bootstrap (exploratory with fewer than five distinct sites).
- `repository/benchmarks/mycon_2026/evidence_export.py`: retains old metrics and output schemas; integrates additive paired outputs, removes unsafe NVS ablation of case-level aggregates.
- `repository/tests/test_research_validation_evidence_export.py`: 6 new synthetic-only tests (23 total) exercising exact matches, omissions, camera mismatch, preregistration, synthetic exclusion and site-cluster CI.
- `repository/docs/research_validation/methods_and_parameters.md` appended protocol, `round2_runbook.md`, `repository/configs/research_validation/paired_view_manifest_template.v1.json`, this handoff and machine-readable test evidence under `repository/benchmarks/mycon_2026/evidence/`.

## APIs / schema / reproducibility

Input remains `mycon.thesis_evidence.v1` with SHA256 project-relative immutable files, explicit `coordinate_frame`, `units`, `capture_source`, independent real GT and metric control gates. Additive per case `paired_reference`, `paired_test_view_ids`, and per-view `camera_receipt_sha256`, `mask_sha256`, `color_space`. New `paired_view_audit.csv`, `paired_view_ablations.csv` and `paired_site_confidence_intervals.csv`. Old `ablation_results.csv` no longer emits unsafe NVS case-mean pairing. All unexecuted real comparisons and LPIPS explicitly remain `not_evaluated`.

## Actual test execution

- Command from `repository/`: `pytest -q tests/test_research_validation_evidence_export.py`
- Result: **23 passed, 0 failed, exit 0** on local Linux CPU, Python 3.13.5. Environment and local source SHA256/Git blob SHA1 are recorded in `benchmarks/mycon_2026/evidence/validation_round2_cpu_test_20261114.json` and `.txt`. `python -m compileall -q ...` exit 0.
- Test inputs: synthetic-only. Local isolated C-owned candidate snapshot; no entire checked-out historical Git repo in this offline container. Exact GitHub code/blob bytes must be compared after PR #46 push; CI/GitHub workstream run should be checked on the **final** PR commit, not an earlier green Integration status.
- Empty-manifest production exporter CLI executed with exit 0; `scientific_evaluated_experiments=0`, `n_real_metric_cells=0`, 0 completed paired comparisons; **no field measurement**. Schematic methodology plot is not a performance graph.

## Explicit NOT EXECUTED / validation blockers

- No real same-heldout 6999-versus-new or 192-versus-expanded rendered camera pairs, no independent real NVS PSNR/SSIM/LPIPS or qualitative side-by-side image from matched real source; LPIPS not configured.
- No independent signed surveyed as-built GT, metric scale, held-out registration anchors, real plane/edge/corner truth, repeated noise/occlusion source variants or reliable CI on independent sites.
- No approved IFC/BOQ quantities and construction-progress ground truth, real A/B contracts exercised, GPU/VRAM/time telemetry audit, full historical regression or A+B+C integrated checkout test.
- No external ScanNet++, ETH3D, Tanks and Temples benchmark executed; exact reproducible command and blockers recorded in `evidence_review.md`.

**Coordinator request:** confirm G0 ownership and all changed files; run specialized GitHub Actions workflow and actual end-to-end integration on exact PR head; furnish independently reviewed files with immutable source SHA and a signed, disjoint site/session test set. Only coordinator may merge; no paper accuracy, scale, progress or superiority claims authorized.
