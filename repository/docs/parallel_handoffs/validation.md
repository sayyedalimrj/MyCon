# Owner C Validation Handoff — MyCON, 2026-11-12

- **Work branch:** `research/mycon-validation-20261112`
- **Target branch:** `research/mycon-integration-20261112`; **Issue:** #35; **no merge by C**.
- **Contract base:** `f4fc2b4aad73e8516e031129780e56069e42257e`; branch-head at work start `33f1b68b2dc099db97d9e9101f56dd5ce2100d3f`.
- **Implementation commit SHA:** `07980f5f81e9e89e00f5317ca599ad2973fe058c` (original 13-file commit); follow-up handoff/QA commit recorded in PR #36.

## Owned additions

- `repository/benchmarks/mycon_2026/`: standalone stdlib metrics, PLY adapter, manifest leakage guard, read-only benchmark runner, optional chart exporter and usage guide.
- `repository/tests/test_research_validation_metrics.py`: deterministic lightweight fixture tests.
- `repository/scripts/research_validation_run.py`: CLI shim.
- `repository/configs/research_validation/`: versioned manifest example and eight preregistered ablation variants.
- `repository/docs/research_validation/protocol.md`: GT collection, split, fair baselines, uncertainty, references, thesis text, and failure taxonomy.
- `repository/requirements-research-validation.txt`: isolated optional SciPy, matplotlib, pytest dependencies.
- This file; no A/B or defaults changed.

## Public interfaces and integration

- Input manifest schema: `mycon.validation.v1`; relative read-only file paths. Each verified real test case requires sample site/building/capture identity, immutable capture hash, independent GT review metadata.
- Locked handoff adapters: Stage8 `scan_aligned.ply`, `bim_elements.jsonl`, independent survey anchor CSV; Stage9 `element_metrics.csv` with `global_id`, `completion_state`, optional `physical_progress_estimate`, numeric `evidence_confidence` and `observability_ratio`.
- Result schemas: `mycon.benchmark_result.v1`, detailed cases and hashes; missing fields = `not_executed`, malformed inputs = `failed`. No pipeline imports.
- **Request A:** publish metric-frame PLY, independent survey registration controls and per-method telemetry. A cannot treat design BIM as independent as-built GT.
- **Request B:** publish numeric [0,1] `physical_progress_estimate`, numeric calibrated [0,1] `evidence_confidence`, and explicit observability. Retain `not_evidenced` as unknown. Existing CSV remains accepted; missing optional data simply produces unavailable metrics.
- **Request lead:** review split/GT collection signoffs, paths and license before real experiment. A/B must not silently adopt this schema without agreement.

## Tests and evidence status

- Tests: local `pytest -q tests/test_research_validation_metrics.py` and CLI `python scripts/research_validation_run.py --help`, synthetic unit smoke, no GPU or model calls. 15 tests passed locally; no GPU or training. Full legacy 580+ test suite not runnable here because repository baseline was not present in local sandbox.
- Real reconstruction, registration, IFC, physical progress, calibration, time/memory and ablations: **not_executed** (no independently approved site GT / paired live outputs supplied).
- Synthetic fixture results: **tool-test only**; never in `scientific_results.json`, `paired_comparisons.json`, or paper comparison figures.
- All committed code/config/docs contain no original training data or checkpoints. Input hash evidence is generated *at runtime* from supplied manifest and artifacts, not invented here.

## Blockers

1. Independent dated as-built survey + controls and accepted element progress GT.
2. Real captures with source SHA256 and site-level locked split + reviewer signoff.
3. A/B verified outputs on the same held-out cases, versioned model/checkpoint information and peak VRAM/time sidecars.
4. Correctly qualified traditional/new baseline outputs, no claims using paper's different-dataset geometry numbers.

**Integrator action:** review PR, ensure only owned paths changed, run lightweight regression and verified per-site end-to-end benchmark; merge into Integration only after review. Do not merge to main directly.


---

# Round 2 Owner C Handoff — 2026-11-12

**Coordinator authority:** Read `repository/docs/parallel_research_round2_contract_20261112.md` at Integration commit `14f0fcdfa78a95bc3b990a2253a9b27ee8aeacbb` and coordinator comments in PR #36. **Last code/evidence commit before this handoff:** `1d1e482281fddbdac72b649f3f23cce191e898a7`. The exact final PR head follows this handoff commit; inspect PR #36 for current SHA. Work branch `research/mycon-validation-20261112` → base `research/mycon-integration-20261112` only. Owner C has not merged or rebased A/B branches.

## Files modified / added (strictly Owner C-owned)

- Modified `repository/benchmarks/mycon_2026/metrics.py`, `runner.py`, `plots.py`: numerically verified progress scored independently of categorical unknowns; counts for missing/abstention/classification/numeric GT; multiple geometry F-scores and orientation; site-cluster instead of building pseudo-replicated intervals, independent real held-out anchors, preregistered GT SHA gating, versioned activity/project/NVS tasks; runtime and package evidence; plot site counts.
- New `repository/benchmarks/mycon_2026/adapters.py`: read-only A `mycon.bim_geometry_evidence.v1` manifest validation with independent survey SHA and registration gates; B additive Stage9 verified-quantity fraction and B standalone Stage11 `mycon.scientific_progress.v2` with explicit percent units, approved BOQ weights and deterministic bounds distinct from CI.
- New `repository/benchmarks/mycon_2026/nvs.py`: independent held-out PSNR/SSIM and opt-in LPIPS; photometric metric is never converted into geometric validity.
- New `repository/benchmarks/mycon_2026/figures.py`: fail-closed manuscript figure suite for geometry, registration, physical/activity progress, NVS, reliability, selective risk and failure rate.
- New `repository/benchmarks/mycon_2026/evidence/round2_tool_smoke_20261112.json`: machine-readable actual synthetic-only execution evidence and source/data hashes.
- New `repository/tests/test_research_validation_round2_contract.py`: 16 new test functions covering split/quantity/units/unknowns/site CIs/A survey/NVS/GT SHA etc; existing Owner C tests remain unchanged.
- New `repository/scripts/research_validation_preflight.py`: read-only A/B v1/v2 CLI with exit status and SHA.
- Updated `repository/configs/research_validation/manifest_template.json` with held-out anchor IDs, B-v2/NVS paths and GT SHA keys.
- New `repository/configs/research_validation/benchmark_matrix.v2.json`: COLMAP, optional DUSt3R/VGGT, NVS/GS metric surface and Depth/Gravity/Sim3/ICP combinations; preregistered 5 seeds and ablations; all `not_executed`.
- New `repository/docs/research_validation/round2_methods.md`, `manuscript_outline_round2.md`, `execution_evidence_round2.md`: full independent GT, methods, risks, stratification, stats and English paper structure with unfilled results.
- Updated `repository/requirements-research-validation.txt` with optional NVS libraries; no root dependency changes.

**Frozen A/B interface and compatibility:** Stage8 aligned PLY and registration report unchanged. A optional `mycon.bim_geometry_evidence.v1` is only a candidate and not survey GT. B's Stage9 CSV still tolerates missing physical fraction; a non-null numeric fraction requires verified executed/planned quantity and authorized source, never geometric coverage. B's v2 activity/project quantity is **percentage 0–100**, not Stage9 **fraction 0–1**, and its deterministic bounds are not Wilson confidence intervals. No A/B production code was edited. The C runner does not invoke upstream training, change originals, overwrite existing runs or mutate producer files.

## Executed QA — evidence (not inferred)

- C source bytes matched **exact GitHub Git blob SHA-1**; local snapshot has no complete historical repository or A/B branches. Actual command `python -m pytest -q tests/test_research_validation_metrics.py tests/test_research_validation_round2_contract.py`: **exit 0; 31 passed, 0 failed**. `python -m compileall -q ...` also completed.
- `scripts/research_validation_run.py --help`, `scripts/research_validation_preflight.py --help`, synthetic B adapter preflight, synthetic Benchmark CLI run and scientific-figure status CLI: **all exit 0**. Exact commands, package versions, hashes and log audit in `repository/docs/research_validation/execution_evidence_round2.md` and `repository/benchmarks/mycon_2026/evidence/round2_tool_smoke_20261112.json`.
- Synthetic dataset: **1 fictional site / 1 building / 1 capture session / 0 real sites**, run `bench-round2-fixture-final2`; **9 task records, 5 synthetic tool-test evaluations, 4 missing input `not_executed`, 0 synthetic runtime errors**. Output status `scientific_status=not_executed`; `scientific_results.json=[]` (SHA-256 `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`). No synthetic numeric values are included in manuscript/paired tables.
- Test-environment Python `3.13.5`, Linux x86_64, CPU cores `5`, GPU `not_measured`, pytest `9.0.2`, numpy `2.3.5`, scipy `1.17.0`. Benchmark code snapshot's last implementation SHA before documentation `155c9b14d03ea2f203b7b53dcba6c3166f2fb52a`.

## NOT EXECUTED / still blocked (scientific publication gates)

- Actual baseline, ablation, independent real as-built survey/Ground Truth, real held-out registration TRE, valid site-cluster CIs, real Stage9/11 evaluation and GPU NVS/GS checkpoint **not_executed**.
- Full pre-existing 580+ lightweight repository regression and A+B+C integrated runtime **not_executed** because there was no full checkout available to execute; A/B PRs are not merged. GitHub connector does not run pytest/CLI.
- Required to proceed: signed independent survey with held-out control anchors, per-element accepted GlobalId and LoD, approved BOQ executed quantities and cross-activity weights, approved baseline and dated reviews; immutable site-disjoint test captures and SHA-256/GT hashes; real paired output files from A/B/COLMAP/optional new methods, checkpoints+hardware telemetry, valid confidence calibration, sufficiently many independent sites and complete regression.
- A real registration task now refuses scoring absent a disjoint fit/held-out anchor split; a real task also refuses evaluation absent matching independent GT SHA in the manifest. **Any claiming `gpu_validated`, `real_case_validated` or `publication_ready` would be unsupported.**

**Lead integration checklist:** confirm changed paths all belong to C; assess A/B interface contract and documented optional fields, sign off locked GT/splits, run actual full repository + three-way integration and paired verified-site benchmark; only the lead may merge C PR #36 into Integration (never straight to main).



---

# Model-agnostic geometry readiness — C handoff, 2026-11-12

**Coordinator contract read:** `repository/docs/integration/model_agnostic_geometry_readiness_contract_20261112.md` from Integration. **Latest coordinator comment read:** PR #36 priority correction at `https://github.com/sayyedalimrj/MyCon/pull/36#issuecomment-6828447013`. The prior 14021208/room MAX contract is explicitly deferred. **No reference MAX/CAD/BIM model was fitted, imported or used as GT.** No owner A/B files changed, no checkpoints/captures deleted, no training started. PR #36 targets Integration and remains unmerged.

## Changed C-owned paths

- `repository/benchmarks/mycon_2026/model_agnostic.py` — SHA-checked read-only adapter for optional `mycon.comparison_ready_geometry.v1`; finite bounded plane and edge scoring, normal orientation when stable IDs exist, real-structure cleanup retention and isolated-outlier removal, 3D corner TP/FP/FN, explicit extrapolation/unknown, geometry accuracy/Chamfer/F-score in correctly declared units.
- `repository/benchmarks/mycon_2026/grounding.py` — frame ROI/feature grounding and separate hallucination/abstention; reviewer-approved measurement IDs required for all asserted dimensions/progress. VLM never serves as GT.
- `repository/benchmarks/mycon_2026/runner.py` — two additive tasks `geometry_ready` and `vlm_grounding` with existing `bench-` run ID, test-split/hash and `not_executed` gates. Existing Stage8/9/11/NVS workflows preserved.
- `repository/tests/test_research_validation_model_agnostic.py` — local synthetic scoring and v1-schema regression definitions, including sloped planes, non-observed corners, tampered source hash, VLM and no-scientific-results gate.
- `repository/configs/research_validation/model_agnostic_stress.v1.json` — 23 registered perturbations, five seeds, classical/A/optional GS baselines and ablations, all `not_executed`.
- `repository/docs/research_validation/model_agnostic_geometry_readiness.md` — methods, independent GT, annotation protocol, RQs/Hs, corner unknown and statistical-denominator constraints, CLI and paper requirements.
- `repository/benchmarks/mycon_2026/evidence/model_agnostic_tool_test_20261112.json` — machine-readable local prototype test logs, versions, exact exit codes and SHA-256 input/outputs, with explicit limitations.
- This handoff file. No A/B files modified and no Integration/main writes.

## Real execution vs code only (mandatory separation)

**Actually executed:** local C source snapshot plus an expanded local model-agnostic prototype, **42 lightweight synthetic tests passed, 0 failed, pytest exit code 0**. Python 3.13.5, Linux x86_64, CPU count 5, pytest 9.0.2, NumPy 2.3.5, SciPy 1.17.0, matplotlib 3.10.8. Local `python -m benchmarks.mycon_2026.readiness --help`, synthetic `--include-fixtures` CLI, report figure-gating CLI and existing Runner `--help`: each exit 0. Synthetic `bench-bimfree-fixture-20261112`: 1 fictional site, 1 synthetic tool-scored case, 0 independent real sites; `scientific_results.json=[]` and no scientific images. Fixture source manifest SHA256 `7a881304f4f1dbf6c0a4803f03caf91cd0bc7dce897a5d6924ebddccdcda88a0`; independent synthetic GT SHA256 `a2655a87b816a9abc4c746678a5df628f18b7bfd75162927776d5440de5527b1`. Full SHA ledger is in evidence JSON.

**Important: these 42 local tests are NOT presented as a checkout regression of the new GitHub commits.** The locally tested expanded prototype has additional modules/implementation and is **not byte-identical** to the newly committed compact `model_agnostic.py`, `grounding.py` and GitHub Runner integration. Attempted `git clone --single-branch --branch research/mycon-validation-20261112 https://github.com/sayyedalimrj/MyCon.git` failed with DNS `Could not resolve host: github.com` (exit 128). Therefore new GitHub-source pytest/CLI and full relevant repository regression remain `not_executed`. Connector permits GitHub file writes but cannot invoke pytest. The coordinator must execute the checked-in `tests/test_research_validation_model_agnostic.py` and the relevant existing lightweight suite in a real full checkout before merging.

## Scientific and implementation blockers

A's actual versioned comparison-ready producer is not yet present/validated on the C integration checkout. Without paired independently reviewed real survey/camera/feature GT and same-site COLMAP/GS outputs, all numeric real geometry-quality, feature precision/recall, cross-view stability, calibration, GPU evaluation, CI and generalizable performance remain `not_executed`. Synthetic controlled stress scenarios demonstrate scorer behavior only; they **cannot** establish reconstruction accuracy, production readiness or publication readiness. Arbitrary reconstruction units are not meters; NVS image fidelity is not metric geometry validity; corner extrapolation is not confirmed observation.

**Required lead actions:** review A versioned schema against C adapter; run checked-in GitHub tests and lightweight/full A+B+C integration smoke; provide independent real sample with signed GT, source/model SHA, calibrated cameras, splits, registered heldout controls and fail/attempted log; pair baselines and repeat sites/seeds. The integration coordinator alone may merge PR #36 into Integration, never directly into main.


---

# MyCON Owner C — independent thesis evidence follow-up (2026-11-13)

- **Workstream:** Validation/C, add-only extension of previously integrated C code.
- **Work branch:** `research/mycon-validation-evidence-20261113`.
- **Target PR:** `research/mycon-integration-20261112`, not `main`. **No merge by C.**
- **Git commit SHA:** see originating follow-up PR head when connector write completes; branch may advance as this handoff is published. `MYCON_COMMIT_SHA` in runtime manifest records the actual evaluated repository code only when supplied.
- **Independent output run:** `validation-c-20261113-demo-no-real-data`. Google Drive workstream folder: `MyCON_Validation_C_20261113_validation-c-evidence-001` (no shared writers).
- **Contract authority:** `repository/docs/parallel_research_contract_20261112.md`, `repository/docs/parallel_research_round2_contract_20261112.md`, `repository/docs/integration/model_agnostic_geometry_readiness_contract_20261112.md`, read on Integration. Original PR #36 was found **already merged** 2026-11-12; this is a new additive follow-up, not a rewrite/rebase of #36.

## Files and interfaces

- New `repository/benchmarks/mycon_2026/evidence_export.py`: read-only exporter; `mycon.thesis_evidence.v1` input; hashed relative paths; immutable-unique `validation-c-*` output; independent GT and metric survey anchors gates, structural/camera metrics, per-view PSNR/SSIM and LPIPS `not_evaluated`, tail summary, site-bootstrap, pairs, telemetry audit fields, scientific figure gating. **Does not train or re-register anything**.
- New `repository/tests/test_research_validation_evidence_export.py`: 17 isolated synthetic tests including intentionally fictional flagged-real unit fixtures which test *only code paths*; they are never published as real case studies.
- New `repository/docs/research_validation/methods_and_parameters.md`: definitions, boundaries, survey requirements and thesis methodological limitations.
- New `repository/configs/research_validation/no_real_data_manifest.v1.json`: deliberately zero real cases and cannot be misread as measured field outcomes.
- This handoff. Existing C modules, Stage A/B code and frozen primary datasets/archives remain untouched.

## Tests actually executed

Environment: Linux x86_64 / Python 3.11+ tool environment; exact actual installed dependency versions are in generated `reproducibility_manifest.json` (do not substitute an asserted platform version). On local isolated authored source, not a full MyCON checkout:

1. `python -m compileall -q benchmarks/mycon_2026/evidence_export.py tests/test_research_validation_evidence_export.py` → **exit 0**.
2. `python -m pytest -q tests/test_research_validation_evidence_export.py` → **17 passed, 0 failed, exit 0**.
3. `python -m benchmarks.mycon_2026.evidence_export --manifest configs/research_validation/no_real_data_manifest.v1.json --output-root /mnt/data/Thesis_Evidence_final` → **exit 0**, unique `validation-c-20261113-demo-no-real-data` output; 0 real cases, 0 scientific metric cells; all science CSVs contain header only. `methodology_flow.png/svg` is a **non-empirical flow diagram**.

Module SHA-256 for this tested local snapshot: `evidence_export.py` `5c9d67f6843f4330367105b3bc4596213df161d251d7bb69c7347a0dcf725695`; test file `1aaf5d3196b59c04f2977801fc7e040b2bb9865a5e3ee773ad820d2bce4df027`.

## NOT tested / scientific limitations

- **GPU, actual MyCON reconstruction/checkpoint, real-site holdout, metric field survey, A/B integrated cases, full historical repository pytest, end-to-end ablation and actual NVS renders: not_executed.**
- **LPIPS: not_evaluated**, pending pinned model weights/normalization and audited software version. Actual execution time/VRAM/size from model training likewise `not_evaluated` without audited telemetry.
- No certified 3D Accuracy, Completeness, F1@τ, Chamfer, point-to-plane, planes/corners, PSNR, real-site site-bootstrap CI, or claim of superiority. Test fixtures are not empirical metrics.
- Current evidence exporter is intentionally additive and accepts independently reviewed case manifests but has not undergone external GT review or full-source integration testing. For scholarly claims, further blinded independent verification and sufficient independent sites are required.

## Coordinator/integration requests

1. Confirm additive files do not conflict with existing C or A/B paths; inspect GitHub branch content against tested SHA-256 bytes.
2. Execute all C existing lightweight tests and three-way A/B/C compatibility tests from full MyCON checkout; audit dataset manifests, GT signature/source instruments and real held-out split.
3. Supply immutable A comparison-ready geometry, camera poses/render pairs, independent surveyed GT, signed scale control with distinct fit and heldout IDs, verified telemetry, paired baselines; pre-register ablations before real test.
4. Refuse release/publication until all gates G0–G5 are satisfied. No PR is merged here.
