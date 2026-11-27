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