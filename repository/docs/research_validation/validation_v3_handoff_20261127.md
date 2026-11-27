# Owner C — validation round 3 / read-only real A intake / 2026-11-27

- **PR:** #46, same branch `research/mycon-validation-evidence-20261113` → Integration `research/mycon-integration-20261112`, Draft / unmerged. Branch source at start `cf9aea1dd968955044989107948438767fefc7e1`. Final SHA should be read from PR #46 after submission. No force/rebase, no new PR, no other team's code changes.
- **Contract:** `repository/docs/integration/parallel_6_streams_approval_20261113.md` as amended; C-only sources, tests, configs, benchmarks, reports.
- **Source A:** read-only Drive `MyCON_Civil_Geometry_Round3_Camera_Registration_20261114.zip`; SHA256 `902e11df91b7c159f787b5730cd946ba220def57ed631882fd341ad9a26f996c`; A producer PR #42. Original source/captures, MyCon_R4, checkpoints, historical COLMAP and A-owned artifacts untouched.

## New Owner C paths

- `repository/benchmarks/mycon_2026/real_intake_audit.py` — validate 25 output/source/photo SHA receipts and independently recompute in-sample sparse COLMAP SIMPLE_RADIAL reprojection from A original text. Unique run, refuses overwrite, fail-closed hash/schema/invalid path; does not read or write survey GT.
- `repository/benchmarks/mycon_2026/diagnostic_figures.py` — appendix-only figure, refuses any non-internal metric CSV.
- `repository/tests/test_research_validation_real_intake_audit.py` — nine new isolated synthetic correctness tests, including wrong hash, unsafe path/run ID, unregistered receipt, no overwrite and projection. Existing `tests/test_research_validation_evidence_export.py` unchanged.
- `repository/docs/research_validation/{independent_benchmark_protocol_v3.md,ground_truth_sop_v3.md,real_intake_results_20261127.md,validation_v3_handoff_20261127.md}` — independent G0–G5, GT SOP, locked test, baseline/ablation/engineering interpretation, actual real-source audit.
- `repository/configs/research_validation/{frozen_test_set_template.v1.json,gt_manifest_template.v1.json,claim_evidence_matrix_v3.csv,ablation_plan_v3.csv}` — schemas and prespecified comparisons; unpopulated ground truth and no invented metric cells.
- `repository/benchmarks/mycon_2026/evidence/{validation_c_real_intake_20261127.json,real_source_a_20261127/{real_input_audit.json,colmap_internal_reprojection.csv,source_asset_hashes.csv}}` — run receipts and 192 per-image **in-sample** pixel diagnostics, not field-validated paper values.
- `repository/docs/parallel_handoffs/validation.md` appended for coordinator.

## Actual executed evidence

1. `python -m compileall -q benchmarks/mycon_2026/real_intake_audit.py tests/test_research_validation_real_intake_audit.py` exit 0.
2. `python -m pytest -q tests/test_research_validation_evidence_export.py tests/test_research_validation_real_intake_audit.py` **32 passed, 0 failed, 0 skipped; exit 0**, Python 3.13.5 Linux CPU. This is the local C source snapshot. GitHub exact-updated-HEAD CI still to be verified, not already claimed.
3. `PYTHONPATH=. python -m benchmarks.mycon_2026.real_intake_audit --package <A zip> --expected-package-sha256 <verified SHA> --output-root <new> --run-id validation-c-real-source-a-20261127-003` exit 0; actual SHA receipts and independent internal reprojection:
   - 192 COLMAP image records, 51,894 sparse points, 283,059 projected correspondences, 25 correct packaged SHA receipts.
   - Median of per-image median **in-sample** error 0.744059 px; median of per-image p95 2.088479 px. Output rows explicitly `scientific_eligible=false`, `data_kind=real_internal_only`, metric scale and registration false.
4. Thesis Evidence v3 zip packaged, hash-audited, 30 files. Actual original A ZIP NOT included (contains real photos and producer-owned geometry), only C-derived diagnostic outputs/hash receipts and C-owned code/docs.

## Unrun / scientific failures to overcome

- **NOT EVALUATED:** independently surveyed metric GT, fit-vs-holdout TRE, as-built surface/plane/corner F1 and dimensional accuracy, 6999-vs-new and 192-vs-expanded same-test NVS, LPIPS, real site-cluster CI and approved BOQ/IFC physical progress; no valid field/generalization claims.
- Missing independent frozen test-camera roster, 6999 training membership, matched real R new render outputs, metric bridge `A depth↔COLMAP camera UNREGISTERED`, independent control survey and BOQ approvals. The current A source is real, but from the same reconstruction optimization, so it only verifies **internal** camera-fitting consistency. A geometry-to-image overlay is not inferred.
- **Lead integration asks:** preserve all A/B/R/V/D ownership, approve survey and blind test roster, obtain signed GT + matched producer manifests with SHA, execute exact PR-head dedicated CI/combined regressions, and merge #46 only after verified G0–G5. No this-chat merge.

**Scientific rule:** A descriptive real-source diagnostic is not an independent field experiment; synthetic fixtures are not real. Every primary paper comparison row remains `not_evaluated` until explicit signed independent GT and matched frozen tests.