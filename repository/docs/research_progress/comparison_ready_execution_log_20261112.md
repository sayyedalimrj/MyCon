# MyCON Owner B — Comparison-ready geometry executable verification (2026-11-12)

## Evidence level

**Executed in a local CPU Python environment**, not a full checked-out GitHub repository. The newly tested B adapter, test source, standalone CLI and all seven A→B synthetic fixture files were checked for **byte-identical Git blob SHA** against PR #37. The GitHub API was used for source commits; local git network/DNS is unavailable. Legacy tests requiring full pipeline and Open3D were **not executed** this round.

The inspected Integration contract is repository/docs/integration/model_agnostic_geometry_readiness_contract_20261112.md. The latest coordinator comment in PR #37 explicitly supersedes the earlier 14021208 room/MAX instruction.

## Runtime

- Linux 6.18.57 x86_64, glibc 2.41
- Python **3.13.5**, pytest **9.0.2**, NumPy **2.3.5**, SciPy **1.17.0**
- CPU-only; no Open3D in this environment; no real building or IFC model loaded
- Test workdir: /mnt/data/mycon_B_round3/repository
- Synthetic fixture ID: geom-ready-synthetic-01, project: mycon-synthetic-only; arbitrary reconstruction coordinate units
- Experiment seed: 0 (synthetic fixture)

## Exact commands and observed exit codes

Workdir for each: /mnt/data/mycon_B_round3/repository.

1. **Exit 0 — compileall**

       python -m compileall -q pipeline/stage_09_progress

   Observed stdout: empty.

2. **Exit 0 — pytest**

       python -m pytest -q -ra tests/test_comparison_ready_progress_contract.py

   Observed: 21 passed in 0.16s. Tests include: nonmetric geometry, false/degenerate corners, candidate-vs-observed, duplicate IDs, invalid planes, tampered SHA, traversal, missing independent scale, invalid registration, IFC checksum mismatch, repeated feature ownership, camera QA, occlusion, missing association and explicit metric mapping. There is no automatic numerical progress from any geometry feature.

3. **Exit 0 — Stage9 geometry diagnostic CLI**

       python -m pipeline.stage_09_progress.run_comparison_ready_diagnostics --manifest configs/research_progress/comparison_ready_fixture/comparison_ready.json --project-root . --out-json /mnt/data/mycon_B_round3/comparison_ready_audit_second.json --expected-project-id mycon-synthetic-only

   Observed: COMPARISON_READY_DIAGNOSTICS_OK planes=4 edges=1 corners=2 IFC=unknown progress=null. The JSON output has SHA-256 bf810d39b860d81c940f3be52e373104d4d32a82131d36bbfce700037cad04f9. It has diagnostic_quality_gate=pass, comparison_gate=blocked_without_independent_ifc_mapping, metric_scale_verified=false, registration_verified=false, physical_progress_estimate=null. One of the two nominal corner candidates is identified as degenerate and zero are audit-supported corners.

4. **Exit 2 — negative CLI: same output path**

       python -m pipeline.stage_09_progress.run_comparison_ready_diagnostics --manifest configs/research_progress/comparison_ready_fixture/comparison_ready.json --project-root . --out-json /mnt/data/mycon_B_round3/comparison_ready_audit_second.json --expected-project-id mycon-synthetic-only

   stderr: COMPARISON_READY_FAILED: output path already exists. No previous output was overwritten.

5. **Exit 2 — negative CLI: wrong/invalid manifest**

       python -m pipeline.stage_09_progress.run_comparison_ready_diagnostics --manifest configs/research_progress/comparison_ready_fixture/planes.json --project-root . --out-json /mnt/data/mycon_B_round3/comparison_ready_bad.json --expected-project-id mycon-synthetic-only

   stdout: COMPARISON_READY_DIAGNOSTICS_BLOCKED planes=0 edges=0 corners=0 IFC=unknown progress=null. No fabricated BIM/progress claim.

## Git-blob verification

These SHAs were compared against remote GitHub blobs on branch research/mycon-progress-20261112:

- pipeline/stage_09_progress/comparison_ready_adapter.py: 88914df68884fbd214e19af49df7751d199d8145
- pipeline/stage_09_progress/run_comparison_ready_diagnostics.py: 75f4c8e2f47f0c6401763eb641b4eee623b9e0d9
- tests/test_comparison_ready_progress_contract.py: 7acf92ff4d70ea3a0fa295eb8a2f96d73c4c49d0
- fixture clean_cloud.npy: 726224b85bdb8c717988e3bca5f4d4f25fc070a4
- fixture clean_cloud.ply: 3bd452fdc3bffa3d013ab3acbd1a15fdcedb8622
- fixture comparison_ready.json: 270e7e7fb227a94a0c553b85dc44c691d35e4d9b
- fixture planes.json: e169756b867bb8348c2993430909e88d5e36cd6e
- fixture edges.json: 36970e0d94a104c693a6deecb5b19a48f2504b91
- fixture corners.json: 753ca12fe68022d598ff9b3c11c4526f0b9a784d
- fixture cameras.json: ca6d1bd120af6be7bf0b0f24cb3c0d6cde27a51e

Git SHA-1 blob IDs are file-identity checks; the separate manifest SHA-256 hashes verify sidecar bytes. No real-site performance metric was measured.

## NOT executed or validated

- Original full checked-out MyCON Stage9 Open3D tests, Stage11 legacy regression, integration pytest, actual Stage9 scientific enrichment on a real point cloud and full Stage11 CLI on a real approved baseline: **NOT EXECUTED IN THIS ROUND**.
- Original Stage8 A-produced comparison-ready manifest and real image/camera/registration/raycast QA: **NOT AVAILABLE / NOT EXECUTED**.
- A real approved BOQ, executed quantity, signed as-of measurement, independent surveyed anchor, metric scale, matched IFC and baseline: **BLOCKED_MISSING_INPUT**.
- No benchmark precision, recall, publication-ready result or statistical CI claimed. B's separate earlier Round2 28-passing-test log remains in docs/research_progress/round2_execution_log_20261112.md; it is **not a new full-repo pass**.

Coordinator must run full integrated tests in a checkout and supply actual A/IFC/site evidence before merge or publication.
