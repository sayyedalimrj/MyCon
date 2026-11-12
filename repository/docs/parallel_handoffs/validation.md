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

- Tests: local `pytest -q tests/test_research_validation_metrics.py`, synthetic unit smoke, no GPU or model calls. 14 tests passed locally; no GPU or training. Full legacy 580+ test suite not runnable here because repository baseline was not present in local sandbox.
- Real reconstruction, registration, IFC, physical progress, calibration, time/memory and ablations: **not_executed** (no independently approved site GT / paired live outputs supplied).
- Synthetic fixture results: **tool-test only**; never in `scientific_results.json`, `paired_comparisons.json`, or paper comparison figures.
- All committed code/config/docs contain no original training data or checkpoints. Input hash evidence is generated *at runtime* from supplied manifest and artifacts, not invented here.

## Blockers

1. Independent dated as-built survey + controls and accepted element progress GT.
2. Real captures with source SHA256 and site-level locked split + reviewer signoff.
3. A/B verified outputs on the same held-out cases, versioned model/checkpoint information and peak VRAM/time sidecars.
4. Correctly qualified traditional/new baseline outputs, no claims using paper's different-dataset geometry numbers.

**Integrator action:** review PR, ensure only owned paths changed, run lightweight regression and verified per-site end-to-end benchmark; merge into Integration only after review. Do not merge to main directly.
