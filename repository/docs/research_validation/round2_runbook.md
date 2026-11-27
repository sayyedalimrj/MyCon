# MyCON C — reproducible runbook (no real scores supplied)

Repository cwd: `repository/` in a real checkout on `research/mycon-validation-evidence-20261113`.

```bash
python -m pytest -q tests/test_research_validation_evidence_export.py
python -m compileall -q benchmarks/mycon_2026/evidence_export.py benchmarks/mycon_2026/paired_views.py
python -m benchmarks.mycon_2026.evidence_export \
  --manifest configs/research_validation/no_real_data_manifest.v1.json \
  --output-root /path/to/unique_C_outputs
```

The no-real-data example produces zero measured scientific cells and `not_evaluated` chart statuses. Never edit or overwrite any source/checkpoint or run directory; choose a new unique `validation-c-*` run ID and output root. Package output only after independent reviewer approval of verified real GT, frozen heldout roster, camera/pose/intrinsics receipts, reference image SHA, capture SHA, units, coordinate frame and separate metric-scale survey. For each run save stdout/exit codes, Python/dependency versions, Git commit, input SHA and output `reproducibility_manifest.json`. Report failures and missing cases, not only successful ones.

Reference implementation used for evaluation *definitions*, not copied: [ScanNet++ official NVS evaluation](https://github.com/scannetpp/scannetpp/blob/21ba1dc724e0d09c6a1f72b0d63944f5363eb675/eval/nvs.py) and [ETH3D multi-view benchmark](https://github.com/ETH3D/multi-view-evaluation/tree/0daa4f4de036d47d37a6ba054c51fd66642ef896). Both external evaluation commands `not_executed` for this run.

Real validation prerequisites and safeguards: independent signed ground truth with split fit/heldout anchors; identical test camera views and reference image hashes; no train/test leakage; site-based paired CI; LPIPS only with pinned weights; object progress separate from rendered appearance. The figure `methodology_flow.png` is a schematic diagram, not a measured performance plot.
