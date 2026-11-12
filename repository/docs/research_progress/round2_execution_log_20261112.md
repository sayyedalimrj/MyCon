# MyCON B — Round 2 executable verification log (2026-11-12)

**Honest scope:** This is a **real local Python/pytest/CLI execution** against B modules and all 8 synthetic fixture files, each verified with the same Git blob SHA as the GitHub branch. The execution used a local directory with *interface-compatible copies* of \`pipeline.common.schedule_io\` and \`pipeline.common.bim_schedule_mapping\` because the container cannot clone/fetch GitHub over the network. Therefore this does **not** establish a full-checkout or integrated-repository test pass; the coordinator must run the existing full relevant suite in a genuine checkout before merge. Open3D is not installed; GPU geometry/real site tests were not executed.

## Environment

- OS: Linux 6.18.57 x86_64, glibc 2.41
- Python 3.13.5, executable \`/opt/pyvenv/bin/python\`
- pytest 9.0.2; NumPy 2.3.5; SciPy 1.17.0
- \`open3d\`: unavailable in local test environment.
- No real building/camera/depth/BOQ input was used.
- Fixture IDs: \`0AbCdeFgHiJkLmNoPqRsTu\`, \`1BcDefGhIjKlMnOpQrStUv\`; capture \`SYNTH-SESSION-1\`.
- Test root: \`/mnt/data/mycon-B-round2/repository\` (local mirror).

## Executed commands and observed exit codes

| # | Actual command | Exit code | Result |
|---|---|---:|---|
| 1 | \`python -m compileall -q pipeline/stage_09_progress pipeline/stage_11_schedule_variance\` | **0** | No compile errors for present local B modules |
| 2 | \`python -m pytest -q -ra tests/test_round2_progress_contract.py\` | **0** | \`28 passed, 1 warning in 0.10s\` |
| 3 | \`python -m pipeline.stage_09_progress.enrich_scientific_evidence\` (full args below) | **0** | \`STAGE9_SCIENTIFIC_OK elements=2 geometry_gate=pass\` |
| 4 | \`python -m pipeline.stage_11_schedule_variance.run_scientific_progress\` (full args below) | **0** | \`STAGE11_SCIENTIFIC_OK run_id=progress-round2-27 quality_gate=blocked status=unknown_evidence\` |

The one pytest warning was \`PytestUnknownMarkWarning: Unknown pytest.mark.lightweight\` because the local partial mirror had no upstream pytest marker registration. **Not** a test failure.

### Stage 9 CLI arguments (all actually executed)

\`\`\`bash
python -m pipeline.stage_09_progress.enrich_scientific_evidence \
 --element-metrics-csv configs/research_progress/round2_contract_fixture/elements_stage9.csv \
 --geometry-manifest configs/research_progress/round2_contract_fixture/bim_geometry_evidence.v1.json \
 --project-root . \
 --out-element-csv /mnt/data/mycon-B-round2/results/progress-round2-20261112/stage9_scientific_27.csv \
 --out-summary-json /mnt/data/mycon-B-round2/results/progress-round2-20261112/stage9_scientific_27.json \
 --data-date-utc 2026-01-16 \
 --expected-ifc-sha256 1111111111111111111111111111111111111111111111111111111111111111 \
 --expected-scan-sha256 2222222222222222222222222222222222222222222222222222222222222222 \
 --expected-registration-sha256 3333333333333333333333333333333333333333333333333333333333333333 \
 --expected-project-id mycon-round2-synthetic
\`\`\`

### Stage 11 CLI arguments (all actually executed)

\`\`\`bash
python -m pipeline.stage_11_schedule_variance.run_scientific_progress \
 --schedule-csv configs/research_progress/round2_contract_fixture/schedule.csv \
 --mapping-csv configs/research_progress/round2_contract_fixture/mapping.csv \
 --element-metrics-csv /mnt/data/mycon-B-round2/results/progress-round2-20261112/stage9_scientific_27.csv \
 --out-json /mnt/data/mycon-B-round2/results/progress-round2-20261112/scientific_progress_27.v2.json \
 --data-date-utc 2026-01-16 --research-run-id progress-round2-27
\`\`\`

Results: E1 physical fraction 0.8 with synthetic audit fields, E2 physical null and \`occluded\` (observability 0), weighted observability 0.375, planned 50%, bounds [60%, 85%], variance bounds [+10,+35] percentage points. The schedule-decision quality gate was **blocked**, as it must be for missing E2 quantity. The output is a *synthetic software test* and not a measured construction progress result.

## SHA-256 of fixture bytes in the executed local mirror

\`\`\`
bim_geometry_evidence.v1.json  80768472c67c45291318bcfe30faca00ca01951ef93f9d3e2851b168f2e7681a
camera_poses.jsonl             4060c5741b08ea48b0283af1bc17c48ab002270fe6d846919ef3df8161e8e522
element_visibility.jsonl       1b196b22d3d53a2f1ef91cf288b549b5979cb3dc042dce30bb8a2a3a6c76278b
surface_samples.jsonl          542851fca3e9ef039389dc3d760999f429523438950d86c82e997cdfa977c3b6
elements_stage9.csv            223ba25fa0e17fa5347af3fad932b8b553c0c387b54b04c918ff1a41852c659e
mapping.csv                    e71c73da8ee02e3843b2a8306b47b9e7dfbfbb1131ef246a641d0dde3d9539b7
schedule.csv                   f69680206e91ba94beb23455e032dfbfcbd443d5a60a435dd8ed82479989edb3
expected.json                  089cbba562bf925ccbbdcb0179c37e4948be391306f72286fdba877f9e3b239a
\`\`\`

The 28th test additionally validates the complete B→C exemplar JSON schema, stable IFC IDs, fraction-vs-percent units, null states, and independent one-view geometry QA. Executed locally with `python -m pytest -q -ra tests/test_round2_progress_contract.py`; observed **exit code 0, `28 passed, 1 warning in 0.10s`**.

## Outstanding acceptance gates

- **Not executed:** all original Stage 9, Stage 11, Stage 8 Open3D/GPU tests on a full checked-out repository, including existing \`tests/test_stage_11_schedule_variance.py\`; no CI workflow exists on the branch.
- **Not executed:** genuine A-produced Stage 8 manifest/real capture and real approved BOQ/baseline end-to-end, independent held-out control, site split or calibrated-confidence statistical evaluation.
- **Coordinator instructions:** in a real checkout, run \`python -m pytest -q -m lightweight tests/test_research_progress_geometry.py tests/test_research_progress_scientific.py tests/test_round2_progress_contract.py tests/test_stage_09_progress_interpretation.py tests/test_stage_11_schedule_variance.py\`; install and run Open3D Stage 9 tests separately, review regression, and retain merge blocked until successful.

This log never claims real research validation or overall full-suite success.
