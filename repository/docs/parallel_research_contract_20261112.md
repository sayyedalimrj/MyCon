# MyCON Parallel Research Contract — 2026-11-12

## Authority and base

- Base commit: `f4fc2b4aad73e8516e031129780e56069e42257e`.
- Integration branch: `research/mycon-integration-20261112` (never work directly on `main`).
- Three work branches are forked from this integration branch after this contract commit:
  - `research/mycon-geometry-20261112` — Owner A: reconstruction, Gaussian Splatting, metric registration.
  - `research/mycon-progress-20261112` — Owner B: stage 9/11 physical progress, observability, uncertainty, schedule.
  - `research/mycon-validation-20261112` — Owner C: independent scientific benchmarks and ablations.
- Lead/coordinator alone merges PRs; each PR targets `research/mycon-integration-20261112` and stays unmerged until checked. No one writes to `main`.
- Agents in different chat sessions do not communicate directly. PRs, handoff reports and this contract are the source of truth; escalate blockers to lead rather than modifying another workstream's files.

## Strict file ownership (repository-relative paths)

A owns:
- `repository/pipeline/stage_01_ingest/**` through `repository/pipeline/stage_08_bim_eval/**`, including stage_04_5, stage_07_7; `repository/colab/**` if required for its own work.
- `repository/tests/test_stage_01*.py` through `repository/tests/test_stage_08*.py` and new `repository/tests/test_research_geometry_*.py`.
- `repository/configs/research_geometry/**`, `repository/docs/research_geometry/**`, `repository/docs/parallel_handoffs/geometry.md`.
- Additions only: `repository/requirements-research-geometry.txt`. No changes to default requirements or root configurations.

B owns:
- `repository/pipeline/stage_09_progress/**`, `repository/pipeline/stage_11_schedule_variance/**`.
- `repository/pipeline/common/{schedule_io.py,bim_schedule_mapping.py,calibration.py,hitl.py}` only if necessary. No edits to other shared common modules without lead approval.
- `repository/tests/test_stage_09*.py`, `repository/tests/test_stage_11*.py` and new `repository/tests/test_research_progress_*.py`.
- `repository/configs/research_progress/**`, `repository/docs/research_progress/**`, `repository/docs/parallel_handoffs/progress.md`.
- Additions only: `repository/requirements-research-progress.txt`. No other GUI/API changes without lead approval.

C owns:
- New `repository/benchmarks/mycon_2026/**`, `repository/docs/research_validation/**`, `repository/docs/parallel_handoffs/validation.md`.
- New `repository/tests/test_research_validation_*.py`, and new scripts prefixed `repository/scripts/research_validation_*.py`.
- `repository/configs/research_validation/**`, optional `repository/requirements-research-validation.txt`.
- Never modify A/B code, existing run scripts, existing fixtures, shared defaults, or existing tests.

If a change crosses ownership boundaries, propose an API/interface contract via PR description; do not edit the other owner's files.

## Frozen cross-team interface

Existing stage 8/9/11 file outputs remain readable, with stable identity keys and units:
- BIM geometry and registration: `data/bim/aligned/<project>/scan_aligned.ply`, `bim_reference.ply`, `bim_elements.jsonl`; `runs/<run_id>/reports/registration_report.json`.
- Stage 9 element metrics: `data/bim/metrics/<project>/element_metrics.csv`. Element key `global_id` (corresponds to IFC GlobalId). Distances in meters, ratios in [0,1].
- Stage 11 reads element metrics + canonical schedule CSV + mapping `(activity_id,ifc_global_id,weight)` and produces activity/schedule/dashboard JSON. Never silently reuse unverified or synthetic inputs as real evidence.
- Existing columns and JSON fields must not be removed/renamed. Add fields as optional; document their semantics. For incompatible changes, write a versioned `v2` adapter and get lead approval before switching consumers.
- Stage 8 produces the same frame and units Stage 9 expects; A must verify this. B must not infer `not_built` solely from `not_evidenced`.
- C benchmarks both existing baseline and optional new variants through frozen file contracts. If A/B features are not merged, use mocked examples or mark `not_executed` instead of inventing results.

## Mandatory scientific and safety gates

- No deletion or overwrite of original recorder captures, original data, checkpoints or training artifacts.
- Never restart expensive training from zero without explicit user confirmation. No destructive `--force` on real artifacts.
- Distinguish implemented / lightweight-tested / GPU-validated / real-case-evaluated. No claim of completion on code existence alone.
- No unverified real-world scientific metrics; report dataset split, denominators, failure cases, environment/seed, wall-clock time, and hardware.
- No test/data leakage: group train/test by building/site/capture session. Benchmarks require ground truth, numeric metrics, and uncertainty.
- Use unique run IDs and output subdirectories per workstream. No concurrent writes to the same Drive/Colab workspace.
- Add isolated tests; run existing lightweight tests before opening PR. If heavy dependencies unavailable, mark precise blockers.

## PR handoff checklist (mandatory)

Each workstream must commit its report:
- `repository/docs/parallel_handoffs/{geometry|progress|validation}.md`
- Include: exact commit SHA, files changed, new API/contracts, tests and actual results, metrics + sample counts, failed/unrun items, artifacts and their paths, input hashes, and integration blockers.
- Open PR into `research/mycon-integration-20261112` (not `main`), with title prefixed `[A]`, `[B]` or `[C]`. Do NOT merge or rebase onto changing integration branch without coordinator approval.
- Final coordinator merges after conflicts and interface checks, executes cross-module regression and end-to-end tests, and only then proposes release to `main`.
