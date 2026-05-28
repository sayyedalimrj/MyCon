# Phase 4 — Algorithmic Novelty + End-to-End Finishing Layer (Summary)

This file is the project-level summary of what Phase 4 produced. It is
the bookend to `docs/literature_q1_2024_2026.md` (the literature map)
and `docs/end_to_end_finishing_plan.md` (the target architecture).

## What Phase 4 delivered

### A. Algorithmic novelty (Q1-grounded, white-space)

| Module | Role | Lit grounding |
|---|---|---|
| `pipeline/common/calibration.py` | ECE / MCE / Brier / smooth-ECE on the discrete confidence labels | Naeini AAAI'15; Brier MWR'50; Roelofs AISTATS'22; Błasiok-Nakkiran ICLR'24 |
| `pipeline/common/hitl.py` | Append-only HITL corrections store + last-write-wins replay + audit trail | Beck WACV'24; Rožanec arXiv 2307.05508 |
| `pipeline/stage_09_progress/multiview_fusion.py` | Trusted multi-view evidential fusion with explicit Dempster conflict mass at the per-IFC-element level | Han et al. ICLR'21 (Trusted MVC); Sensoy NeurIPS'18; LREC-COLING'24 |
| `pipeline/stage_10_copilot/grounding_guard.py` | Claim-decomposition + numeric-tolerance VLM hallucination guardrail | Pelican arXiv 2407.02352; CoRGI arXiv 2508.00378; Liu/Liang arXiv 2403.14003 |

These four modules collectively close the **three white-space areas**
identified in `docs/literature_q1_2024_2026.md` §6:

1. Calibrated, evidence-linked, HITL-replayable progress decisions.
2. Explicit Dempster conflict mass at the per-BIM-element level (first
   application of Trusted MVC to scan-vs-BIM construction progress
   monitoring).
3. VLM claim-grounding against a deterministic evidence package with
   numeric-tolerance verification.

### B. End-to-end finishing layer

| Module | Role |
|---|---|
| `pipeline/common/schedule_io.py` | Canonical schedule CSV loader (`schedule.v1`) — strict-on-output, lenient-on-input. |
| `pipeline/common/bim_schedule_mapping.py` | BIM ↔ schedule mapping CSV loader + validator (`bim_schedule_mapping.v1`). |
| `pipeline/stage_11_schedule_variance/` | New Stage 11 pipeline stage: per-activity rollup (Wilson 95 % CIs) + run-wide variance + dashboard JSON. Lightweight — no Open3D / OpenCV. |
| `scripts/import_schedule_{msp_xml,p6_xer,generic_csv}.py` | Side-car importers; pipeline never parses `.mpp`. |
| `pipeline/service/schedule_api.py` | 5 read endpoints exposing the artefacts: activities, activity detail, variance, dashboard, element status. |
| `gui/src/pages/ScheduleCompare.tsx` + `KPIStrip` + `ActivityTable` + `ReliabilityCard` + typed `scheduleEndpoints` client | Dashboard *Schedule Compare* page that ties every layer above into a single glanceable view. |

### C. Documentation

- `docs/literature_q1_2024_2026.md` — full Q1 literature map for 2024–2026.
- `docs/end_to_end_finishing_plan.md` — the architectural target.
- `docs/schedule_format.md` — canonical CSV reference.
- `docs/hitl_workflow.md` — HITL corrections workflow.
- `docs/calibration_workflow.md` — calibration / reliability workflow.
- `docs/phase_4_summary.md` — this file.

## Testing

| Module | New tests | Total Phase 4 lightweight |
|---|---|---|
| Calibration | 24 | |
| HITL | 22 | |
| Multi-view fusion | 25 | |
| Grounding guard | 31 | |
| Schedule I/O | 24 | |
| BIM↔schedule mapping | 15 | |
| Stage 11 | 26 | |
| Schedule API | 17 | |
| Importers | 8 | |
| **Total Phase 4 module tests** | | **192** |

Repo-wide lightweight test sweep: **508 passed, 7 skipped, 2 deselected**
(equal to the registry's `geometry`-marked tests). The 4 pre-existing
Open3D smoke-test failures are unrelated to Phase 4 work and were
present before the branch opened.

GUI tests: **13 files, 42 tests passing** (was 36 before Phase 4; +6 for
`ScheduleCompare`).

`tsc -b --noEmit` passes; `vite build` succeeds.

## Stage registry

The canonical stage registry now has **16 entries** (was 15). Stage 11
is wired in with `capabilities = {OPTIONAL}` so it stays
laptop-runnable.

## Architectural notes

- Every Phase 4 module is *additive*. Nothing in Stages 1–10 was
  rewritten or had its public contract broken.
- Every new artefact carries the standard provenance envelope from
  Phase 1: `schema_version`, `config_hash`, `git_sha`, `seeds`,
  `inputs`, `generated_at_utc`.
- The `geometry` marker is honoured everywhere: every Phase 4 module
  runs in the laptop test set; nothing here pulls Open3D / OpenCV /
  IfcOpenShell at module load.

## What Phase 5 should pick up

Phase 5 is the polish + reproducibility + paper-readiness phase. The
hand-off from Phase 4:

1. **Per-bin chart in `ReliabilityCard`** — currently only the four
   metrics are rendered. The bin-by-bin reliability chart is the next
   visual.
2. **HITL submit form on the Schedule Compare drilldown** — the
   `POST /api/v1/hitl/corrections` endpoint is described in
   `docs/end_to_end_finishing_plan.md` §6 but is not yet implemented.
3. **Calibration replay button** — a one-click "rebuild calibration
   report against latest HITL log" action on the Reliability card.
4. **End-to-end walkthrough fixture** — a tiny but realistic synthetic
   site (schedule + mapping + element_metrics) that exercises every
   Phase 4 module and is committed to the repo for the paper's
   reproducibility section.
5. **Method-comparison framework** — schema for `method × metric × CI`
   tables exportable as LaTeX, called out in the literature map §5
   row 6.
6. **Extra unit conversion** for the grounding guard (millimetres to
   inches; degrees to radians) and an option to plug in a learned NER
   for claim extraction behind the same interface.

## Branch and PR

Branch: `feat/phase-4-novelty-foundations`
PR: https://github.com/sayyedalimrj/MyCon/pull/8
