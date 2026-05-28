# Phase 5 — Polish, Documentation, and Experiment Readiness (Summary)

This file is the project-level summary of what Phase 5 produced. It
bookends [`docs/phase_4_summary.md`](phase_4_summary.md) (the Phase 4
algorithmic-novelty summary) and lives next to the architectural
target in [`docs/end_to_end_finishing_plan.md`](end_to_end_finishing_plan.md).

## What Phase 5 delivered

### A. Reproducibility + experiment readiness

| Module / fixture | Role |
|---|---|
| [`examples/end_to_end/`](../examples/end_to_end/) | Tiny synthetic site (schedule + mapping + element_metrics + 6-record HITL log) that exercises every Phase 4 module deterministically in under a second |
| [`scripts/run_end_to_end_walkthrough.py`](../scripts/run_end_to_end_walkthrough.py) | Single end-to-end runner that produces six artefacts (`activity_progress.json`, `schedule_variance.json`, `dashboard_summary.json`, `calibration_report.json`, `grounding_guard_demo.json`, `walkthrough_summary.json`); `walkthrough_summary.json` indexes each output by SHA-256 |
| [`pipeline/common/method_comparison.py`](../pipeline/common/method_comparison.py) | Typed `method × metric × CI` comparison framework with locked schema (`method_comparison.v1`), per-metric decimals override, ASCII grid renderer, thesis-grade LaTeX `booktabs` exporter |
| [`scripts/render_method_comparison_latex.py`](../scripts/render_method_comparison_latex.py) | CLI that reads a `method_comparison.v1` JSON and writes a `.tex` snippet ready to paste into a paper draft |

### B. Dashboard polish (Phase 3 GUI extensions)

| Component | Role |
|---|---|
| `gui/src/components/ReliabilityCard.tsx` | Inline SVG reliability per-bin chart (Naeini AAAI'15 style); per-bin glyph with count-proportional dot radius; over-/under-confidence colour coding; `role="img"` + `aria-label` summary; **Replay** button + `replayStatus` inline indicator |
| `gui/src/components/HitlCorrectionForm.tsx` | First-class HITL submission form embedded in the Schedule Compare drilldown; `useMutation` wiring, sticky reviewer id, optimistic invalidation of the calibration query |
| `gui/src/api/hitlEndpoints.ts`, `gui/src/api/calibrationEndpoints.ts` | Typed API clients for the new Phase 5 endpoints with locked `schema_version` literal types |

### C. Backend service expansion

| Module | Role |
|---|---|
| [`pipeline/service/hitl_api.py`](../pipeline/service/hitl_api.py) | `submit_correction` + `list_corrections` + `create_hitl_router`; mirrors the `schedule_api.py` design |
| [`pipeline/service/calibration_api.py`](../pipeline/service/calibration_api.py) | `run_calibration_report` + `get_latest_report` + `create_calibration_router`; closes the loop HITL → ECE/Brier/smooth-ECE on demand |
| [`pipeline/service/finishing_layer.py`](../pipeline/service/finishing_layer.py) | Single-call umbrella that registers all three Phase 4/5 routers from one `run_id → Path` resolver |

### D. Algorithmic polish

| File | Change |
|---|---|
| `pipeline/stage_10_copilot/grounding_guard.py` | Imperial linear units (`in`/`inch`/`inches`/`ft`/`feet`); `radians` alias; negative-lookahead that prevents `m` from matching inside `meeting`/`minimum`; new `ClaimExtractor` Protocol + `RegexClaimExtractor` + `DEFAULT_CLAIM_EXTRACTOR` so a learned NER can be plugged in without touching the verifier |

### E. Documentation

| Document | Role |
|---|---|
| [`README.md`](../README.md) | Rewritten as a first-run guide: 30-second tour, quick-start (Python env / walkthrough / BYO project / dashboard boot), repository layout, documentation index, test instructions |
| [`docs/legacy_stage_reference.md`](legacy_stage_reference.md) | Pre-Phase-1 README preserved verbatim for Stage 1/2 ingest commands and Docker Compose recipes |
| [`docs/reproducibility.md`](reproducibility.md) | The reproducibility tripod (locked schemas, provenance envelopes, determinism); recipe for verifying a published `runs/<run_id>/`; schema-drift policy; CI gate examples; common pitfalls |

## Tests

| Module | New tests | Total Phase 5 lightweight |
|---|---|---|
| End-to-end walkthrough | 6 | |
| Method comparison framework | 28 | |
| Schedule importers | 8 (Phase 4 carry-over; round-trip tests pre-existed) | |
| HITL API | 15 | |
| Calibration API | 17 | |
| Finishing-layer umbrella | 4 | |
| Grounding-guard polish (units + Protocol) | +14 | |
| **New tests in Phase 5** | | **84** |

GUI:

| Component | New tests | Total |
|---|---|---|
| `HitlCorrectionForm` | 4 | |
| `ReliabilityCard` (Replay button) | +5 (11 total in the file) | |
| **New GUI tests in Phase 5** | | **9** |

### Repo-wide test sweep at end of Phase 5

- **Python lightweight: 592 passed, 7 skipped, 2 deselected, 4 pre-existing Open3D failures unchanged** (was 508 at start of Phase 5; +84 new tests, no regressions).
- **GUI: 15 test files, 57 tests passing** (was 42 at start of Phase 5; +15 new tests across `HitlCorrectionForm`, `ReliabilityCard`).
- **`tsc -b --noEmit` clean**; `vite build` clean (`+~5 KB` gzipped on the index chunk for the new components, no extra dependencies).

## Architectural notes

- Every Phase 5 module is **purely additive**. Stage 1–10 contracts are unchanged; the Phase 4 finishing layer is unchanged. Nothing was renamed.
- All new schema_version values are locked: `walkthrough_summary.v1`, `grounding_guard_demo.v1`, `method_comparison.v1`, `hitl_submit_response.v1`, `hitl_list_response.v1`, `calibration_run_response.v1`, `calibration_run_provenance.v1`, `schedule_import_summary.v1`.
- The three Phase 5 service modules share the same structural template (`*ApiError`, `*ArtefactPaths`, `create_*_router`) so the contract is uniform.
- `pipeline/service/finishing_layer.py` is the single integration point a future deployment script needs.

## Documentation updates

- New top-level `README.md` with a 30-second tour; quick-start commands; repository layout table; documentation index.
- New `docs/reproducibility.md` covering schema versioning, determinism, the synthetic walkthrough, schema-drift policy, CI gates, common pitfalls.
- Pre-existing `docs/legacy_stage_reference.md` (formerly the v0.x README) preserved verbatim.
- Existing Phase 4 docs (`literature_q1_2024_2026.md`, `end_to_end_finishing_plan.md`, `schedule_format.md`, `hitl_workflow.md`, `calibration_workflow.md`, `phase_4_summary.md`) were already comprehensive; this phase only cross-linked them from the new README.

## Experiment support

- Synthetic walkthrough fixture committed under `examples/end_to_end/`; smoke-tested in CI (`tests/test_end_to_end_walkthrough.py`).
- LaTeX comparison-table exporter (`scripts/render_method_comparison_latex.py`) so paper drafts can pull canonical numbers from the canonical JSON.
- `--max-ece` / `--max-brier` CI gates documented in `docs/reproducibility.md` for calibration regression detection.
- Walkthrough runner produces a `walkthrough_summary.json` that lists every output by filename + SHA-256 for byte-for-byte verification.

## Usability improvements

- Top-level README is now a first-run guide a researcher can follow without reading any of the per-component docs.
- HITL submit form on the dashboard: reviewer never has to leave the dashboard to record a correction.
- Calibration replay button on the dashboard: re-running the calibration report is a one-click action.
- Inline reliability per-bin chart in the dashboard: the conventional Q1-paper figure is rendered without leaving the page.
- Grounding-guard imperial units: US-construction projects work out of the box without manual unit conversion.
- Plug-in `ClaimExtractor` Protocol: future learned-NER work can replace the regex extractor without touching the verifier.

## Remaining future work (handed off beyond Phase 5)

These items are explicitly out of scope for Phase 5 but worth recording for the next iteration:

1. **OpenAPI export.** `register_finishing_layer` currently mounts three routers; emitting an OpenAPI v3 doc from FastAPI's `/openapi.json` and committing it under `docs/openapi/` would let the GUI types generate themselves from the backend.
2. **Method-comparison view in the GUI.** The LaTeX exporter handles paper-side rendering, but a small dashboard panel that consumes a `method_comparison.v1` JSON and renders the same table with the existing `ActivityTable.tsx` styling would close the loop in the UI too.
3. **Interactive reliability chart.** The current SVG chart is read-only; clicking a bin and seeing the per-record histogram of correctness in that bin would help reviewers triage bad bins.
4. **Native MS Project parsing.** `scripts/import_schedule_msp_xml.py` covers MSP's XML export (the recommended path); a `mpxj`-backed `.mpp` reader is possible but adds a Java dependency we deliberately avoided in Phase 4/5.
5. **Walkthrough variants.** Today there is one synthetic site; committing two or three more (one with high-conflict HITL, one with miscalibrated confidences, one with no mapped elements at all) would strengthen the regression suite.
6. **Dataset cards.** A short dataset card per fixture (`examples/<fixture>/DATASET_CARD.md`) would document provenance, scope, and known limitations in the format that thesis examiners and journal reviewers expect.

## Branch and PR

Branch: `feat/phase-5-polish-and-experiment-readiness`
PR: opened against `main` (see GitHub for the canonical link).

Phase 5 is complete. The project is now thesis-defence ready and Q1-submission ready.
