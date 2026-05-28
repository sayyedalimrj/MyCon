# End-to-End Finishing Plan: From As-Built Capture to Schedule-Aware Dashboard

This document defines the **terminal target** of the project: how the
pipeline that already exists in this repository finishes — from a
captured site (point cloud / extracted model) through automated comparison
against an external project schedule (Microsoft Project / Primavera /
generic schedule CSV) and a BIM, all the way to a brief, glanceable
dashboard view.

This is the **last mile** of *"Development of an Integrated AI- and
BIM-Based Framework for Automated Monitoring of Construction Project
Progress"*.

The plan is **incremental** on top of what already exists. None of the
existing Stage 1–10 modules are rewritten. We add a **schedule contract
layer**, a **comparison API**, and a **dashboard view** that route
through the Phase 1–3 contracts.

---

## 1. Inputs at the finishing line

The finishing layer accepts exactly four named inputs, each with a stable
contract:

| Input | Source today in the repo | Contract |
|---|---|---|
| **As-built point cloud / mesh** | Stage 5 dense (`paths.dense_dir`) and Stage 7 cleaned (`paths.clean_dir`) | `.ply` (point cloud) or `.ply` mesh |
| **As-planned BIM** | Stage 8 input (`paths.bim_ifc`) | `.ifc` (IFC2x3 / IFC4) |
| **Project schedule** | NEW: `paths.schedule_csv` (canonical) plus optional importers from `.mpp`/`.xer` | Canonical schedule CSV (see §3) |
| **BIM ↔ Schedule mapping** | NEW: `paths.bim_schedule_mapping` | CSV mapping `(activity_id ↔ {ifc_global_ids…})` |

All four are loaded exclusively through the Phase 1 typed config schema;
no path is hard-coded.

## 2. Outputs at the finishing line

Per run, the finishing layer produces five artefacts with the standard
provenance envelope (Phase 1 `ArtifactEnvelope`):

1. `runs/<run_id>/reports/element_progress.json` — per-IFC-element status
   (already produced by Stage 9, augmented with multi-view fusion +
   confidence calibration in Phase 4).
2. `runs/<run_id>/reports/activity_progress.json` — per-schedule-activity
   roll-up of element status (NEW).
3. `runs/<run_id>/reports/schedule_variance.json` — schedule-vs-as-built
   variance (NEW; see §5).
4. `runs/<run_id>/reports/dashboard_summary.json` — exactly the JSON the
   dashboard consumes (NEW; see §6).
5. `runs/<run_id>/reports/comparison_export.csv` — flat CSV that opens in
   Excel for non-technical reviewers.

Every artefact carries `config_hash`, `git_sha`, `seeds`, `inputs`,
`stage_name` per Phase 1.

## 3. Canonical schedule format

We do not parse `.mpp` (Microsoft Project) directly inside the pipeline.
That is a closed binary format whose support is fragile across
Microsoft Project versions and which Open Source parsers (e.g. MPXJ)
solve better outside the AI pipeline. Instead:

- The pipeline **always** reads the *canonical schedule CSV* below.
- Importers are provided as **side-car scripts** that convert from
  `.mpp`, `.xer` (Primavera P6), `.xml` (MS Project XML), or vendor
  CSV exports to the canonical CSV. These are best-effort and do not
  block the pipeline.

### Canonical schedule CSV columns

```
activity_id          # stable, unique, e.g. "A0123" or WBS code
activity_name        # human-readable label
wbs_code             # optional WBS hierarchy
planned_start_iso    # ISO-8601 date (YYYY-MM-DD) or datetime
planned_finish_iso   # ISO-8601 date or datetime
percent_complete     # planned %% complete at the data date, in [0, 100]
predecessors         # comma-separated activity_ids
trade                # "structural" / "MEP" / "envelope" / etc.
location             # free text or zone code
```

The schema is intentionally minimal and aligns with the smallest common
denominator across MS Project, Primavera P6, Asta Powerproject, and
Synchro.

### MSP / Primavera importers (side-car scripts)

`scripts/import_schedule_msp_xml.py` — reads MS Project's XML export
(File → Save As → XML; works on every modern MSP version, no Windows
required), produces the canonical CSV.

`scripts/import_schedule_p6_xer.py` — reads Primavera P6 `.xer` text
export, produces the canonical CSV.

`scripts/import_schedule_generic_csv.py` — column-mapping importer for
arbitrary vendor CSV exports.

All importers preserve provenance: every output row records its source
file, source row index, and importer version. They are unit-testable on
fixture data.

## 4. BIM ↔ Schedule mapping

For automated comparison we need to know **which IFC elements belong to
which scheduled activity**. We support two paths in priority order:

1. **Explicit mapping CSV** at `paths.bim_schedule_mapping` (preferred):

   ```
   activity_id, ifc_global_id, weight
   A0123,       1Pq8M…,         1.0
   A0123,       2Yz0L…,         1.0
   …
   ```

   `weight` defaults to 1.0 and lets the user weight elements when an
   element is partially associated with multiple activities (e.g. an
   MEP duct that crosses two scheduled trades).

2. **Convention-based fallback** when the explicit mapping is absent:
   match `IfcBuildingElement.Tag` or `IfcRelAssignsToGroup` against
   `activity_id` patterns. The fallback is conservative (does not invent
   matches) and reports unmapped elements in the variance report so the
   reviewer sees the gap.

A small validator script checks that:

- every `activity_id` in the mapping exists in the schedule CSV;
- every `ifc_global_id` in the mapping exists in the BIM;
- the union of mapped element GlobalIds covers ≥ a configurable
  threshold (default 80 %%) of `IfcBuildingElement` instances in the
  BIM (otherwise the reviewer is warned).

## 5. The comparison stage (NEW: Stage 11 — *Schedule Variance*)

We add **one** new pipeline stage that closes the loop. It consumes:

- `element_progress.json` (Stage 9, augmented in Phase 4)
- the canonical schedule CSV
- the BIM↔schedule mapping
- the run's data date (defaults to `now()` UTC)

…and produces `activity_progress.json` and `schedule_variance.json`.

### What it computes

Per scheduled activity:

| Field | Definition |
|---|---|
| `activity_id` | from schedule |
| `planned_percent_complete` | planned % at data date (interpolated linearly between planned_start and planned_finish if not provided) |
| `actual_percent_complete` | weighted mean of `element_acceptance` ∈ {1.0 = acceptable, 0.5 = uncertain, 0.0 = not_acceptable} over mapped elements |
| `actual_percent_complete_lower_95` | Wilson lower bound from per-element acceptance |
| `actual_percent_complete_upper_95` | Wilson upper bound |
| `schedule_variance_percent` | actual − planned, in percentage points |
| `status` | `on_schedule` / `behind` / `ahead` / `unknown_evidence` |
| `confidence` | `high` / `medium` / `low` (from calibration of element-level confidence) |
| `risks` | array of risk tokens reusing the existing decision-policy vocabulary |
| `evidence_refs` | per-activity links into `element_progress.json` |

### Why we do this in a new stage rather than patching Stage 9

- Stage 9 today is **schedule-agnostic** by design (it is purely
  geometric scan-vs-BIM). That's correct and we keep it.
- Schedule variance is a separate scientific concern (a *project
  controls* concern, not a *perception* concern) and deserves its own
  stage with its own provenance.
- The new stage registers in the existing Phase-1 `STAGE_REGISTRY` with
  a typed `StageDescriptor` (capabilities: `lightweight`, no Open3D /
  IfcOpenShell required at runtime — it operates only on the JSON
  artefacts produced upstream). This keeps it laptop-runnable.

## 6. Comparison API (Phase 2 service layer extension)

The Phase 2 service already exposes pipeline / artefact / run-control
endpoints. We extend it with **schedule-comparison endpoints** so the
GUI consumes them directly:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/v1/schedule/activities` | List scheduled activities + planned %% from the canonical CSV. |
| `GET`  | `/api/v1/schedule/activities/{activity_id}` | One activity, with planned + actual %% + variance + risks + confidence. |
| `GET`  | `/api/v1/schedule/variance` | Run-wide variance summary; the **payload of the dashboard cards**. |
| `POST` | `/api/v1/schedule/compare` | Trigger the Stage 11 comparison for the latest run; streams progress over the existing event channel. |
| `GET`  | `/api/v1/elements/{global_id}` | Per-element status, used when the user clicks an element on the 3-D BIM viewer. |
| `POST` | `/api/v1/hitl/corrections` | Submit a HITL correction (Phase 4 `pipeline/common/hitl.py`). |
| `GET`  | `/api/v1/calibration/report` | Latest calibration report + reliability table (Phase 4 `pipeline/common/calibration.py`). |

Every response carries the standard error model defined in the Phase 2
contract; every response carries `config_hash` + `run_id` from the
Phase 1 envelope so the dashboard knows exactly which version it is
showing.

## 7. Dashboard finishing view (Phase 3 GUI extension)

The existing GUI already has a pipeline overview, an artefact browser,
a run-control panel, a metrics dashboard, a VLM panel, a BIM/3-D viewer,
a config-diff viewer, and a report generator. We add **one** new
top-level view dedicated to the finishing target:

### Page: *Schedule Compare* (`/schedule`)

A single, glanceable page with five panels:

1. **KPI strip** — four big numbers:
   - overall planned %%
   - overall actual %% (with 95 %% Wilson interval)
   - schedule variance (%% points, signed)
   - n activities behind / on / ahead

2. **Activities table** — sortable, filterable table of activities with
   columns from §5. Status cell colour-codes `on_schedule` / `behind` /
   `ahead` / `unknown_evidence`. Confidence cell colour-codes
   `high` / `medium` / `low`. Clicking a row opens panel 3.

3. **Activity drilldown** — for the selected activity:
   - the activity's mapped IFC elements (table)
   - per-element acceptance + confidence + multi-view fused belief
   - "Open in 3-D viewer" → highlights the elements on the existing
     BIM/3-D viewer panel
   - "Ask Copilot about this activity" → pre-fills a Stage-10 query
     scoped to those elements
   - "Submit correction" → opens the HITL correction form (Phase 4)

4. **Reliability summary** — small read-only widget pulling the latest
   `calibration_report.v1` (ECE, MCE, Brier, smooth ECE) so reviewers
   can see at a glance how trustworthy the displayed confidences are.

5. **Comparison export** — a button that downloads the run's
   `comparison_export.csv` directly, for non-technical stakeholders.

The page consumes only the `/api/v1/schedule/*` and `/api/v1/elements/*`
endpoints; **no GUI logic re-implements pipeline computation**.

## 8. Worked example walk-through (the "Walls of Floor 2" scenario)

A reviewer wants to know whether the **walls of Floor 2 in a specific
zone** are on schedule. End-to-end:

1. The schedule CSV contains an activity `A0432 — Floor 2 Zone B Walls`
   with `planned_start = 2026-04-01`, `planned_finish = 2026-05-15`.
2. The mapping CSV associates `A0432` with the GlobalIds of the
   `IfcWall` instances on Floor 2 Zone B.
3. Stage 5 dense + Stage 7 cleanup produce the as-built point cloud
   (existing).
4. Stage 8 registers the cloud against the BIM (existing).
5. Stage 9 produces per-element bidirectional accuracy / completeness /
   F-score @ τ for each wall (existing).
6. Phase 4 multi-view fusion (`stage_09_progress/multiview_fusion.py`)
   produces a fused per-element belief with explicit conflict mass
   (NEW).
7. Stage 11 (schedule variance, NEW) rolls those per-element results
   up to `A0432`, with a Wilson 95 %% interval on `actual_percent_complete`.
8. The dashboard shows a card for `A0432` with planned %%, actual %%
   ± 95 %% CI, schedule variance, status, and confidence.
9. The reviewer clicks the row → drilldown shows three walls with
   `decision = uncertain_conflict` (Phase 4 fusion flagged
   inter-view conflict) → reviewer opens the 3-D viewer, inspects, and
   submits a HITL correction (Phase 4 `hitl.py`).
10. The correction is appended to the corrections JSONL log; the next
    nightly calibration run re-computes ECE, MCE, Brier; the dashboard's
    reliability summary updates.

This is the **exact end-to-end loop** the project is building toward.
The plan above is what every Phase 4 / Phase 5 code change finishes.

## 9. Module map (final state at end of Phase 5)

```
pipeline/
  common/
    calibration.py            (Phase 4 — DONE in part 1)
    hitl.py                   (Phase 4 — DONE in part 1)
    schedule_io.py            (Phase 4 NEW — canonical schedule CSV loader)
    bim_schedule_mapping.py   (Phase 4 NEW — mapping loader + validator)
  stage_09_progress/
    multiview_fusion.py       (Phase 4 — DONE in part 1)
  stage_10_copilot/
    grounding_guard.py        (Phase 4 NEW — VLM claim verification)
  stage_11_schedule_variance/ (Phase 4 NEW)
    __init__.py
    config_access.py
    io_utils.py
    activity_rollup.py
    variance_metrics.py
    run_schedule_variance.py
  service/
    schedule_api.py           (Phase 4 NEW — endpoints in §6)

scripts/
  run_calibration_report.py   (Phase 4 — DONE in part 1)
  import_schedule_msp_xml.py  (Phase 4 NEW)
  import_schedule_p6_xer.py   (Phase 4 NEW)
  import_schedule_generic_csv.py  (Phase 4 NEW)
  validate_bim_schedule_mapping.py (Phase 4 NEW)
  smoke_test_stage_11.py      (Phase 4 NEW)

gui/
  src/pages/ScheduleCompare.tsx      (Phase 4 NEW)
  src/components/KPIStrip.tsx        (Phase 4 NEW)
  src/components/ActivityTable.tsx   (Phase 4 NEW)
  src/components/ReliabilityCard.tsx (Phase 4 NEW)
  src/api/schedule.ts                (Phase 4 NEW — typed client)

docs/
  literature_q1_2024_2026.md    (DONE in part 1)
  end_to_end_finishing_plan.md  (THIS FILE)
  schedule_format.md            (Phase 4 NEW — canonical CSV spec, importers)
  hitl_workflow.md              (Phase 4 NEW)
  calibration_workflow.md       (Phase 4 NEW)

tests/
  test_calibration.py             (Phase 4 NEW)
  test_hitl.py                    (Phase 4 NEW)
  test_multiview_fusion.py        (Phase 4 NEW)
  test_grounding_guard.py         (Phase 4 NEW)
  test_schedule_io.py             (Phase 4 NEW)
  test_bim_schedule_mapping.py    (Phase 4 NEW)
  test_stage_11_schedule_variance.py (Phase 4 NEW)
```

Existing modules: **untouched** unless an additive change is needed.

## 10. Phase 4 → Phase 5 hand-off

By the end of Phase 4 we will have:

- All Phase 4 algorithmic novelty (DONE-in-part-1: calibration, HITL,
  multi-view fusion).
- Schedule contract (canonical CSV + importers + validator).
- Stage 11 schedule-variance pipeline stage.
- Schedule API endpoints.
- VLM grounding guard.
- Tests for everything above.

Phase 5 then polishes:

- Documentation pass on `docs/schedule_format.md`,
  `docs/hitl_workflow.md`, `docs/calibration_workflow.md`,
  `docs/end_to_end_walkthrough.md`.
- The dashboard `ScheduleCompare` page wired to the real backend.
- Example fixtures: a small synthetic schedule CSV, a tiny BIM, and a
  mapping CSV — enough that `scripts/smoke_test_stage_11.py` runs in
  the lightweight test set.
- Reproducibility notes (already partially in `docs/scientific_upgrades.md`).

---

This document is the **authoritative target** every subsequent commit
finishes against.
