# Canonical Schedule CSV Format

This document is the user-facing reference for the canonical schedule
CSV consumed by Stage 11 (schedule variance) and the dashboard
**Schedule Compare** page. The pipeline does **not** parse `.mpp`
(Microsoft Project) directly; it only reads the canonical CSV defined
here. Side-car importers convert MS Project / Primavera / generic
vendor exports to this format.

See also:

- `pipeline/common/schedule_io.py` — loader and dataclass.
- `pipeline/common/bim_schedule_mapping.py` — BIM ↔ schedule mapping.
- `docs/end_to_end_finishing_plan.md` — the bigger picture.

## 1. Required + optional columns

| Column | Required | Type | Notes |
|---|---|---|---|
| `activity_id` | yes | string | Stable, unique, project-wide. Convention: `A####` or WBS code (`1.2.3.4`). |
| `activity_name` | yes | string | Human-readable label (e.g. *"Floor 2 Zone B walls"*). |
| `planned_start_iso` | yes | ISO-8601 | `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS[Z|+HH:MM]`. Naive datetimes are treated as UTC. |
| `planned_finish_iso` | yes | ISO-8601 | Same accepted forms; must be ≥ `planned_start_iso`. |
| `wbs_code` | optional | string | Free-form. Used only for grouping / display. |
| `percent_complete` | optional | float `[0, 100]` | Planner-asserted progress. Stage 11 prefers this when present; otherwise it linearly interpolates between start and finish. |
| `predecessors` | optional | comma-joined `activity_id`s | Quote the cell in CSV: `"A0001,A0002"`. |
| `trade` | optional | string | e.g. *structural*, *MEP*, *envelope*. |
| `location` | optional | string | Free text or zone code (e.g. *"Floor 2 Zone B"*). |

The schema version is locked as `schedule.v1` in
`pipeline.common.schedule_io.SCHEDULE_SCHEMA_VERSION`.

## 2. Minimal example

```csv
activity_id,activity_name,planned_start_iso,planned_finish_iso
A0001,Foundations,2026-03-01,2026-04-01
A0432,Floor 2 Zone B walls,2026-04-01,2026-05-01
```

This file alone is enough to run Stage 11.

## 3. Full example (all columns)

```csv
activity_id,activity_name,wbs_code,planned_start_iso,planned_finish_iso,percent_complete,predecessors,trade,location
A0001,Foundations,1.1,2026-03-01,2026-04-01,100,,structural,Site
A0432,Floor 2 Zone B walls,1.2.3,2026-04-01T08:00:00Z,2026-05-01T17:00:00+02:00,25,"A0001",structural,Floor 2 Zone B
```

## 4. Loader behaviour (loud-but-resumable)

`load_schedule_csv` is *strict on output, lenient on input*:

- Rows missing `activity_id`, `planned_start_iso`, `planned_finish_iso`,
  or with `finish < start`, are **skipped** with a per-reason counter.
- Duplicate `activity_id` rows are kept once; subsequent ones are
  skipped (`duplicate_activity_id`).
- Out-of-range `percent_complete` values become `None` instead of
  failing the row.
- UTF-8 BOMs from Excel exports are stripped automatically.
- The returned `Schedule.provenance` records `source_path`,
  `source_sha256`, total / kept / skipped row counts, and the per-reason
  skip dictionary.

This means a single bad row never blocks a run.

## 5. Side-car importers

The pipeline never parses vendor binary formats. Convert them first:

### 5.1 MS Project XML

In MSP, `File → Save As → XML` produces a stable export that every
modern MSP version supports.

```bash
python3 scripts/import_schedule_msp_xml.py \
    --input  schedules/site01.xml \
    --output configs/schedules/site01.csv \
    --summary-json runs/imports/site01_msp.json
```

The summary JSON records the importer name, input sha-256, kept / skipped
row counts, and per-reason skip dictionary.

### 5.2 Primavera P6 XER

`.xer` is P6's flat-text export. The importer reads only the `TASK` and
`TASKPRED` tables and resolves predecessor links from internal
`task_id`s back to human-readable `task_code`s.

```bash
python3 scripts/import_schedule_p6_xer.py \
    --input  schedules/site01.xer \
    --output configs/schedules/site01.csv \
    --summary-json runs/imports/site01_p6.json
```

### 5.3 Vendor CSV (generic column-mapping)

For vendors whose CSVs use different column names:

```bash
python3 scripts/import_schedule_generic_csv.py \
    --input  schedules/vendor.csv \
    --output configs/schedules/site01.csv \
    --activity-id-column "Task ID" \
    --activity-name-column "Task Name" \
    --planned-start-column "Start Date" \
    --planned-finish-column "Finish Date" \
    --percent-complete-column "% Complete" \
    --wbs-column "WBS" \
    --predecessors-column "Predecessors"
```

The generic importer **does not** attempt locale-specific date parsing.
If the vendor uses `31/04/2026` (DMY) you must convert to ISO-8601
*before* running the importer; silently swapping day/month is a class
of error not worth risking on a construction schedule.

## 6. BIM ↔ schedule mapping

The schedule alone is not enough — Stage 11 also needs to know which
IFC elements belong to which scheduled activity. See
`pipeline/common/bim_schedule_mapping.py` and the BIM↔schedule mapping
CSV format documented inline there.

Minimal mapping CSV:

```csv
activity_id,ifc_global_id,weight
A0432,1Pq8MeKvD2vQ8XYZabcdef,1.0
A0432,2Pq8MeKvD2vQ8XYZabcdef,1.0
A0432,3Pq8MeKvD2vQ8XYZabcdef,1.0
A0001,9Yz0LqvMmN3pP0QRstuvw,1.0
```

`weight` defaults to 1.0 and lets you partially associate one element
with multiple activities (e.g. an MEP duct that crosses two trades).
`validate_mapping()` will surface activities or elements that don't
exist in the schedule / BIM, so mismatches are loud rather than silent.
