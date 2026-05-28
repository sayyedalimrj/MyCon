# Human-in-the-Loop (HITL) Corrections — Workflow

This document describes how reviewer corrections enter the pipeline,
how they are persisted, and how they are replayed to recalibrate
confidence.

The HITL workflow is the third leg of the Phase 4 reliability tripod:

1. **Multi-view evidential fusion** — per-element decisions with
   explicit conflict mass (`pipeline/stage_09_progress/multiview_fusion.py`).
2. **Decision policy + Wilson intervals** — per-activity actual %
   complete with 95 % confidence intervals (`pipeline/stage_11_*`).
3. **HITL corrections + calibration** — measurement of whether the
   reported confidences are *trustworthy* (this document +
   `docs/calibration_workflow.md`).

Code:

- `pipeline/common/hitl.py` — schema, store, replay.
- `pipeline/common/calibration.py` — calibration metrics that consume
  the replay output.
- `scripts/run_calibration_report.py` — CLI bridge.

## 1. What gets corrected

The HITL store accepts five kinds of corrections (locked vocabulary in
`VALID_TARGET_KINDS`):

| `target_kind` | Convention for `target_id` |
|---|---|
| `element_acceptance` | IFC GlobalId |
| `activity_completion` | Schedule `activity_id` |
| `vlm_answer` | `run_id::query_id` |
| `anchor_validation` | `anchor_id` |
| `registration_quality` | `run_id` |

The decision vocabulary (locked in `VALID_DECISION_VALUES`) is
`{accept, reject, uncertain, rework}`.

## 2. Submitting a correction

### 2.1 From Python

```python
from pipeline.common.hitl import Correction, CorrectionStore

store = CorrectionStore("runs/<run_id>/reports/hitl_corrections.jsonl")
store.append(
    Correction.from_dict(
        {
            "target_kind": "element_acceptance",
            "target_id": "1Pq8MeKvD2vQ8XYZabcdef",
            "predicted_value": "accept",
            "predicted_confidence": "high",
            "corrected_value": "reject",
            "reviewer_id": "alice@example.com",
            "rationale": "Wall is in tolerance but missing rebar capping.",
            "evidence_refs": [
                "runs/<run_id>/reports/element_progress.json",
                "runs/<run_id>/reports/contact_sheets/wall_2.jpg",
            ],
            "run_id": "<run_id>",
        }
    )
)
```

`timestamp_utc` and `record_id` are auto-filled if not supplied. The
record_id is a deterministic 12-char SHA-256 prefix of the canonicalised
payload, so the same payload always produces the same id.

### 2.2 From the dashboard (planned)

The Schedule Compare page will get a "Submit correction" button
inside the Activity drilldown (Phase 5) that POSTs to
`/api/v1/hitl/corrections`. The backend route already validates and
appends through the same `CorrectionStore`, so dashboard-submitted
corrections share the same audit log as Python-submitted ones.

## 3. Append-only semantics

The store is append-only. To "edit" a correction, append a new record;
the previous record is preserved. On replay (§4) the latest record
*wins*, and any disagreement between adjacent records surfaces as a
`ConflictRecord` so the audit trail is loud rather than silent.

```jsonl
{"schema_version": "hitl_correction.v1", "target_kind": "element_acceptance", "target_id": "1Pq8...", "corrected_value": "reject", "timestamp_utc": "2026-05-01T08:00:00Z", "reviewer_id": "alice", ...}
{"schema_version": "hitl_correction.v1", "target_kind": "element_acceptance", "target_id": "1Pq8...", "corrected_value": "accept", "timestamp_utc": "2026-05-02T15:30:00Z", "reviewer_id": "bob",   ...}
```

After replay:

- effective: `corrected_value=accept` (last write wins, by `bob`).
- conflicts: one `ConflictRecord` recording that `alice` and `bob`
  disagreed.

## 4. Replay

```python
result = store.replay(
    target_kinds=["element_acceptance"],   # optional filter
    run_id="<run_id>",                     # optional filter
)
print(result.n_total_records)
print(len(result.effective))
print(len(result.conflicts))
```

`result.effective` is one `Correction` per `(target_kind, target_id)`
group, sorted deterministically by `(target_kind, target_id)` so output
is diffable. `result.conflicts` lists every adjacent disagreement so the
operator can audit them.

## 5. Bridge into calibration

```python
from pipeline.common import calibration
from pipeline.common.hitl import build_calibration_records

cal_records = build_calibration_records(result, target_kinds=["element_acceptance"])
report = calibration.calibration_report(cal_records, n_bins=10, strategy="equal_mass")
```

Each effective correction maps to one calibration record:

- `confidence` = the *predicted* confidence label.
- `correct` = `predicted_value == corrected_value`.

The resulting calibration report is described in
`docs/calibration_workflow.md`.

## 6. Recommended directory layout

```
runs/
  <run_id>/
    reports/
      element_progress.json
      hitl_corrections.jsonl       <-- single append-only JSONL
      calibration_report.json      <-- output of run_calibration_report.py
```

For multi-reviewer workflows, prefer one JSONL **per host** and merge
periodically (POSIX append semantics are atomic per record only when
the record is smaller than `PIPE_BUF`; longer records may interleave
under contention).

## 7. Operational guidance

- Treat the `evidence_refs` field as load-bearing. Reviewers are far
  more useful when their corrections cite the artifacts they looked
  at; the `rationale` field is the second most-useful column for
  retrospective analysis (cf. Beck et al., WACV 2024).
- Run `scripts/run_calibration_report.py` against the corrections
  log on every run. The calibration report is itself an artefact you
  should commit alongside the rest of `reports/`.
- A handful of corrections is already useful. ECE is a meaningful
  number once `n_samples ≥ 30` per confidence bin. The reliability
  table reports per-bin counts, so you can see whether the headline
  ECE is supported by enough evidence.

## 8. Common pitfalls

- **Corrections logged before the run that produced the prediction.**
  If you replay across multiple runs and the timestamps are in the
  wrong order, last-write-wins resolves to the wrong record. Always
  filter by `run_id` when calibrating against a specific run.
- **Stale `predicted_confidence`.** When a Phase 4 multi-view fusion
  re-evaluates an element, the original `predicted_confidence` in the
  HITL record may no longer match the latest pipeline output. The
  correction is still authoritative as ground truth, but the
  calibration report measures the *historical* confidence, not the
  current one. Re-run the pipeline and re-collect corrections if you
  want a fresh measurement.
- **Empty `rationale`.** Allowed but discouraged. Without it, the
  audit trail is opaque to a future reviewer. Treat the field as a
  mandatory cultural convention even though the schema does not
  enforce non-empty.
