"""Canonical schedule CSV loader and dataclass for the finishing layer.

This module is the single point of contact between the pipeline's
schedule-aware logic (Stage 11 schedule variance, dashboard
``ScheduleCompare`` page, comparison API) and the *outside world* of
project-management software. The pipeline never parses ``.mpp`` (Microsoft
Project) directly — that's a closed binary format whose support is
fragile across MSP versions. Instead the pipeline reads the **canonical
schedule CSV** documented here, and side-car importer scripts in
``scripts/import_schedule_*.py`` convert from `.mpp` XML, `.xer`
(Primavera P6), or vendor CSV exports to this canonical form.

The canonical CSV columns are deliberately minimal and align with the
smallest common denominator across MS Project, Primavera P6, Asta
Powerproject, and Synchro:

    activity_id, activity_name, wbs_code, planned_start_iso,
    planned_finish_iso, percent_complete, predecessors, trade, location

See ``docs/end_to_end_finishing_plan.md`` Section 3 for the full spec
and ``docs/schedule_format.md`` for end-user documentation (Phase 5).

Design notes
------------

- Pure stdlib (csv + datetime). Runs in the lightweight test set.
- All parsing is **lenient on input** but **strict on output**: invalid
  rows are skipped with a counter, but every *returned* :class:`Activity`
  satisfies all invariants.
- Date parsing accepts both date-only (``YYYY-MM-DD``) and
  date-time (``YYYY-MM-DDTHH:MM:SS`` with optional offset / ``Z``).
  Internally we always store ``datetime`` at UTC.
- The loader also computes a ``planned_percent_complete_at(date)``
  helper that linearly interpolates between planned start and finish.
  This is the function Stage 11 uses to compute schedule variance.
- The loader records its own provenance — input file, byte size, sha-256
  of the raw bytes, n_skipped — so every downstream report can prove
  exactly which schedule it consumed.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SCHEDULE_SCHEMA_VERSION",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "Activity",
    "ScheduleProvenance",
    "Schedule",
    "load_schedule_csv",
    "parse_iso_datetime",
    "planned_percent_complete_at",
]


SCHEDULE_SCHEMA_VERSION = "schedule.v1"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "activity_id",
    "activity_name",
    "planned_start_iso",
    "planned_finish_iso",
)
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "wbs_code",
    "percent_complete",
    "predecessors",
    "trade",
    "location",
)


# ---------------------------------------------------------------------------
# Datatypes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Activity:
    """One schedule activity in the canonical form.

    All datetimes are stored at UTC. ``percent_complete`` is in [0, 100]
    if explicitly recorded in the schedule (planner-asserted value);
    when the schedule omits the column, the field is ``None`` and
    callers should derive ``planned_percent_complete`` from the date
    interpolation via :func:`planned_percent_complete_at`.
    """

    activity_id: str
    activity_name: str
    planned_start: _dt.datetime
    planned_finish: _dt.datetime
    wbs_code: str = ""
    percent_complete: float | None = None
    predecessors: tuple[str, ...] = ()
    trade: str = ""
    location: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["planned_start"] = self.planned_start.isoformat()
        d["planned_finish"] = self.planned_finish.isoformat()
        d["predecessors"] = list(self.predecessors)
        return d

    def planned_percent_complete_at(self, when: _dt.datetime) -> float:
        """Linear interpolation between ``planned_start`` (0%%) and
        ``planned_finish`` (100%%); clipped to [0, 100].

        If the schedule already records a ``percent_complete`` value
        (planner-asserted), this method *ignores* it and returns the
        date-interpolated value. The planner-asserted value is available
        as a separate field; callers that want it should read
        ``activity.percent_complete`` directly.
        """
        return planned_percent_complete_at(self.planned_start, self.planned_finish, when)


@dataclass(frozen=True)
class ScheduleProvenance:
    """Self-recorded provenance for one parsed schedule CSV."""

    source_path: str
    source_sha256: str
    source_bytes: int
    schema_version: str
    n_rows_total: int
    n_rows_kept: int
    n_rows_skipped: int
    skip_reasons: tuple[tuple[str, int], ...] = ()  # (reason, count) pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "schema_version": self.schema_version,
            "n_rows_total": self.n_rows_total,
            "n_rows_kept": self.n_rows_kept,
            "n_rows_skipped": self.n_rows_skipped,
            "skip_reasons": [list(p) for p in self.skip_reasons],
        }


@dataclass(frozen=True)
class Schedule:
    """A parsed schedule = ordered tuple of activities + provenance."""

    activities: tuple[Activity, ...]
    provenance: ScheduleProvenance
    activity_index: Mapping[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.activities)

    def __iter__(self):
        return iter(self.activities)

    def get(self, activity_id: str) -> Activity | None:
        idx = self.activity_index.get(activity_id)
        if idx is None:
            return None
        return self.activities[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "provenance": self.provenance.to_dict(),
            "activities": [a.to_dict() for a in self.activities],
        }


# ---------------------------------------------------------------------------
# Datetime parsing
# ---------------------------------------------------------------------------


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_iso_datetime(value: str) -> _dt.datetime:
    """Parse an ISO-8601 date or datetime into a UTC :class:`datetime`.

    Accepted forms:

    - ``YYYY-MM-DD`` → midnight UTC on that date
    - ``YYYY-MM-DDTHH:MM:SS``
    - ``YYYY-MM-DDTHH:MM:SS+HH:MM``
    - ``YYYY-MM-DDTHH:MM:SSZ``

    The function is strict: anything else raises :class:`ValueError`
    so the loader can record the row as skipped.
    """
    if not isinstance(value, str):
        raise ValueError(f"datetime value must be string, got {type(value).__name__}")
    s = value.strip()
    if not s:
        raise ValueError("empty datetime")
    if _DATE_ONLY.match(s):
        d = _dt.date.fromisoformat(s)
        return _dt.datetime.combine(d, _dt.time(0, 0, 0), tzinfo=_dt.timezone.utc)
    # Python 3.11+ accepts the trailing 'Z' in fromisoformat.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    parsed = _dt.datetime.fromisoformat(s)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    else:
        parsed = parsed.astimezone(_dt.timezone.utc)
    return parsed


def planned_percent_complete_at(
    planned_start: _dt.datetime,
    planned_finish: _dt.datetime,
    when: _dt.datetime,
) -> float:
    """Linear interpolation between start (0) and finish (100), clipped."""
    if planned_finish <= planned_start:
        # Zero-duration activity: any time on/after start counts as 100%.
        return 100.0 if when >= planned_start else 0.0
    if when <= planned_start:
        return 0.0
    if when >= planned_finish:
        return 100.0
    span = (planned_finish - planned_start).total_seconds()
    elapsed = (when - planned_start).total_seconds()
    return max(0.0, min(100.0, 100.0 * elapsed / span))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _split_predecessors(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    parts = [p.strip() for p in raw.split(",")]
    return tuple(p for p in parts if p)


def _parse_percent_complete(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        v = float(s.rstrip("%").strip())
    except ValueError:
        return None
    if v < 0.0 or v > 100.0:
        return None
    return v


def load_schedule_csv(path: Path | str) -> Schedule:
    """Load and validate a canonical schedule CSV.

    Required columns are :data:`REQUIRED_COLUMNS`; optional columns are
    :data:`OPTIONAL_COLUMNS`. Rows that fail validation are *skipped*
    with a per-reason counter recorded in the returned schedule's
    provenance, so a single bad row does not abort the whole load.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the CSV header is missing one of :data:`REQUIRED_COLUMNS`,
        or if the file is empty / has no header row.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"schedule CSV not found: {path}")

    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"schedule CSV is empty: {path}")

    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig")  # tolerate BOM from Excel exports
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError(f"schedule CSV has no header: {path}")
    field_set = {(f or "").strip() for f in reader.fieldnames}
    missing = [c for c in REQUIRED_COLUMNS if c not in field_set]
    if missing:
        raise ValueError(
            f"schedule CSV {path} is missing required columns: {missing}"
        )

    activities: list[Activity] = []
    skip_counts: dict[str, int] = {}
    n_total = 0
    seen_ids: dict[str, int] = {}

    for row in reader:
        n_total += 1
        # csv.DictReader returns None for missing trailing fields when
        # rows have fewer columns than the header.
        activity_id = (row.get("activity_id") or "").strip()
        if not activity_id:
            skip_counts["missing_activity_id"] = skip_counts.get("missing_activity_id", 0) + 1
            continue
        if activity_id in seen_ids:
            skip_counts["duplicate_activity_id"] = skip_counts.get("duplicate_activity_id", 0) + 1
            continue
        try:
            ps = parse_iso_datetime(row.get("planned_start_iso") or "")
        except ValueError:
            skip_counts["bad_planned_start"] = skip_counts.get("bad_planned_start", 0) + 1
            continue
        try:
            pf = parse_iso_datetime(row.get("planned_finish_iso") or "")
        except ValueError:
            skip_counts["bad_planned_finish"] = skip_counts.get("bad_planned_finish", 0) + 1
            continue
        if pf < ps:
            skip_counts["finish_before_start"] = skip_counts.get("finish_before_start", 0) + 1
            continue

        seen_ids[activity_id] = len(activities)
        activities.append(
            Activity(
                activity_id=activity_id,
                activity_name=(row.get("activity_name") or "").strip(),
                planned_start=ps,
                planned_finish=pf,
                wbs_code=(row.get("wbs_code") or "").strip(),
                percent_complete=_parse_percent_complete(row.get("percent_complete")),
                predecessors=_split_predecessors(row.get("predecessors") or ""),
                trade=(row.get("trade") or "").strip(),
                location=(row.get("location") or "").strip(),
            )
        )

    n_skipped = n_total - len(activities)
    provenance = ScheduleProvenance(
        source_path=str(path.resolve()),
        source_sha256=sha,
        source_bytes=len(raw),
        schema_version=SCHEDULE_SCHEMA_VERSION,
        n_rows_total=n_total,
        n_rows_kept=len(activities),
        n_rows_skipped=n_skipped,
        skip_reasons=tuple(sorted(skip_counts.items())),
    )

    return Schedule(
        activities=tuple(activities),
        provenance=provenance,
        activity_index=dict(seen_ids),
    )
