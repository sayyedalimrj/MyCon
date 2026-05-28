"""Tests for :mod:`pipeline.common.schedule_io`.

Verify the canonical schedule CSV loader, ISO datetime parsing, and the
planned-percent-complete interpolation that Stage 11 uses.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from pipeline.common.schedule_io import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SCHEDULE_SCHEMA_VERSION,
    Activity,
    Schedule,
    load_schedule_csv,
    parse_iso_datetime,
    planned_percent_complete_at,
)


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# parse_iso_datetime
# ---------------------------------------------------------------------------


def test_parse_iso_date_only_treated_as_midnight_utc() -> None:
    d = parse_iso_datetime("2026-04-01")
    assert d == _dt.datetime(2026, 4, 1, 0, 0, 0, tzinfo=_dt.timezone.utc)


def test_parse_iso_datetime_with_z_zone() -> None:
    d = parse_iso_datetime("2026-04-01T08:30:00Z")
    assert d == _dt.datetime(2026, 4, 1, 8, 30, 0, tzinfo=_dt.timezone.utc)


def test_parse_iso_datetime_with_offset_normalised_to_utc() -> None:
    d = parse_iso_datetime("2026-04-01T10:00:00+02:00")
    assert d == _dt.datetime(2026, 4, 1, 8, 0, 0, tzinfo=_dt.timezone.utc)


def test_parse_iso_datetime_naive_treated_as_utc() -> None:
    d = parse_iso_datetime("2026-04-01T08:30:00")
    assert d == _dt.datetime(2026, 4, 1, 8, 30, 0, tzinfo=_dt.timezone.utc)


def test_parse_iso_datetime_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        parse_iso_datetime("")


def test_parse_iso_datetime_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_iso_datetime("hello")


def test_parse_iso_datetime_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        parse_iso_datetime(2026)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# planned_percent_complete_at
# ---------------------------------------------------------------------------


_START = _dt.datetime(2026, 4, 1, 0, 0, 0, tzinfo=_dt.timezone.utc)
_FINISH = _dt.datetime(2026, 5, 1, 0, 0, 0, tzinfo=_dt.timezone.utc)
_MID = _dt.datetime(2026, 4, 16, 0, 0, 0, tzinfo=_dt.timezone.utc)


def test_planned_percent_zero_at_or_before_start() -> None:
    assert planned_percent_complete_at(_START, _FINISH, _START) == 0.0
    earlier = _START - _dt.timedelta(days=1)
    assert planned_percent_complete_at(_START, _FINISH, earlier) == 0.0


def test_planned_percent_one_hundred_at_or_after_finish() -> None:
    assert planned_percent_complete_at(_START, _FINISH, _FINISH) == 100.0
    later = _FINISH + _dt.timedelta(days=10)
    assert planned_percent_complete_at(_START, _FINISH, later) == 100.0


def test_planned_percent_linear_at_midpoint() -> None:
    pct = planned_percent_complete_at(_START, _FINISH, _MID)
    assert pct == pytest.approx(50.0, abs=0.5)


def test_planned_percent_zero_duration_activity_jumps_to_100_after_start() -> None:
    same = _START
    just_after = _START + _dt.timedelta(seconds=1)
    assert planned_percent_complete_at(same, same, _START) == 100.0
    assert planned_percent_complete_at(same, same, just_after) == 100.0
    earlier = _START - _dt.timedelta(seconds=1)
    assert planned_percent_complete_at(same, same, earlier) == 0.0


# ---------------------------------------------------------------------------
# load_schedule_csv — happy path
# ---------------------------------------------------------------------------


def _write_csv(tmp_path: Path, header: str, rows: list[str]) -> Path:
    p = tmp_path / "schedule.csv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_load_schedule_minimal_required_columns(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        "activity_id,activity_name,planned_start_iso,planned_finish_iso",
        [
            "A0001,Floor 1 walls,2026-04-01,2026-04-15",
            "A0002,Floor 2 walls,2026-04-15,2026-05-01",
        ],
    )
    s = load_schedule_csv(p)
    assert isinstance(s, Schedule)
    assert len(s) == 2
    assert s.provenance.schema_version == SCHEDULE_SCHEMA_VERSION
    assert s.provenance.n_rows_kept == 2
    assert s.provenance.n_rows_skipped == 0
    a = s.get("A0001")
    assert a is not None
    assert a.activity_name == "Floor 1 walls"
    assert a.planned_start.tzinfo is not None
    assert a.percent_complete is None  # column omitted


def test_load_schedule_all_columns(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        "activity_id,activity_name,wbs_code,planned_start_iso,planned_finish_iso,percent_complete,predecessors,trade,location",
        [
            'A0432,Floor 2 Zone B walls,1.2.3,2026-04-01,2026-05-15,25,"A0001,A0002",structural,Floor 2 Zone B',
        ],
    )
    s = load_schedule_csv(p)
    a = s.get("A0432")
    assert a is not None
    assert a.wbs_code == "1.2.3"
    assert a.percent_complete == 25.0
    assert a.predecessors == ("A0001", "A0002")
    assert a.trade == "structural"
    assert a.location == "Floor 2 Zone B"


def test_load_schedule_records_provenance(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        "activity_id,activity_name,planned_start_iso,planned_finish_iso",
        ["A1,Wall,2026-04-01,2026-04-02"],
    )
    s = load_schedule_csv(p)
    prov = s.provenance
    assert prov.source_path == str(p.resolve())
    assert prov.source_bytes > 0
    assert len(prov.source_sha256) == 64
    assert prov.n_rows_total == 1


# ---------------------------------------------------------------------------
# load_schedule_csv — error / skip paths
# ---------------------------------------------------------------------------


def test_load_schedule_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_schedule_csv(tmp_path / "nope.csv")


def test_load_schedule_raises_when_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_bytes(b"")
    with pytest.raises(ValueError):
        load_schedule_csv(p)


def test_load_schedule_raises_when_required_columns_missing(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        "activity_id,activity_name",  # missing planned_start_iso, planned_finish_iso
        ["A1,Wall"],
    )
    with pytest.raises(ValueError):
        load_schedule_csv(p)


def test_load_schedule_skips_rows_with_invalid_dates(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        "activity_id,activity_name,planned_start_iso,planned_finish_iso",
        [
            "A1,Good,2026-04-01,2026-04-02",
            "A2,BadStart,not-a-date,2026-04-02",
            "A3,BadFinish,2026-04-01,not-a-date",
        ],
    )
    s = load_schedule_csv(p)
    assert len(s) == 1
    reasons = dict(s.provenance.skip_reasons)
    assert reasons.get("bad_planned_start") == 1
    assert reasons.get("bad_planned_finish") == 1


def test_load_schedule_skips_finish_before_start(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        "activity_id,activity_name,planned_start_iso,planned_finish_iso",
        ["A1,Backwards,2026-04-15,2026-04-01"],
    )
    s = load_schedule_csv(p)
    assert len(s) == 0
    reasons = dict(s.provenance.skip_reasons)
    assert reasons.get("finish_before_start") == 1


def test_load_schedule_skips_duplicate_activity_id(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        "activity_id,activity_name,planned_start_iso,planned_finish_iso",
        [
            "A1,First,2026-04-01,2026-04-02",
            "A1,Duplicate,2026-04-03,2026-04-04",
        ],
    )
    s = load_schedule_csv(p)
    assert len(s) == 1
    assert s.activities[0].activity_name == "First"
    reasons = dict(s.provenance.skip_reasons)
    assert reasons.get("duplicate_activity_id") == 1


def test_load_schedule_skips_invalid_percent_complete(tmp_path: Path) -> None:
    """Out-of-range percent_complete is dropped (kept as None) but the
    row itself is preserved."""
    p = _write_csv(
        tmp_path,
        "activity_id,activity_name,planned_start_iso,planned_finish_iso,percent_complete",
        [
            "A1,Wall,2026-04-01,2026-04-02,150",
            "A2,Beam,2026-04-01,2026-04-02,75",
            "A3,Slab,2026-04-01,2026-04-02,not-a-number",
        ],
    )
    s = load_schedule_csv(p)
    assert len(s) == 3
    assert s.get("A1").percent_complete is None
    assert s.get("A2").percent_complete == 75.0
    assert s.get("A3").percent_complete is None


def test_load_schedule_tolerates_utf8_bom(tmp_path: Path) -> None:
    """Excel CSV exports often carry a BOM; we must not let it corrupt
    the first column name."""
    p = tmp_path / "schedule.csv"
    content = (
        "\ufeffactivity_id,activity_name,planned_start_iso,planned_finish_iso\n"
        "A1,Wall,2026-04-01,2026-04-02\n"
    )
    p.write_text(content, encoding="utf-8")
    s = load_schedule_csv(p)
    assert s.get("A1") is not None


# ---------------------------------------------------------------------------
# Activity helper
# ---------------------------------------------------------------------------


def test_activity_planned_percent_complete_at_method() -> None:
    a = Activity(
        activity_id="A1",
        activity_name="Wall",
        planned_start=_START,
        planned_finish=_FINISH,
    )
    assert a.planned_percent_complete_at(_MID) == pytest.approx(50.0, abs=0.5)


def test_activity_to_dict_serialises_datetimes_and_predecessors() -> None:
    a = Activity(
        activity_id="A1",
        activity_name="Wall",
        planned_start=_START,
        planned_finish=_FINISH,
        predecessors=("A0", "B1"),
    )
    d = a.to_dict()
    assert d["planned_start"].endswith("+00:00")
    assert d["predecessors"] == ["A0", "B1"]
