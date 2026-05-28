"""Tests for :mod:`pipeline.stage_11_schedule_variance`.

These tests pin the contract of the schedule-variance pipeline stage:

- ``ActivityRollup`` correctly aggregates per-element acceptance into a
  weighted percent + Wilson 95 %% interval.
- ``classify_status`` honours the on-schedule band, ahead/behind
  semantics, and the unknown-evidence escape hatch.
- ``build_variance_report`` and ``build_dashboard_summary`` return
  stable, JSON-shaped artefacts with the documented schema versions.
"""

from __future__ import annotations

import datetime as _dt
import json
import math

import pytest

from pipeline.common.bim_schedule_mapping import BimScheduleMapping, MappingEntry
from pipeline.common.schedule_io import (
    Activity,
    Schedule,
    ScheduleProvenance,
    SCHEDULE_SCHEMA_VERSION,
)
from pipeline.stage_11_schedule_variance import (
    DASHBOARD_SUMMARY_SCHEMA_VERSION,
    DEFAULT_ON_SCHEDULE_BAND_PCT,
    SCHEDULE_VARIANCE_SCHEMA_VERSION,
    ActivityRollup,
    ActivityVariance,
    build_dashboard_summary,
    build_variance_report,
    classify_status,
    rollup_activities,
)
from pipeline.stage_11_schedule_variance.activity_rollup import wilson_interval


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _activity(aid: str, name: str = "Activity") -> Activity:
    return Activity(
        activity_id=aid,
        activity_name=name,
        planned_start=_dt.datetime(2026, 4, 1, tzinfo=_dt.timezone.utc),
        planned_finish=_dt.datetime(2026, 5, 1, tzinfo=_dt.timezone.utc),
    )


def _schedule(activities: list[Activity]) -> Schedule:
    return Schedule(
        activities=tuple(activities),
        provenance=ScheduleProvenance(
            source_path="",
            source_sha256="",
            source_bytes=0,
            schema_version=SCHEDULE_SCHEMA_VERSION,
            n_rows_total=len(activities),
            n_rows_kept=len(activities),
            n_rows_skipped=0,
        ),
        activity_index={a.activity_id: i for i, a in enumerate(activities)},
    )


def _mapping(pairs: list[tuple[str, str, float]]) -> BimScheduleMapping:
    entries = tuple(MappingEntry(a, e, w) for a, e, w in pairs)
    by_a: dict = {}
    by_e: dict = {}
    for entry in entries:
        by_a.setdefault(entry.activity_id, []).append(entry)
        by_e.setdefault(entry.ifc_global_id, []).append(entry)
    return BimScheduleMapping(
        entries=entries,
        by_activity={k: tuple(v) for k, v in by_a.items()},
        by_element={k: tuple(v) for k, v in by_e.items()},
    )


# ---------------------------------------------------------------------------
# Wilson interval sanity
# ---------------------------------------------------------------------------


def test_wilson_zero_sample_returns_full_unit_interval() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_full_success_interval_below_one() -> None:
    lo, hi = wilson_interval(5, 5)
    assert hi == pytest.approx(1.0, abs=1e-6)
    assert lo < 1.0
    assert lo > 0.5  # the lower bound stays well above 0.5 for n=5


def test_wilson_zero_success_interval_above_zero() -> None:
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0
    assert hi > 0.0


def test_wilson_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_interval(-1, 5)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)


# ---------------------------------------------------------------------------
# rollup_activities
# ---------------------------------------------------------------------------


def test_rollup_three_elements_one_each_status() -> None:
    """1 likely_completed + 1 partially_observed + 1 not_evidenced ->
    50% (1.0 + 0.5 + 0.0) / 3.

    Wilson 95% on 1 success out of 3 trials is roughly [0.06, 0.79]."""
    mapping = _mapping([("A1", "X1", 1.0), ("A1", "X2", 1.0), ("A1", "X3", 1.0)])
    rows = [
        {"global_id": "X1", "status": "likely_completed"},
        {"global_id": "X2", "status": "partially_observed"},
        {"global_id": "X3", "status": "not_evidenced"},
    ]
    rollups = rollup_activities(rows, mapping=mapping)
    assert len(rollups) == 1
    r = rollups[0]
    assert r.activity_id == "A1"
    assert r.n_mapped_elements == 3
    assert r.n_evaluated_elements == 3
    assert r.n_completed_elements == 1
    assert r.n_partial_elements == 1
    assert r.n_not_evidenced_elements == 1
    assert r.actual_percent_complete == pytest.approx(50.0, abs=1e-9)
    assert r.actual_percent_complete_lower_95 == pytest.approx(6.0, abs=2.0)
    assert r.actual_percent_complete_upper_95 == pytest.approx(79.0, abs=2.0)


def test_rollup_consumes_fused_decision_when_present() -> None:
    """Phase 4 multi-view fusion may attach ``fused_decision`` next to
    the legacy ``status``; the rollup should prefer the fused decision."""
    mapping = _mapping([("A1", "X1", 1.0)])
    rows = [
        {"global_id": "X1", "status": "not_evidenced", "fused_decision": "acceptable"},
    ]
    rollup = rollup_activities(rows, mapping=mapping)[0]
    assert rollup.actual_percent_complete == pytest.approx(100.0, abs=1e-9)
    assert rollup.n_completed_elements == 1


def test_rollup_handles_missing_element_row() -> None:
    """An element that's mapped but absent from Stage 9 output is
    counted as not_evidenced and reported in the not-evidenced bucket."""
    mapping = _mapping([("A1", "X1", 1.0), ("A1", "X_MISSING", 1.0)])
    rows = [{"global_id": "X1", "status": "likely_completed"}]
    rollup = rollup_activities(rows, mapping=mapping)[0]
    assert rollup.n_mapped_elements == 2
    assert rollup.n_evaluated_elements == 1
    assert rollup.n_completed_elements == 1
    assert rollup.n_not_evidenced_elements == 1


def test_rollup_weighted_aggregation() -> None:
    """A 2x weight on one of two completed elements raises actual to 100%
    relative to a 0.5-weighted partial."""
    mapping = _mapping([("A1", "X1", 2.0), ("A1", "X2", 1.0)])
    rows = [
        {"global_id": "X1", "status": "likely_completed"},
        {"global_id": "X2", "status": "not_evidenced"},
    ]
    rollup = rollup_activities(rows, mapping=mapping)[0]
    # Weighted: (1.0 * 2 + 0.0 * 1) / (2 + 1) = 2/3 = 66.67%.
    assert rollup.actual_percent_complete == pytest.approx(66.666, abs=0.01)


def test_rollup_unknown_status_marked_uncertain_and_noted() -> None:
    mapping = _mapping([("A1", "X1", 1.0)])
    rows = [{"global_id": "X1", "status": "frobnicate"}]
    rollup = rollup_activities(rows, mapping=mapping)[0]
    assert rollup.n_uncertain_elements == 1
    assert any(n.startswith("unknown_element_status") for n in rollup.notes)


def test_rollup_accepts_all_synonym_status_keys() -> None:
    """The rollup tolerates the full vocabulary from Phase 4 fusion +
    Stage 9 legacy."""
    mapping = _mapping(
        [
            ("A1", "X1", 1.0),
            ("A1", "X2", 1.0),
            ("A1", "X3", 1.0),
            ("A1", "X4", 1.0),
            ("A1", "X5", 1.0),
        ]
    )
    rows = [
        {"global_id": "X1", "status": "acceptable"},
        {"global_id": "X2", "status": "uncertain"},
        {"global_id": "X3", "status": "uncertain_conflict"},
        {"global_id": "X4", "status": "not_acceptable"},
        {"global_id": "X5", "status": "likely_completed"},
    ]
    rollup = rollup_activities(rows, mapping=mapping)[0]
    # Mean acceptance = (1 + 0.5 + 0.5 + 0 + 1) / 5 = 0.6
    assert rollup.actual_percent_complete == pytest.approx(60.0, abs=1e-9)


def test_rollup_returns_empty_for_empty_mapping() -> None:
    mapping = _mapping([])
    rollups = rollup_activities([{"global_id": "X1", "status": "likely_completed"}], mapping=mapping)
    assert rollups == []


def test_rollup_explicit_activity_id_filter() -> None:
    """Caller can request rollups for activity IDs not in the mapping;
    those still appear (with zero evidence) so the dashboard can show
    'no evidence' cells."""
    mapping = _mapping([("A1", "X1", 1.0)])
    rollups = rollup_activities(
        [{"global_id": "X1", "status": "likely_completed"}],
        mapping=mapping,
        activity_ids=["A1", "A_GHOST"],
    )
    by_id = {r.activity_id: r for r in rollups}
    assert by_id["A_GHOST"].n_mapped_elements == 0
    assert by_id["A_GHOST"].n_evaluated_elements == 0


# ---------------------------------------------------------------------------
# classify_status
# ---------------------------------------------------------------------------


def test_classify_status_on_schedule_when_within_band() -> None:
    assert classify_status(50.0, 52.0, n_evaluated_elements=3) == "on_schedule"
    assert classify_status(50.0, 48.0, n_evaluated_elements=3) == "on_schedule"


def test_classify_status_ahead_when_actual_far_above() -> None:
    assert classify_status(50.0, 70.0, n_evaluated_elements=3) == "ahead"


def test_classify_status_behind_when_actual_far_below() -> None:
    assert classify_status(50.0, 20.0, n_evaluated_elements=3) == "behind"


def test_classify_status_unknown_evidence_when_no_evaluation() -> None:
    assert classify_status(50.0, 0.0, n_evaluated_elements=0) == "unknown_evidence"


def test_classify_status_band_is_configurable() -> None:
    assert classify_status(50.0, 53.0, n_evaluated_elements=3, on_schedule_band_pct=2.0) == "ahead"
    assert classify_status(50.0, 47.0, n_evaluated_elements=3, on_schedule_band_pct=2.0) == "behind"


def test_default_band_is_documented() -> None:
    assert DEFAULT_ON_SCHEDULE_BAND_PCT == 5.0


# ---------------------------------------------------------------------------
# build_variance_report
# ---------------------------------------------------------------------------


def test_build_variance_report_for_single_activity_on_schedule() -> None:
    sched = _schedule([_activity("A1", "Floor 1 walls")])
    rollups = [
        ActivityRollup(
            activity_id="A1",
            n_mapped_elements=10,
            n_evaluated_elements=10,
            n_completed_elements=5,
            n_partial_elements=0,
            n_uncertain_elements=0,
            n_not_evidenced_elements=5,
            actual_percent_complete=50.0,
            actual_percent_complete_lower_95=45.0,
            actual_percent_complete_upper_95=55.0,
        ),
    ]
    when = _dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc)  # mid-month -> 50% planned
    report = build_variance_report(schedule=sched, rollups=rollups, data_date=when)
    assert report.schema_version == SCHEDULE_VARIANCE_SCHEMA_VERSION
    assert report.n_activities == 1
    assert report.n_on_schedule == 1
    assert report.n_behind == 0
    assert report.activities[0].confidence == "high"  # width = 10 -> high
    assert report.activities[0].status == "on_schedule"
    # Variance 0 -> overall variance 0.
    assert report.overall_schedule_variance_percent == pytest.approx(0.0, abs=0.5)


def test_build_variance_report_flags_behind_activity() -> None:
    sched = _schedule([_activity("A1")])
    rollups = [
        ActivityRollup(
            activity_id="A1",
            n_mapped_elements=10,
            n_evaluated_elements=10,
            n_completed_elements=2,
            n_partial_elements=0,
            n_uncertain_elements=0,
            n_not_evidenced_elements=8,
            actual_percent_complete=20.0,
            actual_percent_complete_lower_95=10.0,
            actual_percent_complete_upper_95=30.0,
        ),
    ]
    when = _dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc)  # planned ~50%
    report = build_variance_report(schedule=sched, rollups=rollups, data_date=when)
    assert report.activities[0].status == "behind"
    assert "schedule_behind" in report.activities[0].risks
    assert report.n_behind == 1


def test_build_variance_report_uses_planner_asserted_percent_when_present() -> None:
    """If the schedule explicitly sets percent_complete, that wins over
    the date-interpolated value."""
    a = Activity(
        activity_id="A1",
        activity_name="Walls",
        planned_start=_dt.datetime(2026, 4, 1, tzinfo=_dt.timezone.utc),
        planned_finish=_dt.datetime(2026, 5, 1, tzinfo=_dt.timezone.utc),
        percent_complete=80.0,
    )
    sched = _schedule([a])
    rollups = [
        ActivityRollup(
            activity_id="A1",
            n_mapped_elements=1,
            n_evaluated_elements=1,
            n_completed_elements=1,
            n_partial_elements=0,
            n_uncertain_elements=0,
            n_not_evidenced_elements=0,
            actual_percent_complete=80.0,
            actual_percent_complete_lower_95=70.0,
            actual_percent_complete_upper_95=100.0,
        ),
    ]
    when = _dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc)  # interpolated ~50%
    report = build_variance_report(schedule=sched, rollups=rollups, data_date=when)
    assert report.activities[0].planned_percent_complete == 80.0


def test_build_variance_report_handles_activity_without_rollup() -> None:
    """An activity in the schedule with no rollup should appear with
    ``unknown_evidence`` status and zero overall-aggregate weight."""
    sched = _schedule([_activity("A1")])
    report = build_variance_report(schedule=sched, rollups=[], data_date=_dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc))
    assert report.n_unknown_evidence == 1
    assert report.activities[0].status == "unknown_evidence"
    assert "no_evidence_for_activity" in report.activities[0].risks


def test_build_variance_report_data_date_defaults_to_now() -> None:
    sched = _schedule([_activity("A1")])
    report = build_variance_report(schedule=sched, rollups=[])
    # Just ensure it parses; we don't pin a specific value.
    parsed = _dt.datetime.fromisoformat(report.data_date_utc)
    assert parsed.tzinfo is not None


def test_build_variance_report_overall_aggregate_weighted_by_n_mapped() -> None:
    sched = _schedule([_activity("A1"), _activity("A2")])
    rollups = [
        ActivityRollup(
            activity_id="A1",
            n_mapped_elements=10,  # heavy weight
            n_evaluated_elements=10,
            n_completed_elements=10,
            n_partial_elements=0,
            n_uncertain_elements=0,
            n_not_evidenced_elements=0,
            actual_percent_complete=100.0,
            actual_percent_complete_lower_95=90.0,
            actual_percent_complete_upper_95=100.0,
        ),
        ActivityRollup(
            activity_id="A2",
            n_mapped_elements=1,  # light weight
            n_evaluated_elements=1,
            n_completed_elements=0,
            n_partial_elements=0,
            n_uncertain_elements=0,
            n_not_evidenced_elements=1,
            actual_percent_complete=0.0,
            actual_percent_complete_lower_95=0.0,
            actual_percent_complete_upper_95=20.0,
        ),
    ]
    when = _dt.datetime(2026, 5, 1, tzinfo=_dt.timezone.utc)  # planned 100% for both
    report = build_variance_report(schedule=sched, rollups=rollups, data_date=when)
    # Overall actual = (100 * 10 + 0 * 1) / 11 ~ 90.9
    assert report.overall_actual_percent_complete == pytest.approx(90.9, abs=0.5)


# ---------------------------------------------------------------------------
# build_dashboard_summary
# ---------------------------------------------------------------------------


def test_dashboard_summary_schema_version_stable() -> None:
    sched = _schedule([_activity("A1")])
    report = build_variance_report(schedule=sched, rollups=[], data_date=_dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc))
    summary = build_dashboard_summary(report)
    assert summary.schema_version == DASHBOARD_SUMMARY_SCHEMA_VERSION


def test_dashboard_summary_kpi_payload_round_trips_through_json() -> None:
    sched = _schedule([_activity("A1"), _activity("A2")])
    rollups = [
        ActivityRollup(
            activity_id="A1",
            n_mapped_elements=4,
            n_evaluated_elements=4,
            n_completed_elements=2,
            n_partial_elements=0,
            n_uncertain_elements=0,
            n_not_evidenced_elements=2,
            actual_percent_complete=50.0,
            actual_percent_complete_lower_95=20.0,
            actual_percent_complete_upper_95=80.0,
        ),
    ]
    report = build_variance_report(schedule=sched, rollups=rollups, data_date=_dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc))
    summary = build_dashboard_summary(report)
    s = json.dumps(summary.to_dict())
    parsed = json.loads(s)
    assert parsed["schema_version"] == DASHBOARD_SUMMARY_SCHEMA_VERSION
    assert parsed["kpi"]["n_activities"] == 2
    assert "activities" in parsed
    assert len(parsed["activities"]) == 2
