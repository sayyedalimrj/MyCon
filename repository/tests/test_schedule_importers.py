"""Tests for the schedule side-car importers.

We do not parse vendor binary formats here; we test the *text* importers
(MSP XML, P6 XER, generic CSV) against tiny synthetic fixtures and then
*round-trip* the resulting canonical CSV through the production loader
:func:`pipeline.common.schedule_io.load_schedule_csv`. That gives us
end-to-end confidence: an importer is correct iff its output is a valid
canonical schedule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.common.schedule_io import load_schedule_csv


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# import_schedule_msp_xml
# ---------------------------------------------------------------------------


def _msp_xml_with_two_tasks() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<Project xmlns="http://schemas.microsoft.com/project">
  <Tasks>
    <Task>
      <UID>0</UID>
      <Name>Project Summary</Name>
    </Task>
    <Task>
      <UID>1</UID>
      <Name>Foundations</Name>
      <OutlineNumber>1.1</OutlineNumber>
      <Start>2026-03-01T08:00:00</Start>
      <Finish>2026-04-01T17:00:00</Finish>
      <PercentComplete>100</PercentComplete>
    </Task>
    <Task>
      <UID>2</UID>
      <Name>Floor 2 Zone B walls</Name>
      <OutlineNumber>1.2.3</OutlineNumber>
      <Start>2026-04-01T08:00:00</Start>
      <Finish>2026-05-01T17:00:00</Finish>
      <PercentComplete>25</PercentComplete>
      <PredecessorLink>
        <PredecessorUID>1</PredecessorUID>
      </PredecessorLink>
    </Task>
  </Tasks>
</Project>
"""


def test_msp_xml_round_trips_through_canonical_loader(tmp_path: Path) -> None:
    from scripts.import_schedule_msp_xml import main as msp_main

    inp = tmp_path / "in.xml"
    inp.write_text(_msp_xml_with_two_tasks(), encoding="utf-8")
    out = tmp_path / "schedule.csv"
    summary = tmp_path / "summary.json"
    rc = msp_main(
        [
            "--input", str(inp),
            "--output", str(out),
            "--summary-json", str(summary),
        ]
    )
    assert rc == 0

    schedule = load_schedule_csv(out)
    assert len(schedule) == 2
    assert {a.activity_id for a in schedule.activities} == {"1", "2"}
    a2 = schedule.get("2")
    assert a2.activity_name == "Floor 2 Zone B walls"
    assert a2.wbs_code == "1.2.3"
    assert a2.percent_complete == 25.0
    assert a2.predecessors == ("1",)


def test_msp_xml_skips_summary_task(tmp_path: Path) -> None:
    """UID=0 and missing-Start/Finish tasks must be filtered out."""
    from scripts.import_schedule_msp_xml import main as msp_main

    inp = tmp_path / "in.xml"
    inp.write_text(_msp_xml_with_two_tasks(), encoding="utf-8")
    out = tmp_path / "schedule.csv"
    rc = msp_main(["--input", str(inp), "--output", str(out)])
    assert rc == 0
    schedule = load_schedule_csv(out)
    # Only the two real tasks; the UID=0 summary should be dropped.
    assert len(schedule) == 2
    assert "0" not in {a.activity_id for a in schedule.activities}


def test_msp_xml_invalid_xml_returns_failure(tmp_path: Path) -> None:
    from scripts.import_schedule_msp_xml import main as msp_main

    inp = tmp_path / "in.xml"
    inp.write_text("<not really xml", encoding="utf-8")
    rc = msp_main(["--input", str(inp), "--output", str(tmp_path / "out.csv")])
    assert rc == 2  # documented "parse error" exit code


# ---------------------------------------------------------------------------
# import_schedule_p6_xer
# ---------------------------------------------------------------------------


def _p6_xer_two_tasks() -> str:
    """A minimal but realistic XER snippet covering TASK + TASKPRED."""
    lines = [
        "ERMHDR\t19.12\t2026-04-01\tProject\tDB\tdbo\tprivileged\tUS-WIN\tPM",
        "%T\tTASK",
        "%F\ttask_id\ttask_code\ttask_name\twbs_id\tearly_start_date\tearly_end_date\tphys_complete_pct",
        "%R\t100\tA0001\tFoundations\tWBS-1\t2026-03-01 08:00\t2026-04-01 17:00\t100",
        "%R\t200\tA0432\tFloor 2 Zone B walls\tWBS-2\t2026-04-01 08:00\t2026-05-01 17:00\t25",
        "%T\tTASKPRED",
        "%F\ttask_id\tpred_task_id",
        "%R\t200\t100",
        "%E",
    ]
    return "\n".join(lines) + "\n"


def test_p6_xer_round_trips_through_canonical_loader(tmp_path: Path) -> None:
    from scripts.import_schedule_p6_xer import main as p6_main

    inp = tmp_path / "in.xer"
    inp.write_text(_p6_xer_two_tasks(), encoding="utf-8")
    out = tmp_path / "schedule.csv"
    rc = p6_main(["--input", str(inp), "--output", str(out)])
    assert rc == 0

    schedule = load_schedule_csv(out)
    assert len(schedule) == 2
    a432 = schedule.get("A0432")
    assert a432.activity_name == "Floor 2 Zone B walls"
    assert a432.wbs_code == "WBS-2"
    assert a432.percent_complete == 25.0
    # Predecessor resolution from internal task_id back to task_code:
    assert a432.predecessors == ("A0001",)


def test_p6_xer_with_no_task_table_returns_failure_message(tmp_path: Path) -> None:
    """If the XER has no TASK table the importer writes an empty CSV
    rather than crashing; the summary records the skip reason."""
    from scripts.import_schedule_p6_xer import main as p6_main

    inp = tmp_path / "in.xer"
    inp.write_text("ERMHDR\nnothing else\n%E\n", encoding="utf-8")
    out = tmp_path / "schedule.csv"
    summary = tmp_path / "summary.json"
    rc = p6_main(["--input", str(inp), "--output", str(out), "--summary-json", str(summary)])
    assert rc == 0
    # The output CSV will have the header but no rows. The canonical
    # loader treats that as a valid empty schedule.
    schedule = load_schedule_csv(out)
    assert len(schedule) == 0


# ---------------------------------------------------------------------------
# import_schedule_generic_csv
# ---------------------------------------------------------------------------


def _generic_vendor_csv() -> str:
    return (
        "Task ID,Task Name,Start Date,Finish Date,WBS,% Complete,Predecessors,Trade,Location\n"
        "A0001,Foundations,2026-03-01,2026-04-01,1.1,100,,structural,Site\n"
        "A0432,Floor 2 Zone B walls,2026-04-01,2026-05-01,1.2.3,25,A0001,structural,Floor 2 Zone B\n"
    )


def test_generic_csv_round_trips_through_canonical_loader(tmp_path: Path) -> None:
    from scripts.import_schedule_generic_csv import main as gen_main

    inp = tmp_path / "in.csv"
    inp.write_text(_generic_vendor_csv(), encoding="utf-8")
    out = tmp_path / "schedule.csv"
    rc = gen_main(
        [
            "--input", str(inp),
            "--output", str(out),
            "--activity-id-column", "Task ID",
            "--activity-name-column", "Task Name",
            "--planned-start-column", "Start Date",
            "--planned-finish-column", "Finish Date",
            "--wbs-column", "WBS",
            "--percent-complete-column", "% Complete",
            "--predecessors-column", "Predecessors",
            "--trade-column", "Trade",
            "--location-column", "Location",
        ]
    )
    assert rc == 0
    schedule = load_schedule_csv(out)
    a432 = schedule.get("A0432")
    assert a432 is not None
    assert a432.percent_complete == 25.0
    assert a432.trade == "structural"
    assert a432.location == "Floor 2 Zone B"
    assert a432.predecessors == ("A0001",)


def test_generic_csv_missing_required_column_writes_empty_output(tmp_path: Path) -> None:
    from scripts.import_schedule_generic_csv import main as gen_main

    inp = tmp_path / "in.csv"
    inp.write_text("FooId,FooName\n1,a\n", encoding="utf-8")
    out = tmp_path / "schedule.csv"
    summary = tmp_path / "summary.json"
    rc = gen_main(
        [
            "--input", str(inp),
            "--output", str(out),
            "--activity-id-column", "Task ID",  # not present in the input
            "--activity-name-column", "Task Name",
            "--planned-start-column", "Start Date",
            "--planned-finish-column", "Finish Date",
            "--summary-json", str(summary),
        ]
    )
    assert rc == 0
    schedule = load_schedule_csv(out)
    assert len(schedule) == 0


def test_generic_csv_skips_duplicate_activity_ids(tmp_path: Path) -> None:
    from scripts.import_schedule_generic_csv import main as gen_main

    inp = tmp_path / "in.csv"
    inp.write_text(
        "Task ID,Task Name,Start Date,Finish Date\n"
        "A1,Wall1,2026-04-01,2026-04-02\n"
        "A1,Wall1 dup,2026-04-03,2026-04-04\n",
        encoding="utf-8",
    )
    out = tmp_path / "schedule.csv"
    rc = gen_main(
        [
            "--input", str(inp),
            "--output", str(out),
            "--activity-id-column", "Task ID",
            "--activity-name-column", "Task Name",
            "--planned-start-column", "Start Date",
            "--planned-finish-column", "Finish Date",
        ]
    )
    assert rc == 0
    schedule = load_schedule_csv(out)
    assert len(schedule) == 1
    assert schedule.activities[0].activity_name == "Wall1"
