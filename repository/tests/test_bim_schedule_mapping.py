"""Tests for :mod:`pipeline.common.bim_schedule_mapping`."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.common.bim_schedule_mapping import (
    MAPPING_SCHEMA_VERSION,
    REQUIRED_MAPPING_COLUMNS,
    BimScheduleMapping,
    MappingEntry,
    load_mapping_csv,
    validate_mapping,
)


pytestmark = pytest.mark.lightweight


def _write(tmp_path: Path, header: str, rows: list[str]) -> Path:
    p = tmp_path / "mapping.csv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_load_mapping_minimal(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "activity_id,ifc_global_id",
        [
            "A0432,1Pq8MeKvD2vQ8XYZabcdef",
            "A0432,2Pq8MeKvD2vQ8XYZabcdef",
            "A0001,3Pq8MeKvD2vQ8XYZabcdef",
        ],
    )
    m = load_mapping_csv(p)
    assert isinstance(m, BimScheduleMapping)
    assert len(m) == 3
    assert m.schema_version == MAPPING_SCHEMA_VERSION
    assert len(m.elements_for_activity("A0432")) == 2
    assert len(m.elements_for_activity("A0001")) == 1
    assert m.elements_for_activity("A_UNKNOWN") == ()


def test_load_mapping_with_explicit_weight(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "activity_id,ifc_global_id,weight",
        [
            "A0432,1Pq8MeKvD2vQ8XYZabcdef,1.0",
            "A0432,2Pq8MeKvD2vQ8XYZabcdef,0.5",
        ],
    )
    m = load_mapping_csv(p)
    weights = [e.weight for e in m.elements_for_activity("A0432")]
    assert sorted(weights) == [0.5, 1.0]


def test_load_mapping_default_weight_is_one(tmp_path: Path) -> None:
    p = _write(tmp_path, "activity_id,ifc_global_id", ["A1,X1"])
    m = load_mapping_csv(p)
    assert m.entries[0].weight == 1.0


def test_load_mapping_invalid_weight_falls_back_to_default(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "activity_id,ifc_global_id,weight",
        ["A1,X1,not-a-number", "A2,X2,-3.0"],
    )
    m = load_mapping_csv(p)
    by_id = {e.activity_id: e.weight for e in m.entries}
    # Garbage weight -> default 1.0; negative weight -> clamped to 0.0.
    assert by_id["A1"] == 1.0
    assert by_id["A2"] == 0.0


def test_load_mapping_skips_rows_with_blank_ids(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "activity_id,ifc_global_id",
        ["A1,X1", ",X2", "A3,"],
    )
    m = load_mapping_csv(p)
    assert len(m) == 1


def test_load_mapping_raises_when_required_columns_missing(tmp_path: Path) -> None:
    p = _write(tmp_path, "wrong,columns", ["A1,X1"])
    with pytest.raises(ValueError):
        load_mapping_csv(p)


def test_load_mapping_raises_when_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mapping_csv(tmp_path / "nope.csv")


def test_load_mapping_records_provenance(tmp_path: Path) -> None:
    p = _write(tmp_path, "activity_id,ifc_global_id", ["A1,X1"])
    m = load_mapping_csv(p)
    assert m.source_path == str(p.resolve())
    assert len(m.source_sha256) == 64


def test_required_columns_constant_is_stable() -> None:
    assert REQUIRED_MAPPING_COLUMNS == ("activity_id", "ifc_global_id")


# ---------------------------------------------------------------------------
# By-element index
# ---------------------------------------------------------------------------


def test_activities_for_element_returns_all_activities(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "activity_id,ifc_global_id,weight",
        [
            "A0432,1Pq8MeKvD2vQ8XYZabcdef,1.0",
            "A0500,1Pq8MeKvD2vQ8XYZabcdef,0.3",
        ],
    )
    m = load_mapping_csv(p)
    aff = m.activities_for_element("1Pq8MeKvD2vQ8XYZabcdef")
    assert {e.activity_id for e in aff} == {"A0432", "A0500"}


# ---------------------------------------------------------------------------
# validate_mapping
# ---------------------------------------------------------------------------


def test_validate_mapping_full_coverage(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "activity_id,ifc_global_id",
        ["A1,X1", "A1,X2", "A2,X3"],
    )
    m = load_mapping_csv(p)
    rep = validate_mapping(
        m,
        schedule_activity_ids=["A1", "A2"],
        bim_element_global_ids=["X1", "X2", "X3"],
    )
    assert rep.coverage_ratio == 1.0
    assert rep.activities_in_mapping_not_in_schedule == ()
    assert rep.elements_in_mapping_not_in_bim == ()
    assert rep.bim_elements_not_in_mapping == ()


def test_validate_mapping_partial_coverage(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "activity_id,ifc_global_id",
        ["A1,X1", "A1,X2"],
    )
    m = load_mapping_csv(p)
    rep = validate_mapping(
        m,
        schedule_activity_ids=["A1", "A2"],
        bim_element_global_ids=["X1", "X2", "X3", "X4"],
    )
    # 2 of 4 BIM elements are mapped.
    assert rep.coverage_ratio == 0.5
    assert rep.bim_elements_not_in_mapping == ("X3", "X4")


def test_validate_mapping_orphan_activities_and_elements(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "activity_id,ifc_global_id",
        ["A_GHOST,X1", "A1,X_GHOST"],
    )
    m = load_mapping_csv(p)
    rep = validate_mapping(
        m,
        schedule_activity_ids=["A1"],
        bim_element_global_ids=["X1"],
    )
    assert rep.activities_in_mapping_not_in_schedule == ("A_GHOST",)
    assert rep.elements_in_mapping_not_in_bim == ("X_GHOST",)


def test_validate_mapping_empty_bim_yields_zero_coverage() -> None:
    m = BimScheduleMapping(entries=())
    rep = validate_mapping(m, schedule_activity_ids=[], bim_element_global_ids=[])
    assert rep.coverage_ratio == 0.0


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_mapping_to_dict_is_json_round_trippable(tmp_path: Path) -> None:
    import json

    p = _write(tmp_path, "activity_id,ifc_global_id,weight", ["A1,X1,0.7"])
    m = load_mapping_csv(p)
    s = json.dumps(m.to_dict())
    parsed = json.loads(s)
    assert parsed["schema_version"] == MAPPING_SCHEMA_VERSION
    assert parsed["entries"][0]["weight"] == 0.7
