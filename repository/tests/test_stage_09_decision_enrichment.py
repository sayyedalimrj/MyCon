from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeline.stage_09_progress.decision_enrichment import enrich_progress_decisions_from_files


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_low_registration_enrichment_blocks_element_and_activity_acceptance(tmp_path: Path) -> None:
    element_csv = tmp_path / "element_metrics.csv"
    activity_csv = tmp_path / "activity_progress.csv"
    reg_json = tmp_path / "registration_quality.json"
    summary_json = tmp_path / "progress_decision_summary.json"

    _write_csv(
        element_csv,
        [
            {
                "global_id": "E1",
                "activity_id": "A1",
                "coverage": "1.0",
                "in_tolerance_ratio": "1.0",
                "confidence": "1.0",
                "status": "ok",
            }
        ],
    )
    _write_csv(
        activity_csv,
        [
            {
                "activity_id": "A1",
                "activity_name": "Walls",
                "observed_percent": "100.0",
                "status": "ok",
            }
        ],
    )
    reg_json.write_text(json.dumps({"confidence_label": "low"}), encoding="utf-8")

    summary = enrich_progress_decisions_from_files(
        element_metrics_csv=element_csv,
        activity_progress_csv=activity_csv,
        registration_quality_json=reg_json,
        output_summary_json=summary_json,
    )

    element_row = _read_csv(element_csv)[0]
    activity_row = _read_csv(activity_csv)[0]

    assert summary["registration_confidence"] == "low"
    assert element_row["acceptable"] == "false"
    assert element_row["completion_state"] == "uncertain_low_registration"
    assert "registration_confidence_low" in element_row["decision_risks"]
    assert activity_row["acceptable"] == "false"
    assert activity_row["completion_state"] == "uncertain_low_registration"
    assert summary_json.exists()


def test_high_registration_enrichment_allows_good_element_and_activity(tmp_path: Path) -> None:
    element_csv = tmp_path / "element_metrics.csv"
    activity_csv = tmp_path / "activity_progress.csv"
    reg_json = tmp_path / "registration_quality.json"

    _write_csv(
        element_csv,
        [
            {
                "global_id": "E1",
                "activity_id": "A1",
                "coverage": "0.95",
                "in_tolerance_ratio": "0.90",
                "confidence": "0.92",
                "status": "ok",
            }
        ],
    )
    _write_csv(
        activity_csv,
        [
            {
                "activity_id": "A1",
                "activity_name": "Walls",
                "observed_percent": "100.0",
                "status": "ok",
            }
        ],
    )
    reg_json.write_text(json.dumps({"confidence_label": "high"}), encoding="utf-8")

    enrich_progress_decisions_from_files(
        element_metrics_csv=element_csv,
        activity_progress_csv=activity_csv,
        registration_quality_json=reg_json,
    )

    element_row = _read_csv(element_csv)[0]
    activity_row = _read_csv(activity_csv)[0]

    assert element_row["acceptable"] == "true"
    assert element_row["completion_state"] == "completed"
    assert activity_row["acceptable"] == "true"
    assert activity_row["completion_state"] == "completed"
