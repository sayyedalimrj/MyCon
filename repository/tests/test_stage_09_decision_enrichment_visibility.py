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


def test_decision_enrichment_writes_visibility_fields(tmp_path: Path) -> None:
    element_csv = tmp_path / "element_metrics.csv"
    activity_csv = tmp_path / "activity_progress.csv"
    reg_json = tmp_path / "registration_quality.json"

    _write_csv(
        element_csv,
        [
            {
                "global_id": "E1",
                "activity_id": "A1",
                "coverage": "0.0",
                "in_tolerance_ratio": "0.0",
                "confidence": "0.2",
                "status": "ok",
                "visibility_confidence": "high",
            }
        ],
    )
    _write_csv(
        activity_csv,
        [
            {
                "activity_id": "A1",
                "activity_name": "Walls",
                "observed_percent": "0.0",
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

    row = _read_csv(element_csv)[0]

    assert row["acceptable"] == "false"
    assert row["visibility_status"] == "visible"
    assert row["visibility_evidence_status"] == "visible_area_low_or_zero_coverage"
    assert row["construction_state_interpretation"] == "not_observed_in_visible_area"
    assert "visible_area_coverage_below_observation_threshold" in row["decision_risks"]
