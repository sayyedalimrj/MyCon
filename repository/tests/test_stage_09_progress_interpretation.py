from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeline.stage_09_progress.progress_interpretation import (
    infer_completion_state,
    upgrade_element_row,
    upgrade_progress_outputs,
)


def test_low_registration_blocks_completion_claim() -> None:
    row = {
        "coverage": "0.95",
        "in_tolerance_ratio": "0.95",
        "confidence": "0.95",
        "point_count_evaluated": "5000",
        "status": "uncertain_low_registration",
        "registration_confidence": "low",
    }

    completion_state, evidence_state, notes = infer_completion_state(row)

    assert completion_state == "uncertain_low_registration"
    assert evidence_state == "uncertain"
    assert "registration_low_progress_interpretation_blocked" in notes


def test_not_observed_is_not_marked_not_built() -> None:
    row = {
        "coverage": "0.0",
        "in_tolerance_ratio": "0.0",
        "confidence": "0.0",
        "point_count_evaluated": "0",
        "status": "candidate",
        "registration_confidence": "high",
    }

    upgraded = upgrade_element_row(row)

    assert upgraded["completion_state"] == "not_evidenced"
    assert upgraded["evidence_state"] == "not_evidenced"
    assert "not_enough_observed_surface_to_infer_progress" in upgraded["interpretation_notes"]


def test_upgrade_progress_outputs_writes_interpreted_csvs(tmp_path: Path) -> None:
    element_csv = tmp_path / "element_metrics.csv"
    activity_csv = tmp_path / "activity_progress.csv"
    out_element = tmp_path / "element_metrics_interpreted.csv"
    out_activity = tmp_path / "activity_progress_interpreted.csv"
    out_summary = tmp_path / "summary.json"

    element_rows = [
        {
            "global_id": "E1",
            "name": "Wall",
            "activity_id": "A1",
            "coverage": "0.80",
            "in_tolerance_ratio": "0.80",
            "confidence": "0.80",
            "point_count_evaluated": "1000",
            "status": "candidate",
            "registration_confidence": "high",
        }
    ]
    activity_rows = [
        {
            "activity_id": "A1",
            "activity_name": "Walls",
            "observed_percent": "80",
            "planned_percent": "100",
            "status": "candidate",
        }
    ]

    with element_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(element_rows[0].keys()))
        writer.writeheader()
        writer.writerows(element_rows)

    with activity_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(activity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(activity_rows)

    summary = upgrade_progress_outputs(
        element_metrics_csv=element_csv,
        activity_progress_csv=activity_csv,
        output_element_csv=out_element,
        output_activity_csv=out_activity,
        output_summary_json=out_summary,
    )

    assert summary["element_count"] == 1
    assert out_element.exists()
    assert out_activity.exists()
    assert json.loads(out_summary.read_text(encoding="utf-8"))["status"] == "complete"

    with out_element.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["completion_state"] == "complete_candidate"
    assert row["metric_truth_source"] == "stage8_registration_and_stage9_deterministic_metrics"
