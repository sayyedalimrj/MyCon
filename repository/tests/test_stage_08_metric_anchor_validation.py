from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeline.stage_08_bim_eval.metric_anchor_validation import (
    prepare_metric_anchor_template,
    validate_metric_anchor_files,
)


def _write_anchor_csv(path: Path, complete: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"anchor_id": "A", "description": "A", "bim_x_m": "0", "bim_y_m": "0", "bim_z_m": "0", "scan_x": "0", "scan_y": "0", "scan_z": "0", "use_for_scale": "true", "use_for_registration": "true"},
        {"anchor_id": "B", "description": "B", "bim_x_m": "10", "bim_y_m": "0", "bim_z_m": "0", "scan_x": "5", "scan_y": "0", "scan_z": "0", "use_for_scale": "true", "use_for_registration": "true"},
        {"anchor_id": "C", "description": "C", "bim_x_m": "0", "bim_y_m": "10", "bim_z_m": "0", "scan_x": "0", "scan_y": "5", "scan_z": "0", "use_for_scale": "true", "use_for_registration": "true"},
    ]
    if not complete:
        rows[1]["scan_x"] = ""
        rows[1]["scan_y"] = ""
        rows[1]["scan_z"] = ""

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_known_distances(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"distance_id": "AB", "anchor_a": "A", "anchor_b": "B", "distance_m": "10"},
        {"distance_id": "AC", "anchor_a": "A", "anchor_b": "C", "distance_m": "10"},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_metric_anchor_template_normalizes_columns(tmp_path: Path) -> None:
    source = tmp_path / "metric_anchors.csv"
    output = tmp_path / "metric_anchors_working.csv"
    _write_anchor_csv(source, complete=False)

    prepare_metric_anchor_template(source, output, force=True)

    with output.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert "bim_x_m" in rows[0]
    assert "scan_x" in rows[0]
    assert rows[0]["use_for_registration"] == "true"


def test_validate_metric_anchors_reports_incomplete_when_scan_coordinates_missing(tmp_path: Path) -> None:
    anchors = tmp_path / "metric_anchors.csv"
    distances = tmp_path / "known_distances.csv"
    output = tmp_path / "metric_anchor_validation.json"
    _write_anchor_csv(anchors, complete=False)
    _write_known_distances(distances)

    result = validate_metric_anchor_files(anchors, distances, output_json=output)

    assert result.status == "incomplete"
    assert result.ready_for_metric_alignment is False
    assert result.complete_registration_anchor_count == 2
    assert "B" in result.anchors_missing_scan_coordinates
    assert output.exists()


def test_validate_metric_anchors_ready_with_three_complete_points(tmp_path: Path) -> None:
    anchors = tmp_path / "metric_anchors.csv"
    distances = tmp_path / "known_distances.csv"
    output = tmp_path / "metric_anchor_validation.json"
    _write_anchor_csv(anchors, complete=True)
    _write_known_distances(distances)

    result = validate_metric_anchor_files(anchors, distances, output_json=output)

    assert result.status == "ready"
    assert result.ready_for_metric_alignment is True
    assert result.complete_registration_anchor_count == 3
    assert result.usable_known_distance_count == 2
    assert result.estimated_scale_from_known_distances == 2.0

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["status"] == "ready"
