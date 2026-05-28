from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pipeline.stage_08_bim_eval.metric_alignment import (
    build_metric_alignment_report,
    estimate_sim3_umeyama,
    estimate_scale_from_known_distances,
    read_known_distances_csv,
    read_metric_anchors_csv,
)


def _write_anchor_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    scale = 2.0
    translation = np.asarray([1.0, -3.0, 0.5])

    scan_points = {
        "A": np.asarray([0.0, 0.0, 0.0]),
        "B": np.asarray([1.0, 0.0, 0.0]),
        "C": np.asarray([0.0, 2.0, 0.0]),
        "D": np.asarray([0.0, 0.0, 1.0]),
    }

    for anchor_id, scan in scan_points.items():
        bim = scale * scan + translation
        rows.append({
            "anchor_id": anchor_id,
            "description": f"anchor {anchor_id}",
            "bim_x_m": bim[0],
            "bim_y_m": bim[1],
            "bim_z_m": bim[2],
            "scan_x": scan[0],
            "scan_y": scan[1],
            "scan_z": scan[2],
            "use_for_scale": "true",
            "use_for_registration": "true",
        })

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_distances_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"distance_id": "AB", "anchor_a": "A", "anchor_b": "B", "distance_m": 2.0},
        {"distance_id": "AC", "anchor_a": "A", "anchor_b": "C", "distance_m": 4.0},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_read_metric_anchors_and_known_distances(tmp_path: Path) -> None:
    anchors_path = tmp_path / "metric_anchors.csv"
    distances_path = tmp_path / "known_distances.csv"
    _write_anchor_csv(anchors_path)
    _write_distances_csv(distances_path)

    anchors = read_metric_anchors_csv(anchors_path)
    distances = read_known_distances_csv(distances_path)

    assert len(anchors) == 4
    assert anchors["A"].scan_xyz is not None
    assert len(distances) == 2


def test_known_distance_scale_estimation(tmp_path: Path) -> None:
    anchors_path = tmp_path / "metric_anchors.csv"
    distances_path = tmp_path / "known_distances.csv"
    _write_anchor_csv(anchors_path)
    _write_distances_csv(distances_path)

    anchors = read_metric_anchors_csv(anchors_path)
    distances = read_known_distances_csv(distances_path)
    result = estimate_scale_from_known_distances(anchors, distances)

    assert result["status"] == "ok"
    assert abs(result["scale"] - 2.0) < 1.0e-9
    assert result["valid_record_count"] == 2


def test_build_metric_alignment_report_estimates_sim3(tmp_path: Path) -> None:
    anchors_path = tmp_path / "metric_anchors.csv"
    distances_path = tmp_path / "known_distances.csv"
    report_path = tmp_path / "metric_alignment_report.json"
    _write_anchor_csv(anchors_path)
    _write_distances_csv(distances_path)

    report = build_metric_alignment_report(
        anchors_csv=anchors_path,
        known_distances_csv=distances_path,
        output_json=report_path,
    )

    assert report_path.exists()
    assert report["status"] == "ok"
    assert report["confidence"] == "high"
    assert report["can_feed_stage8"] is True
    assert abs(report["transform"]["scale"] - 2.0) < 1.0e-9
    assert report["residuals"]["rmse_m"] < 1.0e-9


def test_build_metric_alignment_report_attaches_selection_and_persists(tmp_path: Path) -> None:
    anchors_path = tmp_path / "metric_anchors.csv"
    distances_path = tmp_path / "known_distances.csv"
    report_path = tmp_path / "metric_alignment_report.json"
    _write_anchor_csv(anchors_path)
    _write_distances_csv(distances_path)

    report = build_metric_alignment_report(
        anchors_csv=anchors_path,
        known_distances_csv=distances_path,
        output_json=report_path,
    )

    assert "metric_alignment_selection" in report
    assert report["metric_alignment_selection"]["input_anchor_count"] == 4
    assert report_path.exists()

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert "metric_alignment_selection" in saved
    assert saved["metric_alignment_selection"]["input_anchor_count"] == 4
