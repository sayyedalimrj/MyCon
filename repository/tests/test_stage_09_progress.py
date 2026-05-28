from __future__ import annotations

import pytest
import csv
import json
from pathlib import Path

import numpy as np
o3d = pytest.importorskip("open3d")

from pipeline.stage_09_progress.config_access import cfg_get
from pipeline.stage_09_progress.run_progress import _dashboard_html, _status_from_metrics, run_progress


def _write_cloud(path: Path, offset=(0.0, 0.0, 0.0)) -> None:
    xs, ys = np.meshgrid(np.linspace(0, 2, 16), np.linspace(0, 2, 16))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)]) + np.asarray(offset)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cfg(root: Path, fitness: float = 1.0) -> dict:
    _write_cloud(root / "data/bim/aligned/site01/scan_aligned.ply")
    _write_cloud(root / "data/bim/aligned/site01/bim_reference.ply")
    elements = root / "data/bim/aligned/site01/bim_elements.jsonl"
    elements.parent.mkdir(parents=True, exist_ok=True)
    elements.write_text(json.dumps({
        "global_id": "E1",
        "ifc_class": "IfcWall",
        "name": "WALL_1",
        "bounds_min": [0, 0, -0.1],
        "bounds_max": [2, 2, 0.1],
    }) + "\n", encoding="utf-8")
    report = root / "runs/test/reports/registration_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"icp": {"fitness": fitness, "inlier_rmse": 0.01}, "quality_gate": {"status": "pass", "warnings": []}}), encoding="utf-8")
    _write_csv(root / "data/bim/design/schedule.csv", [{"activity_id": "A1", "activity_name": "Wall activity", "planned_start": "2026-01-01", "planned_finish": "2026-01-02", "planned_weight": "1.0"}])
    _write_csv(root / "data/bim/design/element_activity_map.csv", [{"global_id": "E1", "activity_id": "A1", "activity_name": "Wall activity", "quantity_type": "area_m2", "quantity_value": "4", "weight": "1.0"}])
    return {
        "project": {"root": str(root), "name": "site01", "run_id": "test"},
        "bim": {
            "scan_aligned_ply": "data/bim/aligned/site01/scan_aligned.ply",
            "bim_reference_ply": "data/bim/aligned/site01/bim_reference.ply",
            "element_metadata_jsonl": "data/bim/aligned/site01/bim_elements.jsonl",
            "registration_report_json": "runs/test/reports/registration_report.json",
        },
        "progress": {
            "schedule_csv": "data/bim/design/schedule.csv",
            "element_activity_map_csv": "data/bim/design/element_activity_map.csv",
            "coverage_threshold": 0.65,
            "deviation_threshold_m": 0.05,
        },
        "copilot": {"paths": {
            "element_metrics_csv": "data/bim/metrics/site01/element_metrics.csv",
            "activity_progress_csv": "data/bim/metrics/site01/activity_progress.csv",
            "deviation_summary_json": "data/bim/metrics/site01/deviation_summary.json",
            "coverage_summary_json": "data/bim/metrics/site01/coverage_summary.json",
            "registration_quality_json": "data/bim/metrics/site01/registration_quality.json",
        }},
    }


def test_cfg_get_reads_nested_dict() -> None:
    cfg = {"project": {"root": "/tmp/x"}}
    assert cfg_get(cfg, "project.root") == "/tmp/x"


def test_run_progress_writes_metrics(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, fitness=1.0)
    report = run_progress(cfg, force=True, log_level="ERROR")
    assert report["element_count"] == 1
    assert (tmp_path / "data/bim/metrics/site01/element_metrics.csv").exists()
    assert (tmp_path / "data/bim/metrics/site01/activity_progress.csv").exists()
    assert (tmp_path / "runs/test/reports/progress_dashboard.html").exists()


def test_low_registration_confidence_is_reported(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, fitness=0.001)
    report = run_progress(cfg, force=True, log_level="ERROR")
    assert report["registration_quality"]["confidence_label"] == "low"
    rows = list(csv.DictReader((tmp_path / "data/bim/metrics/site01/element_metrics.csv").open()))
    assert rows[0]["status"] == "uncertain_low_registration"

pytestmark = pytest.mark.geometry


def test_status_from_metrics_uses_configurable_partial_threshold() -> None:
    assert _status_from_metrics(0.30, "high", 0.65, partial_threshold=0.20) == "partially_observed"
    assert _status_from_metrics(0.30, "high", 0.65, partial_threshold=0.50) == "not_evidenced"


def test_dashboard_html_escapes_ifc_strings() -> None:
    html_text = _dashboard_html(
        {
            "project": {"name": "site01", "run_id": "test"},
            "status": "complete",
            "registration_quality": {"confidence_label": "high", "fitness": 1.0, "rmse_m": 0.01},
            "mode_note": "safe",
        },
        [
            {
                "name": "Wall <A> & Beam",
                "ifc_class": "IfcWall",
                "coverage": "1.0",
                "status": "likely_completed",
                "confidence": "1.0",
            }
        ],
        [],
    )

    assert "Wall &lt;A&gt; &amp; Beam" in html_text
    assert "Wall <A> & Beam" not in html_text
