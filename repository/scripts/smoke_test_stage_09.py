from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
try:
    import open3d as o3d
except ModuleNotFoundError:
    print("SMOKE_SKIP_MISSING_DEPENDENCY script=smoke_test_stage_09.py dependency=open3d")
    raise SystemExit(0)

from pipeline.stage_09_progress.run_progress import run_progress


def _write_cloud(path: Path, offset=(0.0, 0.0, 0.0)) -> None:
    xs, ys = np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)]) + np.asarray(offset)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.5, 0.5, 0.5]], (len(pts), 1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="stage09_smoke_"))
    scan = root / "data/bim/aligned/site01/scan_aligned.ply"
    bim = root / "data/bim/aligned/site01/bim_reference.ply"
    _write_cloud(scan)
    _write_cloud(bim)

    elements = root / "data/bim/aligned/site01/bim_elements.jsonl"
    elements.parent.mkdir(parents=True, exist_ok=True)
    elements.write_text(json.dumps({
        "global_id": "E1",
        "ifc_class": "IfcSlab",
        "name": "SMOKE_SLAB",
        "bounds_min": [0, 0, -0.1],
        "bounds_max": [2, 2, 0.1],
    }) + "\n", encoding="utf-8")

    report = root / "runs/test/reports/registration_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"icp": {"fitness": 1.0, "inlier_rmse": 0.01}, "quality_gate": {"status": "pass", "warnings": []}}), encoding="utf-8")

    _write_csv(root / "data/bim/design/schedule.csv", [{"activity_id": "A1", "activity_name": "Smoke activity", "planned_start": "2026-01-01", "planned_finish": "2026-01-02", "planned_weight": "1.0"}])
    _write_csv(root / "data/bim/design/element_activity_map.csv", [{"global_id": "E1", "activity_id": "A1", "activity_name": "Smoke activity", "quantity_type": "area_m2", "quantity_value": "4", "weight": "1.0"}])

    cfg = {
        "project": {"root": str(root), "name": "site01", "run_id": "test"},
        "bim": {
            "scan_aligned_ply": "data/bim/aligned/site01/scan_aligned.ply",
            "bim_reference_ply": "data/bim/aligned/site01/bim_reference.ply",
            "element_metadata_jsonl": "data/bim/aligned/site01/bim_elements.jsonl",
            "registration_report_json": "runs/test/reports/registration_report.json",
        },
        "progress": {
            "deviation_threshold_m": 0.05,
            "coverage_threshold": 0.65,
            "schedule_csv": "data/bim/design/schedule.csv",
            "element_activity_map_csv": "data/bim/design/element_activity_map.csv",
        },
        "copilot": {"paths": {
            "element_metrics_csv": "data/bim/metrics/site01/element_metrics.csv",
            "activity_progress_csv": "data/bim/metrics/site01/activity_progress.csv",
            "deviation_summary_json": "data/bim/metrics/site01/deviation_summary.json",
            "coverage_summary_json": "data/bim/metrics/site01/coverage_summary.json",
            "registration_quality_json": "data/bim/metrics/site01/registration_quality.json",
        }},
    }
    summary = run_progress(cfg, force=True, log_level="ERROR")
    dashboard = root / "runs/test/reports/progress_dashboard.html"
    if not dashboard.exists():
        raise SystemExit("STAGE_09_SMOKE_FAILED missing dashboard")
    print(f"STAGE_09_SMOKE_OK elements={summary['element_count']} activities={summary['activity_count']} dashboard={dashboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
