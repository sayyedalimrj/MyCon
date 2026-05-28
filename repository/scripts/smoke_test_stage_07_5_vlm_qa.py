from __future__ import annotations

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
    print("SMOKE_SKIP_MISSING_DEPENDENCY script=smoke_test_stage_07_5_vlm_qa.py dependency=open3d")
    raise SystemExit(0)

from pipeline.stage_07_5_vlm_qa.run_vlm_qa import run_vlm_qa


def _write_cloud(path: Path) -> None:
    xs, ys = np.meshgrid(np.linspace(0, 3, 45), np.linspace(0, 2, 35))
    floor = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    wall = np.column_stack([xs.ravel(), np.zeros(xs.size), np.ones(xs.size)])
    pts = np.vstack([floor, wall])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.5, 0.5, 0.5]], (len(pts), 1)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _write_mesh(path: Path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_box(width=3.0, height=2.0, depth=0.2)
    mesh.compute_vertex_normals()
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="stage075_smoke_"))
    clean = root / "data/clean/site01/cleaned_cloud.ply"
    mesh = root / "data/clean/site01/mesh.ply"
    planes = root / "data/clean/site01/planes.json"
    report = root / "runs/test/reports/cleanup_summary.json"

    _write_cloud(clean)
    _write_mesh(mesh)

    planes.parent.mkdir(parents=True, exist_ok=True)
    planes.write_text(json.dumps({
        "status": "ok",
        "plane_count": 2,
        "planes": [
            {"plane_id": "plane_001", "label": "floor_or_ceiling", "centroid": [1.5, 1.0, 0.0], "point_count": 1200},
            {"plane_id": "plane_002", "label": "wall", "centroid": [1.5, 0.0, 1.0], "point_count": 900},
        ],
    }), encoding="utf-8")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "stage": "stage_07_cleanup",
        "status": "ok",
        "cleanup": {"cleaned_count": 3150},
        "planes": {"count": 2, "records": []},
        "mesh": {"status": "ok", "vertex_count": 8, "triangle_count": 12},
        "quality_gate": {"passed": True, "failures": [], "warnings": []},
    }), encoding="utf-8")

    cfg = {
        "project": {"root": str(root), "name": "site01", "run_id": "test"},
        "paths": {
            "clean_cloud": "data/clean/site01/cleaned_cloud.ply",
            "clean_mesh": "data/clean/site01/mesh.ply",
            "clean_planes_json": "data/clean/site01/planes.json",
            "cleanup_report_json": "runs/test/reports/cleanup_summary.json",
        },
        "vlm_qa": {
            "output_dir": "data/vlm_qa/site01",
            "render_dir": "data/vlm_qa/site01/renders",
            "evidence_json": "data/vlm_qa/site01/vlm_qa_evidence.json",
            "summary_json": "runs/test/reports/vlm_qa_summary.json",
            "quality_min_points": 1000,
            "quality_min_planes": 1,
            "max_render_points": 5000,
        },
    }

    summary = run_vlm_qa(cfg, force=True, log_level="ERROR")
    evidence = root / "data/vlm_qa/site01/vlm_qa_evidence.json"
    if not evidence.exists():
        raise SystemExit("STAGE_07_5_SMOKE_FAILED missing evidence")
    for path in summary["render_paths"].values():
        if not Path(path).exists():
            raise SystemExit(f"STAGE_07_5_SMOKE_FAILED missing render: {path}")

    print(f"STAGE_07_5_SMOKE_OK status={summary['status']} confidence={summary['confidence']} evidence={evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
