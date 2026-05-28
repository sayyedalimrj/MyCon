from __future__ import annotations

import pytest
import json
from pathlib import Path

import numpy as np
o3d = pytest.importorskip("open3d")

from pipeline.stage_07_5_vlm_qa.config_access import cfg_get
from pipeline.stage_07_5_vlm_qa.qa_metrics import evaluate_stage75_quality
from pipeline.stage_07_5_vlm_qa.run_vlm_qa import run_vlm_qa


def _write_cloud(path: Path, n: int = 1200) -> None:
    pts = np.column_stack([
        np.linspace(0, 1, n),
        np.sin(np.linspace(0, 3.14, n)),
        np.zeros(n),
    ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.4, 0.4, 0.4]], (len(pts), 1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _write_mesh(path: Path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)


def _cfg(root: Path, n: int = 1200) -> dict:
    _write_cloud(root / "data/clean/site01/cleaned_cloud.ply", n=n)
    _write_mesh(root / "data/clean/site01/mesh.ply")
    planes = root / "data/clean/site01/planes.json"
    planes.write_text(json.dumps({
        "status": "ok",
        "planes": [{"plane_id": "p1", "label": "floor_or_ceiling", "centroid": [0.5, 0.5, 0.0], "point_count": n}],
    }), encoding="utf-8")
    report = root / "runs/test/reports/cleanup_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "status": "ok",
        "cleanup": {"cleaned_count": n},
        "mesh": {"status": "ok"},
        "planes": {"count": 1, "records": []},
        "quality_gate": {"passed": True, "failures": [], "warnings": []},
    }), encoding="utf-8")
    return {
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
            "max_render_points": 3000,
        },
    }


def test_cfg_get_reads_nested_dict() -> None:
    assert cfg_get({"project": {"root": "/tmp/x"}}, "project.root") == "/tmp/x"


def test_quality_gate_fails_low_points() -> None:
    metrics = {
        "cleaned_cloud": {"point_count": 10, "finite_ratio": 1.0},
        "mesh": {"status": "ok"},
        "planes": {"plane_count": 1},
        "cleanup_report": {"quality_gate": {"passed": True}},
    }
    qg = evaluate_stage75_quality({"vlm_qa": {"quality_min_points": 1000}}, metrics)
    assert qg["passed"] is False
    assert qg["confidence"] == "low"


def test_run_vlm_qa_writes_evidence_and_renders(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, n=1500)
    summary = run_vlm_qa(cfg, force=True, log_level="ERROR")
    assert summary["status"] == "ok"
    evidence = tmp_path / "data/vlm_qa/site01/vlm_qa_evidence.json"
    assert evidence.exists()
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["stage"] == "stage_07_5_vlm_qa"
    assert payload["metrics"]["cleaned_cloud"]["point_count"] == 1500
    for path in summary["render_paths"].values():
        assert Path(path).exists()

pytestmark = pytest.mark.geometry
