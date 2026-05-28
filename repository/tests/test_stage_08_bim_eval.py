from __future__ import annotations

import pytest
from pathlib import Path

import numpy as np
o3d = pytest.importorskip("open3d")

from pipeline.stage_08_bim_eval.coarse_registration import coarse_register
from pipeline.stage_08_bim_eval.input_selection import scan_input_candidates, select_scan_input
from pipeline.stage_08_bim_eval.registration_quality import evaluate_registration_quality, nearest_neighbor_summary
from pipeline.stage_08_bim_eval.run_registration import run_registration


def _write_cloud(path: Path, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(0.0, 1.0, 20)
    ys = np.linspace(0.0, 1.0, 20)
    pts = []
    for x in xs:
        for y in ys:
            pts.append([x + offset[0], y + offset[1], offset[2]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
    assert o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _cfg(root: Path, cloud: Path) -> dict:
    return {
        "project": {"name": "site01", "run_id": "test", "root": str(root), "random_seed": 42},
        "inputs": {"video": "data/raw/site01.mp4", "ifc": "data/bim/design/model.ifc", "schedule": "data/bim/design/schedule.csv"},
        "paths": {
            "bim_aligned_dir": "data/bim/aligned/site01",
            "clean_dir": "data/clean/site01",
            "fused_ply": "data/dense/site01/fused.ply",
        },
        "video": {},
        "video_quality": {},
        "keyframes": {},
        "colmap": {},
        "refinement": {},
        "dense": {},
        "da3": {},
        "cleanup": {},
        "bim": {
            "scan_input_candidates": [str(cloud.relative_to(root))],
            "bim_reference_ply": "data/bim/aligned/site01/bim_reference.ply",
            "bim_reference_mesh": "data/bim/aligned/site01/bim_reference_mesh.ply",
            "scan_aligned_ply": "data/bim/aligned/site01/scan_aligned.ply",
            "transform_json": "data/bim/aligned/site01/transform_scan_to_bim.json",
            "element_metadata_jsonl": "data/bim/aligned/site01/bim_elements.jsonl",
            "registration_report_json": "runs/test/reports/registration_report.json",
            "allow_synthetic_ifc_fallback_for_tests": True,
            "estimate_initial_scale_from_bbox": False,
            "coarse_fpfh_enabled": False,
            "icp_enabled": True,
            "icp_estimation": "point_to_point",
            "icp_stages": ["point_to_point"],
            "icp_max_corr_distance_m": 0.3,
            "quality_min_icp_fitness": 0.01,
            "quality_max_icp_rmse_m": 0.5,
            "fail_on_low_registration_quality": False,
        },
        "progress": {},
    }


def test_scan_input_candidates_resolve_relative_paths(tmp_path: Path) -> None:
    cloud = tmp_path / "data" / "clean" / "site01" / "cleaned_cloud.ply"
    cfg = _cfg(tmp_path, cloud)
    candidates = scan_input_candidates(cfg)
    assert candidates[0] == cloud


def test_select_scan_input(tmp_path: Path) -> None:
    cloud = tmp_path / "data" / "clean" / "site01" / "cleaned_cloud.ply"
    _write_cloud(cloud)
    selected = select_scan_input(_cfg(tmp_path, cloud))
    assert selected.path == cloud
    assert selected.source == "stage_07_clean_geometry"


def test_bbox_coarse_registration_centers_clouds(tmp_path: Path) -> None:
    source_path = tmp_path / "source.ply"
    target_path = tmp_path / "target.ply"
    _write_cloud(source_path, offset=(0.0, 0.0, 0.0))
    _write_cloud(target_path, offset=(2.0, 3.0, 0.0))
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
    cfg = {"project": {"root": str(tmp_path)}, "bim": {"estimate_initial_scale_from_bbox": False, "coarse_fpfh_enabled": False}}
    import logging

    result = coarse_register(cfg, source, target, logging.getLogger("test"))
    assert result.method == "center_rigid"
    assert np.allclose(result.transformation[:3, 3], np.asarray([2.0, 3.0, 0.0]), atol=1e-6)


def test_nearest_neighbor_summary(tmp_path: Path) -> None:
    source_path = tmp_path / "source.ply"
    target_path = tmp_path / "target.ply"
    _write_cloud(source_path)
    _write_cloud(target_path)
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
    summary = nearest_neighbor_summary(source, target)
    assert summary["count"] > 0
    assert summary["mean_m"] is not None
    assert summary["mean_m"] < 1e-9


def test_registration_quality_warns_not_fails_by_default(tmp_path: Path) -> None:
    cfg = {"project": {"root": str(tmp_path)}, "bim": {"quality_min_icp_fitness": 0.5, "fail_on_low_registration_quality": False}}
    report = {"icp": {"fitness": 0.01, "inlier_rmse": 0.01}, "coarse_registration": {"scale_factor": 1.0}}
    quality = evaluate_registration_quality(cfg, report)
    assert quality["passed"] is True
    assert quality["warnings"]


def test_run_registration_smoke(tmp_path: Path) -> None:
    cloud = tmp_path / "data" / "clean" / "site01" / "cleaned_cloud.ply"
    _write_cloud(cloud)
    cfg = _cfg(tmp_path, cloud)
    report = run_registration(cfg, force=True, log_level="ERROR")
    assert report["status"] == "complete"
    assert Path(report["outputs"]["scan_aligned_ply"]).exists()
    assert Path(report["outputs"]["bim_reference_ply"]).exists()
    assert report["ifc"]["source"] == "synthetic_test_fallback"
    assert report["ifc"]["synthetic_bim_fallback_used"] is True
    assert report["ifc"]["real_progress_evidence_allowed"] is False


def test_legacy_bbox_scale_flag_is_ignored_by_default(tmp_path: Path) -> None:
    source_path = tmp_path / "source.ply"
    target_path = tmp_path / "target.ply"
    _write_cloud(source_path, offset=(0.0, 0.0, 0.0))
    # Same geometry translated, not scaled; bbox extents would be same anyway, but
    # the legacy flag must not change the safe method name or introduce warnings
    # that imply bbox scale was used.
    _write_cloud(target_path, offset=(4.0, 1.0, 0.0))
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
    cfg = {
        "project": {"root": str(tmp_path)},
        "bim": {
            "estimate_initial_scale_from_bbox": True,
            "initial_scale_strategy": "fixed_1",
            "coarse_fpfh_enabled": False,
        },
    }
    import logging

    result = coarse_register(cfg, source, target, logging.getLogger("test"))
    assert result.scale_factor == 1.0
    assert result.method == "center_rigid"



pytestmark = pytest.mark.geometry
