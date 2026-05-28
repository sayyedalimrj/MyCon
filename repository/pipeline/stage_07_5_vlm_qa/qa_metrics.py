from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config_access import cfg_bool, cfg_int
from .io_utils import read_json


def _o3d():
    import open3d as o3d
    return o3d


def _point_cloud_stats(path: Path) -> dict[str, Any]:
    o3d = _o3d()
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return {
            "path": path.as_posix(),
            "status": "empty",
            "point_count": 0,
            "finite_ratio": 0.0,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_extent": None,
            "has_colors": bool(pcd.has_colors()),
            "has_normals": bool(pcd.has_normals()),
        }

    finite_mask = np.isfinite(pts).all(axis=1)
    bbox = pcd.get_axis_aligned_bounding_box()
    return {
        "path": path.as_posix(),
        "status": "ok",
        "point_count": int(len(pts)),
        "finite_count": int(finite_mask.sum()),
        "finite_ratio": float(finite_mask.mean()),
        "bbox_min": bbox.get_min_bound().tolist(),
        "bbox_max": bbox.get_max_bound().tolist(),
        "bbox_extent": bbox.get_extent().tolist(),
        "has_colors": bool(pcd.has_colors()),
        "has_normals": bool(pcd.has_normals()),
    }


def _mesh_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return {"status": "missing", "path": path.as_posix() if path else None}

    o3d = _o3d()
    mesh = o3d.io.read_triangle_mesh(str(path))
    return {
        "status": "ok" if len(mesh.vertices) > 0 else "empty",
        "path": path.as_posix(),
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": int(len(mesh.triangles)),
        "has_vertex_normals": bool(mesh.has_vertex_normals()),
        "has_triangle_normals": bool(mesh.has_triangle_normals()),
    }


def _plane_stats(path: Path | None, cleanup_report: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}

    if path is not None and path.exists() and path.stat().st_size > 0:
        payload = read_json(path)
        raw = payload.get("planes") or payload.get("records") or []
        if isinstance(raw, list):
            records = [r for r in raw if isinstance(r, dict)]

    if not records:
        raw = cleanup_report.get("planes", {}).get("records", [])
        if isinstance(raw, list):
            records = [r for r in raw if isinstance(r, dict)]

    labels = sorted(set(str(r.get("label", "unknown")) for r in records))
    point_counts = [int(r.get("point_count", 0) or 0) for r in records]
    return {
        "status": "ok" if records else "missing_or_empty",
        "path": path.as_posix() if path else None,
        "plane_count": int(len(records)),
        "labels": labels,
        "max_plane_points": max(point_counts) if point_counts else 0,
        "records_preview": records[:8],
    }


def collect_stage75_metrics(paths: dict[str, Path]) -> dict[str, Any]:
    cleanup_report = read_json(paths["cleanup_report"])
    return {
        "cleaned_cloud": _point_cloud_stats(paths["cleaned_cloud"]),
        "mesh": _mesh_stats(paths.get("mesh")),
        "planes": _plane_stats(paths.get("planes_json"), cleanup_report),
        "cleanup_report": {
            "path": paths["cleanup_report"].as_posix(),
            "status": cleanup_report.get("status"),
            "cleanup": cleanup_report.get("cleanup", {}),
            "quality_gate": cleanup_report.get("quality_gate", {}),
            "mesh": cleanup_report.get("mesh", {}),
            "planes": {
                "count": cleanup_report.get("planes", {}).get("count"),
                "planes_json": cleanup_report.get("planes", {}).get("planes_json"),
            },
        },
    }


def evaluate_stage75_quality(cfg: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    min_points = cfg_int(cfg, "vlm_qa.quality_min_points", 10000)
    min_planes = cfg_int(cfg, "vlm_qa.quality_min_planes", 1)
    require_mesh = cfg_bool(cfg, "vlm_qa.require_mesh", False)

    failures: list[str] = []
    warnings: list[str] = []

    cloud = metrics.get("cleaned_cloud", {})
    mesh = metrics.get("mesh", {})
    planes = metrics.get("planes", {})
    cleanup_qg = metrics.get("cleanup_report", {}).get("quality_gate", {})

    point_count = int(cloud.get("point_count", 0) or 0)
    finite_ratio = float(cloud.get("finite_ratio", 0.0) or 0.0)
    plane_count = int(planes.get("plane_count", 0) or 0)

    if point_count < min_points:
        failures.append(f"cleaned_point_count_below_min:{point_count}<{min_points}")
    if finite_ratio < 0.999:
        failures.append(f"finite_ratio_below_min:{finite_ratio:.6f}<0.999000")
    if plane_count < min_planes:
        warnings.append(f"plane_count_below_min:{plane_count}<{min_planes}")

    if cleanup_qg and cleanup_qg.get("passed") is False:
        warnings.append("upstream_cleanup_quality_gate_not_passed")

    if require_mesh and mesh.get("status") != "ok":
        failures.append("mesh_required_but_not_ok")
    elif mesh.get("status") != "ok":
        warnings.append(f"mesh_status:{mesh.get('status')}")

    if failures:
        status = "failed"
        confidence = "low"
    elif warnings:
        status = "warning"
        confidence = "medium"
    else:
        status = "ok"
        confidence = "high"

    return {
        "status": status,
        "passed": not failures,
        "confidence": confidence,
        "failures": failures,
        "warnings": warnings,
        "thresholds": {
            "quality_min_points": min_points,
            "quality_min_planes": min_planes,
            "require_mesh": require_mesh,
        },
    }
