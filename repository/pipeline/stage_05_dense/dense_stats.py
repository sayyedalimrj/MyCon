"""Dense workspace statistics and quality gates."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config_access import cfg_float, cfg_int
from .io_utils import dir_size_bytes


def parse_ply_vertex_count(path: Path) -> int | None:
    """Parse the vertex count from a PLY header without loading the point cloud."""
    if not path.exists() or path.stat().st_size <= 0:
        return None
    with path.open("rb") as handle:
        header = bytearray()
        limit = 16384
        while len(header) < limit:
            line = handle.readline()
            if not line:
                break
            header.extend(line)
            if line.strip() == b"end_header":
                break
    text = header.decode("utf-8", errors="ignore")
    match = re.search(r"^element\s+vertex\s+(\d+)\s*$", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def build_dense_stats(workspace: Path, fused_ply: Path, input_image_count: int) -> dict[str, Any]:
    stereo_dir = workspace / "stereo"
    depth_dir = stereo_dir / "depth_maps"
    normal_dir = stereo_dir / "normal_maps"
    consistency_dir = stereo_dir / "consistency_graphs"
    undistorted_images = workspace / "images"
    sparse_dir = workspace / "sparse"
    vertex_count = parse_ply_vertex_count(fused_ply)
    depth_maps = count_files(depth_dir, "*.bin")
    normal_maps = count_files(normal_dir, "*.bin")
    consistency_graphs = count_files(consistency_dir, "*.bin")
    undistorted_count = count_files(undistorted_images, "*")
    points_per_input_image = _safe_ratio(int(vertex_count or 0), input_image_count)
    return {
        "workspace": str(workspace),
        "fused_ply": str(fused_ply),
        "workspace_size_bytes": dir_size_bytes(workspace),
        "input_image_count": input_image_count,
        "undistorted_image_count": undistorted_count,
        "depth_map_count": depth_maps,
        "normal_map_count": normal_maps,
        "consistency_graph_count": consistency_graphs,
        "has_patch_match_cfg": (stereo_dir / "patch-match.cfg").exists(),
        "has_fusion_cfg": (stereo_dir / "fusion.cfg").exists(),
        "has_dense_sparse_model": (sparse_dir / "cameras.txt").exists() or (sparse_dir / "cameras.bin").exists(),
        "fused_ply_exists": fused_ply.exists(),
        "fused_ply_size_bytes": fused_ply.stat().st_size if fused_ply.exists() else 0,
        "fused_vertex_count": vertex_count,
        "points_per_input_image": points_per_input_image,
        "depth_map_ratio": _safe_ratio(depth_maps, input_image_count),
    }


def evaluate_quality_gate(cfg: Any, stats: dict[str, Any]) -> dict[str, Any]:
    min_points = cfg_int(cfg, "dense.quality_min_fused_points", 1000)
    min_points_per_image = cfg_float(cfg, "dense.quality_min_fused_points_per_image", 20.0)
    depth_ratio_warning = cfg_float(cfg, "dense.quality_min_depth_map_ratio_warning", 0.25)
    failures: list[str] = []
    warnings: list[str] = []

    input_image_count = int(stats.get("input_image_count") or 0)
    if input_image_count <= 0:
        failures.append("input_image_count_zero")

    vertex_count = stats.get("fused_vertex_count")
    if vertex_count is None:
        failures.append("fused_vertex_count_unavailable")
    elif int(vertex_count) < min_points:
        failures.append(f"fused_vertex_count_below_min:{vertex_count}<{min_points}")

    points_per_image = stats.get("points_per_input_image")
    if points_per_image is None:
        failures.append("points_per_input_image_unavailable")
    elif float(points_per_image) < min_points_per_image:
        failures.append(f"points_per_input_image_below_min:{float(points_per_image):.3f}<{min_points_per_image:.3f}")

    depth_ratio = stats.get("depth_map_ratio")
    if depth_ratio is None:
        warnings.append("depth_map_ratio_unavailable")
    elif float(depth_ratio) < depth_ratio_warning:
        warnings.append(f"depth_map_ratio_low:{float(depth_ratio):.3f}<{depth_ratio_warning:.3f}")

    if not stats.get("fused_ply_exists"):
        failures.append("fused_ply_missing")
    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "thresholds": {
            "quality_min_fused_points": min_points,
            "quality_min_fused_points_per_image": min_points_per_image,
            "quality_min_depth_map_ratio_warning": depth_ratio_warning,
        },
    }
