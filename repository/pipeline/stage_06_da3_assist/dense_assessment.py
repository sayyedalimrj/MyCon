from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_access import bool_value, cfg_get


@dataclass(frozen=True)
class DenseAssessment:
    should_activate: bool
    status: str
    reasons: list[str]
    dense_stats: dict[str, Any]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_dense_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "missing": True,
            "dense_stats": {},
            "quality_gate": {"passed": False, "warnings": ["dense summary is missing"]},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def assess_dense_coverage(cfg: Any, dense_summary_path: Path) -> DenseAssessment:
    summary = load_dense_summary(dense_summary_path)
    stats = dict(summary.get("dense_stats", {}))
    quality_gate = summary.get("quality_gate", {}) if isinstance(summary.get("quality_gate", {}), dict) else {}

    enabled = str(cfg_get(cfg, "da3.enabled", "auto")).strip().lower()
    if enabled in {"false", "off", "never"}:
        return DenseAssessment(False, "disabled_by_config", ["da3.enabled is disabled"], stats)
    if enabled in {"true", "on", "always"}:
        return DenseAssessment(True, "forced_on", ["da3.enabled forces assistance"], stats)

    reasons: list[str] = []
    if summary.get("missing"):
        reasons.append("dense_summary_missing")

    if bool_value(cfg_get(cfg, "da3.activate_if_quality_gate_failed", True)) and quality_gate.get("passed") is False:
        reasons.append("stage5_quality_gate_failed")

    vertices = _num(stats.get("fused_vertex_count"), 0.0)
    min_vertices = _num(cfg_get(cfg, "da3.activate_if_fused_vertices_below", 50000), 0.0)
    if vertices < min_vertices:
        reasons.append(f"fused_vertices_below_threshold:{vertices:.0f}<{min_vertices:.0f}")

    ppi = _num(stats.get("points_per_input_image"), 0.0)
    min_ppi = _num(cfg_get(cfg, "da3.activate_if_points_per_image_below", 100.0), 0.0)
    if ppi < min_ppi:
        reasons.append(f"points_per_image_below_threshold:{ppi:.3f}<{min_ppi:.3f}")

    depth_ratio = _num(stats.get("depth_map_ratio"), 0.0)
    min_depth_ratio = _num(cfg_get(cfg, "da3.activate_if_depth_map_ratio_below", 0.5), 0.0)
    if depth_ratio < min_depth_ratio:
        reasons.append(f"depth_map_ratio_below_threshold:{depth_ratio:.3f}<{min_depth_ratio:.3f}")

    if reasons:
        return DenseAssessment(True, "activated_by_dense_coverage", reasons, stats)
    return DenseAssessment(False, "dense_coverage_sufficient", ["dense output meets DA3 activation thresholds"], stats)
