from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricInitialTransform:
    matrix4x4: np.ndarray
    method: str
    scale: float
    confidence: str
    status: str
    report_path: Path
    warnings: list[str]


def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    data = getattr(cfg, "data", cfg)
    if isinstance(data, dict):
        cur: Any = data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    if hasattr(cfg, "get"):
        try:
            return cfg.get(dotted, default)
        except TypeError:
            return default

    return default


def _project_root(cfg: Any) -> Path:
    raw = _cfg_get(cfg, "project.root", ".")
    return Path(str(raw)).expanduser().resolve()


def _resolve_path(cfg: Any, dotted: str, default: str) -> Path:
    raw = _cfg_get(cfg, dotted, default)
    p = Path(str(raw))
    return p if p.is_absolute() else _project_root(cfg) / p


def _matrix_from_report(report: dict[str, Any]) -> np.ndarray:
    transform = report.get("transform") or {}

    if transform.get("matrix4x4") is not None:
        mat = np.asarray(transform["matrix4x4"], dtype=float)
        if mat.shape != (4, 4):
            raise ValueError(f"Invalid metric alignment matrix shape: {mat.shape}")
        return mat

    scale = float(transform.get("scale", 1.0))
    rotation = np.asarray(transform.get("rotation", np.eye(3)), dtype=float)
    translation = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=float)

    if rotation.shape != (3, 3):
        raise ValueError(f"Invalid metric alignment rotation shape: {rotation.shape}")
    if translation.shape != (3,):
        raise ValueError(f"Invalid metric alignment translation shape: {translation.shape}")

    mat = np.eye(4, dtype=float)
    mat[:3, :3] = scale * rotation
    mat[:3, 3] = translation
    return mat


def load_metric_initial_transform(
    cfg: Any,
    logger: logging.Logger | None = None,
) -> MetricInitialTransform | None:
    enabled = bool(_cfg_get(cfg, "metric_alignment.stage8_prefer_metric_alignment", True))
    if not enabled:
        return None

    report_path = _resolve_path(
        cfg,
        "metric_alignment.report_json",
        "runs/2026-04-30_site01_baseline/reports/metric_alignment_report.json",
    )

    if not report_path.exists():
        if logger:
            logger.info("Stage 8 metric alignment report not found; using normal coarse registration.")
        return None

    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = str(report.get("status", "unknown"))
    confidence = str(report.get("confidence", "low"))
    can_feed = bool(report.get("can_feed_stage8", False))
    quality_gate = report.get("quality_gate") or {}
    qg_passed = bool(quality_gate.get("passed", False))

    usable_status = status in {"ok", "alignment_warning"}
    if not usable_status or not can_feed or not qg_passed:
        if logger:
            logger.info(
                "Stage 8 metric alignment report is not usable as initial transform: "
                "status=%s confidence=%s can_feed=%s qg_passed=%s",
                status,
                confidence,
                can_feed,
                qg_passed,
            )
        return None

    mat = _matrix_from_report(report)
    transform = report.get("transform") or {}
    scale = float(transform.get("scale", np.cbrt(abs(np.linalg.det(mat[:3, :3])))))

    warnings = list(quality_gate.get("warnings") or [])
    warnings.append(f"metric_alignment_initial_transform_used:{report_path}")

    return MetricInitialTransform(
        matrix4x4=mat,
        method=f"metric_alignment_sim3:{status}",
        scale=scale,
        confidence=confidence,
        status=status,
        report_path=report_path,
        warnings=warnings,
    )
