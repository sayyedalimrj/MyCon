from __future__ import annotations

import pytest
import json
import logging
from pathlib import Path

import numpy as np
o3d = pytest.importorskip("open3d")

from pipeline.stage_08_bim_eval.coarse_registration import coarse_register
from pipeline.stage_08_bim_eval.metric_initial_transform import load_metric_initial_transform


def _write_metric_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "stage_08_metric_alignment",
        "status": "ok",
        "confidence": "high",
        "can_feed_stage8": True,
        "transform": {
            "source": "scan_to_bim",
            "scale": 2.0,
            "rotation": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "translation": [1.0, -3.0, 0.5],
            "matrix4x4": [
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, -3.0],
                [0.0, 0.0, 2.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "quality_gate": {
            "passed": True,
            "warnings": [],
            "failures": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _cloud() -> o3d.geometry.PointCloud:
    pts = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc


def test_load_metric_initial_transform_reads_usable_report(tmp_path: Path) -> None:
    report = tmp_path / "metric_alignment_report.json"
    _write_metric_report(report)

    cfg = {
        "project": {"root": str(tmp_path)},
        "metric_alignment": {
            "stage8_prefer_metric_alignment": True,
            "report_json": "metric_alignment_report.json",
        },
    }

    loaded = load_metric_initial_transform(cfg)
    assert loaded is not None
    assert loaded.method == "metric_alignment_sim3:ok"
    assert abs(loaded.scale - 2.0) < 1.0e-12
    assert loaded.matrix4x4.shape == (4, 4)
    assert loaded.matrix4x4[0, 0] == 2.0
    assert loaded.matrix4x4[0, 3] == 1.0


def test_load_metric_initial_transform_ignores_bad_report(tmp_path: Path) -> None:
    report = tmp_path / "metric_alignment_report.json"
    report.write_text(
        json.dumps({
            "status": "skipped_insufficient_anchors",
            "confidence": "low",
            "can_feed_stage8": False,
            "quality_gate": {"passed": False},
            "transform": None,
        }),
        encoding="utf-8",
    )

    cfg = {
        "project": {"root": str(tmp_path)},
        "metric_alignment": {
            "stage8_prefer_metric_alignment": True,
            "report_json": "metric_alignment_report.json",
        },
    }

    assert load_metric_initial_transform(cfg) is None


def test_coarse_register_prefers_metric_alignment_when_available(tmp_path: Path) -> None:
    report = tmp_path / "metric_alignment_report.json"
    _write_metric_report(report)

    cfg = {
        "project": {"root": str(tmp_path)},
        "metric_alignment": {
            "stage8_prefer_metric_alignment": True,
            "report_json": "metric_alignment_report.json",
        },
        "bim": {
            "estimate_initial_scale_from_bbox": False,
            "initial_scale_strategy": "fixed_1",
            "coarse_fpfh_enabled": False,
        },
    }

    result = coarse_register(cfg, _cloud(), _cloud(), logging.getLogger("test"))

    assert result.method == "metric_alignment_sim3:ok"
    assert abs(result.scale_factor - 2.0) < 1.0e-12
    assert result.transformation.shape == (4, 4)
    assert result.transformation[0, 0] == 2.0
    assert any("metric_alignment_initial_transform_used" in item for item in result.warnings)

pytestmark = pytest.mark.geometry
