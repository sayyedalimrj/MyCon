from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pipeline.stage_04_refinement.bundle_adjustment import build_bundle_adjuster_args
from pipeline.stage_04_refinement.model_io import copy_sparse_model, has_sparse_model, resolve_sparse_component_dir
from pipeline.stage_04_refinement.refinement_stats import (
    compute_refinement_delta,
    evaluate_quality_gate,
)


class DummyConfig:
    def __init__(self, root: Path) -> None:
        self.data = {
            "project": {"name": "site01", "run_id": "test_run", "root": str(root)},
            "paths": {"sparse_dir": "data/sparse/site01", "sparse_refined_dir": "data/sparse_refined/site01"},
            "refinement": {
                "ba_max_num_iterations": 7,
                "ba_max_linear_solver_iterations": 11,
                "ba_function_tolerance": 1e-6,
                "ba_gradient_tolerance": 1e-10,
                "ba_parameter_tolerance": 1e-8,
                "ba_num_threads": -1,
                "refine_focal_length": True,
                "refine_principal_point": False,
                "refine_extra_params": True,
                "quality_gate_min_registered_images": 2,
                "quality_gate_min_points": 1,
                "quality_gate_max_point_loss_ratio": 0.40,
                "quality_gate_fail_on_point_loss": True,
                "quality_gate_max_reprojection_error_increase_ratio": 0.10,
                "quality_gate_max_reprojection_error_increase_abs_px": 0.25,
                "quality_gate_fail_on_reprojection_error_increase": True,
            },
        }

    def get(self, dotted: str, default=None):
        current = self.data
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def require(self, dotted: str):
        value = self.get(dotted, None)
        if value is None:
            raise KeyError(dotted)
        return value


def _write_sparse(path: Path, points_multiplier: int = 1) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "cameras.bin").write_bytes(b"camera")
    (path / "images.bin").write_bytes(b"image" * 8)
    (path / "points3D.bin").write_bytes(b"points" * points_multiplier)


def test_resolve_sparse_component_accepts_component_dir(tmp_path: Path) -> None:
    component = tmp_path / "data" / "sparse" / "site01" / "0"
    _write_sparse(component)
    assert resolve_sparse_component_dir(component) == component


def test_resolve_sparse_component_selects_best_child(tmp_path: Path) -> None:
    parent = tmp_path / "sparse"
    _write_sparse(parent / "0", points_multiplier=1)
    _write_sparse(parent / "1", points_multiplier=20)
    assert resolve_sparse_component_dir(parent).name == "1"


def test_copy_sparse_model_writes_non_symlink_contract(tmp_path: Path) -> None:
    src = tmp_path / "src" / "0"
    dst = tmp_path / "dst" / "0"
    _write_sparse(src)
    result = copy_sparse_model(src, dst, force=True)
    assert result == dst
    assert has_sparse_model(dst)
    assert not (dst / "cameras.bin").is_symlink()


def test_bundle_adjuster_args_filter_unsupported_options(tmp_path: Path) -> None:
    cfg = DummyConfig(tmp_path)
    supported = {"--input_path", "--output_path", "--BundleAdjustmentCeres.max_num_iterations"}
    args = build_bundle_adjuster_args(cfg, tmp_path / "in", tmp_path / "out", supported_options=supported)
    assert "bundle_adjuster" in args
    assert "--BundleAdjustmentCeres.max_num_iterations" in args
    assert "7" in args
    assert "--BundleAdjustment.refine_focal_length" not in args


def test_bundle_adjuster_args_skip_optional_when_help_parse_fails(tmp_path: Path) -> None:
    cfg = DummyConfig(tmp_path)
    args = build_bundle_adjuster_args(
        cfg,
        tmp_path / "in",
        tmp_path / "out",
        supported_options=set(),
        logger=logging.getLogger("test"),
    )
    assert args == ["bundle_adjuster", "--input_path", str(tmp_path / "in"), "--output_path", str(tmp_path / "out")]


def test_compute_refinement_delta_and_quality_gate(tmp_path: Path) -> None:
    cfg = DummyConfig(tmp_path)
    before = {
        "registered_image_count": 4,
        "sparse_point_count": 100,
        "registered_ratio": 0.5,
        "mean_reprojection_error_px": 0.8,
    }
    after = {
        "registered_image_count": 4,
        "sparse_point_count": 96,
        "registered_ratio": 0.5,
        "mean_reprojection_error_px": 0.7,
    }
    delta = compute_refinement_delta(before, after)
    assert delta["registered_image_delta"] == 0
    assert delta["sparse_point_delta"] == -4
    assert delta["point_loss_ratio"] == pytest.approx(0.04)
    gate = evaluate_quality_gate(cfg, before, after)
    assert gate["passed"] is True


def test_quality_gate_allows_moderate_point_loss_when_reprojection_improves(tmp_path: Path) -> None:
    cfg = DummyConfig(tmp_path)
    before = {"registered_image_count": 4, "sparse_point_count": 100, "mean_reprojection_error_px": 1.0}
    after = {"registered_image_count": 4, "sparse_point_count": 80, "mean_reprojection_error_px": 0.8}
    gate = evaluate_quality_gate(cfg, before, after)
    assert gate["passed"] is True
    assert not gate["failures"]


def test_quality_gate_flags_extreme_point_loss(tmp_path: Path) -> None:
    cfg = DummyConfig(tmp_path)
    before = {"registered_image_count": 4, "sparse_point_count": 100, "mean_reprojection_error_px": 1.0}
    after = {"registered_image_count": 4, "sparse_point_count": 50, "mean_reprojection_error_px": 0.8}
    gate = evaluate_quality_gate(cfg, before, after)
    assert gate["passed"] is False
    assert any("point loss" in item for item in gate["failures"])


def test_quality_gate_flags_reprojection_error_increase(tmp_path: Path) -> None:
    cfg = DummyConfig(tmp_path)
    before = {"registered_image_count": 4, "sparse_point_count": 100, "mean_reprojection_error_px": 0.5}
    after = {"registered_image_count": 4, "sparse_point_count": 100, "mean_reprojection_error_px": 1.0}
    gate = evaluate_quality_gate(cfg, before, after)
    assert gate["passed"] is False
    assert any("reprojection error" in item for item in gate["failures"])
