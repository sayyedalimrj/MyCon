from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.stage_06_da3_assist.depth_alignment import fit_affine_depth, fit_scale_depth_ransac
from pipeline.stage_06_da3_assist.depth_fusion import _depth_edge_mask, _parse_bbox
from pipeline.stage_06_da3_assist.dense_assessment import assess_dense_coverage
from pipeline.stage_06_da3_assist.io_utils import clean_dir_guarded


class DummyCfg(dict):
    def get(self, key, default=None):
        cur = self
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def test_fit_scale_depth_ransac_recovers_scale_without_shift() -> None:
    raw = np.linspace(1, 10, 80)
    target = 2.5 * raw
    scale, shift, rmse, used, inlier_ratio = fit_scale_depth_ransac(raw, target, iterations=50)
    assert abs(scale - 2.5) < 1e-6
    assert shift == 0.0
    assert rmse < 1e-6
    assert used == 80
    assert inlier_ratio > 0.95


def test_fit_scale_depth_ransac_rejects_outliers() -> None:
    rng = np.random.default_rng(7)
    raw = np.linspace(1, 15, 120)
    target = 1.8 * raw
    target += rng.normal(0, 0.01, size=target.shape)
    target[::9] += 8.0
    scale, shift, rmse, used, inlier_ratio = fit_scale_depth_ransac(
        raw,
        target,
        iterations=150,
        inlier_abs_threshold_m=0.08,
        inlier_rel_threshold=0.03,
        random_seed=4,
    )
    assert abs(scale - 1.8) < 0.02
    assert shift == 0.0
    assert rmse < 0.05
    assert used == 120
    assert 0.75 <= inlier_ratio < 1.0


def test_fit_affine_depth_wrapper_is_scale_only() -> None:
    raw = np.linspace(1, 10, 50)
    target = 2.0 * raw
    scale, shift, rmse, used = fit_affine_depth(raw, target)
    assert abs(scale - 2.0) < 1e-6
    assert shift == 0.0
    assert rmse < 1e-6
    assert used == 50


def test_depth_edge_mask_detects_flying_pixel_boundary() -> None:
    depth = np.ones((20, 20), dtype=np.float32)
    depth[:, 10:] = 5.0
    mask = _depth_edge_mask(depth, threshold_m=0.5, relative_threshold=0.05)
    assert mask[:, 9:12].any()
    assert not mask[:, :4].any()


def test_parse_bbox_accepts_list_and_dict() -> None:
    a = _parse_bbox([0, 0, 0, 1, 2, 3])
    assert a is not None
    assert np.allclose(a[0], [0, 0, 0])
    assert np.allclose(a[1], [1, 2, 3])
    b = _parse_bbox({"min": [1, 1, 1], "max": [3, 2, 1]})
    assert b is not None
    assert np.allclose(b[0], [1, 1, 1])
    assert np.allclose(b[1], [3, 2, 1])


def test_dense_assessment_skips_when_dense_good(tmp_path: Path) -> None:
    p = tmp_path / "dense_summary.json"
    p.write_text(
        '{"dense_stats":{"fused_vertex_count":1000000,"points_per_input_image":5000,"depth_map_ratio":1.0},"quality_gate":{"passed":true}}',
        encoding="utf-8",
    )
    cfg = DummyCfg({"da3": {"enabled": "auto", "activate_if_fused_vertices_below": 50000, "activate_if_points_per_image_below": 100, "activate_if_depth_map_ratio_below": 0.5, "activate_if_quality_gate_failed": True}})
    result = assess_dense_coverage(cfg, p)
    assert result.should_activate is False
    assert result.status == "dense_coverage_sufficient"


def test_dense_assessment_activates_when_dense_weak(tmp_path: Path) -> None:
    p = tmp_path / "dense_summary.json"
    p.write_text(
        '{"dense_stats":{"fused_vertex_count":10,"points_per_input_image":1,"depth_map_ratio":0.1},"quality_gate":{"passed":false}}',
        encoding="utf-8",
    )
    cfg = DummyCfg({"da3": {"enabled": "auto", "activate_if_fused_vertices_below": 50000, "activate_if_points_per_image_below": 100, "activate_if_depth_map_ratio_below": 0.5, "activate_if_quality_gate_failed": True}})
    result = assess_dense_coverage(cfg, p)
    assert result.should_activate is True
    assert result.status == "activated_by_dense_coverage"


def test_clean_dir_guard_rejects_non_da3_path(tmp_path: Path) -> None:
    bad = tmp_path / "data" / "sfm" / "site01"
    bad.mkdir(parents=True)
    import logging

    try:
        clean_dir_guarded(bad, force=True, required_token="da3", logger=logging.getLogger("test"))
    except ValueError as exc:
        assert "Refusing to clean unsafe path" in str(exc)
    else:
        raise AssertionError("Expected unsafe path rejection")
