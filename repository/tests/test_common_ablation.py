"""Tests for ``pipeline.common.ablation``.

The harness is pure: no Open3D / COLMAP. Tests pin the contract that the
ablation runner relies on.
"""

from __future__ import annotations

import pytest

from pipeline.common.ablation import (
    AblationCell,
    AblationGrid,
    apply_overlay,
    build_grid,
    grid_summary,
)


def test_build_grid_cartesian_size() -> None:
    grid = build_grid("g", {
        "a.x": [1, 2, 3],
        "b.y": ["p", "q"],
        "c.z": [True, False],
    })
    assert isinstance(grid, AblationGrid)
    assert len(grid.cells) == 3 * 2 * 2


def test_build_grid_cell_names_unique_and_filesystem_safe() -> None:
    grid = build_grid("g", {
        "bim.icp_robust_loss": ["none", "huber", "tukey"],
        "bim.icp_max_corr_distance_m": [0.05, 0.08, 0.12],
        "project.random_seed": [42, 43, 44],
    })
    names = [c.name for c in grid.cells]
    # 27 distinct cells, all names usable as directory components.
    assert len(set(names)) == len(names) == 27
    for n in names:
        assert "/" not in n and " " not in n


def test_build_grid_is_deterministic() -> None:
    axes = {"a.x": [1, 2], "b.y": ["p", "q"]}
    g1 = build_grid("g", axes)
    g2 = build_grid("g", axes)
    assert [c.name for c in g1.cells] == [c.name for c in g2.cells]
    assert [c.overlay for c in g1.cells] == [c.overlay for c in g2.cells]


def test_apply_overlay_does_not_mutate_base() -> None:
    base = {"bim": {"icp_max_corr_distance_m": 0.08}, "project": {"random_seed": 42}}
    overlay = {"bim.icp_robust_loss": "tukey", "project.random_seed": 43}
    out = apply_overlay(base, overlay)

    assert out["bim"]["icp_robust_loss"] == "tukey"
    assert out["bim"]["icp_max_corr_distance_m"] == 0.08
    assert out["project"]["random_seed"] == 43
    # base must be untouched
    assert "icp_robust_loss" not in base["bim"]
    assert base["project"]["random_seed"] == 42


def test_apply_overlay_creates_intermediate_dicts() -> None:
    out = apply_overlay({}, {"a.b.c": 1})
    assert out == {"a": {"b": {"c": 1}}}


def test_apply_overlay_rejects_non_dict_traversal() -> None:
    base = {"a": 7}  # 'a' is a scalar; traversal should fail loudly.
    with pytest.raises(ValueError):
        apply_overlay(base, {"a.b": 1})


def test_build_grid_rejects_non_dotted_axis_key() -> None:
    with pytest.raises(ValueError):
        build_grid("g", {"flat_key_no_dot": [1, 2]})


def test_build_grid_rejects_empty_axis() -> None:
    with pytest.raises(ValueError):
        build_grid("g", {"a.x": []})


def test_build_grid_rejects_non_list_axis_values() -> None:
    with pytest.raises(TypeError):
        build_grid("g", {"a.x": "not a list"})


def test_grid_summary_round_trips_axes() -> None:
    axes = {"a.x": [1, 2], "b.y": ["p", "q"]}
    grid = build_grid("g", axes)
    summary = grid_summary(grid)
    assert summary["name"] == "g"
    assert summary["axes"] == axes
    assert summary["cell_count"] == 4
    assert len(summary["cell_names"]) == 4


def test_ablation_cell_short_label_is_alphabetized_by_axis_short_name() -> None:
    cell = AblationCell(name="x", overlay={"bim.icp_robust_loss": "tukey", "project.random_seed": 42})
    # Sorted by full dotted key, so "bim.icp_robust_loss" precedes "project.random_seed";
    # the human-readable suffix preserves that order.
    assert cell.short_label == "icp_robust_loss=tukey,random_seed=42"
