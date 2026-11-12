"""Integration-level contract: actual A producer output must be readable by B and C.

Synthetic geometry makes this a software interoperability test, not field QA.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.stage_07_cleanup.comparison_ready import _ply_write, run_ready
from pipeline.stage_09_progress.comparison_ready_adapter import (
    load_comparison_ready_geometry, comparison_for_element,
)
from benchmarks.mycon_2026.model_agnostic import load_ready

pytestmark = pytest.mark.lightweight


@pytest.fixture()
def native_a_run(tmp_path: Path):
    grid = np.linspace(0, 1.7, 13)
    u, v = np.meshgrid(grid, grid)
    points = np.vstack([
        np.column_stack([np.zeros(u.size), u.ravel(), v.ravel()]),
        np.column_stack([u.ravel(), np.zeros(u.size), v.ravel()]),
        np.column_stack([u.ravel(), v.ravel(), np.zeros(u.size)]),
    ])
    path = tmp_path / "source.ply"
    rgb = np.tile(np.array([[150, 80, 30]], dtype=np.uint8), (len(points), 1))
    _ply_write(path, points, rgb)
    run_ready(path, tmp_path, "geom-ready-bridge-test", threshold=0.02,
              min_plane_points=30, max_planes=4, trials=350,
              source_frame="test_reconstruction", source_units="reconstruction_units",
              preview=False)
    return tmp_path, tmp_path / "runs/geom-ready-bridge-test/geometry/comparison_ready.json"


def test_a_producer_to_b_and_c_real_format(native_a_run):
    root, manifest = native_a_run
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["schema_version"] == "mycon.comparison_ready_geometry.v1"
    assert meta["metric_scale_verified"] is False
    assert meta["registration_verified"] is False

    bundle = load_comparison_ready_geometry(manifest, project_root=root)
    assert bundle.valid, bundle.reasons
    assert len(bundle.features["planes"]) >= 3
    assert bundle.units == "reconstruction_units"
    assert bundle.summary()["physical_progress_estimate"] is None
    assert bundle.summary()["comparison_gate"] == "blocked_without_independent_ifc_mapping"
    element = comparison_for_element(bundle, "synthetic_not_a_real_global_id")
    assert element["comparison_ifc_quality_gate"] == "blocked"

    source, paths, features = load_ready(manifest, root=root)
    assert source["metric_scale_verified"] is False
    assert len(features["planes"]) >= 3
    assert all(len(f["support_extent"]) == 2 for f in features["planes"])
    assert all("endpoints" in f for f in features["edges"])
    assert all("feature_ids" in f for f in features["corners"])
    assert paths["clean_cloud"].is_file()


def test_tampered_a_feature_sidecar_rejected_by_b_and_c(native_a_run):
    root, manifest = native_a_run
    sidecar = manifest.parent / "planes.json"
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + " ", encoding="utf-8")
    bundle = load_comparison_ready_geometry(manifest, project_root=root)
    assert bundle.valid is False
    assert "sha256" in str(bundle.reasons).lower()
    with pytest.raises(ValueError, match="SHA"):
        load_ready(manifest, root=root)


def test_native_a_false_metric_claim_is_blocked(native_a_run):
    root, manifest = native_a_run
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["metric_scale_verified"] = True
    manifest.write_text(json.dumps(m), encoding="utf-8")
    bundle = load_comparison_ready_geometry(manifest, project_root=root)
    assert bundle.valid is False
    with pytest.raises(ValueError, match="metric"):
        load_ready(manifest, root=root)
