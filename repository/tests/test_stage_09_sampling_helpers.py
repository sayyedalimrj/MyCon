from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from pipeline.stage_09_progress.run_progress import _nearest_distances_with_indices


def _cloud(points: np.ndarray):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(float))
    return pc


def test_nearest_distances_with_indices_preserves_sampled_indices() -> None:
    query = np.asarray([[float(i), 0.0, 0.0] for i in range(20)], dtype=float)
    target = _cloud(query.copy())

    indices, distances = _nearest_distances_with_indices(query, target, limit=5)

    assert len(indices) == 5
    assert len(distances) == 5
    assert np.all(indices >= 0)
    assert np.all(indices < len(query))
    assert np.allclose(distances, 0.0)


def test_nearest_distances_with_indices_full_query_when_under_limit() -> None:
    query = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    target = _cloud(query.copy())

    indices, distances = _nearest_distances_with_indices(query, target, limit=10)

    assert indices.tolist() == [0, 1]
    assert np.allclose(distances, 0.0)


def test_nearest_distances_implementation_mentions_batch_kdtree() -> None:
    source = Path("pipeline/stage_09_progress/run_progress.py").read_text(encoding="utf-8")
    assert "cKDTree" in source
    assert "workers=-1" in source
