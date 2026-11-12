"""Reproducible SYNTHETIC geometry fixture; never field ground truth."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.stage_07_cleanup.comparison_ready import _ply_write, run_ready


def main() -> int:
    p = argparse.ArgumentParser(description="Make independent synthetic fixture: two walls and a floor")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--run-id", default="geom-ready-qa-fixture-001")
    a = p.parse_args()
    root = a.project_root.expanduser().resolve()
    source = root / "synthetic_three_planes.ply"
    if source.exists():
        raise FileExistsError(f"Refusing to overwrite synthetic original: {source}")
    x = np.linspace(0, 2, 28)
    y, z = np.meshgrid(x, x)
    planes = [
        np.c_[np.zeros(z.size), y.ravel(), z.ravel()],
        np.c_[x.repeat(28), np.zeros(z.size), z.ravel()],
        np.c_[y.ravel(), z.ravel(), np.zeros(z.size)],
    ]
    xyz = np.vstack(planes)
    xyz += np.random.default_rng(55).normal(0, .001, xyz.shape)
    rgb = np.vstack([
        np.tile([220, 65, 70], (z.size, 1)),
        np.tile([60, 150, 200], (z.size, 1)),
        np.tile([90, 180, 110], (z.size, 1)),
    ])
    root.mkdir(parents=True, exist_ok=True)
    _ply_write(source, xyz, rgb)
    result = run_ready(source, root, a.run_id, threshold=.02,
                       min_plane_points=90, trials=350)
    print({"schema_version": result["schema_version"],
           "source_sha256": result["source_sha256"],
           "points": result["clean_points"],
           "planes": result["plane_count"],
           "edges": result["edge_count"],
           "corners": result["corner_count"],
           "metric_scale_verified": result["metric_scale_verified"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
