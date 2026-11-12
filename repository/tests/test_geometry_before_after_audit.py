"""Operational old/new geometry QA: separates source fidelity from independent GT."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.compare_geometry_versions import run_compare, main

pytestmark = pytest.mark.lightweight


@pytest.fixture()
def files(tmp_path):
    # Deliberately noisy prior output; the new output preserves the geometric surface.
    xs = np.linspace(0, 1, 12)
    u, v = np.meshgrid(xs, xs)
    gt = np.stack([u.ravel(), v.ravel(), np.zeros(u.size)], axis=1)
    outliers = np.array([[10., 10., 10.], [-10., -12., 14.]])
    old = np.vstack((gt, outliers))
    source = old.copy()
    a = {}
    for name, pts in (("source", source), ("before", old), ("after", gt), ("gt", gt)):
        path = tmp_path / (name + ".npy")
        np.save(path, pts)
        a[name] = path
    a["root"] = tmp_path
    return a


def test_source_similarity_does_not_prove_new_cloud_is_better(files):
    report = run_compare(files["source"], files["before"], files["after"],
                         frame="reconstruction", units="arbitrary", threshold=.05)
    assert report["quality_conclusion"] == "not_determined_without_independent_ground_truth"
    assert report["scientific_field_validation"] is False
    assert "against_supplied_ground_truth" not in report["after"]
    assert report["inputs"]["source"]["sha256"]


def test_supplied_synthetic_truth_separately_scores_old_and_new(files):
    r = run_compare(files["source"], files["before"], files["after"],
                    frame="reconstruction", units="arbitrary", threshold=.05,
                    gt=files["gt"], gt_kind="synthetic")
    a = r["after"]["against_supplied_ground_truth"]
    b = r["before"]["against_supplied_ground_truth"]
    assert a["fscore_within_threshold"] == pytest.approx(1.)
    assert b["fscore_within_threshold"] < a["fscore_within_threshold"]
    assert a["chamfer_l1_source_units"] == pytest.approx(0.)
    assert r["scientific_field_validation"] is False


def test_disagreeing_manifest_frame_is_a_hard_error(files):
    old = files["root"] / "old.json"
    new = files["root"] / "new.json"
    old.write_text(json.dumps({"coordinate_frame": "A", "units": "arbitrary"}))
    new.write_text(json.dumps({"coordinate_frame": "B", "units": "arbitrary"}))
    with pytest.raises(ValueError, match="mismatch"):
        run_compare(files["source"], files["before"], files["after"],
                    frame="A", units="arbitrary", threshold=.05,
                    before_manifest=old, after_manifest=new)


def test_cli_writes_non_destructive_report_and_rejects_overwrite(files):
    dest = files["root"] / "review.json"
    cli = ["--source", str(files["source"]), "--before", str(files["before"]),
           "--after", str(files["after"]), "--frame", "reconstruction",
           "--units", "arbitrary", "--threshold", ".05", "--out", str(dest)]
    assert main(cli) == 0
    assert dest.exists()
    before_sha = dest.read_bytes()
    assert main(cli) == 2
    assert dest.read_bytes() == before_sha


def test_no_metres_or_fake_gt_from_source(files):
    r = run_compare(files["source"], files["before"], files["after"],
                    frame="unknown", units="reconstruction_units", threshold=.05)
    assert r["source_is_not_ground_truth"] is True
    assert r["units"] == "reconstruction_units"
    assert "metric_scale_verified" not in r
