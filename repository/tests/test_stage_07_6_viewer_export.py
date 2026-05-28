from __future__ import annotations

import json
from pathlib import Path

from pipeline.stage_07_6_viewer_export.input_selection import discover_artifacts
from pipeline.stage_07_6_viewer_export.run_viewer_export import run_viewer_export


def _write(path: Path, content: str = "demo") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _cfg(tmp_path: Path) -> dict:
    return {
        "project": {"root": str(tmp_path), "name": "site01", "run_id": "test"},
        "paths": {
            "clean_cloud": "data/clean/site01/cleaned_cloud.ply",
            "clean_mesh": "data/clean/site01/mesh.ply",
            "clean_planes_json": "data/clean/site01/planes.json",
        },
        "progress": {
            "element_metrics_csv": "data/bim/metrics/site01/element_metrics.csv",
            "dashboard_html": "runs/test/reports/progress_dashboard.html",
        },
        "viewer_export": {
            "output_dir": "exports/viewer/site01",
            "artifacts_dir": "exports/viewer/site01/artifacts",
            "manifest_json": "exports/viewer/site01/viewer_manifest.json",
            "index_html": "exports/viewer/site01/index.html",
        },
    }


def test_discover_artifacts(tmp_path: Path) -> None:
    _write(tmp_path / "data/clean/site01/cleaned_cloud.ply", "ply cleaned")
    _write(tmp_path / "data/clean/site01/mesh.ply", "ply mesh")
    _write(tmp_path / "data/clean/site01/planes.json", json.dumps({"planes": []}))

    found, missing = discover_artifacts(_cfg(tmp_path))

    keys = {item.key for item in found}
    assert "cleaned_cloud" in keys
    assert "clean_mesh" in keys
    assert "planes_json" in keys
    assert any(item.key == "progress_dashboard" for item in missing)


def test_run_viewer_export_writes_manifest_and_html(tmp_path: Path) -> None:
    _write(tmp_path / "data/clean/site01/cleaned_cloud.ply", "ply cleaned")
    _write(tmp_path / "data/clean/site01/mesh.ply", "ply mesh")
    _write(tmp_path / "data/clean/site01/planes.json", json.dumps({"planes": []}))
    _write(tmp_path / "data/bim/metrics/site01/element_metrics.csv", "global_id,name\nE1,Wall\n")
    _write(tmp_path / "runs/test/reports/progress_dashboard.html", "<html>dashboard</html>")

    manifest = run_viewer_export(_cfg(tmp_path), force=True, log_level="ERROR")

    assert manifest["stage"] == "stage_07_6_viewer_export"
    assert manifest["status"] == "ok"
    assert Path(manifest["index_html"]).exists()
    assert Path(manifest["manifest_json"]).exists()
    assert len(manifest["artifacts"]) >= 4
    assert "PotreeConverter" in manifest["external_tools"]


def test_run_viewer_export_is_skip_safe_with_no_artifacts(tmp_path: Path) -> None:
    manifest = run_viewer_export(_cfg(tmp_path), force=True, log_level="ERROR")
    assert manifest["status"] == "no_artifacts"
    assert Path(manifest["index_html"]).exists()
    assert Path(manifest["manifest_json"]).exists()
