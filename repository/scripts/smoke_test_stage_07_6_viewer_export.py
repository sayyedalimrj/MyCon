from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage_07_6_viewer_export.run_viewer_export import run_viewer_export


def _write(path: Path, content: str = "demo") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="stage076_smoke_"))

    _write(root / "data/clean/site01/cleaned_cloud.ply", "ply demo cleaned")
    _write(root / "data/clean/site01/mesh.ply", "ply demo mesh")
    _write(root / "data/clean/site01/planes.json", json.dumps({"planes": []}))
    _write(root / "data/bim/metrics/site01/element_metrics.csv", "global_id,name\nE1,Wall\n")
    _write(root / "runs/test/reports/progress_dashboard.html", "<html>dashboard</html>")

    cfg = {
        "project": {"root": str(root), "name": "site01", "run_id": "test"},
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

    manifest = run_viewer_export(cfg, force=True, log_level="ERROR")

    assert Path(manifest["index_html"]).exists()
    assert Path(manifest["manifest_json"]).exists()
    assert len(manifest["artifacts"]) >= 4

    print(f"STAGE_07_6_SMOKE_OK artifacts={len(manifest['artifacts'])} index={manifest['index_html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
