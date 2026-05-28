from __future__ import annotations

import argparse
import html
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from .config_access import bool_value, cfg_get, project_name, project_root, run_id, stage76_paths
from .input_selection import ViewerArtifact, discover_artifacts
from .io_utils import clean_dir_guarded, copy_file, ensure_dir, write_json_atomic

LOGGER_NAME = "pipeline.stage_07_6_viewer_export"


class Stage76ViewerExportError(RuntimeError):
    """Raised when Stage 7.6 cannot create a viewer package."""


def _load_config(path: Path) -> Any:
    from pipeline.common.config import load_config

    return load_config(path)


def _tool_status() -> dict[str, str | None]:
    return {
        "PotreeConverter": shutil.which("PotreeConverter"),
        "potreeconverter": shutil.which("potreeconverter"),
        "pdal": shutil.which("pdal"),
        "entwine": shutil.which("entwine"),
        "py3dtiles": shutil.which("py3dtiles"),
    }


def _relative_to_root(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def _copy_artifacts(root: Path, artifacts: list[ViewerArtifact], artifacts_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for artifact in artifacts:
        rel = _relative_to_root(root, artifact.path)
        dst = artifacts_dir / rel
        size = copy_file(artifact.path, dst)
        records.append({
            "key": artifact.key,
            "kind": artifact.kind,
            "description": artifact.description,
            "source_path": artifact.path.as_posix(),
            "export_path": dst.as_posix(),
            "relative_export_path": dst.relative_to(artifacts_dir.parent).as_posix(),
            "size_bytes": size,
        })
    return records


def _render_index_html(path: Path, manifest: dict[str, Any]) -> None:
    rows = []
    for artifact in manifest["artifacts"]:
        link = html.escape(artifact["relative_export_path"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(artifact['key'])}</td>"
            f"<td>{html.escape(artifact['kind'])}</td>"
            f"<td>{html.escape(artifact['description'])}</td>"
            f"<td>{artifact['size_bytes']}</td>"
            f"<td><a href='{link}'>open/download</a></td>"
            "</tr>"
        )

    missing_rows = []
    for item in manifest["missing_artifacts"]:
        missing_rows.append(
            "<tr>"
            f"<td>{html.escape(item['key'])}</td>"
            f"<td>{html.escape(item['kind'])}</td>"
            f"<td>{html.escape(item['path'])}</td>"
            "</tr>"
        )

    tool_rows = []
    for name, value in manifest["external_tools"].items():
        status = value or "not installed"
        tool_rows.append(f"<tr><td>{html.escape(name)}</td><td>{html.escape(status)}</td></tr>")

    potree_note = (
        "Potree conversion is not active because PotreeConverter was not found. "
        "This package still exports the raw artifacts and a stable manifest."
        if not manifest["capabilities"]["potree_converter_available"]
        else "PotreeConverter is available; a future converter step can generate Potree octree assets."
    )

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Viewer Export - {html.escape(manifest['project'])}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f4f4f4; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }}
.ok {{ color: #0a7a28; font-weight: bold; }}
.warn {{ color: #9a6200; font-weight: bold; }}
</style>
</head>
<body>
<h1>Stage 7.6 Viewer Export</h1>
<div class="card">
<p><strong>Project:</strong> {html.escape(manifest['project'])}</p>
<p><strong>Run ID:</strong> {html.escape(manifest['run_id'])}</p>
<p><strong>Status:</strong> <span class="ok">{html.escape(manifest['status'])}</span></p>
<p><strong>Artifact count:</strong> {len(manifest['artifacts'])}</p>
<p><strong>Generated at:</strong> {manifest['created_at_unix']}</p>
</div>

<div class="card">
<h2>Viewer package note</h2>
<p>{html.escape(potree_note)}</p>
<p>This MVP viewer package is a portable artifact portal. Use CloudCompare, MeshLab, Open3D, Potree, or Cesium tooling to open or convert the exported point clouds.</p>
</div>

<h2>Exported artifacts</h2>
<table>
<thead><tr><th>Key</th><th>Kind</th><th>Description</th><th>Size bytes</th><th>Link</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>

<h2>Missing optional artifacts</h2>
<table>
<thead><tr><th>Key</th><th>Kind</th><th>Expected path</th></tr></thead>
<tbody>
{''.join(missing_rows) if missing_rows else '<tr><td colspan="3">None</td></tr>'}
</tbody>
</table>

<h2>External viewer tools</h2>
<table>
<thead><tr><th>Tool</th><th>Status</th></tr></thead>
<tbody>
{''.join(tool_rows)}
</tbody>
</table>
</body>
</html>
"""
    ensure_dir(path.parent)
    path.write_text(body, encoding="utf-8")


def run_viewer_export(cfg: Any, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    logger = logging.getLogger(LOGGER_NAME)

    paths = stage76_paths(cfg)
    output_dir = paths["output_dir"]

    clean_dir_guarded(output_dir, force=force, required_token="viewer")
    ensure_dir(paths["artifacts_dir"])

    root = project_root(cfg)
    artifacts, missing = discover_artifacts(cfg)

    if not artifacts and bool_value(cfg_get(cfg, "viewer_export.fail_if_no_artifacts", False)):
        raise Stage76ViewerExportError("No viewer artifacts found.")

    copied = _copy_artifacts(root, artifacts, paths["artifacts_dir"])
    tools = _tool_status()

    manifest = {
        "stage": "stage_07_6_viewer_export",
        "status": "ok" if copied else "no_artifacts",
        "project": project_name(cfg),
        "run_id": run_id(cfg),
        "output_dir": output_dir.as_posix(),
        "index_html": paths["index_html"].as_posix(),
        "manifest_json": paths["manifest_json"].as_posix(),
        "artifacts_dir": paths["artifacts_dir"].as_posix(),
        "artifacts": copied,
        "missing_artifacts": [
            {
                "key": item.key,
                "kind": item.kind,
                "path": item.path.as_posix(),
                "required": item.required,
                "description": item.description,
            }
            for item in missing
        ],
        "external_tools": tools,
        "capabilities": {
            "portable_artifact_portal": True,
            "potree_converter_available": bool(tools.get("PotreeConverter") or tools.get("potreeconverter")),
            "cesium_local_tiler_available": bool(tools.get("py3dtiles") or tools.get("entwine")),
            "raw_ply_export": True,
        },
        "notes": [
            "This MVP does not require PotreeConverter, PDAL, Entwine, py3dtiles, Open3D, or NumPy.",
            "Potree/Cesium conversion can be added later without changing upstream Stage 7/8/9 outputs.",
            "This viewer export is for visualization and delivery; metric truth remains Stage 8/9.",
        ],
        "created_at_unix": time.time(),
    }

    write_json_atomic(paths["manifest_json"], manifest)
    _render_index_html(paths["index_html"], manifest)

    logger.info("Stage 7.6 viewer export complete: %s", paths["index_html"])
    print(
        "STAGE_07_6_VIEWER_EXPORT_OK "
        f"artifacts={len(copied)} "
        f"index={paths['index_html'].as_posix()} "
        f"manifest={paths['manifest_json'].as_posix()}"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 7.6 viewer/export package generation.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    run_viewer_export(cfg, force=args.force, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
