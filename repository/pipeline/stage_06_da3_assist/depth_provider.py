from __future__ import annotations

import csv
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_access import cfg_get, list_value
from .io_utils import ensure_dir


@dataclass(frozen=True)
class DepthMapRecord:
    image_name: str
    image_path: Path
    depth_path: Path
    source: str


def _candidate_depth_names(image_name: str, extensions: list[str]) -> list[str]:
    stem = Path(image_name).stem
    return [f"{stem}{ext if ext.startswith('.') else '.' + ext}" for ext in extensions]


def run_external_depth_provider(cfg: Any, image_dir: Path, output_dir: Path, manifest_csv: Path) -> dict[str, Any]:
    command_template = str(cfg_get(cfg, "da3.external_command", "")).strip()
    if not command_template:
        return {"status": "not_configured", "command": ""}
    ensure_dir(output_dir)
    command = command_template.format(
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        manifest_csv=str(manifest_csv),
        config=str(cfg_get(cfg, "da3.external_config_path", "")),
    )
    use_shell = bool(cfg_get(cfg, "da3.external_command_shell", True))
    args = command if use_shell else shlex.split(command)
    result = subprocess.run(args, shell=use_shell, capture_output=True, text=True, errors="replace", check=False)
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def find_depth_maps(
    image_dir: Path,
    depth_dir: Path,
    *,
    extensions: list[str],
    max_images: int | None = None,
) -> list[DepthMapRecord]:
    if not image_dir.exists() or not depth_dir.exists():
        return []
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]
    rows: list[DepthMapRecord] = []
    for image_path in image_paths:
        depth_path = None
        for name in _candidate_depth_names(image_path.name, extensions):
            candidate = depth_dir / name
            if candidate.exists():
                depth_path = candidate
                break
        if depth_path is not None:
            rows.append(DepthMapRecord(image_path.name, image_path, depth_path, "precomputed_or_external"))
    return rows


def write_depth_manifest(path: Path, records: list[DepthMapRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "image_path", "depth_path", "source"])
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "image_name": row.image_name,
                    "image_path": row.image_path.as_posix(),
                    "depth_path": row.depth_path.as_posix(),
                    "source": row.source,
                }
            )


def configured_extensions(cfg: Any) -> list[str]:
    return [ext.lower() for ext in list_value(cfg_get(cfg, "da3.depth_file_extensions", None), [".npy", ".npz", ".png", ".tif", ".tiff", ".pfm"])]
