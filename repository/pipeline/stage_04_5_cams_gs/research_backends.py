"""Isolated optional VGGT and external DUSt3R COLMAP-export adapters.

The official VGGT demo_colmap.py produces COLMAP-compatible geometry.
DUSt3R requires a separately supplied COLMAP export script; no native exporter
is asserted here. No backend downloads models or runs GPU work in inspect mode.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .input_selection import IMAGE_EXTS

RUN_ID = re.compile(r"^geom-[a-zA-Z0-9][a-zA-Z0-9._-]*$")
PARTS = ("cameras", "images", "points3D")


def colmap_contract(sparse: Path) -> dict:
    """Read-only validation; accepts modern COLMAP binary or text models."""
    for ext in (".bin", ".txt"):
        files = [sparse / (part + ext) for part in PARTS]
        if all(p.is_file() and p.stat().st_size > 0 for p in files):
            return {"valid": True, "format": ext[1:], "paths": [str(p) for p in files]}
    return {"valid": False, "format": None, "paths": []}


def prepare_backend_command(
    backend: str, backend_root: Path, scene_dir: Path,
    *, use_ba: bool = False, dust3r_adapter: Path | None = None,
) -> list[str]:
    backend = backend.lower()
    if backend == "vggt":
        script = backend_root / "demo_colmap.py"
        if not script.is_file():
            raise FileNotFoundError(f"VGGT official demo_colmap.py missing: {script}")
        cmd = [sys.executable, str(script), "--scene_dir", str(scene_dir)]
        if use_ba:
            cmd.append("--use_ba")
        return cmd
    if backend == "dust3r":
        if dust3r_adapter is None or not dust3r_adapter.is_file():
            raise ValueError("DUSt3R has no built-in COLMAP exporter; supply a reviewed --dust3r-adapter")
        return [sys.executable, str(dust3r_adapter.resolve()), "--scene-dir", str(scene_dir)]
    raise ValueError("Backend must be vggt or dust3r")


def run_backend(
    *, backend: str, images: Path, project_root: Path, run_id: str,
    backend_root: Path, execute: bool = False, use_ba: bool = False,
    dust3r_adapter: Path | None = None,
) -> dict:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Run ID must begin geom-")
    source_images = sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if images.is_dir() else []
    if len(source_images) < 2:
        raise ValueError("At least two images are required")
    output_dir = project_root.expanduser().resolve() / "runs" / run_id / "geometry" / f"backend_{backend}"
    scene_dir = output_dir / "scene"
    cmd = prepare_backend_command(backend, backend_root.resolve(), scene_dir, use_ba=use_ba, dust3r_adapter=dust3r_adapter)
    evidence = {
        "schema": "mycon.geometry.backend.v1",
        "backend": backend,
        "run_id": run_id,
        "status": "inspected_only",
        "image_count": len(source_images),
        "images_dir": str(images.resolve()),
        "scene_dir": str(scene_dir),
        "command": cmd,
        "gpu_validated": False,
        "metric_scale_confirmed": False,
        "gravity_aligned": False,
        "registration_validated": False,
        "created_at_unix": time.time(),
    }
    if not execute:
        return evidence
    if output_dir.exists():
        raise FileExistsError(f"Never overwrite an existing experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = output_dir / "backend_evidence.json"
    def save() -> None:
        manifest.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    save()
    try:
        (scene_dir / "images").mkdir(parents=True)
        for image in source_images:
            shutil.copy2(image, scene_dir / "images" / image.name)
        start = time.monotonic()
        log = output_dir / "backend.log"
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
        evidence.update({"exit_code": rc, "wall_time_sec": round(time.monotonic() - start, 3), "log": str(log)})
        sparse = scene_dir / "sparse"
        contract = colmap_contract(sparse)
        if not contract["valid"]:
            contract = colmap_contract(sparse / "0")
        evidence["colmap_contract"] = contract
        evidence["status"] = "colmap_export_validated_files_only" if rc == 0 and contract["valid"] else "backend_failed_or_missing_colmap"
        # A COLMAP export alone never proves metric scale, good cameras, or NVS.
    except Exception as exc:
        evidence["status"] = "backend_failed"
        evidence["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    return evidence


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Optional geometry backend: VGGT or external DUSt3R exporter")
    p.add_argument("--backend", choices=["vggt", "dust3r"], required=True)
    p.add_argument("--backend-root", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--dust3r-adapter", type=Path)
    p.add_argument("--use-ba", action="store_true")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    try:
        payload = run_backend(backend=a.backend, backend_root=a.backend_root, images=a.images,
                              project_root=a.project_root, run_id=a.run_id, execute=a.execute,
                              use_ba=a.use_ba, dust3r_adapter=a.dust3r_adapter)
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
        return 2
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] in {"inspected_only", "colmap_export_validated_files_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
