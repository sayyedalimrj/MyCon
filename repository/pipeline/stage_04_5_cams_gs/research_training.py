"""Opt-in, resume-only Nerfstudio Splatfacto adapter for research geometry.

This is deliberately separate from legacy Stage 4.5 output paths. No new
training is ever started from scratch, and no existing directory is removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from .input_selection import IMAGE_EXTS

RUN_ID = re.compile(r"^geom-[a-zA-Z0-9][a-zA-Z0-9._-]*$")
COLMAP_PARTS = ("cameras", "images", "points3D")


class GeometryTrainingError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(images_dir: Path, sparse_dir: Path, resume_dir: Path | None) -> dict:
    """Read-only inspection. Checkpoints must be actual Nerfstudio .ckpt files."""
    images = sorted(p for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if len(images) < 2:
        raise GeometryTrainingError("At least two source images are required")
    variants = [
        [sparse_dir / (part + ext) for part in COLMAP_PARTS]
        for ext in (".bin", ".txt")
    ]
    model_files = next((set_ for set_ in variants if all(p.is_file() and p.stat().st_size > 0 for p in set_)), None)
    if model_files is None:
        raise GeometryTrainingError("Missing nonempty COLMAP cameras/images/points3D (bin or txt)")
    checkpoints = sorted(resume_dir.glob("step-*.ckpt")) if resume_dir and resume_dir.is_dir() else []
    checkpoints = [p for p in checkpoints if p.is_file() and p.stat().st_size > 0]
    if not checkpoints:
        raise GeometryTrainingError("Resume-only: a Nerfstudio step-*.ckpt is required; no fresh training")
    return {
        "image_count": len(images),
        "image_names": [p.name for p in images],
        "colmap_files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in model_files},
        "resume_checkpoint": str(checkpoints[-1].resolve()),
        "resume_checkpoint_bytes": checkpoints[-1].stat().st_size,
        "resume_checkpoint_sha256": "not_calculated_large_artifact",
    }


def command_for_resume(dataset: Path, output: Path, resume: Path, max_iterations: int) -> list[str]:
    """COLMAP dataparser is required; a bare --data expects transforms.json."""
    if max_iterations <= 0:
        raise GeometryTrainingError("max_iterations must be positive")
    return [
        "ns-train", "splatfacto",
        "--output-dir", str(output),
        "--load-dir", str(resume),
        "--max-num-iterations", str(max_iterations),
        "colmap", "--data", str(dataset),
        "--colmap-path", "sparse/0",
        "--images-path", "images",
        "--auto-scale-poses", "False",
    ]


def _run_logged(argv: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"argv": argv}) + "\n")
        out.flush()
        try:
            proc = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT, check=False)
            return int(proc.returncode)
        except OSError as exc:
            out.write(f"\nLaunch error: {exc}\n")
            return 127


def _manifest_path(root: Path, rid: str) -> Path:
    return root / "runs" / rid / "geometry" / "training_evidence.json"


def run_resume(
    *,
    project_root: Path,
    run_id: str,
    images_dir: Path,
    sparse_dir: Path,
    resume_dir: Path,
    execute: bool = False,
    max_iterations: int = 7000,
    camera_path: Path | None = None,
) -> dict:
    """Inspect first; only execute on explicit opt-in and new geom-* output.

    A completed render is evidence that a renderer produced bytes, NOT proof
    of out-of-sample or independently validated view synthesis.
    """
    if not RUN_ID.fullmatch(run_id):
        raise GeometryTrainingError("Run ID must begin geom- and contain safe characters")
    root = project_root.expanduser().resolve()
    images_dir, sparse_dir = images_dir.resolve(), sparse_dir.resolve()
    resume_dir = resume_dir.resolve()
    inspected = validate_inputs(images_dir, sparse_dir, resume_dir)
    if camera_path is not None and (not camera_path.is_file() or camera_path.suffix.lower() != ".json"):
        raise GeometryTrainingError("A real Nerfstudio camera path JSON is required")
    target = _manifest_path(root, run_id).parent
    cmd = command_for_resume(target / "dataset", target / "nerfstudio_outputs", resume_dir, max_iterations)
    evidence = {
        "schema": "mycon.geometry.training.v1",
        "run_id": run_id,
        "backend": "nerfstudio_splatfacto_resume",
        "stage_04_5_legacy_outputs_untouched": True,
        "is_metric_truth": False,
        "input_images_dir": str(images_dir),
        "input_sparse_dir": str(sparse_dir),
        "resume_dir": str(resume_dir),
        "input_evidence": inspected,
        "command": cmd,
        "gpu_validated": False,
        "real_case_evaluated": False,
        "novel_view_validated": False,
        "status": "inspected_only",
        "render_status": "not_executed",
        "created_at_unix": time.time(),
    }
    if not execute:
        return evidence
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing geometry run: {target}")
    if shutil.which("ns-train") is None:
        raise GeometryTrainingError("ns-train is not available; resume was NOT started")
    target.mkdir(parents=True, exist_ok=False)
    evidence_path = target / "training_evidence.json"
    def save() -> None:
        evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    save()
    start = time.monotonic()
    try:
        dataset = target / "dataset"
        (dataset / "sparse").mkdir(parents=True)
        shutil.copytree(images_dir, dataset / "images")
        shutil.copytree(sparse_dir, dataset / "sparse" / "0")
        evidence["status"] = "training_running"
        save()
        training_log = target / "ns_train.log"
        rc = _run_logged(cmd, training_log)
        evidence["training_return_code"] = rc
        evidence["training_log"] = str(training_log)
        evidence["wall_time_sec"] = round(time.monotonic() - start, 3)
        configs = sorted((target / "nerfstudio_outputs").rglob("config.yml"))
        checkpoints = sorted((target / "nerfstudio_outputs").rglob("step-*.ckpt"))
        configs = [p for p in configs if p.is_file() and p.stat().st_size > 0]
        checkpoints = [p for p in checkpoints if p.is_file() and p.stat().st_size > 0]
        if rc != 0 or not configs or not checkpoints:
            evidence["status"] = "training_failed_or_missing_checkpoint"
            evidence["failure_reason"] = f"rc={rc}, configs={len(configs)}, checkpoints={len(checkpoints)}"
        else:
            evidence["status"] = "trained_checkpoint_present_render_unverified"
            evidence["gpu_validated"] = True
            evidence["config_yml"] = str(configs[-1])
            evidence["trained_checkpoint"] = str(checkpoints[-1])
            if camera_path is not None and shutil.which("ns-render"):
                render_path = target / "novel_camera_path_render.mp4"
                render_cmd = [
                    "ns-render", "camera-path",
                    "--load-config", str(configs[-1]),
                    "--camera-path-filename", str(camera_path.resolve()),
                    "--output-path", str(render_path),
                ]
                render_rc = _run_logged(render_cmd, target / "ns_render.log")
                evidence["render_command"] = render_cmd
                evidence["render_return_code"] = render_rc
                evidence["render_status"] = (
                    "artifact_created_view_independence_unverified"
                    if render_rc == 0 and render_path.exists() and render_path.stat().st_size > 0
                    else "failed"
                )
                if evidence["render_status"].startswith("artifact_created"):
                    evidence["render_artifact"] = str(render_path)
        save()
    except BaseException as exc:
        evidence["status"] = "adapter_failed"
        evidence["failure_reason"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MyCON Geometry: inspect or resume a real Nerfstudio 3DGS checkpoint")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="Unique geom-* run ID")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=7000)
    parser.add_argument("--camera-path", type=Path)
    parser.add_argument("--execute", action="store_true", help="Opt in to GPU resume; NEVER starts training from zero")
    args = parser.parse_args(argv)
    try:
        result = run_resume(project_root=args.project_root, run_id=args.run_id,
                            images_dir=args.images, sparse_dir=args.sparse,
                            resume_dir=args.resume_dir, execute=args.execute,
                            max_iterations=args.max_iterations, camera_path=args.camera_path)
    except (GeometryTrainingError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if not result["status"].endswith("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
