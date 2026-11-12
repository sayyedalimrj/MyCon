"""Read-only research audit for existing *gsplat* checkpoints / PLY / renders.

The official gsplat .pt format is not Nerfstudio's step-*.ckpt format.
This module never loads torch pickles, performs training, or modifies sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

RUN_PATTERN = re.compile(r"^geom-[A-Za-z0-9][A-Za-z0-9._-]*$")
METRICS_PATTERN = re.compile(
    r"PSNR:\s*([0-9]+(?:\.[0-9]+)?)"
    r".*?SSIM:\s*([0-9]+(?:\.[0-9]+)?)"
    r".*?LPIPS:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE
)
FRAME_PATTERN = re.compile(r"Rendering trajectory:\s*100%.*?(\d+)\s*/\s*(\d+)")
GS_PATTERN = re.compile(r"Number of GS:\s*(\d+)")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_gsplat_artifacts(
    checkpoint: Path,
    splat_ply: Path,
    evaluation_log: Path,
    video: Path | None = None,
    *,
    hash_large_inputs: bool = False,
) -> dict:
    """Evaluate artifact existence and historical log claims, never model accuracy."""
    sources = {"checkpoint_pt": checkpoint, "gaussian_ply": splat_ply,
               "evaluation_log": evaluation_log}
    if video is not None:
        sources["render_video"] = video
    evidence = {}
    for kind, source in sources.items():
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty {kind}: {source}")
        evidence[kind] = {
            "path": str(source.resolve()),
            "size_bytes": int(source.stat().st_size),
            "sha256": file_sha256(source) if hash_large_inputs or kind == "evaluation_log" else None,
        }
    text = evaluation_log.read_text(encoding="utf-8", errors="replace")
    # Last value normally corresponds to final eval if the log includes multiple.
    metrics = list(METRICS_PATTERN.finditer(text))
    match = metrics[-1] if metrics else None
    frames = [(int(m.group(1)), int(m.group(2))) for m in FRAME_PATTERN.finditer(text)]
    video_saved_log = "Video saved to" in text
    gaussian_counts = GS_PATTERN.findall(text)
    reported = ({
        "psnr_db": float(match.group(1)),
        "ssim": float(match.group(2)),
        "lpips": float(match.group(3)),
    } if match else None)
    return {
        "schema": "mycon.geometry.gsplat_existing_artifacts.v1",
        "backend": "official_gsplat_existing_checkpoint_audit",
        "sources": evidence,
        "reported_from_log_only": reported,
        "reported_gaussian_count": int(gaussian_counts[-1]) if gaussian_counts else None,
        "trajectory_frames_reported": frames[-1] if frames else None,
        "video_saved_reported": video_saved_log,
        "video_file_verified_nonempty": video is not None,
        "checkpoint_content_verified": False,
        "novel_views_independent_of_train_verified": False,
        "ground_truth_validated": False,
        "metric_scale_verified": False,
        "registration_validated": False,
        "status": "artifact_presence_verified_log_metrics_unvalidated",
        "notes": [
            "PSNR/SSIM/LPIPS extracted from an existing log: data split, GT and provenance not independently established",
            "PLY alone does not establish a functioning novel-view renderer or accurate geometry",
            "Checkpoint .pt is not compatible with Nerfstudio .ckpt resume path",
        ],
    }


def run_audit(
    *, project_root: Path, run_id: str, checkpoint: Path, splat_ply: Path,
    evaluation_log: Path, video: Path | None = None, write: bool = False,
    hash_large_inputs: bool = False,
) -> dict:
    if not RUN_PATTERN.fullmatch(run_id):
        raise ValueError("Run ID must use a unique geom-* name")
    result = inspect_gsplat_artifacts(checkpoint, splat_ply, evaluation_log, video,
                                      hash_large_inputs=hash_large_inputs)
    result["run_id"] = run_id
    result["created_at_unix"] = time.time()
    if not write:
        return result
    output = project_root.resolve() / "runs" / run_id / "geometry" / "existing_gsplat_audit.json"
    if output.exists():
        raise FileExistsError(f"Existing audit cannot be overwritten: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    result["audit_path"] = str(output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect existing official gsplat artifacts without touching them")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--eval-log", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--write", action="store_true", help="Create new geom-* audit, no overwrite")
    parser.add_argument("--hash-large-inputs", action="store_true")
    a = parser.parse_args(argv)
    try:
        result = run_audit(
            project_root=a.project_root, run_id=a.run_id, checkpoint=a.checkpoint,
            splat_ply=a.ply, evaluation_log=a.eval_log, video=a.video,
            write=a.write, hash_large_inputs=a.hash_large_inputs,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
