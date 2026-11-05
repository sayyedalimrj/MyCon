#!/usr/bin/env python3
"""Mandatory REAL MASt3R two-frame CUDA smoke before expensive LongSplat training.

Invokes NVIDIA LongSplat's own MASt3R wrapper on verified prepared frames.
No optimization, checkpoint overwrite, source mutation, or model training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def choose_pair(frames_dir: Path) -> tuple[Path, Path, int]:
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) < 30:
        raise RuntimeError("MAST3R_SMOKE_REQUIRES_COMPLETE_PREPARED_FRAMES")
    first, second = frames[0], frames[min(8, len(frames) - 1)]
    if first == second or any(not p.is_file() or not p.stat().st_size for p in (first, second)):
        raise RuntimeError("MAST3R_SMOKE_FRAME_PAIR_INVALID")
    return first, second, len(frames)


def run(source: Path, frames_dir: Path) -> dict:
    if not (source / "train.py").is_file() or not (source / "submodules/mast3r").is_dir():
        raise RuntimeError("OFFICIAL_NVIDIA_LONGSPLAT_CHECKOUT_REQUIRED")
    first, second, total = choose_pair(frames_dir)
    # Execute from the official NVIDIA repository using the same Python as training.
    # The caller explicitly sets cwd=source and CUDA_HOME/compiled extension paths.
    import torch
    from utils.mast3r_utils import Mast3rMatcher
    from dust3r.utils.image import load_images
    from dust3r.inference import inference
    if not torch.cuda.is_available():
        raise RuntimeError("MAST3R_SMOKE_CUDA_UNAVAILABLE")
    expected = int(os.getenv("MYCON_EXPECTED_SM", "0"))
    capability = torch.cuda.get_device_capability(0)
    actual = 10 * capability[0] + capability[1]
    if expected and actual != expected:
        raise RuntimeError(f"MAST3R_SMOKE_GPU_CHANGED: expected {expected} got {actual}")
    matcher = Mast3rMatcher()
    images = load_images([str(first), str(second)], size=512, verbose=False)
    if len(images) != 2:
        raise RuntimeError("MAST3R_SMOKE_IMAGE_LOADING_FAILED")
    result = inference([(images[0], images[1])], matcher.model,
                       device="cuda", batch_size=1, verbose=False)
    points = result["pred1"]["pts3d"]
    if not bool(torch.isfinite(points).all()):
        raise RuntimeError("MAST3R_SMOKE_NONFINITE_POINTS")
    torch.cuda.synchronize()
    return {
        "schema": "mycon.mast3r.2view.cuda.v1",
        "status": "MAST3R_TWO_VIEW_CUDA_INFERENCE_PASSED",
        "frame_count": total,
        "frames": [first.name, second.name],
        "points_shape": list(points.shape),
        "max_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "sm": actual,
        "torch_version": torch.__version__,
        "training_executed": False,
        "bim_metric_validated": False,
        "interpretation": "Actual pretrained MASt3R two-frame inference, not LongSplat training"
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    a = parser.parse_args(argv)
    report = run(a.source.resolve(), a.frames_dir.resolve())
    a.report.parent.mkdir(parents=True, exist_ok=True)
    if a.report.exists():
        old = json.loads(a.report.read_text())
        if (old.get("schema") != report["schema"] or old.get("frames") != report["frames"]
            or old.get("sm") != report["sm"]):
            raise RuntimeError("OLD_MAST3R_SMOKE_REPORT_CONFLICT: preserve existing report")
    else:
        tmp = a.report.with_suffix(".partial.json")
        if tmp.exists():
            raise RuntimeError("MAST3R_SMOKE_PARTIAL_REPORT_PRESERVED")
        try:
            tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            os.replace(tmp, a.report)
        finally:
            if tmp.exists(): tmp.unlink()
    print("MYCON_MAST3R_TWO_VIEW_PASSED " + json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("MYCON_MAST3R_TWO_VIEW_FAILED:", type(exc).__name__, str(exc), file=sys.stderr)
        raise SystemExit(2)
