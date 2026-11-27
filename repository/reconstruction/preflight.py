"""Non-mutating Colab/local smoke/preflight; never installs or starts models."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from .workflow import Blocked, checked_run, sha256


def audit(run_dir: Path | None = None, *, checkpoint: Path | None = None) -> dict:
    packages = {p: bool(importlib.util.find_spec(p)) for p in
                ("cv2", "numpy", "torch", "gsplat", "lpips", "skimage", "pytest")}
    report = {"schema_version":"mycon.reconstruction.runtime_preflight.v1",
              "python":sys.version.split()[0], "platform":platform.platform(),
              "colmap_executable":shutil.which("colmap"), "ffmpeg_executable":shutil.which("ffmpeg"),
              "packages_available":packages, "cuda_available":False, "gpu_name":None,
              "cuda_test_executed":False, "gs_gpu_train_executed":False,
              "metric_scale_verified":False, "run_id":run_dir.name if run_dir else None}
    if packages["torch"]:
        import torch
        report["torch_version"] = torch.__version__
        report["torch_cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["cuda_test_executed"] = True
    if run_dir:
        checked_run(run_dir)
        report["dataset_manifest_sha256"] = sha256(run_dir / "dataset_manifest.json")
        report["run_preflight_only"] = True
    if checkpoint:
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise Blocked("Missing existing checkpoint")
        report["existing_checkpoint"] = {"size_bytes":checkpoint.stat().st_size,
                                          "sha256":sha256(checkpoint), "read_only_inspection":True}
    return report


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    a=parser.parse_args(argv)
    try:
        print(json.dumps(audit(a.run_dir,checkpoint=a.checkpoint),indent=2))
        return 0
    except (Blocked,OSError,ValueError) as e:
        print(json.dumps({"status":"blocked","reason":str(e)}))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
