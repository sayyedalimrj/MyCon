#!/usr/bin/env python3
"""Install the ACTUAL NVIDIA LongSplat repository with its 4 pinned submodules.
Noncommercial research/evaluation only. Do not overwrite MyCON checkpoints.
"""
import argparse
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

URL = "https://github.com/NVlabs/LongSplat.git"
REV = "19750775a9d19f30aa05a8333c4c6c231b2d5f4a"
REQUIRED = ["train.py", "render.py", "convert_3dgs.py", "LICENSE.md", "requirements.txt", ".gitmodules"]
SUBMODULES = ["submodules/mast3r", "submodules/diff-gaussian-rasterization",
              "submodules/simple-knn", "submodules/fused-ssim"]

def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, text=True, check=True,
                          capture_output=True).stdout.strip()

def verify(root):
    root = Path(root)
    if not (root / ".git").exists():
        raise RuntimeError("No official LongSplat Git checkout: " + str(root))
    origin = git("remote", "get-url", "origin", cwd=root)
    if origin.rstrip("/") not in (URL, "git@github.com:NVlabs/LongSplat.git"):
        raise RuntimeError("Unrecognized upstream origin: " + origin)
    sha = git("rev-parse", "HEAD", cwd=root)
    if sha != REV:
        raise RuntimeError(f"Revision mismatch: {sha}, expected {REV}")
    if any(not (root / name).is_file() for name in REQUIRED):
        raise RuntimeError("Incomplete NVIDIA source checkout")
    licence = (root / "LICENSE.md").read_text(encoding="utf-8").lower()
    if "non-commercially" not in licence or "research or evaluation" not in licence:
        raise RuntimeError("NVIDIA research-use licence could not be verified")
    lines = git("submodule", "status", "--recursive", cwd=root).splitlines()
    found = {}
    for line in lines:
        if line and line[0] != " ":
            raise RuntimeError("Incomplete/unexpected submodule: " + line)
        parts = line.strip().split()
        if len(parts) >= 2:
            found[parts[1]] = parts[0]
    if any(name not in found or not (root / name).is_dir() or
           not any((root / name).iterdir()) for name in SUBMODULES):
        raise RuntimeError("A mandatory CUDA/MASt3R submodule is missing")
    return {"upstream": URL, "commit": sha, "submodules": found,
            "status": "SOURCE_VERIFIED_NOT_CUDA_TESTED",
            "license": "NVIDIA noncommercial research/evaluation"}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    dest = a.dest.expanduser().resolve()
    try:
        if a.verify_only:
            result = verify(dest)
        elif dest.exists():
            result = verify(dest)
            print("REUSED intact official NVIDIA source")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.parent / ("." + dest.name + ".partial-" + uuid.uuid4().hex[:8])
            try:
                git("clone", "--recursive", URL, str(partial))
                git("checkout", "--detach", REV, cwd=partial)
                git("submodule", "sync", "--recursive", cwd=partial)
                git("submodule", "update", "--init", "--recursive", cwd=partial)
                result = verify(partial)
                partial.rename(dest)
            except Exception:
                print("PARTIAL CHECKOUT PRESERVED:", partial, file=sys.stderr)
                raise
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("NVIDIA_SOURCE_NOT_READY:", exc, file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
