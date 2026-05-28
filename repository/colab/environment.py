"""Colab environment bootstrap.

Installs the system + Python dependencies the MyCon pipeline needs,
in an order that avoids known Colab conflicts:

1. APT: ffmpeg, colmap, libgl1, libglib2.0-0, git-lfs (idempotent).
2. Pin numpy<2 first to honour requirements-core (open3d / opencv compat).
3. Install requirements-core.txt.
4. Install requirements-da3.txt (transformers/accelerate). Optional.
5. Install Gradio + ipywidgets for the UI itself.
6. Validate by import-probing each library and printing a summary.

All commands stream output to a ``LogBuffer`` if one is provided, so the
Gradio UI can show the install progress live.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from colab.log_capture import LogBuffer

# Order matters: requirements-core pins numpy<2 which open3d/opencv assume.
APT_PACKAGES = [
    "ffmpeg",
    "colmap",
    "libgl1",
    "libglib2.0-0",
    "git-lfs",
]

# Gradio + Jupyter widgets are not in any requirements*.txt because they
# are Colab-only concerns.
UI_PACKAGES = [
    "gradio>=4.44,<5.0",
    "ipywidgets>=8.1",
]

# Probes that confirm each requirements layer is functional.
CORE_PROBES = [
    "numpy",
    "yaml",
    "cv2",
    "PIL",
    "pandas",
    "scipy",
    "matplotlib",
    "requests",
    "huggingface_hub",
]
GEOMETRY_PROBES = ["open3d", "ifcopenshell"]
DA3_PROBES = ["transformers", "accelerate", "safetensors"]
UI_PROBES = ["gradio"]


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""


def _stream_run(
    command: list[str],
    *,
    log: Optional[LogBuffer],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> int:
    """Run ``command``, tee stdout/stderr line-by-line into ``log``."""
    pretty = " ".join(command)
    if log is not None:
        log.append(f"$ {pretty}")
    else:
        print(f"$ {pretty}", flush=True)
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        msg = f"[error] command not found: {command[0]} ({exc})"
        if log is not None:
            log.append(msg)
        else:
            print(msg, flush=True)
        return 127

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if log is not None:
                log.append(line)
            else:
                print(line, flush=True)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        msg = f"[error] timeout after {timeout}s: {pretty}"
        if log is not None:
            log.append(msg)
        else:
            print(msg, flush=True)
        return 124
    return int(proc.returncode or 0)


def is_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


# ---------------------------------------------------------------------------
# APT
# ---------------------------------------------------------------------------


def install_apt_packages(*, log: Optional[LogBuffer] = None) -> StepResult:
    if shutil.which("apt-get") is None:
        return StepResult("apt", False, "apt-get not available (not a Debian/Ubuntu host)")
    rc_update = _stream_run(["apt-get", "-qq", "update"], log=log, timeout=600)
    if rc_update != 0:
        return StepResult("apt-update", False, f"apt-get update returned {rc_update}")
    rc_install = _stream_run(
        ["apt-get", "-qq", "install", "-y", "--no-install-recommends", *APT_PACKAGES],
        log=log,
        timeout=900,
    )
    return StepResult(
        "apt-install",
        rc_install == 0,
        f"apt-get install returned {rc_install}",
    )


# ---------------------------------------------------------------------------
# Pip
# ---------------------------------------------------------------------------


def _pip_install(
    args: list[str], *, log: Optional[LogBuffer] = None, timeout: int = 1800
) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *args]
    return _stream_run(cmd, log=log, timeout=timeout)


def install_python_dependencies(
    repo_root: Path,
    *,
    install_da3: bool = True,
    install_ui: bool = True,
    log: Optional[LogBuffer] = None,
) -> list[StepResult]:
    repo_root = Path(repo_root).resolve()
    results: list[StepResult] = []

    # Step 1: pin numpy<2 first so opencv/open3d stay happy on Colab.
    rc = _pip_install(["--upgrade", "pip"], log=log)
    results.append(StepResult("pip-upgrade", rc == 0, f"pip upgrade rc={rc}"))

    rc = _pip_install(["numpy<2"], log=log)
    results.append(StepResult("numpy-pin", rc == 0, f"numpy<2 rc={rc}"))

    # Step 2: requirements-core.
    core = repo_root / "requirements-core.txt"
    if core.exists():
        rc = _pip_install(["-r", str(core)], log=log)
        results.append(StepResult("requirements-core", rc == 0, f"rc={rc}"))
    else:
        results.append(StepResult("requirements-core", False, f"missing: {core}"))

    # Step 3: requirements-da3 (heavy, transformers stack). Optional.
    if install_da3:
        da3 = repo_root / "requirements-da3.txt"
        if da3.exists():
            rc = _pip_install(["-r", str(da3)], log=log)
            results.append(StepResult("requirements-da3", rc == 0, f"rc={rc}"))
        else:
            results.append(StepResult("requirements-da3", False, f"missing: {da3}"))

    # Step 4: UI deps.
    if install_ui:
        rc = _pip_install(UI_PACKAGES, log=log)
        results.append(StepResult("ui-deps", rc == 0, f"rc={rc}"))

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _probe(name: str) -> StepResult:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "")
        return StepResult(name, True, ver)
    except Exception as exc:  # pragma: no cover - depends on Colab state
        return StepResult(name, False, str(exc))


def validate_environment(*, log: Optional[LogBuffer] = None) -> list[StepResult]:
    out: list[StepResult] = []
    out.append(StepResult("python", True, sys.version.split()[0]))
    for tool in ("ffmpeg", "colmap", "git"):
        path = shutil.which(tool)
        out.append(StepResult(tool, path is not None, path or "not found"))
    for name in CORE_PROBES + GEOMETRY_PROBES + DA3_PROBES + UI_PROBES:
        out.append(_probe(name))
    # GPU probe.
    try:
        import torch  # type: ignore

        cuda = "yes" if torch.cuda.is_available() else "no"
        device = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
        out.append(StepResult("torch", True, f"{torch.__version__} cuda={cuda} dev={device}"))
    except Exception as exc:
        out.append(StepResult("torch", False, str(exc)))

    if log is not None:
        log.banner("Environment validation")
        for r in out:
            mark = "OK " if r.ok else "FAIL"
            log.append(f"  [{mark}] {r.name:20s} {r.detail}")
    return out


def format_validation(results: list[StepResult]) -> str:
    rows = []
    for r in results:
        mark = "OK " if r.ok else "FAIL"
        rows.append(f"[{mark}] {r.name:22s} {r.detail}")
    return "\n".join(rows)
