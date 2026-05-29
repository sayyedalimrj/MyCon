"""Automated model & binary provisioning for the MyCon Colab pipeline.

This module turns the "manual setup" steps documented for the server
deployment into one-call, idempotent, resumable functions suitable for an
unattended Colab run:

* :func:`ensure_system_binaries` — verify/install COLMAP + ffmpeg (apt) and
  report what is available.
* :func:`ensure_ollama` — install the Ollama runtime (official installer)
  and start the local server, so a *real* Qwen-VL VLM can answer Stage 7.5 /
  Stage 10 questions instead of the deterministic mock.
* :func:`ensure_ollama_model` — pull a VLM model into Ollama (cached on
  Drive via ``OLLAMA_MODELS`` so a reconnect does not re-download GBs).
* :func:`ensure_hf_model` — pre-download a Hugging Face model snapshot (e.g.
  a Depth-Anything-V3 checkpoint) into a Drive-backed cache.
* :func:`provision_vlm` — high level helper returning the config overrides
  needed to switch the pipeline onto the real local VLM.

Everything is best-effort and returns a structured :class:`ProvisionResult`
(never raises for "not available"), so the notebook can degrade gracefully
to the mock provider when a heavy model cannot be fetched on a given runtime.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from colab.log_capture import LogBuffer

# Default local Ollama endpoint reachable from the Colab kernel itself.
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_ENDPOINT = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_ENDPOINT}/api/chat"

# A compact, widely-available vision model that runs on a free/standard
# Colab GPU. Qwen3-VL tags are large; this is the practical Colab default.
DEFAULT_VLM_MODEL = "qwen2.5vl:7b"
DEFAULT_VLM_MODEL_SMALL = "qwen2.5vl:3b"


@dataclass
class ProvisionResult:
    name: str
    ok: bool
    detail: str = ""
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{'OK ' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def _log(log: Optional[LogBuffer], msg: str) -> None:
    if log is not None:
        log.append(msg)
    else:
        print(msg, flush=True)


def _run(
    cmd: list[str],
    *,
    log: Optional[LogBuffer],
    timeout: Optional[int] = None,
    env: Optional[dict] = None,
) -> int:
    _log(log, "$ " + " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except FileNotFoundError as exc:
        _log(log, f"[models] command not found: {cmd[0]} ({exc})")
        return 127
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            _log(log, line.rstrip("\n"))
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        _log(log, f"[models] timeout after {timeout}s: {' '.join(cmd)}")
        return 124
    return int(proc.returncode or 0)


# ---------------------------------------------------------------------------
# System binaries
# ---------------------------------------------------------------------------


def ensure_system_binaries(*, log: Optional[LogBuffer] = None) -> ProvisionResult:
    """Verify COLMAP + ffmpeg + git-lfs are present; report which are missing.

    Actual apt installation is owned by :mod:`colab.environment`; this is the
    fast pre-stage gate the runner calls so a heavy SfM stage fails early with
    a clear message instead of deep inside a subprocess.
    """
    found = {tool: shutil.which(tool) for tool in ("colmap", "ffmpeg", "git", "git-lfs")}
    missing = [k for k, v in found.items() if not v]
    ok = not [t for t in ("colmap", "ffmpeg") if not found.get(t)]
    detail = ", ".join(f"{k}={'ok' if v else 'MISSING'}" for k, v in found.items())
    _log(log, f"[models] system binaries: {detail}")
    return ProvisionResult("system_binaries", ok, detail, {"missing": missing, "found": found})


# ---------------------------------------------------------------------------
# Ollama (real local VLM)
# ---------------------------------------------------------------------------


def _ollama_up(timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_ENDPOINT}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def install_ollama(*, log: Optional[LogBuffer] = None, timeout: int = 600) -> ProvisionResult:
    """Install the Ollama runtime via the official install script (idempotent)."""
    if shutil.which("ollama"):
        return ProvisionResult("ollama_install", True, "already installed")
    # Download the official installer to a temp file, then run it. We avoid
    # piping curl|sh so we can stream/inspect failures.
    script = Path("/tmp/install_ollama.sh")
    try:
        with urllib.request.urlopen("https://ollama.com/install.sh", timeout=60) as resp:
            script.write_bytes(resp.read())
    except (urllib.error.URLError, OSError) as exc:
        return ProvisionResult("ollama_install", False, f"download failed: {exc}")
    rc = _run(["sh", str(script)], log=log, timeout=timeout)
    ok = rc == 0 and shutil.which("ollama") is not None
    return ProvisionResult("ollama_install", ok, f"installer rc={rc}")


def start_ollama_server(
    *,
    models_dir: Optional[Path] = None,
    log: Optional[LogBuffer] = None,
    wait_sec: int = 40,
) -> ProvisionResult:
    """Start ``ollama serve`` in the background; wait until it answers.

    ``models_dir`` (typically on Drive) is exported as ``OLLAMA_MODELS`` so a
    pulled model survives a runtime reset and is reused on the next session.
    """
    if not shutil.which("ollama"):
        return ProvisionResult("ollama_serve", False, "ollama not installed")
    if _ollama_up():
        return ProvisionResult("ollama_serve", True, "already running")

    env = os.environ.copy()
    if models_dir is not None:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        env["OLLAMA_MODELS"] = str(models_dir)
    env.setdefault("OLLAMA_HOST", f"{OLLAMA_HOST}:{OLLAMA_PORT}")

    log_path = Path("/tmp/ollama_serve.log")
    try:
        with log_path.open("ab") as sink:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=env,
            )
    except OSError as exc:
        return ProvisionResult("ollama_serve", False, f"spawn failed: {exc}")

    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if _ollama_up():
            _log(log, f"[models] ollama server is up at {OLLAMA_ENDPOINT}")
            return ProvisionResult(
                "ollama_serve", True, "running", {"endpoint": OLLAMA_CHAT_ENDPOINT}
            )
        time.sleep(2)
    return ProvisionResult("ollama_serve", False, f"did not become ready in {wait_sec}s")


def list_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_ENDPOINT}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    models = payload.get("models") or []
    return [str(m.get("name", "")) for m in models if isinstance(m, dict)]


def ensure_ollama_model(
    model: str = DEFAULT_VLM_MODEL,
    *,
    log: Optional[LogBuffer] = None,
    timeout: int = 3600,
) -> ProvisionResult:
    """Pull a model into Ollama if not already present (resumable)."""
    if not shutil.which("ollama"):
        return ProvisionResult("ollama_pull", False, "ollama not installed")
    existing = list_ollama_models()
    # Ollama reports names with an explicit tag (e.g. "qwen2.5vl:7b"); a bare
    # request like "qwen2.5vl" defaults to ":latest".
    wanted = model if ":" in model else f"{model}:latest"
    if model in existing or wanted in existing:
        return ProvisionResult("ollama_pull", True, f"{model} already present")
    rc = _run(["ollama", "pull", model], log=log, timeout=timeout)
    refreshed = list_ollama_models()
    ok = rc == 0 and (model in refreshed or wanted in refreshed)
    return ProvisionResult("ollama_pull", ok, f"pull {model} rc={rc}")


def provision_vlm(
    *,
    model: str = DEFAULT_VLM_MODEL,
    models_dir: Optional[Path] = None,
    log: Optional[LogBuffer] = None,
    pull_timeout: int = 3600,
) -> ProvisionResult:
    """Install Ollama, start the server, pull ``model``; return config overrides.

    On success ``ProvisionResult.data['config_overrides']`` contains the dotted
    config keys to merge so the pipeline uses the real local VLM. On failure
    the caller should keep the deterministic mock provider.
    """
    steps: list[ProvisionResult] = []
    steps.append(install_ollama(log=log))
    if not steps[-1].ok:
        return ProvisionResult("provision_vlm", False, steps[-1].detail, {"steps": [str(s) for s in steps]})
    steps.append(start_ollama_server(models_dir=models_dir, log=log))
    if not steps[-1].ok:
        return ProvisionResult("provision_vlm", False, steps[-1].detail, {"steps": [str(s) for s in steps]})
    steps.append(ensure_ollama_model(model, log=log, timeout=pull_timeout))
    if not steps[-1].ok:
        return ProvisionResult("provision_vlm", False, steps[-1].detail, {"steps": [str(s) for s in steps]})

    overrides = {
        "copilot.vlm.provider": "ollama_local",
        "copilot.vlm.endpoint": OLLAMA_CHAT_ENDPOINT,
        "copilot.vlm.model": model,
        "copilot.vlm.local_only": True,
        "copilot.vlm.fallback_to_mock_when_unavailable": True,
        "copilot.vlm.require_real_vlm": False,
        "vlm_qa.provider": "ollama_local",
        "vlm_qa.endpoint": OLLAMA_CHAT_ENDPOINT,
        "vlm_qa.model": model,
    }
    return ProvisionResult(
        "provision_vlm",
        True,
        f"real VLM ready: {model} @ {OLLAMA_CHAT_ENDPOINT}",
        {"config_overrides": overrides, "endpoint": OLLAMA_CHAT_ENDPOINT, "model": model},
    )


# ---------------------------------------------------------------------------
# Hugging Face snapshots (DA3 checkpoints, etc.)
# ---------------------------------------------------------------------------


def ensure_hf_model(
    repo_id: str,
    *,
    cache_dir: Path,
    allow_patterns: Optional[list[str]] = None,
    revision: Optional[str] = None,
    log: Optional[LogBuffer] = None,
) -> ProvisionResult:
    """Download a Hugging Face model snapshot into ``cache_dir`` (resumable).

    Uses ``huggingface_hub.snapshot_download`` which resumes partial files, so
    a disconnect mid-download is recovered on the next attempt. The cache dir
    should live on (or be mirrored to) Drive so the snapshot persists.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        return ProvisionResult("hf_model", False, f"huggingface_hub unavailable: {exc}")
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            revision=revision,
            resume_download=True,
        )
    except Exception as exc:  # network / gated repo / etc.
        _log(log, f"[models] hf snapshot failed for {repo_id}: {exc}")
        return ProvisionResult("hf_model", False, f"{type(exc).__name__}: {exc}", {"repo_id": repo_id})
    _log(log, f"[models] hf snapshot ready: {repo_id} -> {local_path}")
    return ProvisionResult("hf_model", True, f"{repo_id} cached", {"path": str(local_path), "repo_id": repo_id})
