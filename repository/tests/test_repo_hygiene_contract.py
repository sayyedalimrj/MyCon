from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _repo_visible_files() -> list[str]:
    """Return tracked + untracked non-ignored files.

    `-c safe.directory` is required inside Docker because /workspace is a
    bind mount and Git may reject it as dubious ownership.
    """
    out = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={ROOT}",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def _is_allowed_placeholder(path: str) -> bool:
    return path.endswith("/.gitkeep")


def test_no_generated_runtime_paths_are_visible_to_git_except_placeholders() -> None:
    forbidden_prefixes = (
        "data/",
        "runs/",
        "exports/",
        "model_cache/",
        "models/",
        "ollama_models/",
        "hf_cache/",
        ".venv/",
    )
    bad = [
        path
        for path in _repo_visible_files()
        if path.startswith(forbidden_prefixes) and not _is_allowed_placeholder(path)
    ]
    assert bad == []


def test_no_python_cache_files_are_visible_to_git() -> None:
    bad = [
        path for path in _repo_visible_files()
        if "__pycache__/" in path or path.endswith((".pyc", ".pyo"))
    ]
    assert bad == []


def test_server_env_template_exists_in_safe_location() -> None:
    files = set(_repo_visible_files())
    assert "env/server.env.example" in files
    assert ".env.server.example" not in files
