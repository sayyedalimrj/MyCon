from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


REQUIRED_HANDOFF_FILES = [
    "requirements-core.txt",
    "requirements-da3.txt",
    "requirements-dev.txt",
    "requirements/requirements-core.txt",
    "requirements/requirements-da3.txt",
    "requirements/requirements-dev.txt",
    "env/server.env.example",
    "docker/docker-compose.yml",
    "docker/Dockerfile.core-dev",
    "configs/site01.yaml",
    "docs/server_handoff_checklist.md",
    "docs/current_project_status.md",
    "docs/operational_readiness_matrix.md",
]

FORBIDDEN_PREFIXES = (
    "data/",
    "runs/",
    "exports/",
    "model_cache/",
    "models/",
    "ollama_models/",
    "hf_cache/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".pytest_cache/",
)

FORBIDDEN_PARTS = (
    "/__pycache__/",
    "/.pytest_cache/",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_ls_files(root: Path) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "ls-files"],
            cwd=root,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git ls-files failed: {exc}") from exc

    return [line.strip() for line in out.splitlines() if line.strip()]


def _is_allowed(path: str) -> bool:
    if path.startswith(FORBIDDEN_PREFIXES):
        return False
    if any(part in path for part in FORBIDDEN_PARTS):
        return False
    if path.endswith((".pyc", ".pyo")):
        return False
    return True


def collect_handoff_files(root: Path) -> list[str]:
    tracked = _git_ls_files(root)
    files = [p for p in tracked if _is_allowed(p)]

    missing_required = [
        p for p in REQUIRED_HANDOFF_FILES
        if not (root / p).exists()
    ]
    if missing_required:
        raise FileNotFoundError(
            "Required handoff files are missing from working tree:\n"
            + "\n".join(missing_required)
        )

    not_tracked = [
        p for p in REQUIRED_HANDOFF_FILES
        if p not in tracked
    ]
    if not_tracked:
        raise RuntimeError(
            "Required handoff files exist but are not tracked by git:\n"
            + "\n".join(not_tracked)
        )

    omitted_required = [
        p for p in REQUIRED_HANDOFF_FILES
        if p not in files
    ]
    if omitted_required:
        raise RuntimeError(
            "Required handoff files are excluded by export filters:\n"
            + "\n".join(omitted_required)
        )

    return sorted(files)


def write_zip(root: Path, output: Path) -> list[str]:
    files = collect_handoff_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(root / rel, rel)

    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a self-contained server handoff source ZIP.")
    parser.add_argument("--output", default="dist/construction-progress-ai-bim_server_handoff.zip")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    root = _repo_root()

    if args.list_only:
        files = collect_handoff_files(root)
        for item in files:
            print(item)
        print(f"HANDOFF_FILE_COUNT={len(files)}")
        return 0

    output = root / args.output
    files = write_zip(root, output)

    print("SERVER_HANDOFF_ZIP_OK")
    print(f"output={output}")
    print(f"files={len(files)}")
    for required in REQUIRED_HANDOFF_FILES:
        print(f"required:{required}=included")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
