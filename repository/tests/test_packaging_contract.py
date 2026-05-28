from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _dockerfile_copy_sources(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    sources: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("COPY "):
            continue
        parts = line.split()
        if len(parts) >= 3:
            sources.extend(parts[1:-1])
    return sources


def test_root_level_requirements_expected_by_dockerfiles_exist() -> None:
    assert (ROOT / "requirements-core.txt").exists()
    assert (ROOT / "requirements-da3.txt").exists()


def test_documented_requirements_folder_mirrors_root_requirements() -> None:
    for name in [
        "requirements-core.txt",
        "requirements-da3.txt",
        "requirements-dev.txt",
        "requirements-service.txt",
    ]:
        root_file = ROOT / name
        folder_file = ROOT / "requirements" / name
        assert root_file.exists(), root_file
        assert folder_file.exists(), folder_file
        assert root_file.read_text(encoding="utf-8").strip() == folder_file.read_text(encoding="utf-8").strip()


def test_dockerfile_copy_sources_exist_in_build_context() -> None:
    for dockerfile in [
        ROOT / "docker" / "Dockerfile.core-dev",
        ROOT / "docker" / "Dockerfile.da3-dev",
    ]:
        assert dockerfile.exists(), dockerfile
        for source in _dockerfile_copy_sources(dockerfile):
            # Skip advanced Docker syntax or remote-ish placeholders if any.
            if source.startswith("--") or "$" in source:
                continue
            assert (ROOT / source).exists(), f"{dockerfile} copies missing build-context file: {source}"
