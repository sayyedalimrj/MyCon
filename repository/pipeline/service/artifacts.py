"""Artifact discovery for finished runs.

The run reports each stage already writes under
``<project_root>/runs/<run_id>/reports/`` are the canonical artifact set
the API exposes to the GUI. This module walks that directory, matches
files against the registry's per-stage ``report_basename``, and returns a
JSON-friendly summary that includes:

- the resolved file path,
- file size and modification time,
- the stage's reported ``status`` (top-level field of every report),
- the parsed ``provenance`` block when present (the new uniform envelope
  introduced in Phase 1),
- a small preview (top-level non-provenance keys) so the GUI can render a
  one-line summary without downloading the whole report.

Intentional non-features
------------------------

- This module does **not** parse stage-specific schemas (deviation_summary,
  registration_quality, etc.). Each stage's report has its own shape; the
  GUI is responsible for rendering whatever fields it knows about. This
  module's job is *enumeration*, not *interpretation*.
- This module does **not** mutate any artifact. It is read-only with
  respect to disk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.common.registry import STAGE_REGISTRY, StageRegistry


LOGGER = logging.getLogger(__name__)


__all__ = [
    "ArtifactSummary",
    "discover_run_artifacts",
]


# When extracting a "preview" from a report we cap its size to keep API
# responses cheap. The full report is always available via the file path.
_PREVIEW_KEY_LIMIT: int = 12
_PREVIEW_VALUE_CHAR_LIMIT: int = 200


@dataclass(frozen=True)
class ArtifactSummary:
    """One artifact (a stage's report) discovered for a run."""

    stage: str
    """Canonical stage name from the registry."""

    artifact_path: str
    """Absolute path to the report file as a POSIX-style string."""

    artifact_basename: str
    """Filename only; equals ``StageDescriptor.report_basename``."""

    exists: bool
    """True if the file is present on disk."""

    size_bytes: int
    """File size in bytes; 0 when ``exists`` is False."""

    modified_at_unix: float | None
    """``stat().st_mtime`` if the file exists, else None."""

    status: str | None
    """The report's top-level ``status`` field, if present."""

    provenance: Mapping[str, Any] | None
    """Parsed provenance block, if the report contains one."""

    preview: Mapping[str, Any]
    """Capped subset of non-``provenance`` top-level keys for GUI summaries."""

    parse_error: str | None = None
    """When the file existed but could not be parsed, the human-readable reason."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "artifact_path": self.artifact_path,
            "artifact_basename": self.artifact_basename,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "modified_at_unix": self.modified_at_unix,
            "status": self.status,
            "provenance": dict(self.provenance) if self.provenance is not None else None,
            "preview": dict(self.preview),
            "parse_error": self.parse_error,
        }


def discover_run_artifacts(
    run_id: str,
    *,
    project_root: Path,
    registry: StageRegistry | None = None,
) -> list[ArtifactSummary]:
    """Return one :class:`ArtifactSummary` per stage that declares a report.

    Stages whose ``report_basename`` is None (currently Stage 10) are
    skipped — they have no canonical report file. Stages that *should*
    have a report but whose file is absent are returned with
    ``exists=False`` so the GUI can render a "not run" state without
    pretending the file is missing for some other reason.
    """
    reg = registry or STAGE_REGISTRY
    reports_dir = project_root / "runs" / run_id / "reports"

    summaries: list[ArtifactSummary] = []
    for descriptor in reg:
        basename = descriptor.report_basename
        if not basename:
            continue
        path = reports_dir / basename
        summary = _summarize(descriptor.name, path, basename)
        summaries.append(summary)
    return summaries


def _summarize(stage_name: str, path: Path, basename: str) -> ArtifactSummary:
    if not path.exists():
        return ArtifactSummary(
            stage=stage_name,
            artifact_path=path.as_posix(),
            artifact_basename=basename,
            exists=False,
            size_bytes=0,
            modified_at_unix=None,
            status=None,
            provenance=None,
            preview={},
        )
    try:
        stat = path.stat()
    except OSError as exc:
        return ArtifactSummary(
            stage=stage_name,
            artifact_path=path.as_posix(),
            artifact_basename=basename,
            exists=True,
            size_bytes=0,
            modified_at_unix=None,
            status=None,
            provenance=None,
            preview={},
            parse_error=f"stat failed: {exc}",
        )
    size_bytes = int(stat.st_size)
    modified_at_unix = float(stat.st_mtime)

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ArtifactSummary(
            stage=stage_name,
            artifact_path=path.as_posix(),
            artifact_basename=basename,
            exists=True,
            size_bytes=size_bytes,
            modified_at_unix=modified_at_unix,
            status=None,
            provenance=None,
            preview={},
            parse_error=f"read failed: {exc}",
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ArtifactSummary(
            stage=stage_name,
            artifact_path=path.as_posix(),
            artifact_basename=basename,
            exists=True,
            size_bytes=size_bytes,
            modified_at_unix=modified_at_unix,
            status=None,
            provenance=None,
            preview={},
            parse_error=f"json decode failed: {exc}",
        )

    if not isinstance(parsed, dict):
        return ArtifactSummary(
            stage=stage_name,
            artifact_path=path.as_posix(),
            artifact_basename=basename,
            exists=True,
            size_bytes=size_bytes,
            modified_at_unix=modified_at_unix,
            status=None,
            provenance=None,
            preview={},
            parse_error="report root is not a JSON object",
        )

    provenance = parsed.get("provenance") if isinstance(parsed.get("provenance"), dict) else None
    status = parsed.get("status") if isinstance(parsed.get("status"), str) else None
    preview = _build_preview(parsed)

    return ArtifactSummary(
        stage=stage_name,
        artifact_path=path.as_posix(),
        artifact_basename=basename,
        exists=True,
        size_bytes=size_bytes,
        modified_at_unix=modified_at_unix,
        status=status,
        provenance=provenance,
        preview=preview,
    )


def _build_preview(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a capped subset of top-level keys for GUI summaries.

    The ``provenance`` block is excluded because callers already receive it
    as a separate field. Long string values are truncated.
    """
    preview: dict[str, Any] = {}
    for key, value in report.items():
        if key in {"provenance", "outputs", "inputs"}:
            continue
        if len(preview) >= _PREVIEW_KEY_LIMIT:
            break
        preview[key] = _summarize_value(value)
    return preview


def _summarize_value(value: Any) -> Any:
    """Recursively cap nested structures so the preview is bounded in size."""
    if isinstance(value, str):
        return value if len(value) <= _PREVIEW_VALUE_CHAR_LIMIT else value[:_PREVIEW_VALUE_CHAR_LIMIT] + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_summarize_value(v) for v in list(value)[:6]]
    if isinstance(value, Mapping):
        return {str(k): _summarize_value(v) for k, v in list(value.items())[:6]}
    return repr(value)
