"""Uniform artifact-provenance envelope for stage outputs.

Goals
-----

Every stage in the pipeline already writes a structured JSON report under
``runs/<run_id>/reports/<stage_basename>.json``. This module adds a
**provenance block** that can be attached to any of those reports without
touching the stage's existing report-shape, so:

1. A future GUI or REST API can answer "what config produced this artifact?"
   "what code version?" "what seeds?" with one read.
2. Two runs that claim to be identical can be cryptographically distinguished
   by their config hash.
3. The artifact aggregator (``scripts/aggregate_run_metrics.py``) can surface
   these fields without each stage having to plumb its own discovery hook.

Design constraints
------------------

- **No new dependencies.** Uses ``hashlib``, ``json``, ``platform``,
  ``socket``, ``subprocess``, ``time`` from the stdlib, plus the existing
  :mod:`pipeline.common.config` typing.
- **No I/O of its own.** Building an envelope is pure; the caller decides
  where (and whether) to write it.
- **Best-effort metadata gathering.** Git / hostname / Python lookups are
  wrapped so a missing ``git`` binary or unreadable hostname never blocks
  a real pipeline run.
- **Stable serialization.** Config hash is computed over a canonical JSON
  rendering with sorted keys and stripped path-prefixes for the
  ``project.root``-relative invariants documented in
  ``docs/data_contracts.md``.

Attach-points
-------------

Stages that opt in call ``attach_provenance(report, envelope)`` right before
they write their report JSON. The envelope is added under the key
``"provenance"`` and never overwrites an existing block.

Stages that do not opt in are still discoverable: their reports remain
unchanged, and the aggregator simply reports ``provenance: null`` for them.
This keeps Phase 1 invasive only where it is genuinely additive value.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeline.common.config import PipelineConfig

__all__ = [
    "ArtifactEnvelope",
    "compute_config_hash",
    "current_provenance",
    "attach_provenance",
    "git_sha",
    "git_dirty",
    "environment_metadata",
]


# Code version sentinel. The Phase 1 PR introduces the provenance schema; we
# bump this any time the envelope layout itself changes in a backwards-
# incompatible way. Aggregators should warn when they encounter an unknown
# code_version so consumers know to update.
PROVENANCE_SCHEMA_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Best-effort environment helpers.
#
# Every helper here returns a value (str / int / bool / None) rather than
# raising. A pipeline run must never fail because git is unavailable.
# ---------------------------------------------------------------------------


def _run_git(args: Sequence[str], cwd: Path | None = None) -> str | None:
    """Run a short git command and return its stripped stdout, or None on any failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def git_sha(cwd: Path | None = None) -> str | None:
    """Return the current git commit SHA (40 hex), or None if unavailable."""
    return _run_git(["rev-parse", "HEAD"], cwd=cwd)


def git_dirty(cwd: Path | None = None) -> bool | None:
    """Return True if the working tree has uncommitted changes, False if clean,
    or None if git status cannot be determined.
    """
    out = _run_git(["status", "--porcelain"], cwd=cwd)
    if out is None:
        return None
    return bool(out)


def environment_metadata() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the run environment.

    Fields included
    ---------------
    - python_version : full ``sys.version`` string
    - python_implementation : CPython / PyPy / etc.
    - platform : ``platform.platform()`` (kernel + distro info)
    - hostname : best-effort from :func:`socket.gethostname`
    - cpu_count : ``os.cpu_count()`` (may be None on unusual kernels)
    - pid : current process PID

    Values that cannot be obtained safely are returned as None rather than
    omitted, so consumers always see the full key set.
    """
    try:
        hostname: str | None = socket.gethostname()
    except OSError:
        hostname = None
    return {
        "python_version": sys.version.split(" ", 1)[0],
        "python_full_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "hostname": hostname,
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
    }


# ---------------------------------------------------------------------------
# Config hashing.
#
# The hash is over the *resolved* config dict, not the YAML text, so two
# YAMLs that parse to the same dict (e.g. different key order, different
# whitespace) hash identically. We strip ``project.root`` because two runs
# of the same logical config from different working directories should be
# considered the same config; ``project.root`` is operator-environment, not
# scientific intent.
# ---------------------------------------------------------------------------


_CONFIG_HASH_STRIP_KEYS: tuple[tuple[str, ...], ...] = (
    ("project", "root"),
)


def _strip_paths(data: Any, paths: Sequence[Sequence[str]]) -> Any:
    """Return a deep copy of ``data`` with each ``paths`` entry removed.

    Each ``paths`` entry is a tuple of dotted-key parts, e.g.
    ``("project", "root")``. Missing keys are tolerated silently. This
    function never mutates its input.
    """
    if not isinstance(data, Mapping):
        return data
    out: dict[str, Any] = {}
    for k, v in data.items():
        out[k] = _strip_paths(v, paths) if isinstance(v, Mapping) else v
    for path in paths:
        if not path:
            continue
        cur: Any = out
        for part in path[:-1]:
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if isinstance(cur, dict):
            cur.pop(path[-1], None)
    return out


def compute_config_hash(cfg: PipelineConfig | Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 hex hash of the given config.

    The hash covers the resolved config dict, with:
    - keys sorted lexicographically at every depth
    - separators ``(",", ":")`` for canonical JSON
    - non-JSON values (paths, etc.) coerced via ``str``
    - ``project.root`` stripped before hashing (see module docstring)

    Two PipelineConfig instances with identical scientific intent therefore
    produce identical hashes.
    """
    if isinstance(cfg, PipelineConfig):
        data: Mapping[str, Any] = cfg.data
    elif isinstance(cfg, Mapping):
        data = cfg
    else:
        raise TypeError(f"compute_config_hash expects a PipelineConfig or Mapping, got {type(cfg).__name__}")

    stripped = _strip_paths(data, _CONFIG_HASH_STRIP_KEYS)
    canonical = json.dumps(stripped, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The envelope itself.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactEnvelope:
    """Uniform metadata block stamped onto every stage report that opts in.

    Fields
    ------
    schema_version : str
        Version of this envelope schema; consumers MUST check it.
    stage : str
        Canonical stage name, matching :class:`StageDescriptor.name`.
    artifact_name : str
        Human-readable artifact identifier within the stage. Usually the
        report file basename without ``.json``.
    created_at_unix : float
        ``time.time()`` at envelope construction.
    created_at_iso : str
        ISO 8601 UTC timestamp; redundant with ``created_at_unix`` but
        present so reports are inspectable without timestamp tooling.
    config_hash : str
        SHA-256 hex of the resolved config (see :func:`compute_config_hash`).
    config_path : str | None
        Path to the loaded YAML, if known.
    git_sha : str | None
        Current commit SHA if running inside a git tree.
    git_dirty : bool | None
        True if the working tree has uncommitted changes.
    project_seed : int
        Project-wide RNG seed (``project.random_seed``).
    seeds : Mapping[str, int]
        Stage-derived seeds, keyed by label. Stages that derive seeds via
        :mod:`pipeline.common.determinism` should include them here for
        full reproducibility.
    inputs : Mapping[str, str]
        Logical name → resolved path string for every declared input the
        stage actually consumed.
    code_version : str
        Optional human-readable version tag (e.g. a branch name); falls
        back to the schema version when no other tag is available.
    environment : Mapping[str, Any]
        Snapshot returned by :func:`environment_metadata`.
    notes : Sequence[str]
        Free-form audit notes the stage wants to attach.
    """

    schema_version: str
    stage: str
    artifact_name: str
    created_at_unix: float
    created_at_iso: str
    config_hash: str
    config_path: str | None
    git_sha: str | None
    git_dirty: bool | None
    project_seed: int
    seeds: Mapping[str, int]
    inputs: Mapping[str, str]
    code_version: str
    environment: Mapping[str, Any]
    notes: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict; usable as a report ``provenance`` block."""
        # asdict deep-converts nested dataclasses but here all fields are
        # primitives or mappings/sequences of primitives, so a manual cast is
        # both faster and clearer than asdict + post-processing.
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "artifact_name": self.artifact_name,
            "created_at_unix": self.created_at_unix,
            "created_at_iso": self.created_at_iso,
            "config_hash": self.config_hash,
            "config_path": self.config_path,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "project_seed": int(self.project_seed),
            "seeds": dict(self.seeds),
            "inputs": dict(self.inputs),
            "code_version": self.code_version,
            "environment": dict(self.environment),
            "notes": list(self.notes),
        }


def current_provenance(
    cfg: PipelineConfig,
    stage: str,
    *,
    artifact_name: str,
    inputs: Mapping[str, Path] | Mapping[str, str] | None = None,
    seeds: Mapping[str, int] | None = None,
    notes: Sequence[str] = (),
    code_version: str | None = None,
    repo_root: Path | None = None,
) -> ArtifactEnvelope:
    """Construct an :class:`ArtifactEnvelope` for the *current moment*.

    Parameters
    ----------
    cfg : PipelineConfig
        The config the stage is running against.
    stage : str
        Canonical stage name.
    artifact_name : str
        Identifier for the artifact this envelope describes (typically the
        report basename without extension).
    inputs : Mapping[str, Path] | Mapping[str, str] | None
        Stage inputs the caller actually consumed, keyed by logical name.
        Path objects are stringified for JSON-friendliness.
    seeds : Mapping[str, int] | None
        Per-call seeds the stage derived (for example via
        ``derived_seed``). Use this so reproducibility is checkable from the
        report alone.
    notes : Sequence[str]
        Free-form audit notes.
    code_version : str | None
        Optional explicit code version tag (e.g. a branch or release tag).
        When None, defaults to the envelope schema version.
    repo_root : Path | None
        Optional working directory for git lookups; defaults to the parent
        of the loaded config file (which is typically the repository root
        when configs live under ``configs/``).
    """
    now = time.time()
    seeds_resolved: dict[str, int] = {str(k): int(v) for k, v in (seeds or {}).items()}
    inputs_resolved: dict[str, str] = {str(k): str(v) for k, v in (inputs or {}).items()}
    cwd_for_git = repo_root or (cfg.path.parent if isinstance(cfg, PipelineConfig) and cfg.path else None)
    return ArtifactEnvelope(
        schema_version=PROVENANCE_SCHEMA_VERSION,
        stage=stage,
        artifact_name=artifact_name,
        created_at_unix=now,
        created_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        config_hash=compute_config_hash(cfg),
        config_path=str(cfg.path) if isinstance(cfg, PipelineConfig) else None,
        git_sha=git_sha(cwd=cwd_for_git),
        git_dirty=git_dirty(cwd=cwd_for_git),
        project_seed=int(cfg.require("project.random_seed")) if isinstance(cfg, PipelineConfig) else 0,
        seeds=seeds_resolved,
        inputs=inputs_resolved,
        code_version=code_version or PROVENANCE_SCHEMA_VERSION,
        environment=environment_metadata(),
        notes=tuple(notes),
    )


def attach_provenance(report: dict[str, Any], envelope: ArtifactEnvelope) -> dict[str, Any]:
    """Add the envelope under ``report["provenance"]`` and return the report.

    If ``report`` already has a ``provenance`` key, the existing block is
    preserved under ``provenance.previous`` and the new envelope replaces
    the top-level value. This is to avoid silent truncation of any custom
    provenance a stage might already maintain.
    """
    if not isinstance(report, dict):
        raise TypeError(f"attach_provenance expects a dict report, got {type(report).__name__}")
    new_block: dict[str, Any] = envelope.to_dict()
    existing = report.get("provenance")
    if isinstance(existing, dict):
        new_block.setdefault("previous", existing)
    report["provenance"] = new_block
    return report
