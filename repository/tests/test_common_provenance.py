"""Tests for ``pipeline.common.provenance`` artifact-envelope layer.

Pinned contracts:

- Config hash is deterministic across calls on the same config.
- Config hash is independent of ``project.root`` (operator-environment
  invariance documented in the module).
- Config hash *changes* when any other key changes.
- Envelope serialization round-trips through JSON.
- ``attach_provenance`` does not silently overwrite an existing block.
- Best-effort metadata helpers (git_sha, git_dirty, environment_metadata)
  do not raise when their underlying binaries / sockets are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.common.config import PipelineConfig, load_config
from pipeline.common.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    ArtifactEnvelope,
    attach_provenance,
    compute_config_hash,
    current_provenance,
    environment_metadata,
    git_dirty,
    git_sha,
)

CONFIG_PATH = Path("configs/site01.yaml")


# ---------------------------------------------------------------------------
# Config hash determinism / sensitivity
# ---------------------------------------------------------------------------


def test_config_hash_is_deterministic() -> None:
    cfg = load_config(CONFIG_PATH)
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(cfg)
    assert h1 == h2


def test_config_hash_is_64_hex_chars() -> None:
    cfg = load_config(CONFIG_PATH)
    h = compute_config_hash(cfg)
    # SHA-256 hex digest → 64 lowercase hex chars.
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_config_hash_is_invariant_under_project_root_change() -> None:
    """Two operators on different machines must compute the same hash for the
    same scientific config — that is the whole point of stripping project.root.
    """
    cfg_a = load_config(CONFIG_PATH)
    cfg_b = load_config(CONFIG_PATH, root_override="/different/operator/path")
    assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)


def test_config_hash_changes_for_unrelated_field_change() -> None:
    cfg_a = load_config(CONFIG_PATH)
    # Build a copy with one field changed.
    data_modified = json.loads(json.dumps(cfg_a.data, default=str))
    data_modified["project"]["random_seed"] = 999
    cfg_b = PipelineConfig(path=cfg_a.path, data=data_modified)
    assert compute_config_hash(cfg_a) != compute_config_hash(cfg_b)


def test_config_hash_accepts_plain_dict() -> None:
    """Useful for testing — a dict-shaped config must hash the same as the
    PipelineConfig that contains it.
    """
    cfg = load_config(CONFIG_PATH)
    h_cfg = compute_config_hash(cfg)
    h_dict = compute_config_hash(dict(cfg.data))
    assert h_cfg == h_dict


def test_config_hash_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        compute_config_hash("not a config")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ArtifactEnvelope
# ---------------------------------------------------------------------------


def test_envelope_round_trips_through_json() -> None:
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(
        cfg,
        stage="stage_09_progress",
        artifact_name="progress_summary",
        inputs={"scan_aligned": Path("/workspace/data/bim/aligned/site01/scan_aligned.ply")},
        seeds={"bootstrap": 42, "ransac": 43},
        notes=["test note"],
    )
    encoded = json.dumps(envelope.to_dict())
    decoded = json.loads(encoded)
    assert decoded["stage"] == "stage_09_progress"
    assert decoded["artifact_name"] == "progress_summary"
    assert decoded["config_hash"] == compute_config_hash(cfg)
    assert decoded["seeds"]["bootstrap"] == 42
    # Path was stringified during envelope construction.
    assert isinstance(decoded["inputs"]["scan_aligned"], str)


def test_envelope_includes_schema_version() -> None:
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress")
    assert envelope.schema_version == PROVENANCE_SCHEMA_VERSION


def test_envelope_includes_environment_block() -> None:
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress")
    env = envelope.environment
    assert "python_version" in env
    assert "platform" in env
    assert "pid" in env


def test_envelope_records_config_path() -> None:
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress")
    assert envelope.config_path is not None
    assert envelope.config_path.endswith("site01.yaml")


def test_envelope_is_frozen_dataclass() -> None:
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress")
    with pytest.raises(Exception):
        envelope.stage = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# attach_provenance
# ---------------------------------------------------------------------------


def test_attach_provenance_adds_block_and_returns_report() -> None:
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress")
    report = {"stage": "stage_09_progress", "status": "complete"}
    out = attach_provenance(report, envelope)
    assert out is report  # in-place + return for chaining
    assert "provenance" in report
    assert report["provenance"]["stage"] == "stage_09_progress"
    assert report["provenance"]["config_hash"] == compute_config_hash(cfg)


def test_attach_provenance_preserves_existing_block_under_previous() -> None:
    """If a stage already had a provenance block, it must be preserved under
    ``provenance.previous`` so we never silently lose audit trail.
    """
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress")
    report = {
        "stage": "stage_09_progress",
        "provenance": {"stage": "stage_09_progress", "schema_version": "0.9", "old": True},
    }
    attach_provenance(report, envelope)
    assert report["provenance"]["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert report["provenance"]["previous"]["schema_version"] == "0.9"
    assert report["provenance"]["previous"]["old"] is True


def test_attach_provenance_rejects_non_dict_report() -> None:
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress")
    with pytest.raises(TypeError):
        attach_provenance(["not", "a", "dict"], envelope)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Best-effort env helpers
# ---------------------------------------------------------------------------


def test_environment_metadata_is_complete_dict() -> None:
    env = environment_metadata()
    for key in ("python_version", "python_full_version", "platform", "hostname", "cpu_count", "pid"):
        assert key in env
    assert isinstance(env["pid"], int)


def test_git_sha_returns_str_or_none(tmp_path: Path) -> None:
    """Calling git in a non-git directory returns None, not a crash."""
    out = git_sha(cwd=tmp_path)
    # Either returns None (not a git tree) or a 40-char hex SHA in this repo.
    if out is not None:
        assert len(out) == 40
        assert all(c in "0123456789abcdef" for c in out)


def test_git_dirty_returns_bool_or_none(tmp_path: Path) -> None:
    out = git_dirty(cwd=tmp_path)
    assert out is None or isinstance(out, bool)


def test_envelope_handles_empty_inputs_and_seeds() -> None:
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress")
    assert envelope.inputs == {}
    assert envelope.seeds == {}
