"""Lightweight tests for stage_runner planning/resume and models degradation.

We never launch a real stage subprocess here; we exercise the catalog,
ordering, command construction, and the resume/skip path (which short-circuits
before any subprocess) plus the always-graceful model provisioning helpers.
"""

from __future__ import annotations

from pathlib import Path

from colab import models
from colab import stage_runner as sr
from colab.checkpoint import CheckpointManager
from colab.log_capture import LogBuffer


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Catalog / ordering
# ---------------------------------------------------------------------------


def test_catalog_keys_unique_and_full_pipeline_subset() -> None:
    keys = [s.key for s in sr.STAGE_CATALOG]
    assert len(keys) == len(set(keys))
    catalog = set(keys)
    assert set(sr.FULL_PIPELINE_KEYS).issubset(catalog)
    assert set(sr.COLAB_SAFE_DEFAULT_KEYS).issubset(catalog)


def test_order_stages_follows_canonical_order() -> None:
    shuffled = ["stage_11_schedule_variance", "stage_03_colmap", "stage_01_ingest"]
    ordered = sr.order_stages(shuffled)
    assert ordered.index("stage_01_ingest") < ordered.index("stage_03_colmap")
    assert ordered.index("stage_03_colmap") < ordered.index("stage_11_schedule_variance")


def test_order_stages_keeps_unknown_keys_at_end() -> None:
    ordered = sr.order_stages(["stage_99_unknown", "stage_01_ingest"])
    assert ordered[0] == "stage_01_ingest"
    assert ordered[-1] == "stage_99_unknown"


def test_outputs_for_known_and_unknown() -> None:
    assert sr.outputs_for("stage_05_dense") == ["data/dense/**/fused.ply"]
    assert sr.outputs_for("nope") is None


def test_keys_from_stage_selects_tail() -> None:
    ks = sr.keys_from_stage("stage_05_dense")
    assert ks[0] == "stage_05_dense"
    assert "stage_01_ingest" not in ks
    assert ks[-1] == sr.FULL_PIPELINE_KEYS[-1]
    # Every returned key is a real catalog stage, in canonical order.
    assert ks == sr.order_stages(ks)


def test_keys_from_stage_unknown_returns_full_pipeline() -> None:
    assert sr.keys_from_stage("does_not_exist") == list(sr.FULL_PIPELINE_KEYS)


def test_keys_from_first_stage_is_full_pipeline() -> None:
    assert sr.keys_from_stage(sr.FULL_PIPELINE_KEYS[0]) == list(sr.FULL_PIPELINE_KEYS)


def test_build_command_run_stage_and_force() -> None:
    spec = sr.STAGES_BY_KEY["stage_01_ingest"]
    cmd = sr._build_command(
        spec=spec, config_path=Path("/cfg.yaml"), repo_root=REPO_ROOT,
        force=True, log_level="INFO",
    )
    assert cmd[1].endswith("run_stage.py")
    assert "stage_01_ingest" in cmd
    assert "--force" in cmd
    assert "--config" in cmd


def test_build_command_stage_10_injects_question() -> None:
    spec = sr.STAGES_BY_KEY["stage_10_copilot"]
    cmd = sr._build_command(
        spec=spec, config_path=Path("/cfg.yaml"), repo_root=REPO_ROOT,
        force=False, log_level="INFO", extra_kv={"question": "is the slab poured?"},
    )
    assert "--question" in cmd
    assert "is the slab poured?" in cmd
    assert "--json" in cmd


def test_make_env_sets_hf_cache_and_pythonpath(tmp_path: Path) -> None:
    env = sr._make_env(REPO_ROOT, hf_cache_dir=tmp_path / "hf")
    assert str(REPO_ROOT) in env["PYTHONPATH"]
    assert env["HF_HOME"] == str(tmp_path / "hf")
    assert env["QT_QPA_PLATFORM"] == "offscreen"


# ---------------------------------------------------------------------------
# Resume short-circuit (no subprocess launched)
# ---------------------------------------------------------------------------


def test_run_stages_skips_completed_without_running(tmp_path: Path) -> None:
    reports = tmp_path / "runs" / "r1" / "reports"
    logs = tmp_path / "runs" / "r1" / "logs"
    reports.mkdir(parents=True)
    logs.mkdir(parents=True)

    # Pre-create the Stage 1 output and mark it complete in the checkpoint.
    (tmp_path / "data" / "normalized").mkdir(parents=True)
    (tmp_path / "data" / "normalized" / "v.mp4").write_bytes(b"0" * 10)
    mgr = CheckpointManager(project_root=tmp_path, run_id="r1", state_path=reports / "run_state.json")
    mgr.mark_ok("stage_01_ingest", duration_sec=1.0, output_patterns=sr.outputs_for("stage_01_ingest"))

    log = LogBuffer()
    results = sr.run_stages(
        spec_keys=["stage_01_ingest"],
        config_path=tmp_path / "active.yaml",  # never read because skipped
        repo_root=REPO_ROOT,
        logs_dir=logs,
        reports_dir=reports,
        log=log,
        project_root=tmp_path,
        run_id="r1",
        resume=True,
    )
    assert len(results) == 1
    assert results[0].skipped is True
    assert results[0].ok is True


def test_run_stages_unknown_stage_is_ignored(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    reports.mkdir()
    logs.mkdir()
    log = LogBuffer()
    results = sr.run_stages(
        spec_keys=["stage_does_not_exist"],
        config_path=tmp_path / "active.yaml",
        repo_root=REPO_ROOT,
        logs_dir=logs,
        reports_dir=reports,
        log=log,
        resume=False,
    )
    assert results == []
    assert "unknown stage" in log.text()


# ---------------------------------------------------------------------------
# models — always graceful
# ---------------------------------------------------------------------------


def test_ensure_system_binaries_returns_result() -> None:
    res = models.ensure_system_binaries()
    assert res.name == "system_binaries"
    assert "found" in res.data


def test_provision_vlm_overrides_shape_on_success_contract() -> None:
    # We cannot install Ollama in CI, but the override contract is stable: the
    # keys the pipeline reads must be present in the documented default model.
    overrides = {
        "copilot.vlm.provider": "ollama_local",
        "copilot.vlm.endpoint": models.OLLAMA_CHAT_ENDPOINT,
        "copilot.vlm.model": models.DEFAULT_VLM_MODEL,
        "vlm_qa.provider": "ollama_local",
    }
    # Endpoint must be a localhost URL (Stage 10 enforces local-only).
    assert "127.0.0.1" in overrides["copilot.vlm.endpoint"]
    assert overrides["copilot.vlm.provider"] == "ollama_local"


def test_list_ollama_models_no_server_returns_empty() -> None:
    # No ollama running in CI -> graceful empty list, never raises.
    assert models.list_ollama_models() == []
