"""Unusable metric alignment must never produce a successful Stage-8 CLI exit."""
import sys
from pathlib import Path
import pytest
from pipeline.stage_08_bim_eval import run_metric_alignment as cli

pytestmark = pytest.mark.lightweight

def test_cli_returns_blocked_when_no_survey_anchors(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "site.yaml"
    cfg.write_text(f"project:\n  root: {tmp_path.as_posix()}\n")
    monkeypatch.setattr(sys, "argv", ["metric_alignment", "--config", str(cfg)])
    monkeypatch.setattr(cli, "build_metric_alignment_report", lambda **kwargs: {
        "status": "skipped_insufficient_anchors", "confidence": "low",
        "usable_registration_anchor_count": 0, "can_feed_stage8": False,
        "quality_gate": {"passed": False},
    })
    assert cli.main() == 2
    assert "STAGE_08_METRIC_ALIGNMENT_BLOCKED" in capsys.readouterr().out

def test_cli_returns_success_for_approved_alignment(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "site.yaml"
    cfg.write_text(f"project:\n  root: {tmp_path.as_posix()}\n")
    monkeypatch.setattr(sys, "argv", ["metric_alignment", "--config", str(cfg)])
    monkeypatch.setattr(cli, "build_metric_alignment_report", lambda **kwargs: {
        "status": "ok", "confidence": "high",
        "usable_registration_anchor_count": 4, "can_feed_stage8": True,
        "quality_gate": {"passed": True},
    })
    assert cli.main() == 0
    assert "STAGE_08_METRIC_ALIGNMENT_OK" in capsys.readouterr().out
