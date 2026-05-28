from __future__ import annotations

from pathlib import Path


def test_run_progress_wraps_decision_enrichment() -> None:
    text = Path("pipeline/stage_09_progress/run_progress.py").read_text(encoding="utf-8")
    assert "enrich_progress_decisions_from_config" in text
    assert "def _run_progress_unenriched(" in text
    assert "def run_progress(*args, **kwargs):" in text
    assert "Stage 9 decision enrichment skipped" in text
