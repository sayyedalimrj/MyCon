from __future__ import annotations

from pathlib import Path


def test_stage_10_api_attaches_answer_validation_contract() -> None:
    text = Path("pipeline/stage_10_copilot/api.py").read_text(encoding="utf-8")
    assert "validate_copilot_answer_payload" in text
    assert "def attach_answer_validation(" in text
    assert "def _ask_copilot_unvalidated(" in text
    assert "def ask_copilot(*args, **kwargs):" in text
    assert '"answer_validation"' in text
    assert "answer_validation_failed:" in text
