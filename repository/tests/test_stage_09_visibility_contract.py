from __future__ import annotations

from pathlib import Path


def test_stage9_decision_enrichment_imports_visibility_policy() -> None:
    text = Path("pipeline/stage_09_progress/decision_enrichment.py").read_text(encoding="utf-8")
    assert "interpret_element_visibility" in text
    assert "visibility_fields = interpret_element_visibility(out)" in text
    assert "visibility_decision_risks" in text
