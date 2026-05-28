from __future__ import annotations

from pathlib import Path


def test_metric_alignment_report_is_quality_enriched() -> None:
    text = Path("pipeline/stage_08_bim_eval/metric_alignment.py").read_text(encoding="utf-8")
    assert "enrich_metric_alignment_report" in text
    assert "def _build_metric_alignment_report_unvalidated(" in text
    assert "def build_metric_alignment_report(*args, **kwargs):" in text
    assert "metric_alignment_quality_warning" in text
