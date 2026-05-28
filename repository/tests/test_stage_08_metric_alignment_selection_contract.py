from __future__ import annotations

from pathlib import Path


def test_build_metric_alignment_report_attaches_selection() -> None:
    text = Path("pipeline/stage_08_bim_eval/metric_alignment.py").read_text(encoding="utf-8")

    assert "attach_metric_alignment_selection" in text
    assert "metric_alignment_selection_failed" in text
    assert "locals().get('anchors', [])" not in text
    assert "read_metric_anchors_csv" in text
