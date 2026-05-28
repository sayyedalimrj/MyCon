from __future__ import annotations

from pathlib import Path


def test_visual_anchor_quality_module_defines_required_gate_terms() -> None:
    text = Path("pipeline/stage_08_bim_eval/visual_anchor_quality.py").read_text(encoding="utf-8")

    for term in [
        "min_ray_angle_deg",
        "reprojection_fail_px",
        "insufficient_observations",
        "ray_angle_too_small",
        "reprojection_error_too_high",
        "summarize_visual_anchor_quality",
    ]:
        assert term in text
