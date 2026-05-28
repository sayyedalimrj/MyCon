from __future__ import annotations

from pathlib import Path

import yaml

from pipeline.common.server_readiness_policy import build_server_readiness_gate


def test_element_activity_map_uses_default_demo_path_when_config_key_missing(tmp_path: Path) -> None:
    for raw in [
        "video.mp4",
        "model.ifc",
        "schedule.csv",
        "metric_anchors.csv",
        "data/bim/design/element_activity_map.csv",
    ]:
        p = tmp_path / raw
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")

    cfg = {
        "project": {"root": str(tmp_path)},
        "inputs": {
            "video": "video.mp4",
            "ifc": "model.ifc",
            "schedule_csv": "schedule.csv",
        },
        "metric_alignment": {
            "metric_anchors_csv": "metric_anchors.csv",
        },
        "copilot": {
            "vlm": {
                "model": "qwen3-vl:8b-thinking",
                "hf_model": "Qwen/Qwen3-VL-8B-Thinking",
            }
        },
    }

    cfg_path = tmp_path / "site.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    report = build_server_readiness_gate(cfg_path)
    status = {item["key"]: item["status"] for item in report["items"]}

    assert status["path.element_activity_map"] == "ok"
    assert status["metricization.route"] == "ok"
    assert status["model.qwen3_vl_8b_thinking_config"] == "ok"
