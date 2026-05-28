from __future__ import annotations

from pathlib import Path

import yaml

from pipeline.common.server_readiness_policy import build_server_readiness_gate


def test_strict_gate_flags_missing_server_inputs(tmp_path: Path) -> None:
    cfg = {
        "project": {"root": str(tmp_path)},
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

    assert report["passed"] is False
    keys = {item["key"] for item in report["server_blockers"]}
    assert "path.raw_video_or_images" in keys
    assert "path.ifc" in keys
    assert "metricization.route" in keys


def test_strict_gate_accepts_configured_project_inputs_except_runtime_tools(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    ifc = tmp_path / "model.ifc"
    schedule = tmp_path / "schedule.csv"
    element_map = tmp_path / "element_activity_map.csv"
    anchors = tmp_path / "metric_anchors.csv"

    for p in [video, ifc, schedule, element_map, anchors]:
        p.write_text("x", encoding="utf-8")

    cfg = {
        "project": {"root": str(tmp_path)},
        "inputs": {
            "video": "video.mp4",
            "ifc": "model.ifc",
            "schedule_csv": "schedule.csv",
            "element_activity_map_csv": "element_activity_map.csv",
            "metric_anchors_csv": "metric_anchors.csv",
        },
        "metric_alignment": {"metric_anchors_csv": "metric_anchors.csv"},
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
    item_status = {item["key"]: item["status"] for item in report["items"]}

    assert item_status["path.raw_video_or_images"] == "ok"
    assert item_status["path.ifc"] == "ok"
    assert item_status["path.schedule_csv"] == "ok"
    assert item_status["path.element_activity_map"] == "ok"
    assert item_status["path.metric_anchors_csv"] == "ok"
    assert item_status["metricization.route"] == "ok"
    assert item_status["model.qwen3_vl_8b_thinking_config"] == "ok"
