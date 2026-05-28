from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.run_pipeline_plan import build_pipeline_plan
from scripts.server_readiness_check import build_readiness_report


def _write_config(path: Path) -> None:
    cfg = {
        "project": {"root": str(path.parent)},
        "paths": {
            "clean_cloud": "data/clean/site01/cleaned_cloud.ply",
            "clean_mesh": "data/clean/site01/mesh.ply",
        },
        "bim": {"ifc_path": "data/bim/design/model.ifc"},
        "metric_alignment": {
            "anchors_csv": "data/bim/design/metric_anchors.csv",
            "known_distances_csv": "data/bim/design/known_distances.csv",
        },
        "cams_gs": {"source_images_dir": "data/sfm/site01/images"},
        "copilot": {
            "vlm": {
                "model": "qwen3-vl:8b-thinking",
                "hf_model": "Qwen/Qwen3-VL-8B-Thinking",
            }
        },
        "model_cache": {"root_dir": "model_cache"},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def test_server_readiness_report_is_non_destructive(tmp_path: Path) -> None:
    config = tmp_path / "site01.yaml"
    _write_config(config)

    report = build_readiness_report(config, check_host_tools=False)

    assert "checks" in report
    assert report["summary"]["missing"] >= 1
    assert report["status"] in {"ready_or_laptop_stub", "server_inputs_missing"}
    assert any(item["key"] == "vlm.qwen_profile" for item in report["checks"])


def test_pipeline_plan_is_dry_run_only(tmp_path: Path) -> None:
    config = tmp_path / "site01.yaml"
    _write_config(config)

    plan = build_pipeline_plan(config)

    assert plan["status"] == "plan_only"
    assert plan["default_behavior"] == "dry_run_only_no_heavy_execution"
    assert len(plan["stages"]) >= 8
    assert "Stage 8 BIM registration" in plan["heavy_stages"]


def test_pipeline_plan_json_roundtrip(tmp_path: Path) -> None:
    config = tmp_path / "site01.yaml"
    _write_config(config)

    plan = build_pipeline_plan(config)
    out = tmp_path / "pipeline_plan.json"
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["status"] == "plan_only"
    assert loaded["stages"][0]["stage"].startswith("Stage")
