from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeline.stage_10_copilot.api import ask_copilot
from pipeline.stage_10_copilot.metric_tools import collect_metrics
from pipeline.stage_10_copilot.model_profiles import recommend_profile
from pipeline.stage_10_copilot.local_vlm_client import LocalVLMConfigError, validate_local_endpoint
from pipeline.stage_10_copilot.query_router import QueryCategory, route_query


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cfg(root: Path) -> dict:
    metrics = root / "data" / "bim" / "metrics" / "site01"
    _write_csv(metrics / "element_metrics.csv", [{"global_id": "W1", "coverage": "0.74", "confidence": "0.7"}])
    _write_csv(metrics / "activity_progress.csv", [{"activity_id": "A1", "actual_percent": "60", "planned_percent": "80"}])
    (metrics / "deviation_summary.json").write_text(json.dumps({"max_deviation_m": 0.09}), encoding="utf-8")
    (metrics / "coverage_summary.json").write_text(json.dumps({"undercovered_regions": ["r1"]}), encoding="utf-8")
    (metrics / "registration_quality.json").write_text(json.dumps({"fitness": 0.8}), encoding="utf-8")
    cloud = root / "data" / "clean" / "site01" / "cleaned_cloud.ply"
    cloud.parent.mkdir(parents=True, exist_ok=True)
    cloud.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    return {
        "project": {"root": str(root), "name": "site01", "run_id": "test"},
        "inputs": {"ifc": "data/bim/design/model.ifc"},
        "paths": {"clean_cloud": "data/clean/site01/cleaned_cloud.ply"},
        "copilot": {
            "default_view": "front",
            "paths": {
                "evidence_dir": "runs/test/copilot/evidence",
                "render_dir": "runs/test/copilot/renders",
                "default_pointcloud": "data/clean/site01/cleaned_cloud.ply",
                "element_metrics_csv": "data/bim/metrics/site01/element_metrics.csv",
                "activity_progress_csv": "data/bim/metrics/site01/activity_progress.csv",
                "deviation_summary_json": "data/bim/metrics/site01/deviation_summary.json",
                "coverage_summary_json": "data/bim/metrics/site01/coverage_summary.json",
                "registration_quality_json": "data/bim/metrics/site01/registration_quality.json",
            },
            "vlm": {"provider": "mock"},
        },
    }


def test_route_query_progress_bim() -> None:
    route = route_query("Has this wall been executed and can I accept it?")
    assert route.category == QueryCategory.PROGRESS_QUESTION
    assert route.needs_metrics
    assert route.needs_bim


def test_metric_tools_read_element_and_activity(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    metrics = collect_metrics(cfg, element_global_id="W1", activity_id="A1")
    assert metrics["element_metrics"]["status"] == "ok"
    assert metrics["activity_progress"]["status"] == "ok"
    assert metrics["registration_quality"]["data"]["fitness"] == 0.8


def test_ask_copilot_builds_evidence_and_answer(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    response = ask_copilot(
        cfg,
        {
            "question": "Where is the highest deviation?",
            "selected_element_id": "W1",
            "selected_activity_id": "A1",
        },
    )
    evidence = Path(response["evidence_package_path"])
    assert evidence.exists()
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["selected_element_id"] == "W1"
    assert payload["selected_activity_id"] == "A1"
    assert response["selected_element_id"] == "W1"
    assert response["selected_activity_id"] == "A1"
    assert "Direct answer" in response["answer"]
    assert response["generated_view_paths"]
    for path in response["generated_view_paths"].values():
        assert Path(path).exists()


def test_server_profile_recommendations() -> None:
    svc1 = recommend_profile("SVC1", "Nvidia 3090 Ti 24GB", ram_gb=58)
    assert "yolo11x" in svc1.recommended_yolo
    assert svc1.recommended_vlm_live == "qwen3-vl:8b-thinking"
    svc2 = recommend_profile("SVC2", "Nvidia 1080", ram_gb=58)
    assert svc2.recommended_vlm_live == "qwen3-vl:4b"


def test_local_vlm_endpoint_guard_rejects_cloud_endpoint() -> None:
    try:
        validate_local_endpoint("https://api.example.com/v1/chat/completions", allow_private_lan=False)
    except LocalVLMConfigError as exc:
        assert "Refusing non-local VLM endpoint" in str(exc)
    else:
        raise AssertionError("cloud endpoint should be rejected by local-only guard")


def test_local_vlm_endpoint_guard_accepts_localhost() -> None:
    endpoint = validate_local_endpoint("http://127.0.0.1:11434/api/chat", allow_private_lan=False)
    assert endpoint.endswith("/api/chat")


def test_server_profile_has_local_offline_provider() -> None:
    profile = recommend_profile("SVC4", "Nvidia A5000 32GB", ram_gb=90)
    assert profile.recommended_provider == "ollama_local"
    assert profile.ollama_model == "qwen3-vl:8b-thinking"
