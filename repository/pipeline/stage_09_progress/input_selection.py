from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_access import cfg_get, project_name, resolve_path, run_id


def stage9_paths(cfg: Any) -> dict[str, Path]:
    name = project_name(cfg)
    rid = run_id(cfg)
    metrics_dir = resolve_path(cfg, cfg_get(cfg, "progress.metrics_dir", f"data/bim/metrics/{name}"), required=True)
    report_dir = resolve_path(cfg, f"runs/{rid}/reports", required=True)

    assert metrics_dir is not None
    assert report_dir is not None

    return {
        "scan_aligned": resolve_path(cfg, cfg_get(cfg, "bim.scan_aligned_ply", f"data/bim/aligned/{name}/scan_aligned.ply"), required=True),
        "bim_reference": resolve_path(cfg, cfg_get(cfg, "bim.bim_reference_ply", f"data/bim/aligned/{name}/bim_reference.ply"), required=True),
        "bim_elements": resolve_path(cfg, cfg_get(cfg, "bim.element_metadata_jsonl", f"data/bim/aligned/{name}/bim_elements.jsonl"), required=True),
        "registration_report": resolve_path(cfg, cfg_get(cfg, "bim.registration_report_json", f"runs/{rid}/reports/registration_report.json"), required=True),
        "schedule_csv": resolve_path(cfg, cfg_get(cfg, "progress.schedule_csv", cfg_get(cfg, "bim.schedule_filter_csv", "data/bim/design/schedule.csv")), required=True),
        "element_activity_map": resolve_path(cfg, cfg_get(cfg, "progress.element_activity_map_csv", "data/bim/design/element_activity_map.csv"), required=True),
        "metrics_dir": metrics_dir,
        "element_metrics_csv": resolve_path(cfg, cfg_get(cfg, "copilot.paths.element_metrics_csv", f"data/bim/metrics/{name}/element_metrics.csv"), required=True),
        "activity_progress_csv": resolve_path(cfg, cfg_get(cfg, "copilot.paths.activity_progress_csv", f"data/bim/metrics/{name}/activity_progress.csv"), required=True),
        "deviation_summary_json": resolve_path(cfg, cfg_get(cfg, "copilot.paths.deviation_summary_json", f"data/bim/metrics/{name}/deviation_summary.json"), required=True),
        "coverage_summary_json": resolve_path(cfg, cfg_get(cfg, "copilot.paths.coverage_summary_json", f"data/bim/metrics/{name}/coverage_summary.json"), required=True),
        "registration_quality_json": resolve_path(cfg, cfg_get(cfg, "copilot.paths.registration_quality_json", f"data/bim/metrics/{name}/registration_quality.json"), required=True),
        "deviation_map_ply": resolve_path(cfg, cfg_get(cfg, "progress.deviation_map_ply", f"data/bim/metrics/{name}/deviation_map.ply"), required=True),
        "progress_summary_json": resolve_path(cfg, cfg_get(cfg, "progress.summary_json", f"runs/{rid}/reports/progress_summary.json"), required=True),
        "dashboard_html": resolve_path(cfg, cfg_get(cfg, "progress.dashboard_html", f"runs/{rid}/reports/progress_dashboard.html"), required=True),
    }


def missing_required_inputs(paths: dict[str, Path]) -> list[str]:
    required = [
        "scan_aligned",
        "bim_reference",
        "bim_elements",
        "registration_report",
        "schedule_csv",
        "element_activity_map",
    ]
    missing = []
    for key in required:
        path = paths[key]
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(f"{key}:{path.as_posix()}")
    return missing
