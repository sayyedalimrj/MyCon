from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_access import cfg_get, project_name, resolve_path, run_id


def stage75_paths(cfg: Any) -> dict[str, Path]:
    name = project_name(cfg)
    rid = run_id(cfg)

    output_dir = resolve_path(cfg, cfg_get(cfg, "vlm_qa.output_dir", f"data/vlm_qa/{name}"), required=True)
    assert output_dir is not None

    render_dir = resolve_path(cfg, cfg_get(cfg, "vlm_qa.render_dir", f"data/vlm_qa/{name}/renders"), required=True)
    assert render_dir is not None

    return {
        "cleaned_cloud": resolve_path(
            cfg,
            cfg_get(cfg, "paths.clean_cloud", cfg_get(cfg, "cleanup.cleaned_cloud", f"data/clean/{name}/cleaned_cloud.ply")),
            required=True,
        ),
        "downsampled_cloud": resolve_path(
            cfg,
            cfg_get(cfg, "paths.clean_downsampled_cloud", cfg_get(cfg, "cleanup.downsampled_cloud", f"data/clean/{name}/downsampled_cloud.ply")),
            required=False,
        ),
        "mesh": resolve_path(
            cfg,
            cfg_get(cfg, "paths.clean_mesh", cfg_get(cfg, "cleanup.mesh_ply", f"data/clean/{name}/mesh.ply")),
            required=False,
        ),
        "planes_json": resolve_path(
            cfg,
            cfg_get(cfg, "paths.clean_planes_json", cfg_get(cfg, "cleanup.planes_json", f"data/clean/{name}/planes.json")),
            required=False,
        ),
        "cleanup_report": resolve_path(
            cfg,
            cfg_get(cfg, "paths.cleanup_report_json", cfg_get(cfg, "cleanup.report_json", f"runs/{rid}/reports/cleanup_summary.json")),
            required=True,
        ),
        "output_dir": output_dir,
        "render_dir": render_dir,
        "evidence_json": resolve_path(
            cfg,
            cfg_get(cfg, "vlm_qa.evidence_json", f"data/vlm_qa/{name}/vlm_qa_evidence.json"),
            required=True,
        ),
        "summary_json": resolve_path(
            cfg,
            cfg_get(cfg, "vlm_qa.summary_json", f"runs/{rid}/reports/vlm_qa_summary.json"),
            required=True,
        ),
    }


def missing_required_inputs(paths: dict[str, Path]) -> list[str]:
    required = ["cleaned_cloud", "cleanup_report"]
    missing: list[str] = []
    for key in required:
        path = paths[key]
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(f"{key}:{path.as_posix()}")
    return missing
