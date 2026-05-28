from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_access import cfg_get, project_root, resolve_path


@dataclass(frozen=True)
class ViewerArtifact:
    key: str
    kind: str
    path: Path
    description: str
    required: bool = False


def _candidate(cfg: Any, key: str, kind: str, dotted_key: str, default: str, description: str, *, required: bool = False) -> ViewerArtifact:
    return ViewerArtifact(
        key=key,
        kind=kind,
        path=resolve_path(cfg, dotted_key, default),
        description=description,
        required=required,
    )


def configured_artifacts(cfg: Any) -> list[ViewerArtifact]:
    run = str(cfg_get(cfg, "project.run_id", "2026-04-30_site01_baseline"))

    artifacts = [
        _candidate(cfg, "cleaned_cloud", "point_cloud", "paths.clean_cloud", "data/clean/site01/cleaned_cloud.ply", "Stage 7 cleaned point cloud", required=True),
        _candidate(cfg, "downsampled_cloud", "point_cloud", "paths.clean_downsampled_cloud", "data/clean/site01/downsampled_cloud.ply", "Stage 7 downsampled point cloud"),
        _candidate(cfg, "clean_mesh", "mesh", "paths.clean_mesh", "data/clean/site01/mesh.ply", "Stage 7 reconstructed mesh"),
        _candidate(cfg, "planes_json", "json", "paths.clean_planes_json", "data/clean/site01/planes.json", "Stage 7 plane extraction records"),
        _candidate(cfg, "vlm_qa_evidence", "json", "vlm_qa.evidence_json", "data/vlm_qa/site01/vlm_qa_evidence.json", "Stage 7.5 visual QA evidence"),
        _candidate(cfg, "vlm_qa_summary", "json", "vlm_qa.summary_json", f"runs/{run}/reports/vlm_qa_summary.json", "Stage 7.5 visual QA summary"),
        _candidate(cfg, "scan_aligned", "point_cloud", "bim.scan_aligned_ply", "data/bim/aligned/site01/scan_aligned.ply", "Stage 8 scan aligned to BIM coordinates"),
        _candidate(cfg, "bim_reference", "point_cloud", "bim.bim_reference_ply", "data/bim/aligned/site01/bim_reference.ply", "Stage 8 BIM reference point cloud"),
        _candidate(cfg, "deviation_map", "point_cloud", "progress.deviation_map_ply", "data/bim/metrics/site01/deviation_map.ply", "Stage 9 deviation map point cloud"),
        _candidate(cfg, "element_metrics", "csv", "progress.element_metrics_csv", "data/bim/metrics/site01/element_metrics.csv", "Stage 9 element metrics"),
        _candidate(cfg, "activity_progress", "csv", "progress.activity_progress_csv", "data/bim/metrics/site01/activity_progress.csv", "Stage 9 activity progress"),
        _candidate(cfg, "progress_dashboard", "html", "progress.dashboard_html", f"runs/{run}/reports/progress_dashboard.html", "Stage 9 progress dashboard"),
    ]

    return artifacts


def discover_artifacts(cfg: Any) -> tuple[list[ViewerArtifact], list[ViewerArtifact]]:
    found: list[ViewerArtifact] = []
    missing: list[ViewerArtifact] = []
    seen: set[Path] = set()

    for artifact in configured_artifacts(cfg):
        path = artifact.path
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            resolved = path.resolve()
            if resolved not in seen:
                found.append(artifact)
                seen.add(resolved)
        else:
            missing.append(artifact)

    render_dir = resolve_path(cfg, "copilot.paths.render_dir", "runs/2026-04-30_site01_baseline/copilot/renders")
    max_renders = int(cfg_get(cfg, "viewer_export.max_copilot_renders", 12) or 12)
    if render_dir.exists():
        for p in sorted(render_dir.glob("*.png"))[:max_renders]:
            resolved = p.resolve()
            if resolved in seen:
                continue
            found.append(ViewerArtifact(
                key=f"copilot_render_{len(found)+1:02d}",
                kind="image",
                path=p,
                description="Stage 10 copilot render image",
                required=False,
            ))
            seen.add(resolved)

    return found, missing
