"""Stage 8 CLI: IFC/BIM extraction and scan-to-BIM registration."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from pipeline.common.config import load_config
from pipeline.common.logging_utils import setup_logging

from .coarse_registration import coarse_register
from .config_access import cfg_bool, cfg_get, project_name, resolve_project_path, run_id
from .geometry_utils import bounds_summary, load_point_cloud, point_count, triangle_count, write_point_cloud
from .ifc_to_mesh import extract_ifc_geometry
from .input_selection import select_ifc_input, select_scan_input
from .io_utils import clean_dir, write_json_atomic
from .refine_icp import refine_icp
from .registration_quality import evaluate_registration_quality, nearest_neighbor_summary


LOGGER_NAME = "pipeline.stage_08_bim_eval"


def _paths(cfg: Any) -> dict[str, Path]:
    name = project_name(cfg)
    rid = run_id(cfg)
    aligned_dir = resolve_project_path(cfg, "paths.bim_aligned_dir", f"data/bim/aligned/{name}")
    return {
        "aligned_dir": aligned_dir,
        "bim_reference_ply": resolve_project_path(cfg, "bim.bim_reference_ply", f"data/bim/aligned/{name}/bim_reference.ply"),
        "bim_reference_mesh": resolve_project_path(cfg, "bim.bim_reference_mesh", f"data/bim/aligned/{name}/bim_reference_mesh.ply"),
        "scan_aligned_ply": resolve_project_path(cfg, "bim.scan_aligned_ply", f"data/bim/aligned/{name}/scan_aligned.ply"),
        "transform_json": resolve_project_path(cfg, "bim.transform_json", f"data/bim/aligned/{name}/transform_scan_to_bim.json"),
        "element_metadata_jsonl": resolve_project_path(cfg, "bim.element_metadata_jsonl", f"data/bim/aligned/{name}/bim_elements.jsonl"),
        "report_json": resolve_project_path(cfg, "bim.registration_report_json", f"runs/{rid}/reports/registration_report.json"),
    }


def _transform_scan(scan_pcd: Any, transform: Any) -> Any:
    transformed = scan_pcd.clone() if hasattr(scan_pcd, "clone") else scan_pcd.select_by_index(list(range(len(scan_pcd.points))))
    transformed.transform(transform)
    return transformed


def _existing_outputs(paths: dict[str, Path]) -> bool:
    required = [
        paths["bim_reference_ply"],
        paths["scan_aligned_ply"],
        paths["transform_json"],
        paths["report_json"],
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def run_registration(cfg: Any, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    t0 = time.perf_counter()
    paths = _paths(cfg)
    if _existing_outputs(paths) and not force:
        return {"status": "skipped_existing_outputs", "report_json": paths["report_json"].as_posix()}
    if force:
        clean_dir(paths["aligned_dir"], force=True, expected_leaf="aligned")
    else:
        paths["aligned_dir"].mkdir(parents=True, exist_ok=True)

    selected_scan = select_scan_input(cfg)
    selected_ifc = select_ifc_input(cfg)
    logger.info("Starting Stage 8 BIM registration for project=%s run_id=%s", project_name(cfg), run_id(cfg))
    logger.info("Selected scan input: %s", selected_scan.path)
    logger.info("Selected IFC input: %s", selected_ifc.path)

    scan_pcd = load_point_cloud(selected_scan.path)
    extraction = extract_ifc_geometry(
        cfg,
        selected_ifc.path,
        scan_pcd,
        paths["bim_reference_mesh"],
        paths["bim_reference_ply"],
        paths["element_metadata_jsonl"],
        logger,
    )
    bim_pcd = extraction.point_cloud
    coarse = coarse_register(cfg, scan_pcd, bim_pcd, logger)
    icp = refine_icp(cfg, scan_pcd, bim_pcd, coarse.transformation, logger)
    scan_aligned = _transform_scan(scan_pcd, icp.transformation)
    write_point_cloud(paths["scan_aligned_ply"], scan_aligned, binary=True)
    nn_scan_to_bim = nearest_neighbor_summary(
        scan_aligned,
        bim_pcd,
        sample_limit=int(cfg_get(cfg, "bim.quality_nn_sample_limit", 200_000)),
    )
    transform_payload = {
        "source_frame": "stage_07_scan_or_selected_scan_input",
        "target_frame": "design_bim_ifc_world_coordinates",
        "matrix_4x4_scan_to_bim": icp.transformation.tolist(),
        "coarse_matrix_4x4": coarse.transformation.tolist(),
        "coarse_method": coarse.method,
        "scale_factor_initial": coarse.scale_factor,
        "icp_method": icp.method,
        "icp_fitness": icp.fitness,
        "icp_inlier_rmse_m": icp.inlier_rmse,
    }
    write_json_atomic(paths["transform_json"], transform_payload)

    report: dict[str, Any] = {
        "stage": "stage_08_bim_eval",
        "status": "complete",
        "project": {"name": project_name(cfg), "run_id": run_id(cfg)},
        "inputs": {
            "scan": selected_scan.__dict__,
            "ifc": selected_ifc.__dict__,
        },
        "outputs": {key: value for key, value in paths.items()},
        "ifc": {
            "source": extraction.source,
            "synthetic_bim_fallback_used": extraction.source == "synthetic_test_fallback",
            "real_progress_evidence_allowed": extraction.source != "synthetic_test_fallback",
            "units": extraction.units,
            "element_count": len(extraction.elements),
            "warnings": extraction.warnings,
            "schedule_filter": extraction.schedule_filter,
            "visibility_filter": extraction.visibility_filter,
        },
        "scan_geometry": {
            "point_count": point_count(scan_pcd),
            "bounds": bounds_summary(scan_pcd).__dict__,
        },
        "bim_geometry": {
            "point_count": point_count(bim_pcd),
            "mesh_vertex_count": point_count(extraction.mesh),
            "mesh_triangle_count": triangle_count(extraction.mesh),
            "bounds": bounds_summary(bim_pcd).__dict__,
        },
        "coarse_registration": {
            "method": coarse.method,
            "scale_factor": coarse.scale_factor,
            "fitness": coarse.fitness,
            "inlier_rmse": coarse.inlier_rmse,
            "warnings": coarse.warnings,
        },
        "icp": {
            "method": icp.method,
            "fitness": icp.fitness,
            "inlier_rmse": icp.inlier_rmse,
            "correspondence_set_size": icp.correspondence_set_size,
            "warnings": icp.warnings,
        },
        "nearest_neighbor_scan_to_bim": nn_scan_to_bim,
        "elapsed_sec": time.perf_counter() - t0,
    }
    report["quality_gate"] = evaluate_registration_quality(cfg, report)
    write_json_atomic(paths["report_json"], report)
    if report["quality_gate"]["status"] == "fail" and cfg_bool(cfg, "bim.fail_on_low_registration_quality", False):
        raise RuntimeError("Stage 8 registration quality gate failed: " + "; ".join(report["quality_gate"].get("errors", [])))
    logger.info(
        "Stage 8 complete in %.3fs; fitness=%.6f rmse=%.6f outputs=%s",
        report["elapsed_sec"],
        icp.fitness,
        icp.inlier_rmse,
        paths["aligned_dir"],
    )
    print(
        "STAGE_08_BIM_REGISTRATION_OK "
        f"fitness={icp.fitness:.6f} rmse={icp.inlier_rmse:.6f} "
        f"scan={paths['scan_aligned_ply']} bim={paths['bim_reference_ply']}"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 8: IFC/BIM extraction and scan-to-BIM registration")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(name=LOGGER_NAME, level=args.log_level)
    cfg = load_config(args.config)
    try:
        run_registration(cfg, force=args.force, log_level=args.log_level)
    except Exception as exc:  # noqa: BLE001
        print(f"STAGE_08_BIM_REGISTRATION_FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
