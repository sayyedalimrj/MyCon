from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PlannedStage:
    stage: str
    command: str
    mode: str
    heavy: bool
    server_required: bool
    expected_marker: str
    notes: str


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def build_pipeline_plan(config_path: Path) -> dict[str, Any]:
    _load_yaml(config_path)

    stages = [
        PlannedStage(
            "Stage 4.5 CAMS-GS prepare",
            "python3 -m pipeline.stage_04_5_cams_gs.run_cams_gs_prepare --config configs/site01.yaml --force",
            "prepare_only",
            False,
            False,
            "STAGE_04_5_CAMS_GS_PREPARE_OK",
            "No training. Prepares future 3DGS/Nerfstudio assets.",
        ),
        PlannedStage(
            "Stage 6 DA3 assist",
            "python3 -m pipeline.stage_06_da3_assist.run_da3_assist --config configs/site01.yaml --force",
            "skip_safe",
            False,
            False,
            "STAGE_06_DA3_OK",
            "May skip when dense is sufficient.",
        ),
        PlannedStage(
            "Stage 7 cleanup",
            "python3 -m pipeline.stage_07_cleanup.run_cleanup --config configs/site01.yaml --force",
            "pointcloud_processing",
            True,
            True,
            "STAGE_07_CLEANUP_OK",
            "Can be moderate/heavy on full server dense cloud.",
        ),
        PlannedStage(
            "Stage 7.5 VLM QA",
            "python3 -m pipeline.stage_07_5_vlm_qa.run_vlm_qa --config configs/site01.yaml --force",
            "mock_or_real_vlm",
            False,
            False,
            "STAGE_07_5_VLM_QA_OK",
            "Currently mock-safe. Real Qwen requires server model cache.",
        ),
        PlannedStage(
            "Stage 7.6 viewer export",
            "python3 -m pipeline.stage_07_6_viewer_export.run_viewer_export --config configs/site01.yaml --force",
            "package_export",
            False,
            False,
            "STAGE_07_6_VIEWER_EXPORT_OK",
            "Lightweight artifact packaging unless external tilers are enabled.",
        ),
        PlannedStage(
            "Stage 7.7 CAMS-GS evidence",
            "python3 -m pipeline.stage_07_7_cams_gs_evidence.run_cams_gs_evidence --config configs/site01.yaml --force",
            "package_export",
            False,
            False,
            "STAGE_07_7_CAMS_GS_EVIDENCE_OK",
            "Reports prepared_no_training until real 3DGS training exists.",
        ),
        PlannedStage(
            "Stage 8 metric alignment",
            "python3 -m pipeline.stage_08_bim_eval.run_metric_alignment --config configs/site01.yaml --force",
            "requires_metric_or_visual_anchors",
            False,
            True,
            "STAGE_08_METRIC_ALIGNMENT_OK",
            "On laptop may be skipped_insufficient_anchors. Server/project data required.",
        ),
        PlannedStage(
            "Stage 8 BIM registration",
            "python3 -m pipeline.stage_08_bim_eval.run_registration --config configs/site01.yaml --force",
            "registration",
            True,
            True,
            "STAGE_08_BIM_REGISTRATION_OK",
            "Needs real/project BIM for defensible progress evidence.",
        ),
        PlannedStage(
            "Stage 9 progress",
            "python3 -m pipeline.stage_09_progress.run_progress --config configs/site01.yaml --force",
            "metrics",
            False,
            True,
            "STAGE_09_PROGRESS_OK",
            "Useful only if Stage 8 registration confidence is defensible.",
        ),
        PlannedStage(
            "Stage 10 copilot ask",
            "python3 -m pipeline.stage_10_copilot.run_ask --config configs/site01.yaml --question \"Based on the available BIM progress metrics, can this element be accepted?\" --json",
            "mock_or_real_vlm",
            False,
            False,
            "answer",
            "Mock-safe on laptop. Real Qwen requires server cache and endpoint.",
        ),
    ]

    return {
        "status": "plan_only",
        "config": str(config_path),
        "default_behavior": "dry_run_only_no_heavy_execution",
        "stages": [asdict(s) for s in stages],
        "server_required_stages": [s.stage for s in stages if s.server_required],
        "heavy_stages": [s.stage for s in stages if s.heavy],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the intended pipeline execution plan without running heavy stages.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--output", default="runs/2026-04-30_site01_baseline/reports/pipeline_plan.json")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()

    plan = build_pipeline_plan(Path(args.config))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print(
        "PIPELINE_PLAN_OK "
        f"stages={len(plan['stages'])} "
        f"heavy={len(plan['heavy_stages'])} "
        f"server_required={len(plan['server_required_stages'])} "
        f"output={output}"
    )

    if args.print_commands:
        for item in plan["stages"]:
            print(f"\n# {item['stage']}")
            print(f"# mode={item['mode']} heavy={item['heavy']} server_required={item['server_required']}")
            print(item["command"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
