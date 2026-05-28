#!/usr/bin/env python3
"""Generic stage launcher.

Resolves the canonical ``python3 -m pipeline.stage_XX...`` invocation for each
shipped pipeline stage. Stage 10 ("copilot ask") requires a ``--question``
argument that this generic launcher cannot synthesize, so it is intentionally
excluded; invoke it directly with
``python3 -m pipeline.stage_10_copilot.run_ask --config <yaml> --question "..."``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# B4: previously this dict knew only stage_01 and stage_02; calling for any
# other stage raised an argparse error. The generic launcher now resolves
# every stage that takes (--config, --force, --log-level) without extra args.
# Stage 8 has two entry points and is exposed under explicit names.
_STAGE_MODULES: dict[str, str] = {
    "stage_01_ingest": "pipeline.stage_01_ingest.run_ingest",
    "stage_02_keyframes": "pipeline.stage_02_keyframes.select_keyframes",
    "stage_03_colmap": "pipeline.stage_03_colmap.run_sparse",
    "stage_04_refinement": "pipeline.stage_04_refinement.run_refinement",
    "stage_04_5_cams_gs": "pipeline.stage_04_5_cams_gs.run_cams_gs_prepare",
    "stage_05_dense": "pipeline.stage_05_dense.run_dense",
    "stage_06_da3_assist": "pipeline.stage_06_da3_assist.run_da3_assist",
    "stage_07_cleanup": "pipeline.stage_07_cleanup.run_cleanup",
    "stage_07_5_vlm_qa": "pipeline.stage_07_5_vlm_qa.run_vlm_qa",
    "stage_07_6_viewer_export": "pipeline.stage_07_6_viewer_export.run_viewer_export",
    "stage_07_7_cams_gs_evidence": "pipeline.stage_07_7_cams_gs_evidence.run_cams_gs_evidence",
    "stage_08_metric_alignment": "pipeline.stage_08_bim_eval.run_metric_alignment",
    "stage_08_bim_registration": "pipeline.stage_08_bim_eval.run_registration",
    "stage_09_progress": "pipeline.stage_09_progress.run_progress",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a configured pipeline stage.")
    parser.add_argument("stage", choices=sorted(_STAGE_MODULES.keys()))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    command = [sys.executable, "-m", _STAGE_MODULES[args.stage], "--config", str(args.config), "--log-level", args.log_level]
    if args.force:
        command.append("--force")
    return int(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
