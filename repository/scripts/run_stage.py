#!/usr/bin/env python3
"""Generic stage launcher stub."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_STAGE_MODULES: dict[str, str] = {
    "stage_01_ingest": "pipeline.stage_01_ingest.run_ingest",
    "stage_02_keyframes": "pipeline.stage_02_keyframes.select_keyframes",
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
