"""CLI entrypoint for Stage 10 Copilot questions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.config import load_config  # noqa: E402
from pipeline.stage_10_copilot.api import ask_copilot, run_stdlib_server  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask the local Construction Copilot.")
    parser.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/site01.yaml")
    parser.add_argument("--question", help="Question to ask. Required unless --serve is used.")
    parser.add_argument("--selected-element-id", default=None)
    parser.add_argument("--selected-activity-id", default=None)
    parser.add_argument("--selected-bbox", default=None, help="Comma-separated minx,miny,minz,maxx,maxy,maxz")
    parser.add_argument("--current-view", default=None)
    parser.add_argument("--pointcloud-path", default=None)
    parser.add_argument("--ifc-path", default=None)
    parser.add_argument("--serve", action="store_true", help="Start stdlib local HTTP API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--json", action="store_true", help="Print full JSON response.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.serve:
        run_stdlib_server(cfg, host=args.host, port=args.port)
        return 0
    if not args.question:
        raise SystemExit("--question is required unless --serve is used")
    payload: dict[str, Any] = {
        "question": args.question,
        "selected_element_id": args.selected_element_id,
        "selected_activity_id": args.selected_activity_id,
        "selected_bbox": args.selected_bbox,
        "current_view": args.current_view,
        "pointcloud_path": args.pointcloud_path,
        "ifc_path": args.ifc_path,
    }
    response = ask_copilot(cfg, payload)
    if args.json:
        print(json.dumps(response, indent=2, sort_keys=True))
    else:
        print("STAGE_10_COPILOT_OK")
        print(response["answer"])
        print(f"evidence_package={response['evidence_package_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
