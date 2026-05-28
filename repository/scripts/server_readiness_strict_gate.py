from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.common.server_readiness_policy import write_server_readiness_gate_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict server readiness gate for expensive GPU/server runs.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument(
        "--output",
        default="runs/2026-04-30_site01_baseline/reports/server_readiness_strict_gate.json",
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero if server blockers are present.")
    args = parser.parse_args()

    report = write_server_readiness_gate_report(Path(args.config), Path(args.output))
    blockers = report.get("server_blockers", [])

    print(
        "SERVER_READINESS_STRICT_GATE_OK "
        f"passed={str(report.get('passed')).lower()} "
        f"blockers={len(blockers)} "
        f"warnings={report.get('summary', {}).get('warning_count')} "
        f"output={args.output}"
    )

    if blockers:
        print("server_blockers=" + ",".join(str(item.get("key")) for item in blockers))

    return 0 if report.get("passed") or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
