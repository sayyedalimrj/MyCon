"""Read-only A/B interface preflight. Does not write to data, train, or score GT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.mycon_2026.adapters import (  # noqa: E402
    load_scientific_progress_v2, load_stage9_v2, validate_geometry_evidence)
from benchmarks.mycon_2026.protocol import sha256  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect optional A/B versioned contracts, fail closed")
    parser.add_argument("--a-geometry-manifest", type=Path)
    parser.add_argument("--a-data-root", type=Path)
    parser.add_argument("--b-stage9-csv", type=Path)
    parser.add_argument("--b-stage11-v2-json", type=Path)
    args = parser.parse_args(argv)
    if not any((args.a_geometry_manifest, args.b_stage9_csv, args.b_stage11_v2_json)):
        parser.error("Provide at least one artifact")
    checks = {}
    entries = (("a_geometry", args.a_geometry_manifest),
               ("b_stage9", args.b_stage9_csv),
               ("b_stage11", args.b_stage11_v2_json))
    for label, path in entries:
        if path is None:
            continue
        try:
            if label == "a_geometry":
                value = validate_geometry_evidence(path, root=args.a_data_root or path.resolve().parent)
                status = "valid_candidate" if value["valid_for_visibility_candidate"] else "incomplete"
            elif label == "b_stage9":
                rows = load_stage9_v2(path)
                value = {"n_element_rows": len(rows), "n_numeric_progress":
                         sum(r["physical_progress_estimate"] is not None for r in rows)}
                status = "structurally_valid"
            else:
                doc = load_scientific_progress_v2(path)
                value = {"n_activities": len(doc["activities"]),
                         "run_id": doc.get("run_id"), "units": doc["units"]}
                status = "structurally_valid"
            checks[label] = {"status": status, "input_sha256": sha256(path), "checks": value}
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            checks[label] = {"status": "invalid", "error": type(exc).__name__ + ": " + str(exc)}
    print(json.dumps({"schema_version": "mycon.preflight.v1", "checks": checks}, indent=2, allow_nan=False))
    return 2 if any(item["status"] == "invalid" for item in checks.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
