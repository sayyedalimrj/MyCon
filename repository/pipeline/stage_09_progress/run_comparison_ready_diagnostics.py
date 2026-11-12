"""Stage 9 audit-only CLI for mycon.comparison_ready_geometry.v1 (no IFC required)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .comparison_ready_adapter import load_comparison_ready_geometry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--out-json', type=Path, required=True)
    p.add_argument('--expected-project-id')
    a = p.parse_args(argv)
    if a.out_json.exists():
        print('COMPARISON_READY_FAILED: output path already exists',file=sys.stderr)
        return 2
    b = load_comparison_ready_geometry(a.manifest, project_root=a.project_root,
                                        expected_project_id=a.expected_project_id)
    report = b.summary()
    report['scientific_interpretation'] = (
        'This is reconstruction geometry QA only. Features have no IFC ownership. '
        'No metric/BIM/physical-progress claim is authorized without independent gates.'
    )
    try:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        with a.out_json.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
    except (OSError,ValueError) as exc:
        print(f'COMPARISON_READY_FAILED: {exc}', file=sys.stderr)
        return 2
    print('COMPARISON_READY_DIAGNOSTICS_'+('OK' if b.valid else 'BLOCKED')+
          f' planes={report["feature_counts"]["planes"]} '
          f'edges={report["feature_counts"]["edges"]} '
          f'corners={report["feature_counts"]["corners"]} '
          f'IFC=unknown progress=null output={a.out_json}')
    return 0 if b.valid else 2


if __name__ == '__main__':
    raise SystemExit(main())
