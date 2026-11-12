"""Stage 9 additive A->B scientific evidence adapter CLI; no Open3D dependency.

Consumes Stage9 CSV plus optional A geometry manifest and produces versioned
Stage9 element CSV; independent quantities are retained on geometry failure.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .evidence_metrics import element_scientific_fields, valid_sha256
from .geometry_evidence_adapter import load_geometry_evidence_manifest

SCHEMA = 'mycon.stage9.scientific_evidence.v2'


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich_scientific_evidence_rows(
    rows: list[dict[str,Any]], *, manifest_path: Path | None,
    project_root: Path, expected_ifc_sha256: str | None = None,
    expected_scan_sha256: str | None = None,
    expected_registration_sha256: str | None = None,
    expected_project_id: str | None = None,
    data_date: dt.datetime | None = None,
) -> tuple[list[dict[str,Any]], dict[str,Any]]:
    ids=[str(r.get('global_id') or '').strip() for r in rows]
    count={gid:ids.count(gid) for gid in set(ids)}
    independent_inputs_valid = (all(valid_sha256(x) for x in (expected_ifc_sha256, expected_scan_sha256, expected_registration_sha256)) and bool(expected_project_id))
    bundle=(load_geometry_evidence_manifest(
        manifest_path, project_root=project_root, expected_global_ids=set(ids),
        expected_ifc_sha256=expected_ifc_sha256,
        expected_scan_sha256=expected_scan_sha256,
        expected_registration_sha256=expected_registration_sha256,
        expected_project_id=expected_project_id,
    ) if manifest_path and independent_inputs_valid else None)
    out=[]
    for row in rows:
        gid=str(row.get('global_id') or '').strip()
        source=dict(row)
        # Do not trust legacy CSV's reported physical_progress_estimate;
        # revalidate signed quantity from the *raw* audit fields below.
        pre=element_scientific_fields(source,require_audit=True,data_date=data_date)
        if not bundle:
            geom={'geometry_adapter_valid':False,'geometry_evidence_state':'unknown_manifest',
                  'observability_ratio':None,'observability_source':'',
                  'geometry_manifest_sha256':None,'geometry_evidence_reason_codes':'manifest_not_supplied'}
        else:
            geom=bundle.for_element(gid,pre['physical_progress_estimate'])
        if count.get(gid,0) != 1 or not gid:
            geom.update({'geometry_adapter_valid':False,
                         'geometry_evidence_state':'unknown_duplicate_or_missing_global_id',
                         'observability_ratio':None,
                         'geometry_evidence_reason_codes':'duplicate_or_missing_global_id'})
        source.update(geom)
        parsed=element_scientific_fields(source,require_audit=True,data_date=data_date)
        # Preserve geometry state and all raw per-element input provenance.
        source.update({k:v for k,v in parsed.items() if k!='quantity_audit'})
        source['quantity_audit_json']=json.dumps(parsed['quantity_audit'],sort_keys=True)
        source['stage9_scientific_schema']=SCHEMA
        out.append(source)
    summary={
        'schema_version':SCHEMA,
        'input_element_count':len(rows),
        'distinct_global_id_count':len(count),
        'geometry_adapter_valid':bool(bundle and bundle.valid),
        'geometry_adapter_reasons':list(bundle.reason_codes) if bundle else ([
            'independent_ifc_scan_registration_hashes_or_project_id_missing'
        ] if manifest_path else ['manifest_not_supplied']),
        'manifest_sha256':bundle.manifest_sha256 if bundle else None,
        'verified_quantity_count':sum(r['physical_progress_status']=='verified_quantity' for r in out),
        'unknown_geometry_count':sum(str(r['geometry_evidence_state']).startswith('unknown') for r in out),
        'note':'Authoritative quantity must be approved independently. Geometry is only evidence for visibility/conformance.',
    }
    return out,summary


def _write_csv(path: Path,rows: list[dict[str,Any]]) -> None:
    fields=list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='',encoding='utf-8') as handle:
        w=csv.DictWriter(handle,fieldnames=fields);w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--element-metrics-csv',type=Path,required=True)
    p.add_argument('--geometry-manifest',type=Path)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--out-element-csv',type=Path,required=True)
    p.add_argument('--out-summary-json',type=Path,required=True)
    p.add_argument('--data-date-utc',required=True)
    p.add_argument('--expected-ifc-sha256')
    p.add_argument('--expected-scan-sha256')
    p.add_argument('--expected-registration-sha256')
    p.add_argument('--expected-project-id')
    args=p.parse_args(argv)
    try:
        date=dt.datetime.fromisoformat(args.data_date_utc.replace('Z','+00:00'))
        if date.tzinfo is None: date=date.replace(tzinfo=dt.timezone.utc)
        with args.element_metrics_csv.open(encoding='utf-8-sig',newline='') as fh:
            rows=list(csv.DictReader(fh))
        updated,report=enrich_scientific_evidence_rows(
            rows,manifest_path=args.geometry_manifest,project_root=args.project_root,
            expected_ifc_sha256=args.expected_ifc_sha256,
            expected_scan_sha256=args.expected_scan_sha256,
            expected_registration_sha256=args.expected_registration_sha256,
            expected_project_id=args.expected_project_id,data_date=date)
        if args.out_element_csv.exists() or args.out_summary_json.exists():
            raise FileExistsError('scientific evidence output exists; use a unique progress- run ID')
        report['inputs']={'element_metrics_csv_sha256':_hash(args.element_metrics_csv),
                           'manifest_path':str(args.geometry_manifest) if args.geometry_manifest else None}
        _write_csv(args.out_element_csv,updated)
        args.out_summary_json.parent.mkdir(parents=True,exist_ok=True)
        with args.out_summary_json.open('x',encoding='utf-8') as handle:
            json.dump(report,handle,indent=2,allow_nan=False)
        print(f"STAGE9_SCIENTIFIC_OK elements={len(updated)} geometry_gate={'pass' if report['geometry_adapter_valid'] else 'blocked'} output={args.out_element_csv}")
        return 0
    except (OSError,ValueError,TypeError,json.JSONDecodeError) as exc:
        print(f'STAGE9_SCIENTIFIC_FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
