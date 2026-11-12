"""Independent authoritative v2 Stage 11 CLI; never serves deprecated v1 KPIs.

Inputs are canonical schedule/mapping CSV and Stage9 scientific element CSV.
Approved dated baselines may be provided as a separate time-series CSV.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from pipeline.common.bim_schedule_mapping import load_mapping_csv
from pipeline.common.schedule_io import load_schedule_csv, parse_iso_datetime
from .scientific_progress import build_scientific_progress_report


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule-csv',type=Path,required=True)
    p.add_argument('--mapping-csv',type=Path,required=True)
    p.add_argument('--element-metrics-csv',type=Path,required=True)
    p.add_argument('--baseline-csv',type=Path)
    p.add_argument('--out-json',type=Path,required=True)
    p.add_argument('--data-date-utc',required=True)
    p.add_argument('--research-run-id',required=True)
    p.add_argument('--on-schedule-band-pct',type=float,default=5.0)
    args=p.parse_args(argv)
    try:
        if args.out_json.exists(): raise FileExistsError('output already exists')
        date=parse_iso_datetime(args.data_date_utc)
        schedule=load_schedule_csv(args.schedule_csv)
        mapping=load_mapping_csv(args.mapping_csv)
        with args.element_metrics_csv.open('r',encoding='utf-8-sig',newline='') as handle:
            elements=list(csv.DictReader(handle))
        baseline_path=args.baseline_csv or args.schedule_csv
        with baseline_path.open('r',encoding='utf-8-sig',newline='') as handle:
            baselines=list(csv.DictReader(handle))
        report=build_scientific_progress_report(
            schedule=schedule,mapping=mapping,element_rows=elements,schedule_rows=baselines,
            data_date=date,run_id=args.research_run_id,
            on_schedule_band_pct=args.on_schedule_band_pct,
        )
        report['provenance']={'stage':'stage_11_schedule_variance',
                              'input_sha256':{k:_sha(path) for k,path in {
                                  'schedule_csv':args.schedule_csv,
                                  'mapping_csv':args.mapping_csv,
                                  'element_metrics_csv':args.element_metrics_csv,
                                  'baseline_csv':baseline_path}.items()},
                              'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
        args.out_json.parent.mkdir(parents=True,exist_ok=True)
        with args.out_json.open('x',encoding='utf-8') as fh:
            json.dump(report,fh,indent=2,allow_nan=False)
        print(f"STAGE11_SCIENTIFIC_OK run_id={args.research_run_id} quality_gate={report['quality_gates']['schedule_decision']} status={report['overall']['schedule_variance_status']} output={args.out_json}")
        return 0
    except (OSError,TypeError,ValueError,json.JSONDecodeError) as exc:
        print(f'STAGE11_SCIENTIFIC_FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
