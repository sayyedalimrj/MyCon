#!/usr/bin/env python3
"""Read-only Recorder evidence inventory; writes reports only to a new file."""
import argparse
import collections
import csv
import io
import json
import zipfile
from pathlib import Path


def inventory(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as bundle:
        entries=[{'name':x.filename,'bytes':x.file_size} for x in bundle.infolist()]
        sidecars={};poses=[]
        for item in bundle.infolist():
            if item.filename.endswith('.json') and item.file_size<100000:
                sidecars[item.filename]=json.loads(bundle.read(item))
            if Path(item.filename).name=='pose.csv':
                poses=list(csv.DictReader(io.StringIO(bundle.read(item).decode('utf-8-sig'))))
    groups=collections.defaultdict(list)
    for row in poses:
        try:
            t=int(row['timestamp_ns'])
            if t>0:groups[t].append(row)
        except (KeyError,ValueError,TypeError):pass
    fields=('tx_m','ty_m','tz_m','qx','qy','qz','qw','tracking')
    conflicting=sum(len({tuple(row.get(k) for k in fields) for row in group})>1 for group in groups.values())
    return {'archive_bytes':archive.stat().st_size,'files':entries,'sidecars':sidecars,
            'source_unchanged':True,'pose_audit':{'source_rows':len(poses),
            'unique_positive_timestamps':len(groups),
            'duplicate_timestamp_groups':sum(len(x)>1 for x in groups.values()),
            'duplicate_groups_with_different_pose_or_tracking':conflicting,'pose_is_survey_truth':False}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=inventory(args.capture)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(args.output)

if __name__=='__main__':main()
