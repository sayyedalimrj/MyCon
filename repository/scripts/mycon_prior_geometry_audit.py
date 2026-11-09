#!/usr/bin/env python3
"""Read existing candidate headers/reports; never certify geometry from file presence."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root, inventory):
    records=[]
    for entry in inventory['artifacts']:
        path=(root/entry['path']).resolve()
        if root not in path.parents or path.suffix.lower()!='.ply':
            continue
        record={'path':entry['path'],'status':'UNVALIDATED_EXISTING_CANDIDATE'}
        try:
            with path.open('rb') as stream:
                digest=hashlib.sha256()
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                stream.seek(0);header=[]
                for _ in range(256):
                    line=stream.readline(4096).decode('ascii').strip();header.append(line)
                    if line=='end_header':break
                if header[0]!='ply' or header[-1]!='end_header':raise ValueError('Invalid or oversized PLY header')
            record.update(sha256=digest.hexdigest(),bytes=path.stat().st_size,header=header)
            # Bounded provenance lookup; preserve exact contents without interpreting success labels.
            sidecars=[]
            for directory in (path.parent,path.parent.parent):
                for candidate in sorted(directory.glob('*.json'))[:20]:
                    if candidate.stat().st_size<=250_000:
                        sidecars.append({'path':str(candidate.relative_to(root)),
                                         'content':json.loads(candidate.read_text())})
            record['provenance_sidecars']=sidecars
        except Exception as exc:
            record.update(status='READ_FAILED',error=str(exc))
        records.append(record)
    groups={}
    for record in records:
        if 'sha256' in record:groups.setdefault(record['sha256'],[]).append(record['path'])
    return {'schema':'mycon.prior_geometry_audit.v1','read_only_inputs':True,
            'acceptance':'NOT_GEOMETRY_OR_NOVEL_VIEW_VALIDATION','candidates':records,
            'identical_copy_groups':[paths for paths in groups.values() if len(paths)>1],
            'note':'Headers and provenance identify reuse opportunities, not accuracy or coherent coverage.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--drive-root',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=audit(a.drive_root.resolve(),json.loads(a.inventory.read_text()))
    with a.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'candidates':len(report['candidates']),
                      'identical_copy_groups':len(report['identical_copy_groups']),
                      'acceptance':report['acceptance']}))
