#!/usr/bin/env python3
"""Orthographic diagnostic of an existing ASCII XYZRGB candidate, not novel-view QA."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():
        raise RuntimeError('Refusing to overwrite existing diagnostics')
    with a.candidate.open() as stream:
        header=[]
        for _ in range(256):
            line=stream.readline().strip();header.append(line)
            if line=='end_header':break
        if 'format ascii 1.0' not in header or header[-1]!='end_header':
            raise ValueError('This diagnostic accepts ASCII PLY only')
        columns=[x.split()[-1] for x in header if x.startswith('property ')]
        if columns!=['x','y','z','red','green','blue']:
            raise ValueError('Unexpected PLY columns')
        expected=int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
        points=np.loadtxt(stream)
    if points.shape!=(expected,6) or not np.isfinite(points).all():
        raise ValueError('Point data are incomplete or nonfinite')
    xyz=points[:,:3];colors=np.clip(points[:,3:]/255,0,1)
    lo,hi=xyz.min(0),xyz.max(0)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,(i,j) in zip(axes,((0,1),(0,2),(1,2))):
        ax.scatter(xyz[:,i],xyz[:,j],c=colors,s=.15,rasterized=True)
        ax.set_aspect('equal');ax.set_xlabel('XYZ'[i]+' (arbitrary units)')
        ax.set_ylabel('XYZ'[j]+' (arbitrary units)');ax.grid(alpha=.15)
    fig.suptitle('Existing DA3 candidate: orthographic diagnostics only\nNot a Gaussian render; no metric or completeness certification')
    fig.savefig(a.output,dpi=140);plt.close(fig)
    report={'candidate':str(a.candidate),'sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
            'points':expected,'bbox_min':lo.tolist(),'bbox_max':hi.tolist(),
            'acceptance':'ORTHOGRAPHIC_DIAGNOSTIC_ONLY','training_executed':False,
            'metric_validated':False,'novel_view_validated':False}
    with a.output.with_suffix('.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
