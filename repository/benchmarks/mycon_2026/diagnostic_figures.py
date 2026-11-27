"""Appendix-only real in-sample diagnostic figure; never benchmark field accuracy."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def render(in_csv,output_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader(Path(in_csv).open(encoding='utf-8',newline='')))
    if not rows or any(r.get('evidence_class')!='real_internal_colmap_fit_not_heldout' or r.get('scientific_eligible')!='False' for r in rows):
        raise ValueError('only_in_sample_COLMAP_internal_diagnostics_allowed')
    a=sorted((int(r['image_id']),float(r['median_error_px'])) for r in rows)
    xs=[v[0] for v in a];ys=[v[1] for v in a]
    path=Path(output_png)
    if path.exists():raise FileExistsError('Refusing to overwrite existing figure')
    fig,ax=plt.subplots(figsize=(11,4.7))
    ax.plot(xs,ys,marker='.',markersize=3,linewidth=1)
    ax.set_xlabel('COLMAP image ID (not the chronological frame order)')
    ax.set_ylabel('Internal sparse reprojection median (px)')
    ax.set_title('Real MyCON source: COLMAP internal fit ONLY — NOT held-out / physical accuracy')
    ax.grid(True,alpha=.25)
    fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(render(a.csv,a.out))