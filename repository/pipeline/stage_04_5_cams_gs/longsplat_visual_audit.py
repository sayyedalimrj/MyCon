"""Independent rendered held-out frame comparison for LongSplat research."""
from __future__ import annotations
import json
from pathlib import Path


def evaluate_holdout_views(model: Path, iteration: int | None, report_path: Path) -> dict:
    import numpy as np
    from PIL import Image
    if iteration is None:
        raise RuntimeError('Trained iteration unavailable')
    folder=model/'test'/f'ours_{iteration}'
    images=folder/'renders'
    gt=folder/'gt'
    records=[]
    if images.is_dir() and gt.is_dir():
        for file in sorted(images.glob('*.png')):
            truth=gt/file.name
            if not truth.is_file():continue
            with Image.open(file) as a, Image.open(truth) as b:
                a=np.asarray(a.convert('RGB'),dtype=np.float64)/255.
                b=np.asarray(b.convert('RGB'),dtype=np.float64)/255.
            if a.shape!=b.shape:continue
            mse=float(np.mean((a-b)**2))
            psnr=float(-10*np.log10(max(mse,1e-12)))
            records.append({'file':file.name,'psnr_db':round(psnr,3)})
    # Do not promise a beautiful result solely because the train views render.
    psnr=float(np.mean([x['psnr_db'] for x in records])) if records else None
    status='HELDOUT_VIEWS_MEASURED' if len(records)>=5 else 'INSUFFICIENT_HELDOUT_EVIDENCE'
    result={'schema':'mycon.longsplat.novel_view_qa.v1','status':status,
            'heldout_pairs':len(records),'psnr_mean_db':psnr,'per_view':records,
            'unobserved_camera_positions_verified':False,
            'metric_geometry_verified':False,
            'interpretation':'PSNR on withheld source-camera views is an appearance check, not proof of accurate geometry, pose or all novel viewpoints.'}
    from .longsplat_research import atomic_json
    atomic_json(report_path,result)
    return result
