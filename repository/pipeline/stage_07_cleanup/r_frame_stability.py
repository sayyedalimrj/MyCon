"""Owner A: independent re-detection perturbation audit, not ground-truth accuracy.

Uses one already reconstructed unscaled COLMAP sparse point set; perturbations
are synthetic even when the original input is real. Never certify a building.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from . import research_dense_candidates as candidates
from .r_frame_geometry import _read_tracks, _sha

SCHEMA='mycon.civil_sfm_repeatability.v1'


def evaluate(xyz, *,threshold=.07,min_points=250,max_planes=9,trials=90,seed=17):
    p=np.asarray(xyz,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<min_points or not np.isfinite(p).all():
        raise ValueError('Invalid source cloud')
    params={'threshold':threshold,'min_points':min_points,'max_planes':max_planes,
            'trials':trials,'max_sample':min(16000,len(p)),'seed':seed}
    base,_,_,_=candidates.extract(p,**params)
    rng=np.random.default_rng(seed+99)
    cen=np.median(p,axis=0)
    half=np.sort(rng.choice(len(p),size=len(p)//2,replace=False))
    quadrant=np.flatnonzero(~((p[:,0]>cen[0])&(p[:,1]>cen[1])))
    allidx=np.arange(len(p))
    trials_case=[('half_density',half,p[half],threshold),
         ('occluded_xy_quadrant',quadrant,p[quadrant],threshold),
         ('gaussian_noise_source_units_0p025',allidx,p+rng.normal(0,.025,p.shape),threshold),
         ('threshold_low_0p85x',allidx,p,threshold*.85),
         ('threshold_high_1p15x',allidx,p,threshold*1.15)]
    results=[]
    for name,mapidx,test,thr in trials_case:
        pars=dict(params);pars['threshold']=thr;pars['max_sample']=min(len(test),16000)
        planes,_,_,_=candidates.extract(test,**pars)
        original_sets=[set(map(int,mapidx[np.asarray(x['support_indices'],dtype=int)])) for x in planes]
        registered=set(map(int,mapidx))
        outcomes=[]
        for b in base:
            bs=set(map(int,b['support_indices']))
            available=bs&registered
            options=[]
            for k,c in enumerate(planes):
                dot=float(abs(np.dot(b['normal'],c['normal'])))
                angle=float(np.degrees(np.arccos(np.clip(dot,-1,1))))
                shared=len(available&original_sets[k])
                overlap=shared/max(1,len(available))
                if angle<=12 and overlap>=.2:
                    options.append((overlap,-angle,k,angle,shared))
            if options:
                best=max(options)
                outcome={'base_plane_id':b['plane_id'],'matched':True,
                         'best_repeat_plane_index':best[2],
                         'normal_angle_deg_unsigned':round(best[3],4),
                         'original_support_recovered_fraction':round(best[0],4),
                         'recovered_source_support_points':int(best[4])}
            else:
                outcome={'base_plane_id':b['plane_id'],'matched':False,
                         'best_repeat_plane_index':None,'normal_angle_deg_unsigned':None,
                         'original_support_recovered_fraction':0.,'recovered_source_support_points':0}
            outcomes.append(outcome)
        results.append({'scenario':name,'synthetically_modified_input':True,
                        'retained_source_point_count':len(test),
                        'repeat_detected_patch_count':len(planes),
                        'matched_reference_patches':sum(x['matched'] for x in outcomes),
                        'baseline_patch_count':len(base),
                        'per_reference_patch':outcomes})
    return {'schema_version':SCHEMA,'input_type':'original point positions real-or-synthetic supplied by caller',
        'experiment':'repeat feature DETECTION on same input after synthetic density/occlusion/noise/threshold perturbations',
        'not_independent_ground_truth':True,'not_metric':True,'not_physical_surface_confirmation':True,
        'baseline_patch_count':len(base),'threshold_reconstruction_units':threshold,
        'density_and_noise_units':'arbitrary_COLMAP_units','seed':seed,
        'matching_criteria':'unsigned normal angle <=12deg and >=0.2 recovered visible original inlier fraction; no GT F1',
        'scenarios':results}


def main(argv=None):
    p=argparse.ArgumentParser(description='Civil A repeat-detection stress experiment (not a site QA certification)')
    p.add_argument('--points3d-txt',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--threshold',type=float,default=.07);p.add_argument('--min-points',type=int,default=250)
    p.add_argument('--max-planes',type=int,default=9);p.add_argument('--trials',type=int,default=90)
    args=p.parse_args(argv)
    try:
        ids,xyz,_,_,_,reject=_read_tracks(args.points3d_txt,error_px=2,min_track_length=4)
        ret=evaluate(xyz,threshold=args.threshold,min_points=args.min_points,
                     max_planes=args.max_planes,trials=args.trials)
        ret.update({'points3d_source_sha256':_sha(args.points3d_txt),
            'quality_gated_point_count':len(ids),'quality_rejected_count':reject})
        if args.output.exists():raise FileExistsError('Refuse overwrite')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf8') as f:json.dump(ret,f,indent=2,allow_nan=False)
    except (ValueError,OSError) as exc:
        print(json.dumps({'status':'blocked','reason':str(exc)}));return 2
    print(json.dumps({'status':'repeatability_report_generated','patches':ret['baseline_patch_count'],
              'repeat_cases':len(ret['scenarios']),'not_ground_truth':True}));return 0

if __name__=='__main__':raise SystemExit(main())
