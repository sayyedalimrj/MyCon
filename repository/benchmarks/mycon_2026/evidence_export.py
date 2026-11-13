"""MyCON Owner C: immutable, fail-closed thesis-evidence exporter.

Only reads input paths and creates a unique output directory. Does not run a
model, register a scan, infer progress, or elevate synthetic tests to field GT.
Optional numpy/Pillow/skimage/scipy provide actual numeric measurements.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'mycon.thesis_evidence.v1'
COLS = {
 'per_view_metrics.csv': ['schema_version','run_id','experiment_id','case_id','site_id','building_id','session_id','split','method','variant','view_id','metric','value','unit','status','reason','reference_path','reference_sha256','prediction_path','prediction_sha256','coordinate_frame','confidence','provenance'],
 'geometry_metrics.csv': ['schema_version','run_id','experiment_id','case_id','site_id','building_id','session_id','split','method','variant','metric','threshold_m','value','unit','status','reason','gt_path','gt_sha256','prediction_path','prediction_sha256','coordinate_frame','metric_scale_verified','confidence','provenance'],
 'ablation_results.csv': ['schema_version','run_id','case_id','site_id','reference_method','candidate_method','reference_variant','candidate_variant','metric','reference_value','candidate_value','difference','unit','status','reason','source_sha256','provenance'],
 'confidence_intervals.csv': ['schema_version','run_id','method','variant','metric','unit','mean','ci95_lower','ci95_upper','n_sites','n_cases','cluster_unit','bootstrap_draws','status','reason','provenance'],
 'view_tail_summary.csv': ['schema_version','run_id','method','variant','metric','percentile','worst_decile_mean','n_views','n_sites','unit','status','reason','provenance'],
 'performance_metrics.csv': ['schema_version','run_id','experiment_id','case_id','site_id','method','variant','metric','value','unit','status','reason','source_path','source_sha256','provenance'],
 'failure_cases.csv': ['schema_version','run_id','experiment_id','case_id','site_id','method','variant','task','status','reason','source_sha256','provenance'],
}
METRIC_UNITS = {'psnr_db':'dB','ssim':'fraction','lpips':'fraction','reprojection_rmse_px':'px','registered_camera_coverage':'fraction', 'accuracy_m':'m','completeness_m':'m','chamfer_l1_m':'m','point_to_plane_mae_m':'m','corner_mae_m':'m','plane_angular_mae_deg':'deg','corner_precision':'fraction','corner_recall':'fraction'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def secure_path(root, ref):
    """Only immutable relative regular files under the declared manifest root."""
    if not isinstance(ref,dict) or not isinstance(ref.get('path'),str) or not isinstance(ref.get('sha256'),str):
        raise ValueError('path_and_sha256_required')
    rel=Path(ref['path'])
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('unsafe_relative_path')
    path=(root/rel).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('out_of_root_or_not_file')
    if len(ref['sha256'])!=64 or digest(path)!=ref['sha256'].lower():
        raise ValueError('sha256_mismatch')
    return path

def _fnum(x):
    x=float(x)
    if not math.isfinite(x): raise ValueError('nonfinite_number')
    return x

def read_xyz(path, with_normals=False):
    import numpy as np
    if path.suffix.lower()=='.npy':
        arr=np.load(path,allow_pickle=False)
    elif path.suffix.lower()=='.csv':
        with path.open(newline='',encoding='utf-8') as f:
            arr=np.array([[ _fnum(row[c]) for c in (('x','y','z','nx','ny','nz') if with_normals else ('x','y','z'))] for row in csv.DictReader(f)],dtype=float)
    else: raise ValueError('unsupported_xyz_type')
    if arr.ndim!=2 or len(arr)==0 or arr.shape[1]<(6 if with_normals else 3) or not np.isfinite(arr).all():
        raise ValueError('invalid_xyz')
    return arr

def score_geometry(pred, gt, tau):
    """Two-way NN: Accuracy=pred→GT MAE; Completeness=GT→pred MAE;
    Chamfer L1=mean of two directed distances; never nearest-neighbour GT reuse
    as an independent registration/metric-scale calibration."""
    import numpy as np
    from scipy.spatial import cKDTree
    p=np.asarray(pred,dtype=float); g=np.asarray(gt,dtype=float)
    if p.ndim!=2 or g.ndim!=2 or p.shape[1]<3 or g.shape[1]<3 or not len(p) or not len(g): raise ValueError('invalid_cloud')
    if not np.isfinite(p).all() or not np.isfinite(g).all() or tau<=0: raise ValueError('invalid_geometry_values')
    d_pg, idx=cKDTree(g[:,:3]).query(p[:,:3]); d_gp,_=cKDTree(p[:,:3]).query(g[:,:3]);
    precision=float(np.mean(d_pg<=tau)); recall=float(np.mean(d_gp<=tau))
    f1=0. if precision+recall==0 else 2*precision*recall/(precision+recall)
    out={'accuracy_m':float(d_pg.mean()),'completeness_m':float(d_gp.mean()),'chamfer_l1_m':float((d_pg.mean()+d_gp.mean())/2),f'f1_at_{tau:g}m':f1}
    if g.shape[1]>=6:
        n=g[idx,3:6]; lengths=np.linalg.norm(n,axis=1)
        good=lengths>1e-9
        if good.any(): out['point_to_plane_mae_m']=float(np.mean(np.abs(np.sum((p[good,:3]-g[idx[good],:3])*n[good]/lengths[good,None],axis=1))))
    return out

def score_corners(pred_path,gt_path,tau):
    """Only supported observed corners count as predictions; extrapolated are NOT verified."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    def load(p):
        with p.open(encoding='utf-8',newline='') as f:
            return [r for r in csv.DictReader(f) if r.get('state','observed_supported')=='observed_supported']
    a,b=load(pred_path),load(gt_path)
    if not b: raise ValueError('no_verified_corners')
    A=np.array([[ _fnum(x[k]) for k in ('x','y','z')] for x in a],dtype=float).reshape(-1,3)
    B=np.array([[ _fnum(x[k]) for k in ('x','y','z')] for x in b],dtype=float).reshape(-1,3)
    if len(A):
        D=cdist(A,B); r,c=linear_sum_assignment(D); pairs=[float(D[i,j]) for i,j in zip(r,c) if D[i,j]<=tau]
    else: pairs=[]
    tp=len(pairs)
    return {'corner_precision':tp/len(A) if len(A) else 0.,'corner_recall':tp/len(B),'corner_mae_m':statistics.mean(pairs) if pairs else None}

def score_planes(pred_path,gt_path):
    """Only finite-support, explicitly matched plane IDs yield angular accuracy."""
    import numpy as np
    def load(path):
        with path.open(encoding='utf-8',newline='') as f:
            return {r['feature_id']:r for r in csv.DictReader(f) if r.get('finite_support')=='true' and r.get('reviewed')=='true'}
    p,g=load(pred_path),load(gt_path); errors=[]
    for k in sorted(p.keys()&g.keys()):
        n=np.array([_fnum(p[k][c]) for c in ('nx','ny','nz')]);m=np.array([_fnum(g[k][c]) for c in ('nx','ny','nz')]);
        if np.linalg.norm(n)<=0 or np.linalg.norm(m)<=0: continue
        cosine=np.clip(abs(np.dot(n,m)/(np.linalg.norm(n)*np.linalg.norm(m))),-1.,1.)
        errors.append(float(np.degrees(np.arccos(cosine))))
    return {'plane_angular_mae_deg':statistics.mean(errors) if errors else None}

def score_image(ref_path,pred_path):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity
    with Image.open(ref_path) as a,Image.open(pred_path) as b:
        a=np.asarray(a.convert('RGB'),dtype=np.float64)/255
        b=np.asarray(b.convert('RGB'),dtype=np.float64)/255
    if a.shape!=b.shape or min(a.shape[:2])<7: raise ValueError('image_shape_or_size_mismatch')
    mse=float(np.mean((a-b)**2))
    psnr=float('inf') if mse==0 else -10*math.log10(mse)
    # Explicit channel_axis, RGB [0,1] data range.
    ssim=float(structural_similarity(a,b,channel_axis=2,data_range=1.0))
    return {'psnr_db':None if not math.isfinite(psnr) else psnr,'ssim':ssim}

def score_reprojection(path):
    import numpy as np
    with path.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    good=[r for r in rows if r.get('registered')=='true']
    if not rows: raise ValueError('missing_camera_attempts')
    out={'registered_camera_coverage':len(good)/len(rows)}
    if good:
        d=[(_fnum(r['u_pred'])-_fnum(r['u_gt']))**2+(_fnum(r['v_pred'])-_fnum(r['v_gt']))**2 for r in good]
        out['reprojection_rmse_px']=float(np.sqrt(np.mean(d)))
    return out

def leakage_guard(cases):
    """Whole sites/buildings/sessions and capture hashes stay in one partition."""
    split_for={}
    ids=set()
    for c in cases:
        split=c.get('split');
        if split not in ('train','validation','test'): raise ValueError('invalid_split')
        for field in ('site_id','building_id','session_id','case_id'):
            if not isinstance(c.get(field),str) or not c[field]: raise ValueError('missing_'+field)
        for key in [('site',c['site_id']),('building',c['site_id'],c['building_id']),('session',c['site_id'],c['building_id'],c['session_id']),('capture_sha',c.get('capture_sha256'))]:
            if not key[-1]: continue
            if key in split_for and split_for[key]!=split: raise ValueError('train_validation_test_leakage:'+str(key))
            split_for[key]=split
        eid=(c['case_id'],c.get('method'),c.get('variant'),c.get('seed'))
        if eid in ids: raise ValueError('duplicate_experiment_identity')
        ids.add(eid)

def _write_csv(path,cols,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def _json(path,value):
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2,sort_keys=True)

def _site_ci(rows,run_id,seed=20261113,draws=2000):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r.get('status')=='evaluated' and isinstance(r.get('value'),(int,float)) and math.isfinite(r['value']):
            groups[(r['method'],r['variant'],r['metric'],r['unit'])][r['site_id']].append(r['value'])
    result=[]
    for (method,variant,metric,unit),sites in sorted(groups.items()):
        data=[statistics.mean(x) for x in sites.values()]; n=len(data)
        low=high=None
        if n>=2:
            rng=random.Random(f'{seed}:{method}:{variant}:{metric}');boots=sorted(statistics.mean(rng.choices(data,k=n)) for _ in range(draws))
            low,high=boots[int(draws*.025)],boots[min(draws-1,int(draws*.975))]
        result.append(dict(schema_version=SCHEMA,run_id=run_id,method=method,variant=variant,metric=metric,unit=unit,mean=statistics.mean(data),ci95_lower=low,ci95_upper=high,n_sites=n,n_cases=sum(map(len,sites.values())),cluster_unit='site',bootstrap_draws=draws if n>=2 else 0,status='evaluated' if n>=2 else 'not_evaluated',reason='' if n>=2 else 'fewer_than_two_independent_sites',provenance='independent_real_test_only'))
    return result

def render_figures(output_dir, eligible, run_id):
    """Methodological diagram is not empirical evidence; plots require real test data."""
    status={'schema_version':SCHEMA,'run_id':run_id,'rendered':[], 'not_evaluated':[]}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        status['not_evaluated'].append({'figure':'methodology_flow','reason':'matplotlib_not_installed'})
        return status
    fig,ax=plt.subplots(figsize=(10,2.6));ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
    stages=['Immutable inputs','Site/session split','GT + scale gate','Scoring','Site-cluster CI','Claims audit']
    for i,label in enumerate(stages):
        x=.1+i*1.68
        ax.text(x+.7,1,label,ha='center',va='center',fontsize=9,
                bbox=dict(boxstyle='round',facecolor='white',edgecolor='#555555',pad=.55))
        if i<5:ax.annotate('',xy=(x+1.60,1),xytext=(x+1.42,1),arrowprops={'arrowstyle':'->'})
    fig.suptitle('MyCON C: evaluation protocol (schematic; no measured results)',fontsize=12)
    for ext in ('png','svg'):
        fig.savefig(output_dir/f'methodology_flow.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    status['rendered'].append({'figure':'methodology_flow','type':'non_empirical_method_schematic'})
    if not eligible:
        status['not_evaluated'].extend([{'figure':x,'reason':'no_independent_real_test_metrics'} for x in ('view_quality','geometry_accuracy','ablation_effect','failure_rates','confidence_intervals')])
        return status
    by_metric=defaultdict(list)
    for item in eligible:
        if item['metric'] in ('ssim','psnr_db','accuracy_m','completeness_m','chamfer_l1_m'):
            by_metric[item['metric']].append(item)
    for metric,items in sorted(by_metric.items()):
        fig,ax=plt.subplots(figsize=(7,4))
        labels=sorted(set((e['method'],e['variant']) for e in items))
        grouped=[[e['value'] for e in items if (e['method'],e['variant'])==label] for label in labels]
        ax.boxplot(grouped,tick_labels=[x[0]+'/'+x[1] for x in labels],showmeans=True)
        ax.set(ylabel=metric, title=metric+' | verified real held-out cases (descriptive)')
        fig.autofmt_xdate()
        for ext in ('png','svg'):fig.savefig(output_dir/f'figure_{metric}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
        status['rendered'].append({'figure':'figure_'+metric,'n_test_items':len(items),'type':'descriptive_real_test'})
    return status

def export(manifest_path,output_root):
    manifest_path=Path(manifest_path).resolve(strict=True); root=manifest_path.parent
    m=json.loads(manifest_path.read_text(encoding='utf-8'))
    if m.get('schema_version')!=SCHEMA: raise ValueError('schema_version_mismatch')
    run_id=m.get('run_id')
    if not isinstance(run_id,str) or not run_id.startswith('validation-c-') or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in run_id): raise ValueError('invalid_run_id')
    cases=m.get('cases')
    if not isinstance(cases,list): raise ValueError('cases_must_be_list')
    leakage_guard(cases)
    out=Path(output_root).resolve()/run_id
    # Unique, atomic mkdir; no source file writes, no overwrites.
    out.mkdir(parents=True,exist_ok=False)
    rows={k:[] for k in COLS}; registry=[]; eligible=[]
    mh=digest(manifest_path)
    for ix,c in enumerate(cases):
        cid=c['case_id']; method=c.get('method','unknown');variant=c.get('variant','baseline'); eid=c.get('experiment_id',f'exp-{ix:04d}')
        frame=c.get('coordinate_frame','unknown'); units=c.get('units','unknown');
        base=dict(schema_version=SCHEMA,run_id=run_id,experiment_id=eid,case_id=cid,site_id=c['site_id'],building_id=c['building_id'],session_id=c['session_id'],split=c['split'],method=method,variant=variant,coordinate_frame=frame,confidence='not_calibrated',provenance='manifest_sha256:'+mh)
        scientific = c.get('data_kind')=='real' and c['split']=='test' and c.get('independent_gt_verified') is True and c.get('gt_reviewed') is True and isinstance(c.get('independent_gt_reviewer'),str) and bool(c.get('independent_gt_reviewer'))
        record=dict(base,data_kind=c.get('data_kind','unknown'),scientific_eligible=scientific,reason='' if scientific else 'not_independent_reviewed_real_heldout_gt',source_manifest_sha256=mh,status='not_evaluated',registered_camera_coverage=None)
        # Independent GT gate requires actual local immutable file + declared review.
        # A method's own reconstruction/surface is never accepted as GT.
        if scientific:
            try:
                survey=c['independent_gt_source']
                verified=secure_path(root,survey)
                if not c.get('independent_gt_review_date_utc') or not c.get('ground_truth_source_type') in ('independent_survey','independent_manual_annotation'):
                    raise ValueError('missing_independent_gt_review_provenance')
                if 'capture_source' in c:
                    cap=secure_path(root,c['capture_source'])
                    if c.get('capture_sha256')!=digest(cap):raise ValueError('capture_sha256_mismatch')
                else: raise ValueError('missing_capture_source')
                if digest(verified)==c.get('prediction_source_sha256'):
                    raise ValueError('predictions_not_independent_ground_truth')
            except (KeyError,ValueError,FileNotFoundError) as exc:
                scientific=False;record['scientific_eligible']=False;record['reason']=str(exc)
        if not scientific:
            rows['failure_cases.csv'].append(dict(base,task='scientific_gating',status='not_evaluated',reason=record['reason'],source_sha256=mh))
            registry.append(record);continue
        # No data-dependent tuning: threshold locked in manifest, in metres.
        tau=c.get('tau_m',.05)
        try:
            tau=_fnum(tau)
            if tau<=0: raise ValueError('invalid_tolerance')
        except (TypeError,ValueError) as exc: raise ValueError('invalid_tolerance') from exc
        status_any=False
        for v in c.get('views',[]):
            b=dict(base,view_id=v.get('view_id','unknown'),reference_path='',reference_sha256='',prediction_path='',prediction_sha256='')
            try:
                a_ref=v['reference_rgb'];p_ref=v['prediction_rgb']
                aa=secure_path(root,a_ref); pp=secure_path(root,p_ref)
                b.update(reference_path=a_ref['path'],reference_sha256=a_ref['sha256'],prediction_path=p_ref['path'],prediction_sha256=p_ref['sha256'])
                for metric,val in score_image(aa,pp).items():
                    state='evaluated' if val is not None else 'not_evaluated'
                    rows['per_view_metrics.csv'].append(dict(b,metric=metric,value=val,unit=METRIC_UNITS[metric],status=state,reason='' if val is not None else 'identical_images_infinite_psnr'))
                    if val is not None:eligible.append(dict(base,status='evaluated',metric=metric,value=val,unit=METRIC_UNITS[metric]));status_any=True
                rows['per_view_metrics.csv'].append(dict(b,metric='lpips',value=None,unit='fraction',status='not_evaluated',reason='pinned_lpips_evaluator_not_configured'))
            except (KeyError,ValueError,FileNotFoundError,ImportError) as exc:
                rows['failure_cases.csv'].append(dict(base,task='nvs',status='not_evaluated',reason=str(exc),source_sha256=mh))
        if 'camera_observations' in c:
            try:
                ref=c['camera_observations'];path=secure_path(root,ref)
                for metric,val in score_reprojection(path).items():
                    rows['per_view_metrics.csv'].append(dict(base,view_id='aggregate',metric=metric,value=val,unit=METRIC_UNITS[metric],status='evaluated',reason='',reference_path=ref['path'],reference_sha256=ref['sha256'],prediction_path='',prediction_sha256=''))
                    eligible.append(dict(base,status='evaluated',metric=metric,value=val,unit=METRIC_UNITS[metric]));status_any=True
            except (ValueError,KeyError,ImportError) as exc: rows['failure_cases.csv'].append(dict(base,task='camera',status='not_evaluated',reason=str(exc),source_sha256=mh))
        geom=c.get('geometry')
        if geom:
            b=dict(base,gt_path='',gt_sha256='',prediction_path='',prediction_sha256='',metric_scale_verified=c.get('metric_scale_verified') is True)
            if c.get('metric_scale_verified') is not True or units!='m' or 'metric_scale_evidence' not in c:
                rows['failure_cases.csv'].append(dict(base,task='metric_geometry',status='not_evaluated',reason='independent_metric_scale_unverified',source_sha256=mh))
            else:
                try:
                    control=secure_path(root,c['metric_scale_evidence'])
                    controls=json.loads(control.read_text(encoding='utf-8'))
                    fit=set(controls.get('fit_anchor_ids',[]));hold=set(controls.get('heldout_anchor_ids',[]))
                    if not fit or not hold or fit&hold or not controls.get('independent_survey_reviewer'):
                        raise ValueError('invalid_or_leaking_metric_scale_controls')
                    gtref=geom['gt']; prref=geom['prediction'];a=secure_path(root,gtref);p=secure_path(root,prref)
                    b.update(gt_path=gtref['path'],gt_sha256=gtref['sha256'],prediction_path=prref['path'],prediction_sha256=prref['sha256'])
                    # No ICP or registration fit to held-out test GT is performed here.
                    base_metrics=score_geometry(read_xyz(p),read_xyz(a,with_normals=bool(geom.get('gt_normals'))),tau)
                    taus=c.get('thresholds_m',[0.01,0.02,0.05,0.10])
                    if not isinstance(taus,list) or any(_fnum(t)<=0 for t in taus):raise ValueError('invalid_preregistered_thresholds')
                    for other_tau in sorted(set(map(float,taus))):
                        f1_key=f'f1_at_{other_tau:g}m'
                        base_metrics[f1_key]=score_geometry(read_xyz(p),read_xyz(a,with_normals=bool(geom.get('gt_normals'))),other_tau)[f1_key]
                    base_metrics.pop(f'f1_at_{tau:g}m',None) if float(tau) not in taus else None
                    for metric,val in base_metrics.items():
                        rows['geometry_metrics.csv'].append(dict(b,metric=metric,threshold_m=float(metric[6:-1]) if metric.startswith('f1_at_') else None,value=val,unit='fraction' if metric.startswith('f1_') else 'm',status='evaluated',reason=''))
                        eligible.append(dict(base,status='evaluated',metric=metric,value=val,unit='fraction' if metric.startswith('f1_') else 'm'));status_any=True
                    for kind,scorer in [('corners',score_corners),('planes',score_planes)]:
                        if kind in geom:
                            pred=secure_path(root,geom[kind]['prediction']);truth=secure_path(root,geom[kind]['gt']);vals=scorer(pred,truth,tau) if kind=='corners' else scorer(pred,truth)
                            for metric,val in vals.items():
                                state='evaluated' if val is not None else 'not_evaluated'
                                rows['geometry_metrics.csv'].append(dict(b,metric=metric,threshold_m=tau,value=val,unit=METRIC_UNITS[metric],status=state,reason='' if val is not None else 'no_valid_matches'))
                                if val is not None:eligible.append(dict(base,status='evaluated',metric=metric,value=val,unit=METRIC_UNITS[metric]));status_any=True
                except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:
                    rows['failure_cases.csv'].append(dict(base,task='metric_geometry',status='not_evaluated',reason=str(exc),source_sha256=mh))
        # Explicit sidecar telemetry; values are source-provided, not measured by exporter.
        if 'telemetry' in c:
            try:
                tref=c['telemetry']; tp=secure_path(root,tref)
                telem=json.loads(tp.read_text(encoding='utf-8'))
                for mk,un in (('elapsed_sec','s'),('peak_vram_mb','MiB'),('peak_rss_mb','MiB'),('output_bytes','bytes')):
                    if mk in telem and telem[mk] is not None:
                        val=_fnum(telem[mk]);
                        if val<0:raise ValueError('invalid_telemetry_'+mk)
                        rows['performance_metrics.csv'].append(dict(base,metric=mk,value=val,unit=un,status='reported_unverified_instrumentation',reason='requires_hardware_logger_audit',source_path=tref['path'],source_sha256=tref['sha256']))
            except (ValueError,KeyError,FileNotFoundError) as exc:
                rows['failure_cases.csv'].append(dict(base,task='telemetry',status='not_evaluated',reason=str(exc),source_sha256=mh))
        if not status_any:rows['failure_cases.csv'].append(dict(base,task='scientific_metrics',status='not_evaluated',reason='no_verified_compatible_artifacts',source_sha256=mh))
        record['status']='evaluated' if status_any else 'not_evaluated';registry.append(record)
    # Pair only the same case, exact capture checksum and matched metric units.
    pairkeys=defaultdict(list)
    for e in eligible:
        capture=next((c.get('capture_sha256') for c in cases if c['case_id']==e['case_id'] and c.get('method')==e['method'] and c.get('variant','baseline')==e['variant']),None)
        gt=next((c.get('independent_gt_source',{}).get('sha256') for c in cases if c['case_id']==e['case_id'] and c.get('method')==e['method'] and c.get('variant','baseline')==e['variant']),None)
        pairkeys[(e['case_id'],e['metric'],e['unit'],capture,gt)].append(e)
    for (cid,metric,unit,capture,gt),group in pairkeys.items():
        # Per-view results can create multiple values per case; mean first.
        methods=defaultdict(list)
        for e in group:methods[(e['method'],e['variant'])].append(e['value'])
        combos=sorted(methods)
        if not capture or not gt: continue
        for i,a in enumerate(combos):
            for b in combos[i+1:]:
                aa,bb=statistics.mean(methods[a]),statistics.mean(methods[b]); example=group[0]
                rows['ablation_results.csv'].append(dict(schema_version=SCHEMA,run_id=run_id,case_id=cid,site_id=example['site_id'],reference_method=a[0],candidate_method=b[0],reference_variant=a[1],candidate_variant=b[1],metric=metric,reference_value=aa,candidate_value=bb,difference=bb-aa,unit=unit,status='evaluated',reason='descriptive_not_significance',source_sha256=capture,provenance='matched_case_capture_and_gt_required'))
    rows['confidence_intervals.csv']=_site_ci(eligible,run_id)
    by_views=defaultdict(list)
    for r in rows['per_view_metrics.csv']:
        if r['status']=='evaluated' and r['metric'] in ('psnr_db','ssim') and r['view_id']!='aggregate':
            by_views[(r['method'],r['variant'],r['metric'],r['unit'])].append(r)
    for (method,variant,metric,unit),rr in sorted(by_views.items()):
        values=sorted(_fnum(x['value']) for x in rr)
        dec=max(1,math.ceil(len(values)*.1))
        idx=max(0,math.ceil(len(values)*.1)-1)
        rows['view_tail_summary.csv'].append(dict(schema_version=SCHEMA,run_id=run_id,method=method,variant=variant,metric=metric,percentile='P10',worst_decile_mean=statistics.mean(values[:dec]),n_views=len(values),n_sites=len({x['site_id'] for x in rr}),unit=unit,status='evaluated',reason='descriptive_lower_tail_heldout_views',provenance='real_reviewed_test_only'))
    for name in COLS:_write_csv(out/name,COLS[name],rows[name])
    # Declared experiment grid and not-evaluated science remain inspectable.
    _json(out/'experiment_registry.json',{'schema_version':SCHEMA,'run_id':run_id,'manifest_sha256':mh,'experiments':registry,'scientific_evaluated_experiments':sum(x['status']=='evaluated' for x in registry),'required_ablation_axes':['old_new','frame_count','resolution','cleaning','algorithm','seed'],'untested_axes_status':'not_evaluated'})
    _json(out/'figure_status.json',render_figures(out,eligible,run_id))
    methods=(Path(__file__).resolve().parents[2]/'docs/research_validation/methods_and_parameters.md')
    if methods.is_file(): (out/'methods_and_parameters.md').write_text(methods.read_text(encoding='utf-8'),encoding='utf-8')
    else: (out/'methods_and_parameters.md').write_text('# Scientific evaluation\nMethods source unavailable; no results certified.\n',encoding='utf-8')
    version={}
    for mod in ('numpy','scipy','Pillow','scikit-image','matplotlib','pytest','lpips'):
        try:version[mod]=importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:version[mod]=None
    _json(out/'reproducibility_manifest.json',{'schema_version':SCHEMA,'run_id':run_id,'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_manifest_sha256':mh,'source_manifest_relative_path':manifest_path.name,'code_sha256':digest(__file__),'git_commit_sha':os.environ.get('MYCON_COMMIT_SHA'),'platform':platform.platform(),'python':platform.python_version(),'dependency_versions':version,'environment_hardware':{'gpu':'not_measured','vram':'not_measured'},'input_mutation':False,'test_execution':'separate_test_log','independent_sites':len({r['site_id'] for r in eligible}),'n_real_metric_cells':len(eligible),'files':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='reproducibility_manifest.json'}})
    return out

def main():
    parser=argparse.ArgumentParser(description='MyCON independent thesis-evidence exporter (read-only inputs)')
    parser.add_argument('--manifest',required=True);parser.add_argument('--output-root',required=True)
    a=parser.parse_args(); print(export(a.manifest,a.output_root))
if __name__=='__main__':main()