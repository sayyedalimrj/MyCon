"""Fail-closed, per-view paired independent-test comparisons for MyCON C.

Never aligns, renders, trains, loads checkpoints, or treats synthetic data as
thesis evidence. A case-level mean alone is never an NVS pair.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

VIEW_METRICS = ('psnr_db', 'ssim', 'lpips')
AUDIT_COLUMNS = ['run_id','case_id','site_id','view_id','reference_method','candidate_method',
    'reference_variant','candidate_variant','metric','reference_image_sha256','camera_receipt_sha256',
    'reference_value','candidate_value','difference','status','reason','expected_views','matched_views',
    'reference_prediction_sha256','candidate_prediction_sha256','provenance']
SUMMARY_COLUMNS = ['run_id','case_id','site_id','reference_method','candidate_method',
    'reference_variant','candidate_variant','metric','n_expected_views','n_matched_views',
    'n_missing_views','mean_candidate_minus_reference','status','reason','provenance']
CI_COLUMNS = ['run_id','reference_method','candidate_method','reference_variant','candidate_variant',
    'metric','n_independent_sites','n_paired_cases','site_mean_delta','ci95_lower','ci95_upper',
    'cluster_unit','status','reason','provenance']


def _write(path, columns, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)


def _valid_hash(x):
    return isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)


def evaluate_paired_views(cases, registry, per_view_rows, run_id, output_dir, *, seed=20261114, draws=2000):
    """Returns audit+summary+CI; only scientifically eligible real held-out cases.

    Matching requires a predeclared complete view roster, per-view RGB GT SHA,
    identical camera/intrinsic/extrinsic receipt SHA and masked pixel protocol.
    No sighting on a single method may be counted as a successful pair.
    """
    approved={(r['case_id'],r['method'],r['variant']) for r in registry
              if r.get('status')=='evaluated' and r.get('scientific_eligible') is True
              and r.get('split')=='test' and r.get('data_kind')=='real'}
    bycase=defaultdict(list)
    for c in cases:
        key=(c['case_id'],c.get('method','unknown'),c.get('variant','baseline'))
        if key in approved: bycase[c['case_id']].append(c)
    scores={}
    for r in per_view_rows:
        if r.get('view_id')!='aggregate':
            key=(r['case_id'],r['method'],r['variant'],r['view_id'],r['metric'])
            if key in scores: raise ValueError('duplicate_per_view_metric_record')
            scores[key]=r
    audit=[];summary=[];site_deltas=defaultdict(lambda:defaultdict(list))
    for case_id,cohort in sorted(bycase.items()):
        cohort=sorted(cohort,key=lambda c:(0 if c.get('paired_reference') is True else 1, c.get('method','unknown'), c.get('variant','baseline')))
        for i,left in enumerate(cohort):
            for right in cohort[i+1:]:
                lm,lv=left.get('method','unknown'),left.get('variant','baseline')
                rm,rv=right.get('method','unknown'),right.get('variant','baseline')
                if (lm,lv)==(rm,rv):continue
                site=left['site_id']; pair=(lm,lv,rm,rv)
                lroster=left.get('paired_test_view_ids'); rroster=right.get('paired_test_view_ids')
                roster_ok=(isinstance(lroster,list) and bool(lroster) and lroster==rroster
                            and len(set(lroster))==len(lroster)
                            and all(isinstance(x,str) and x for x in lroster))
                lviews={v.get('view_id'):v for v in left.get('views',[]) if isinstance(v,dict)}
                rviews={v.get('view_id'):v for v in right.get('views',[]) if isinstance(v,dict)}
                for metric in VIEW_METRICS:
                    deltas=[];total=len(lroster) if roster_ok else 0
                    view_ids=lroster if roster_ok else ['<no_frozen_roster>']
                    for vid in view_ids:
                        a=lviews.get(vid);b=rviews.get(vid)
                        one=scores.get((case_id,lm,lv,vid,metric))
                        two=scores.get((case_id,rm,rv,vid,metric))
                        base=dict(run_id=run_id,case_id=case_id,site_id=site,view_id=vid,
                            reference_method=lm,candidate_method=rm,reference_variant=lv,candidate_variant=rv,
                            metric=metric,reference_image_sha256='',camera_receipt_sha256='',
                            reference_value='',candidate_value='',difference='',status='not_evaluated',
                            reason='',expected_views=total,matched_views=0,
                            reference_prediction_sha256='',candidate_prediction_sha256='',
                            provenance='independent_real_heldout_v1')
                        reason=''
                        if not roster_ok:reason='missing_or_inconsistent_frozen_test_roster'
                        elif (left['site_id'],left['building_id'],left['session_id']) != (right['site_id'],right['building_id'],right['session_id']):
                            reason='site_building_session_mismatch'
                        elif (left.get('capture_sha256')!=right.get('capture_sha256') or
                              not _valid_hash(left.get('capture_sha256'))):reason='capture_sha256_mismatch'
                        elif (left.get('independent_gt_source',{}).get('sha256')!=right.get('independent_gt_source',{}).get('sha256')):
                            reason='independent_gt_sha256_mismatch'
                        elif a is None or b is None:reason='missing_prediction_view'
                        elif not isinstance(a.get('reference_rgb'),dict) or not isinstance(b.get('reference_rgb'),dict):
                            reason='missing_reference_rgb'
                        elif (a['reference_rgb'].get('sha256')!=b['reference_rgb'].get('sha256') or
                              not _valid_hash(a['reference_rgb'].get('sha256'))):reason='reference_rgb_sha256_mismatch'
                        elif (not _valid_hash(a.get('camera_receipt_sha256')) or
                              a.get('camera_receipt_sha256')!=b.get('camera_receipt_sha256')):
                            reason='camera_receipt_sha256_missing_or_mismatch'
                        elif (a.get('mask_sha256')!=b.get('mask_sha256') or
                              a.get('color_space')!=b.get('color_space') or not a.get('color_space')):
                            reason='mask_or_color_space_mismatch'
                        elif not one or not two or one.get('status')!='evaluated' or two.get('status')!='evaluated':
                            reason='missing_or_unmeasured_metric'
                        elif one.get('reference_sha256')!=two.get('reference_sha256'):
                            reason='measured_reference_sha256_mismatch'
                        else:
                            try:
                                la=float(one['value']);rb=float(two['value'])
                                if not all(math.isfinite(x) for x in (la,rb)): raise ValueError()
                                delta=rb-la
                                base.update(reference_image_sha256=one['reference_sha256'],
                                    camera_receipt_sha256=a['camera_receipt_sha256'],reference_value=la,
                                    candidate_value=rb,difference=delta,status='evaluated',
                                    reference_prediction_sha256=one.get('prediction_sha256',''),
                                    candidate_prediction_sha256=two.get('prediction_sha256',''),matched_views=1)
                                deltas.append(delta)
                            except (TypeError,ValueError):reason='nonfinite_or_invalid_numeric_metric'
                        if reason:base['reason']=reason
                        audit.append(base)
                    enough=roster_ok and len(deltas)==total and total>0
                    row=dict(run_id=run_id,case_id=case_id,site_id=site,reference_method=lm,
                        candidate_method=rm,reference_variant=lv,candidate_variant=rv,metric=metric,
                        n_expected_views=total,n_matched_views=len(deltas),n_missing_views=max(0,total-len(deltas)),
                        mean_candidate_minus_reference=statistics.mean(deltas) if enough else '',
                        status='evaluated' if enough else 'not_evaluated',
                        reason='' if enough else 'incomplete_or_unmeasured_paired_roster',
                        provenance='site_clustered_complete_paired_views_only')
                    summary.append(row)
                    if enough:
                        site_deltas[(lm,lv,rm,rv,metric)][site].append(statistics.mean(deltas))
    ci=[]
    for (lm,lv,rm,rv,metric),sites in sorted(site_deltas.items()):
        sitevals=[statistics.mean(v) for v in sites.values()]; n=len(sitevals)
        lo=hi=''
        if n>=2:
            rng=random.Random(f'{seed}:{lm}:{lv}:{rm}:{rv}:{metric}')
            samples=sorted(statistics.mean(rng.choices(sitevals,k=n)) for _ in range(draws))
            lo=samples[int(.025*draws)];hi=samples[min(draws-1,int(.975*draws))]
        ci.append(dict(run_id=run_id,reference_method=lm,candidate_method=rm,
            reference_variant=lv,candidate_variant=rv,metric=metric,n_independent_sites=n,
            n_paired_cases=sum(len(v) for v in sites.values()),site_mean_delta=statistics.mean(sitevals),
            ci95_lower=lo,ci95_upper=hi,cluster_unit='site',status='evaluated' if n>=2 else 'not_evaluated',
            reason='exploratory_small_site_count' if n in (2,3,4) else ('' if n>=5 else 'insufficient_independent_sites'),
            provenance='paired_delta_site_bootstrap_2000_draws'))
    root=Path(output_dir)
    _write(root/'paired_view_audit.csv',AUDIT_COLUMNS,audit)
    _write(root/'paired_view_ablations.csv',SUMMARY_COLUMNS,summary)
    _write(root/'paired_site_confidence_intervals.csv',CI_COLUMNS,ci)
    return {'audit_rows':len(audit),'complete_case_metric_pairs':sum(x['status']=='evaluated' for x in summary),
            'rejected_or_missing_view_metric_pairs':sum(x['status']!='evaluated' for x in audit),
            'independent_site_cluster_intervals':sum(x['status']=='evaluated' for x in ci),
            'paired_test_protocol':'exact_reference_image_camera_mask_color_capture_and_gt_sha256',
            'status':'evaluated' if any(x['status']=='evaluated' for x in summary) else 'not_evaluated'}
