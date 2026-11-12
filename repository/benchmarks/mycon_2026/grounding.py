"""Independently annotated per-frame VLM grounding, NOT VLM-generated truth."""
from __future__ import annotations
import math
from collections import defaultdict

def box(raw):
    if not isinstance(raw,(tuple,list)) or len(raw)!=4:
        raise ValueError("expected normalized bbox [x0,y0,x1,y1]")
    x0,y0,x1,y1=(float(x) for x in raw)
    if not all(map(math.isfinite,(x0,y0,x1,y1))) or not (0<=x0<x1<=1 and 0<=y0<y1<=1):
        raise ValueError("invalid ROI bounds")
    return (x0,y0,x1,y1)

def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1])
    x1,y1=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x1-x0)*max(0,y1-y0)
    area_a=(a[2]-a[0])*(a[3]-a[1])
    area_b=(b[2]-b[0])*(b[3]-b[1])
    return intersection/(area_a+area_b-intersection) if area_a+area_b-intersection else 0.0

def evaluate_grounding(claims,annotations,*,approved_measurements=(),min_iou=.5):
    """Measure asserted claims/abstentions against independently reviewed ROI truth.

    Unsupported numeric dimensions and physical progress are hallucinations,
    even if a valid feature ROI is referenced. GT cannot come from the VLM.
    """
    if not 0<min_iou<=1:raise ValueError("invalid IoU acceptance threshold")
    truth={}
    for r in annotations:
        k=(r["view_id"],r["feature_id"],r["label"])
        if k in truth:raise ValueError("duplicated GT ROI")
        truth[k]=box(r["bbox"])
    approved={}
    for r in approved_measurements:
        mid=r.get("measurement_id")
        if not mid or mid in approved:raise ValueError("invalid measurement ID")
        if r.get("reviewed_independently") is True and r.get("source_sha256"):
            approved[mid]=r
    seen=set()
    n_asserted=n_supported=n_hallucinated=n_abstained=0
    per_claim=[]
    repeated=defaultdict(list)
    for c in claims:
        cid=c["claim_id"]
        if cid in seen:raise ValueError("duplicated claim ID")
        seen.add(cid)
        if c.get("state") in ("unknown","abstained"):
            n_abstained+=1
            per_claim.append({"claim_id":cid,"status":"abstained"})
            continue
        if c.get("state","asserted")!="asserted":raise ValueError("invalid state")
        n_asserted+=1
        key=(c.get("view_id"),c.get("feature_id"),c.get("label"))
        measurement=approved.get(c.get("independent_measurement_id"))
        numeric=c.get("assertion_type") in ("dimension","physical_progress")
        valid_measurement=(measurement is not None and
            measurement.get("feature_id")==c.get("feature_id") and
            measurement.get("quantity_type")==c.get("assertion_type") and
            measurement.get("value")==c.get("value") and measurement.get("unit")==c.get("unit"))
        if numeric and not valid_measurement:
            status="unsupported_numeric_claim"
            overlap=0.0
        else:
            roi=truth.get(key)
            overlap=iou(box(c.get("bbox")),roi) if roi else 0.
            status="supported" if overlap>=min_iou else "wrong_feature_or_roi"
        if status=="supported":n_supported+=1
        else:n_hallucinated+=1
        repeated[(c.get("feature_id"),c.get("label"))].append((c.get("view_id"),status=="supported"))
        per_claim.append({"claim_id":cid,"status":status,"bbox_iou":overlap})
    multi=[a for a in repeated.values() if len({x[0] for x in a})>=2]
    stability=(sum(all(y==a[0][1] for _,y in a) for a in multi)/len(multi) if multi else None)
    return {"n_claims":len(claims),"n_asserted":n_asserted,"n_abstained":n_abstained,
        "n_supported":n_supported,"n_hallucinated":n_hallucinated,
        "hallucination_rate_asserted":n_hallucinated/n_asserted if n_asserted else None,
        "abstention_rate":n_abstained/len(claims) if claims else None,
        "grounded_precision":n_supported/n_asserted if n_asserted else None,
        "n_multiview_features":len(multi),"multiview_consistency":stability,
        "per_claim":per_claim}
