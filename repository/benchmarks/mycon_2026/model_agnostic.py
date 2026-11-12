"""Independent BIM-free geometry readiness scorer for A's optional v1 artifact.

Only verified finite surfaces/edges and *observed* corners score as detections.
Synthetic fixtures test scoring logic, never field performance. Coordinate units
remain reconstruction units until independently metrologically verified.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import mean
from .metrics import read_ply_xyz, reconstruction_metrics, nearest_distances
from .protocol import resolve_input, sha256

SCHEMA = "mycon.comparison_ready_geometry.v1"
TRUTH = "mycon.structural_gt.v1"

def vec(x, n=3):
    if not isinstance(x, (list,tuple)) or len(x)!=n:
        raise ValueError("invalid coordinate dimensionality")
    result=tuple(float(y) for y in x)
    if not all(math.isfinite(y) for y in result):
        raise ValueError("nonfinite coordinate")
    return result

def norm(x):
    x=vec(x)
    z=math.sqrt(sum(t*t for t in x))
    if z < 1e-12: raise ValueError("degenerate normal")
    return tuple(t/z for t in x)

def plane(p):
    c=p.get("coefficients",p.get("plane"))
    if c is not None:
        a,b,c,d=vec(c,4)
        length=math.sqrt(a*a+b*b+c*c)
        if length < 1e-12: raise ValueError("degenerate plane")
        return (a/length,b/length,c/length),d/length
    a=vec(p.get("normal"))
    length=math.sqrt(sum(x*x for x in a))
    if length < 1e-12: raise ValueError("degenerate plane")
    return tuple(x/length for x in a),float(p.get("offset",p.get("d",0)))/length

def endpoints(row):
    e=row.get("endpoints",[row.get("start"),row.get("end")])
    if not isinstance(e,list) or len(e)!=2: raise ValueError("finite edge endpoints required")
    a,b=vec(e[0]),vec(e[1])
    if math.dist(a,b) < 1e-8: raise ValueError("zero-length edge")
    return a,b

def load_features(path,key):
    j=json.loads(path.read_text(encoding="utf-8"))
    x=j if isinstance(j,list) else j.get(key)
    if not isinstance(x,list): raise ValueError("invalid "+key+" list")
    return x

def load_ready(path,root=None):
    path=Path(path).resolve()
    root=Path(root or path.parent).resolve()
    m=json.loads(path.read_text(encoding="utf-8"))
    if m.get("schema_version")!=SCHEMA: raise ValueError("unknown A readiness schema")
    if m.get("units") not in {"m","arbitrary","reconstruction_units","unitless"}:
        raise ValueError("source units must be declared")
    if not m.get("coordinate_frame"): raise ValueError("missing source frame")
    if m.get("metric_scale_verified") is True and not m.get("independent_metric_evidence"):
        raise ValueError("metric scaling requires independent controls")
    if m.get("registration_verified") is True and not m.get("independent_registration_evidence"):
        raise ValueError("BIM registration requires independent evidence")
    files={}
    for name in ("clean_cloud","planes","edges","corners"):
        item=(m.get("outputs") or {}).get(name,m.get(name))
        if not isinstance(item,dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError("missing A output hash: "+name)
        f=resolve_input(root,item["path"])
        if not f.is_file() or sha256(f).lower()!=str(item["sha256"]).lower():
            raise ValueError("A output SHA mismatch: "+name)
        files[name]=f
    assets={key:load_features(files[key],key) for key in ("planes","edges","corners")}
    for key,records in assets.items():
        ids=set()
        for r in records:
            name=r.get("id") or r.get(key[:-1]+"_id")
            if not isinstance(name,str) or not name or name in ids:
                raise ValueError("duplicate/invalid feature identifier")
            ids.add(name)
            if key=="planes": plane(r)
            elif key=="edges": endpoints(r)
            else:
                vec(r.get("xyz",r.get("point")))
                if r.get("state") not in {"observed_supported","candidate_extrapolated","unknown"}:
                    raise ValueError("unknown corner state")
    return m,files,assets

def match(pred,truth,metric,tau):
    pairs=[]
    for i,a in enumerate(pred):
        for j,b in enumerate(truth):
            d=metric(a,b)
            if d is not None and math.isfinite(d) and d<=tau: pairs.append((d,i,j))
    out=[]; used_pred=set();used_truth=set()
    for d,i,j in sorted(pairs):
        if i not in used_pred and j not in used_truth:
            out.append(d);used_pred.add(i);used_truth.add(j)
    return out

def counts(prefix,n_pred,n_gt,matched):
    t=len(matched)
    return {prefix+"_tp":t,prefix+"_fp":n_pred-t,prefix+"_fn":n_gt-t,
        prefix+"_n_pred":n_pred,prefix+"_n_gt":n_gt,
        prefix+"_precision":t/n_pred if n_pred else None,
        prefix+"_recall":t/n_gt if n_gt else None,
        prefix+"_f1":2*t/(n_pred+n_gt) if n_pred+n_gt else None,
        prefix+"_localization_mae":mean(matched) if matched else None}

def bounded_plane_error(a,b,max_angle_deg):
    na,da=plane(a);nb,db=plane(b)
    angle=math.degrees(math.acos(min(1.,max(-1.,abs(sum(x*y for x,y in zip(na,nb)))))))
    if angle>max_angle_deg: return None
    if not a.get("support_count") or not b.get("support_count"):return None
    ea,eb=a.get("support_extent"),b.get("support_extent")
    if not ea or not eb or len(ea)!=2 or len(eb)!=2: return None
    if any(math.dist(vec(e[0]),vec(e[1]))<1e-8 for e in (ea,eb)):return None
    if a.get("centroid") is None or b.get("centroid") is None:return None
    sign=1 if sum(x*y for x,y in zip(na,nb))>=0 else -1
    return abs(da-sign*db)+math.dist(vec(a["centroid"]),vec(b["centroid"]))

def finite_edge_error(a,b):
    if not a.get("support_count"):return None
    if len(a.get("plane_ids",[]))<2 and not a.get("validated_feature_line"):return None
    x,y=endpoints(a);u,v=endpoints(b)
    return min((math.dist(x,u)+math.dist(y,v))/2,(math.dist(x,v)+math.dist(y,u))/2)

def evaluate_ready(ready,truth,*,root=None,tau=.05,max_points=20000,seed=42,min_support=3):
    if not math.isfinite(tau) or tau<=0: raise ValueError("tau must be positive")
    m,files,pred=load_ready(ready,root)
    gt=json.loads(Path(truth).read_text(encoding="utf-8"))
    if gt.get("schema_version")!=TRUTH:raise ValueError("unknown independent GT schema")
    if (m["units"],m["coordinate_frame"])!=(gt.get("units"),gt.get("coordinate_frame")):
        raise ValueError("units/frame mismatch, no automatic registration")
    cloud,n_points=read_ply_xyz(files["clean_cloud"],max_points=max_points,seed=seed)
    origin=gt.get("points",[])
    true=[vec(p["xyz"]) for p in origin if p.get("class")=="structure"]
    outliers=[vec(p["xyz"]) for p in origin if p.get("class")=="outlier"]
    result={"n_clean_points":n_points,"n_gt_true_points":len(true),
        "metric_scale_verified":bool(m.get("metric_scale_verified",False)),
        "registration_verified":bool(m.get("registration_verified",False)),
        "units":m["units"]}
    source_count=m.get("input_point_count")
    result["point_removal_fraction"]=(int(source_count)-n_points)/int(source_count) if source_count else None
    if source_count is not None and int(source_count)<n_points:raise ValueError("cleaned points exceed input")
    if true:
        distances=nearest_distances(true,cloud)
        result["structure_retention_fraction"]=sum(x<=tau for x in distances)/len(distances)
        result.update(reconstruction_metrics(cloud,true,tau_m=tau))
        if m["units"]!="m":
            result["chamfer_l1_source_units"]=result.pop("chamfer_l1_m")
            result["chamfer_l2_source_units_squared"]=result.pop("chamfer_l2_m2")
            result["tau_source_units"]=result.pop("tau_m")
            for k in list(result):
                if k.endswith("m") and "_at_0." in k:
                    result[k.replace("m","_source_units")]=result.pop(k)
    else:
        result["structure_retention_fraction"]=None
    isolated=[o for o in outliers if not true or min(math.dist(o,p) for p in true)>2*tau]
    d=nearest_distances(isolated,cloud) if isolated else []
    result["outlier_removal_fraction"]=sum(x>tau for x in d)/len(d) if d else None
    result["n_outlier_gt_eligible"]=len(d)
    result["n_outlier_gt_ambiguous"]=len(outliers)-len(isolated)
    p=match(pred["planes"],gt.get("planes",[]),lambda a,b:bounded_plane_error(a,b,7.5),tau)
    result.update(counts("plane",len(pred["planes"]),len(gt.get("planes",[])),p))
    e=match(pred["edges"],gt.get("edges",[]),finite_edge_error,tau)
    result.update(counts("edge",len(pred["edges"]),len(gt.get("edges",[])),e))
    observed=[c for c in pred["corners"] if c.get("state")=="observed_supported"
        and int(c.get("support_count",0))>=min_support and len(c.get("feature_ids",[]))>=2]
    visible=[c for c in gt.get("corners",[]) if c.get("observable",True) is True]
    c=match(observed,visible,lambda a,b:math.dist(vec(a.get("xyz",a.get("point"))),vec(b.get("xyz",b.get("point")))),tau)
    result.update(counts("corner",len(observed),len(visible),c))
    result["corner_n_unknown_or_extrapolated"]=len(pred["corners"])-len(observed)
    result["corner_unknown_rate"]=(len(pred["corners"])-len(observed))/len(pred["corners"]) if pred["corners"] else None
    result["n_gt_corners_unobservable"]=len(gt.get("corners",[]))-len(visible)
    normal_item=(m.get("outputs") or {}).get("normals")
    if normal_item and normal_item.get("path") and normal_item.get("sha256"):
        nf=resolve_input(Path(root or Path(ready).parent).resolve(),normal_item["path"])
        if not nf.is_file() or sha256(nf)!=normal_item["sha256"]:
            raise ValueError("normal sample SHA mismatch")
        gt_normal={r["sample_id"]:norm(r["normal"]) for r in gt.get("normals",[])}
        pred_normal={r["sample_id"]:norm(r["normal"]) for r in load_features(nf,"normals")}
        shared=sorted(gt_normal.keys() & pred_normal.keys())
        if len(gt_normal)!=len(gt.get("normals",[])) or len(pred_normal)!=len(load_features(nf,"normals")):
            raise ValueError("duplicate normal sample IDs")
        ang=[]
        for k in shared:
            dot=sum(a*b for a,b in zip(gt_normal[k],pred_normal[k]))
            ang.append(math.degrees(math.acos(min(1.,max(-1.,abs(dot))))))
        result["normal_n_pairs"]=len(shared)
        result["normal_angular_mae_deg"]=mean(ang) if ang else None
        result["normal_flip_rate"]=sum(sum(a*b for a,b in zip(gt_normal[k],pred_normal[k]))<0 for k in shared)/len(shared) if shared else None
    else:
        result.update(normal_n_pairs=0,normal_angular_mae_deg=None,normal_flip_rate=None)
    return result
