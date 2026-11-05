#!/usr/bin/env python3
"""Optional measurable COLMAP sparse-cloud -> MyCON Stage 8 handoff.

Requires REAL IFC plus independent non-coplanar fit/holdout site control points;
does not invent anchors or claim that a sparse cloud proves BIM completion.
Use only for Stage 8 review, not automatic Stage 9/11 progress. All outputs
live in a NEW folder and original Recorder, COLMAP, IFC stay immutable.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from mycon_audit_runtime import atomic_json, sha256


def require_real_ifc(path: Path) -> None:
    if not path.is_file() or path.stat().st_size<1024:
        raise RuntimeError("MYCON_REAL_IFC_REQUIRED_NO_SYNTHETIC_FALLBACK")
    with path.open("rb") as handle:
        head=handle.read(4096).lstrip(b"\xef\xbb\xbf\t\r\n ")
        handle.seek(max(0,path.stat().st_size-4096))
        tail=handle.read().rstrip()
    if not head.startswith(b"ISO-10303-21;") or b"END-ISO-10303-21;" not in tail:
        raise RuntimeError("MYCON_REAL_IFC_STEP_HEADER_OR_FOOTER_MISSING")


def aligned_colmap_cloud(cloud: Path, anchors: Path, ifc: Path,
                         output: Path, max_holdout_m: float=0.05,
                         base_config: Path|None=None) -> dict:
    import numpy as np
    from plyfile import PlyData,PlyElement
    from pipeline.stage_04_5_cams_gs.metric_anchor_gate import (
        estimate_sim3,apply_sim3,read_anchors)
    require_real_ifc(ifc)
    if not cloud.is_file() or cloud.stat().st_size<512:
        raise RuntimeError("MYCON_COLMAP_SPARSE_CANDIDATE_MISSING")
    if not anchors.is_file():
        raise RuntimeError("MYCON_INDEPENDENT_SITE_ANCHORS_MISSING")
    if not 0<max_holdout_m<=0.25:
        raise ValueError("Holdout tolerance must be positive and <=0.25m")
    x,y,xt,yt=read_anchors(anchors)
    scale,rotation,translation=estimate_sim3(x,y)
    residual=np.linalg.norm(apply_sim3(xt,scale,rotation,translation)-yt,axis=1)
    holdout_max=float(residual.max())
    rmse=float(np.sqrt(np.mean(residual**2)))
    if not (np.isfinite(residual).all() and holdout_max<=max_holdout_m
            and rmse<=max_holdout_m):
        raise RuntimeError("MYCON_METRIC_SIM3_HOLDOUT_REJECTED")
    source=PlyData.read(str(cloud))
    if "vertex" not in source:raise RuntimeError("MYCON_COLMAP_PLY_NO_VERTEX")
    vertex=source["vertex"].data
    fields=vertex.dtype.names or ()
    if not all(f in fields for f in ("x","y","z")):
        raise RuntimeError("MYCON_COLMAP_PLY_NO_XYZ")
    xyz=np.column_stack([vertex[k] for k in ("x","y","z")]).astype(np.float64)
    if len(xyz)<100 or not np.isfinite(xyz).all():
        raise RuntimeError("MYCON_COLMAP_SPARSE_TOO_SMALL_OR_NONFINITE")
    points=apply_sim3(xyz,scale,rotation,translation)
    if not np.isfinite(points).all():raise RuntimeError("MYCON_ALIGNED_GEOMETRY_NONFINITE")
    structured=np.empty(len(points),dtype=[
        ("x","<f4"),("y","<f4"),("z","<f4"),
        ("red","u1"),("green","u1"),("blue","u1")])
    for index,key in enumerate(("x","y","z")):
        structured[key]=points[:,index].astype(np.float32)
    for color in ("red","green","blue"):
        if color in fields:
            structured[color]=vertex[color].astype(np.uint8)
        else:
            structured[color]=180
    site_cfg=None
    if base_config is not None:
        # Validate the real site config BEFORE writing even a temporary output.
        import yaml
        if not base_config.is_file():
            raise RuntimeError("MYCON_SITE_CONFIG_REQUIRED_FOR_REAL_STAGE8")
        site_cfg=yaml.safe_load(base_config.read_text(encoding="utf-8"))
        if not isinstance(site_cfg,dict):
            raise RuntimeError("MYCON_STAGE8_BASE_CONFIG_NOT_OBJECT")
    if output.exists() and any(output.iterdir()):
        prior=output/"stage8_metric_review.json"
        if prior.is_file():
            old=json.loads(prior.read_text())
            expected={"colmap_sha256":sha256(cloud),"anchors_sha256":sha256(anchors),
                      "ifc_sha256":sha256(ifc)}
            if all(old.get(k)==v for k,v in expected.items()):
                result=output/"colmap_sparse_aligned_m.ply"
                if result.is_file() and old.get("metric_candidate_sha256")==sha256(result):
                    return old
        raise RuntimeError("MYCON_METRIC_HANDOFF_EXISTS_OR_PARTIAL_PRESERVED")
    output.mkdir(parents=True,exist_ok=False)
    candidate=output/"colmap_sparse_aligned_m.ply"
    partial=output/"colmap_sparse_aligned_m.partial.ply"
    PlyData([PlyElement.describe(structured,"vertex")],text=False).write(str(partial))
    partial.replace(candidate)
    transform=np.eye(4,dtype=np.float64)
    transform[:3,:3]=scale*rotation
    transform[:3,3]=translation
    report={"schema":"mycon.colmap.metric_handoff.v1",
            "status":"METRIC_ANCHORS_PASSED_STAGE8_SURFACE_REVIEW_REQUIRED",
            "colmap_sha256":sha256(cloud),"anchors_sha256":sha256(anchors),
            "ifc_sha256":sha256(ifc),
            "metric_candidate_sha256":sha256(candidate),
            "metric_candidate_ply":str(candidate),"ifc_path":str(ifc),
            "coordinate_frame":"BIM_WORLD_METERS",
            "original_coordinate_frame":"COLMAP_SFM_UNSCALED",
            "sim3_scale":float(scale),"sim3_matrix_4x4":transform.tolist(),
            "fit_anchors":len(x),"holdout_anchors":len(xt),
            "holdout_rmse_m":rmse,"holdout_max_m":holdout_max,
            "metric_anchor_threshold_m":max_holdout_m,
            "geometry_representation":"SPARSE_COLMAP_POINTS_NOT_DENSE_SURFACE",
            "stage8_surface_review_required":True,
            "stage9_progress_authorized":False,
            "stage11_schedule_authorized":False,
            "quality_warning":"Sparse SfM may not support element coverage or completion measurement"}
    if site_cfg is not None:
        # Reuse original site configuration; never edit it.
        # We parsed/validated this BEFORE creating the output directory.
        base=site_cfg
        base.setdefault("bim",{}).update({
            "scan_input_candidates":[str(candidate)],
            "initial_scale_strategy":"fixed_1",
            "allow_synthetic_ifc_fallback_for_tests":False,
            "fail_on_low_registration_quality":True})
        base.setdefault("inputs",{})["ifc"]=str(ifc)
        target=output/"stage8_review_config.yaml"
        target.write_text(yaml.safe_dump(base,allow_unicode=True,sort_keys=False),
                          encoding="utf-8")
        report["stage8_config"]=str(target)
        report["stage8_run_command"]=[
            "python","-m","pipeline.stage_08_bim_eval.run_registration",
            "--config",str(target)]
    atomic_json(output/"stage8_metric_review.json",report)
    return report


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--colmap-ply",type=Path,required=True)
    ap.add_argument("--anchors-csv",type=Path,required=True)
    ap.add_argument("--ifc",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--site-config",type=Path)
    ap.add_argument("--max-holdout-m",type=float,default=0.05)
    args=ap.parse_args(argv)
    report=aligned_colmap_cloud(args.colmap_ply,args.anchors_csv,
                                args.ifc,args.out,args.max_holdout_m,
                                args.site_config)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
