"""Safe R/COLMAP -> V viewer adaptation; NO writes to R/A/D or original data.

R world-to-camera (OpenCV, X-right/Y-down/Z-forward) must be inverted
for the V camera-to-world contract. A geometry is withheld unless a
hash-pinned independent coordinate-frame review explicitly approves it.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import shutil
from pathlib import Path
from .core import ViewerInputError, RUN_ID, _matrix, read_path, sha256, load_bundle

R_DATASET = "mycon.reconstruction.dataset.v1"
R_POSES = "mycon.reconstruction.camera_poses.v1"
A_REVIEW = "mycon.civil_geometry_review.v1"
FRAME_BRIDGE = "mycon.viewer.frame_bridge.v1"
R_CONVENTION = "colmap_opencv_world_to_camera"

def invert_colmap_world_to_camera(w2c):
    """Rigid SE3 inverse: R.T and -R.T@t; preserve optical axes, never flip."""
    if not _matrix(w2c):
        raise ViewerInputError("invalid COLMAP world_to_camera: non-rigid or reflected")
    R = [[w2c[i][j] for j in range(3)] for i in range(3)]
    t = [w2c[i][3] for i in range(3)]
    inv = [[R[j][i] for j in range(3)] for i in range(3)]
    result = [row + [-sum(row[j]*t[j] for j in range(3))] for row in inv]
    result.append([0., 0., 0., 1.])
    if not _matrix(result): raise ViewerInputError("COLMAP inverse is not rigid")
    return result

def _read_json(root, relative, schema, run=None):
    p = read_path(root, relative)
    try: obj = json.loads(p.read_text(encoding="utf8"))
    except (ValueError,UnicodeError) as e: raise ViewerInputError("invalid source JSON") from e
    if not isinstance(obj,dict) or obj.get("schema_version")!=schema:
        raise ViewerInputError("source schema mismatch: "+relative)
    if run is not None and obj.get("run_id")!=run:
        raise ViewerInputError("source run_id mismatch: "+relative)
    return obj,p

def _intrinsics(c):
    counts={"SIMPLE_PINHOLE":3,"PINHOLE":4,"SIMPLE_RADIAL":4,"RADIAL":5,
            "OPENCV":8,"OPENCV_FISHEYE":8}
    if not isinstance(c,dict): raise ViewerInputError("missing registered camera")
    model,par=c.get("model"),c.get("parameters")
    if model not in counts or not isinstance(par,list) or len(par)!=counts[model]:
        raise ViewerInputError("unsupported COLMAP intrinsics")
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in par):
        raise ViewerInputError("nonfinite camera parameters")
    if model in ("PINHOLE","OPENCV","OPENCV_FISHEYE"):
        fx,fy,cx,cy=par[:4];dist=par[4:]
    else:
        fx=fy=par[0];cx,cy=par[1:3];dist=par[3:]
    w,h=c.get("width"),c.get("height")
    if not all(isinstance(v,int) and v>0 for v in (w,h)) or min(fx,fy)<=0 or not(0<=cx<=w and 0<=cy<=h):
        raise ViewerInputError("camera intrinsics/dimensions invalid")
    return {"width":w,"height":h,"fx":fx,"fy":fy,"cx":cx,"cy":cy,
            "calibration_model":model,"distortion_coefficients":dist,
            "frustum_projection":"pinhole_exact" if not any(abs(v)>1e-12 for v in dist) else "approximate_ignore_distortion"}

def _xyz(T,p):
    if not isinstance(p,list) or len(p)!=3 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p):
        raise ViewerInputError("geometry XYZ invalid")
    return [sum(T[i][j]*p[j] for j in range(3))+T[i][3] for i in range(3)]

def _import_A(root,relative,bridge_relative,rsha):
    """A source is validated independently; unsupported co-registration stays hidden."""
    a, ap = _read_json(root,relative,A_REVIEW)
    receipt=a.get("outputs",{}).get("viewer_overlay.json")
    if not isinstance(receipt,dict):
        raise ViewerInputError("A viewer_overlay receipt missing")
    overlay_path=read_path(root,receipt.get("relative_path"))
    if sha256(overlay_path)!=receipt.get("sha256"):
        raise ViewerInputError("A overlay SHA256 mismatch")
    overlay=json.loads(overlay_path.read_text(encoding="utf8"))
    if any(overlay.get(k)!=a.get(k) for k in ("schema_version","run_id","coordinate_frame","units")):
        raise ViewerInputError("A overlay schema/frame/units mismatch")
    audit={"a_manifest_sha256":sha256(ap),"a_overlay_sha256":sha256(overlay_path)}
    if not bridge_relative:
        return None, "A_camera_frame_link_missing; geometry withheld", audit
    proof, bp = _read_json(root,bridge_relative,FRAME_BRIDGE)
    match={"r_camera_poses_sha256":rsha,"a_manifest_sha256":sha256(ap),
           "a_coordinate_frame":a["coordinate_frame"],"r_pose_convention":R_CONVENTION}
    if any(proof.get(k)!=v for k,v in match.items()):
        raise ViewerInputError("A/R source digest or coordinate-frame mismatch")
    if (proof.get("independent_review") is not True or not proof.get("reviewer_id")
            or proof.get("metric_scale_verified") is not False
            or proof.get("same_reconstruction_unit_ratio")!=1):
        raise ViewerInputError("A/R frame match not independently reviewed")
    evidence=proof.get("independent_evidence")
    if not isinstance(evidence,dict) or not isinstance(evidence.get("sha256"),str) or len(evidence["sha256"])!=64:
        raise ViewerInputError("A/R independent evidence missing")
    evidence_path=read_path(root,evidence.get("path"))
    if sha256(evidence_path)!=evidence["sha256"]:
        raise ViewerInputError("A/R independent evidence SHA256 mismatch")
    T=proof.get("geometry_to_r_colmap_world_4x4")
    if not _matrix(T):
        raise ViewerInputError("A/R reviewed transform must be rigid, no reflection/scale")
    audit.update({"frame_bridge_sha256":sha256(bp),"evidence_sha256":sha256(evidence_path),
                  "reviewer_id":proof["reviewer_id"]})
    return (overlay,T), "reviewer_confirmed_nonmetric_alignment_not_surveyed", audit

def convert_r_run(run_dir,destination,*,project_root=None,a_manifest=None,frame_bridge=None):
    """Create only a new V bundle, preserving all R/A originals unchanged."""
    root=Path(run_dir).resolve();dest=Path(destination)
    if not RUN_ID.fullmatch(dest.name) or dest.exists():
        raise FileExistsError("viewer output exists or unsafe Run ID")
    dataset,dpath=_read_json(root,"dataset_manifest.json",R_DATASET,root.name)
    poses,ppath=_read_json(root,"camera_poses.json",R_POSES,root.name)
    if dataset.get("metric_scale_verified") is True or poses.get("metric_scale_verified") is True:
        raise ViewerInputError("R source metric claim requires separate independent validation")
    frames={}
    for f in dataset.get("frames",[]):
        fid=f.get("view_id")
        if not isinstance(fid,str) or fid in frames: raise ViewerInputError("duplicate R frame ID")
        src=read_path(root,f.get("image_relpath"))
        if sha256(src)!=f.get("image_sha256"): raise ViewerInputError("R source image SHA256 mismatch")
        frames[fid]=f
    if not frames:raise ViewerInputError("R has no source images")
    selected=[];excluded=[];seen=set()
    for pose in poses.get("poses",[]):
        fid=pose.get("view_id")
        if fid in seen or fid not in frames: raise ViewerInputError("R pose/frame ID invalid or duplicate")
        seen.add(fid)
        f=frames[fid]
        for key in ("image_relpath","image_sha256","capture_session_id","split"):
            if f.get(key)!=pose.get(key): raise ViewerInputError("R camera/image SHA, path, session or split conflict")
        if not pose.get("registered"):
            excluded.append({"view_id":fid,"reason":"not_registered"});continue
        if pose.get("coordinate_frame")!=R_CONVENTION or pose.get("units")!="reconstruction_units_unscaled":
            raise ViewerInputError("R registered camera coordinate frame/units mismatch")
        intr=_intrinsics(pose.get("camera"))
        if intr["width"]!=f.get("width") or intr["height"]!=f.get("height"):
            raise ViewerInputError("R camera vs source image dimensions mismatch")
        c2w=invert_colmap_world_to_camera(pose.get("world_to_camera_4x4"))
        selected.append((pose,f,intr,c2w))
    if not selected:raise ViewerInputError("no registered cameras; cannot fabricate poses")
    geometry=None;state="no_A_geometry_supplied";gaudit=None
    if a_manifest:
        if project_root is None:raise ViewerInputError("A project root missing")
        geometry,state,gaudit=_import_A(Path(project_root).resolve(),a_manifest,frame_bridge,sha256(ppath))
    entries={"camera_poses":[],"frame_manifest":[],"planes":[],"edges":[],"corners":[],
             "mesh":{"vertices":[],"faces":[]},"alignment_result":{
             "metric_scale_verified":False,"registration_verified":False,
             "independent_control_verified":False}}
    if geometry is not None:
        ov,T=geometry
        entries["planes"]=[{"id":str(p.get("plane_id","unknown")),
             "vertices":[_xyz(T,v) for v in p.get("finite_boundary_xyz",[])],
             "state":"unknown","coverage":None,"confidence":None,
             "source_confidence":p.get("confidence"),"human_review_state":p.get("human_review_state","pending")}
             for p in ov.get("planes",[]) if len(p.get("finite_boundary_xyz",[]))>=3]
        entries["edges"]=[{"id":str(e.get("edge_id","unknown")),
             "start":_xyz(T,e["endpoints_xyz"][0]),"end":_xyz(T,e["endpoints_xyz"][1]),
             "confidence":None,"state":"unknown","source_state":e.get("state")}
             for e in ov.get("edges",[]) if len(e.get("endpoints_xyz",[]))==2]
        entries["corners"]=[{"id":str(c.get("corner_id","unknown")),
             "xyz":_xyz(T,c["xyz"]),"confidence":None,"state":"unknown","source_state":c.get("state")}
             for c in ov.get("corners",[])]
        if len(ov.get("points",[]))>25000:raise ViewerInputError("A overlay too large")
        entries["point_cloud"]={"points":[_xyz(T,p) for p in ov.get("points",[])]}
    quality={"state":"reconstruction_not_field_validated","alignment_state":state,
             "image_sha256_checked":True,"survey_certified":False,
             "frustum_distortion_approximation":any(i["frustum_projection"]!="pinhole_exact" for _,_,i,_ in selected)}
    provenance={"r_run_id":root.name,"r_dataset_sha256":sha256(dpath),
                "r_camera_poses_sha256":sha256(ppath),"a_frame_evidence":gaudit,
                "adapter":"viewer_civil.producer_bridge","run_id":dest.name}
    common={"coordinate_frame":"R_"+root.name+"_colmap_world_unverified","units":"reconstruction_units",
            "quality":quality,"provenance":provenance}
    copies=[]
    for i,(pose,f,intr,c2w) in enumerate(selected,1):
        ext=Path(f["image_relpath"]).suffix.lower()
        if ext not in (".png",".jpg",".jpeg",".webp"):raise ViewerInputError("unsupported source frame format")
        cid=f"CAM-{i:04d}"; fid=f"FRAME-{i:04d}"; relative=f"frames/{fid}{ext}"
        entries["frame_manifest"].append({"frame_id":fid,"path":relative,"sha256":f["image_sha256"],
              "source_view_id":pose["view_id"],"capture_session_id":pose["capture_session_id"],
              "split":pose["split"],"quality":{"state":"source_image_SHA256_verified"}})
        entries["camera_poses"].append({"camera_id":cid,"frame_id":fid,
              "source_view_id":pose["view_id"],"camera_to_world":c2w,
              "extrinsic_convention":"camera_to_world",**intr,
              "quality":{"state":"COLMAP_registered_nonmetric"}})
        copies.append((read_path(root,f["image_relpath"]),relative))
    schema={"camera_poses":"mycon.viewer.camera_poses.v1","frame_manifest":"mycon.viewer.frames.v1",
            "planes":"mycon.viewer.planes.v1","edges":"mycon.viewer.edges.v1",
            "corners":"mycon.viewer.corners.v1","mesh":"mycon.viewer.mesh.v1",
            "point_cloud":"mycon.viewer.point_cloud.v1","alignment_result":"mycon.viewer.alignment.v1"}
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"frames").mkdir()
    for src,p in copies:shutil.copyfile(src,dest/p)
    receipts={}
    for role,version in schema.items():
        if role not in entries:continue
        payload={**common,"schema_version":version}
        if role in ("camera_poses","frame_manifest","planes","edges","corners"):
            key={"camera_poses":"cameras","frame_manifest":"frames",
                 "planes":"planes","edges":"edges","corners":"corners"}[role]
            payload[key]=entries[role]
        else:payload.update(entries[role])
        path=dest/(role+".json")
        with path.open("x",encoding="utf8") as fp:json.dump(payload,fp,allow_nan=False)
        receipts[role]={**common,"schema_version":version,"path":path.name,"sha256":sha256(path)}
    synt=dataset.get("synthetic") if isinstance(dataset.get("synthetic"),bool) else None
    manifest={"schema_version":"mycon.viewer_bundle.v1","run_id":dest.name,
       "created_at_utc":datetime.now(timezone.utc).isoformat(),
       "coordinate_frame":common["coordinate_frame"],"units":common["units"],
       "synthetic":synt,"metric_scale_verified":False,"registration_verified":False,
       "geometry_alignment_status":state,"camera_image_count":len(selected),
       "excluded_views":excluded,"provenance":provenance,"files":receipts}
    with (dest/"viewer_manifest.json").open("x",encoding="utf8") as fp:json.dump(manifest,fp,indent=2,allow_nan=False)
    result=load_bundle(dest)
    assert result["deviation_enabled"] is False
    return dest/"viewer_manifest.json"

def main(argv=None):
    p=argparse.ArgumentParser(description="R-COLMAP -> V, unscaled safe standalone viewer")
    p.add_argument("--r-run-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--a-project-root",type=Path)
    p.add_argument("--a-manifest")
    p.add_argument("--frame-bridge")
    a=p.parse_args(argv)
    try:
        output=convert_r_run(a.r_run_dir,a.out_dir,project_root=a.a_project_root,
                             a_manifest=a.a_manifest,frame_bridge=a.frame_bridge)
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(json.dumps({"status":"blocked","reason":str(e)}));return 2
    print(json.dumps({"status":"created","manifest":str(output),"metric_scale_verified":False}));return 0

if __name__=="__main__":raise SystemExit(main())
