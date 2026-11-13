"""MyCON Owner-A: non-destructive COLMAP/Depth-Fusion co-registration audit.

COLMAP reprojection validates internal sparse-camera consistency ONLY, not
Depth-Fusion alignment. The latter is UNREGISTERED without independently
verified correspondences/transform and survey. No photo-feature overlay is
asserted from an in-FOV projection alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = "mycon.civil_camera_geometry_bridge.v1"
BRIDGE = "mycon.civil_geometry_viewer_handoff.v1"


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def confined(root: Path, p: Path) -> Path:
    p = (root/p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError("Source absent/outside the read-only project tree: " + str(p))
    return p


def xyz_from_ply(path: Path) -> np.ndarray:
    import trimesh
    cloud = trimesh.load(path, file_type="ply", process=False)
    pts = np.asarray(cloud.vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts) or not np.isfinite(pts).all():
        raise ValueError("Invalid source PLY XYZ")
    return pts


def parse_cameras(p: Path) -> dict:
    rows = {}
    for line in p.open(encoding="utf8"):
        if not line.strip() or line.startswith("#"): continue
        a = line.split()
        if a[1] != "SIMPLE_RADIAL" or len(a) != 8:
            raise ValueError("Unsupported camera model; require SIMPLE_RADIAL with distortion")
        key = int(a[0]); w,h=int(a[2]),int(a[3]); f,cx,cy,k=map(float,a[4:])
        if not (w>0 and h>0 and f>0 and np.isfinite([f,cx,cy,k]).all()):
            raise ValueError("Invalid camera intrinsics")
        rows[key] = {"camera_id":key,"model":"SIMPLE_RADIAL","width":w,"height":h,
                     "fx":f,"fy":f,"cx":cx,"cy":cy,"k1":k}
    if not rows: raise ValueError("Empty cameras.txt")
    return rows


def parse_points(p: Path) -> dict[int, np.ndarray]:
    result = {}
    for line in p.open(encoding="utf8"):
        if not line.strip() or line.startswith("#"): continue
        a=line.split(); key=int(a[0]); xyz=np.asarray(list(map(float,a[1:4])))
        if key in result or not np.isfinite(xyz).all(): raise ValueError("Duplicate/invalid 3D track")
        result[key]=xyz
    if not result: raise ValueError("Empty points3D.txt")
    return result


def quat_rot(q: np.ndarray) -> np.ndarray:
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-3:
        raise ValueError("Invalid COLMAP quaternion")
    q=q/np.linalg.norm(q)
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=float)


def parse_images(p: Path, selected: set[str]) -> tuple[dict, int]:
    images = {}
    image_count=0
    with p.open(encoding="utf8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):continue
            a=line.split()
            if len(a) < 10: raise ValueError("Malformed COLMAP images.txt pose line")
            image_count+=1
            name=a[9]
            # COLMAP uses alternating metadata and 2D-keypoint lines; stream without storing 40 MB.
            observations=next(f, "").split()
            if len(observations)%3:raise ValueError("Malformed COLMAP image observations")
            if name in selected:
                if name in images: raise ValueError("Duplicate COLMAP view name")
                R=quat_rot(np.asarray(list(map(float,a[1:5]))))
                t=np.asarray(list(map(float,a[5:8])))
                if not np.isfinite(t).all() or not np.allclose(R@R.T,np.eye(3),atol=1e-5):
                    raise ValueError("Nonrigid image pose")
                w2c=np.eye(4);w2c[:3,:3]=R;w2c[:3,3]=t
                c2w=np.eye(4);c2w[:3,:3]=R.T;c2w[:3,3]=-R.T@t
                obs=[]
                for i in range(0,len(observations),3):
                    point_id=int(observations[i+2])
                    if point_id >= 0:
                        obs.append((float(observations[i]),float(observations[i+1]),point_id))
                images[name]={"view_id":int(a[0]),"camera_id":int(a[8]),"name":name,
                              "world_to_camera_4x4":w2c,"camera_to_world_4x4":c2w,
                              "observations":obs}
    if len(images)!=len(selected):
        raise ValueError("Photo filenames missing in COLMAP registered images: "+repr(sorted(selected-set(images))))
    return images,image_count


def project(pts: np.ndarray, w2c: np.ndarray, camera: dict) -> tuple[np.ndarray,np.ndarray]:
    c=pts@w2c[:3,:3].T+w2c[:3,3]
    z=c[:,2]; ok=z>1e-9
    zsafe=np.where(ok,z,1.)
    x,y=c[:,0]/zsafe,c[:,1]/zsafe
    radius=1+camera["k1"]*(x*x+y*y)
    uv=np.column_stack((camera["fx"]*radius*x+camera["cx"],camera["fy"]*radius*y+camera["cy"]))
    ok &= np.isfinite(uv).all(axis=1)
    return uv,ok


def reprojection(im: dict, camera: dict, points: dict) -> tuple[dict,list]:
    valid=[(u,v,points[pid]) for u,v,pid in im["observations"] if pid in points]
    if len(valid)<30:raise ValueError("Insufficient tracked 3D→2D correspondences")
    xy=np.asarray([(u,v) for u,v,_ in valid]);xyz=np.asarray([p for _,_,p in valid])
    uv,zok=project(xyz,im["world_to_camera_4x4"],camera)
    if zok.sum()<30:raise ValueError("3D features behind camera")
    err=np.linalg.norm(uv[zok]-xy[zok],axis=1)
    order=np.linspace(0,len(err)-1,min(90,len(err)),dtype=int)
    # visual dots are COLMAP's OWN correspondences, NOT Depth-Fusion features
    dots=xy[zok][order].tolist()
    metrics={"num_registered_2d_3d":int(len(err)),"median_px":float(np.median(err)),
             "p95_px":float(np.quantile(err,.95)),"max_px":float(np.max(err)),
             "status":"internal_sparse_camera_consistent" if np.median(err)<2 and np.quantile(err,.95)<5 else "internal_sparse_camera_uncertain",
             "is_evidence_of_depth_fusion_registration":False}
    return metrics,dots


def nearest_diagnostics(depth: np.ndarray, sparse: np.ndarray) -> dict:
    rng=np.random.default_rng(20261114)
    # Use bounded deterministic subset; no rescaling, ICP or alignment fitted to leak overfit.
    ids=rng.choice(len(depth),min(len(depth),30000),replace=False)
    test=depth[ids]
    tree=cKDTree(sparse)
    def score(pts):
        d=tree.query(pts,k=1,workers=1)[0]
        return {"n":len(d),"median_source_units":float(np.median(d)),
                "p95_source_units":float(np.quantile(d,.95)),
                "fraction_within_0_1_source_units":float(np.mean(d<.1))}
    ident=score(test)
    shifted=score(test+np.array([1.,0.,0.]))
    reflected=score(test*np.array([-1.,1.,1.]))
    return {"measurement":"as-is nearest-neighbor spatial overlap; not matched point IDs, not calibration or registration proof",
            "depth_sample_count":len(test),"sparse_count":len(sparse),
            "identity_hypothesis":ident,"translation_x_plus_1_control":shifted,
            "reflection_x_control":reflected,
            "interpretation":"proximity/overlap alone cannot certify identical coordinate frame; correspondence/provenance missing"}


def source_features(root:Path, manifest:Path) -> tuple[dict,dict,dict]:
    m=json.loads(manifest.read_text(encoding="utf8"))
    if m.get("schema_version")!="mycon.civil_geometry_review.v1" or m.get("metric_scale_verified") is not False:
        raise ValueError("Invalid civil source receipt or metric assertion")
    data={}
    for filename in ("planes.json","edges.json","corners.json","clean_cloud.npy","viewer_overlay.json"):
        rec=m.get("outputs",{}).get(filename)
        if not rec:raise ValueError("Missing source civil artifact "+filename)
        p=confined(root,Path(rec["relative_path"]))
        if digest(p)!=rec["sha256"]: raise ValueError("Civil source SHA-256 mismatch: "+filename)
        data[filename]=p
    for name in ("planes","edges","corners"):
        doc=json.loads(data[name+".json"].read_text())
        if doc.get("coordinate_frame")!=m["coordinate_frame"] or doc.get("units")!=m["units"]:
            raise ValueError("Inconsistent civil source frame/units")
        data[name]=doc[name]
    xyz=np.load(data["clean_cloud.npy"],allow_pickle=False)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError("Invalid civil support coordinates")
    return m,data,{"xyz":xyz}


def make_review(root:Path, civil_manifest:Path, sparse_ply:Path, cameras_txt:Path, images_txt:Path,
                points3d_txt:Path, photo_files:list[Path], run_id:str) -> dict:
    root=root.resolve()
    if not run_id.startswith("geom-civil-camera-") or not run_id[18:].replace("-","").replace("_","").isalnum():
        raise ValueError("Run ID requires geom-civil-camera-* unique suffix")
    src=[confined(root,p) for p in (civil_manifest,sparse_ply,cameras_txt,images_txt,points3d_txt)]
    photos=[confined(root,p) for p in photo_files]
    if any(not __import__("re").fullmatch(r"frame_[0-9]{6}\.(?:jpg|png)",p.name) for p in photos):
        raise ValueError("Only explicitly indexed COLMAP frame_000000 photos allowed")
    if len(photos)<3 or len(photos)!=len(set(p.name for p in photos)):
        raise ValueError("Require 3+ distinct photos")
    out=root/"runs"/run_id/"geometry"
    if out.parent.exists():raise FileExistsError("Refusing to overwrite a prior camera run")
    civil,files,cloud=source_features(root,src[0])
    cams=parse_cameras(src[2]);pts=parse_points(src[4]);model,num_registered=parse_images(src[3],set(p.name for p in photos))
    from PIL import Image
    review_views=[]
    for photo in sorted(photos):
        im=model[photo.name];camera=cams.get(im["camera_id"])
        if camera is None:raise ValueError("COLMAP image references missing camera")
        with Image.open(photo) as pic:
            if pic.format not in ("JPEG","PNG") or pic.size!=(camera["width"],camera["height"]):
                raise ValueError("Source photograph format/dimension mismatch")
            pic.verify()
        metrics,dots=reprojection(im,camera,pts)
        view={"name":im["name"],"view_id":im["view_id"],"source_image_sha256":digest(photo),
              "original_source_path":photo.relative_to(root).as_posix(),"camera":camera,
              "world_to_camera_4x4":im["world_to_camera_4x4"].tolist(),
              "camera_to_world_4x4":im["camera_to_world_4x4"].tolist(),
              "sparse_reprojection":metrics,"sparse_reprojection_observed_pixels":dots,
              "depth_fusion_projection_status":"WITHHELD_UNREGISTERED"}
        review_views.append(view)
    sparse=xyz_from_ply(src[1]);depth=np.asarray(cloud["xyz"])
    sparse_tracks=np.asarray(list(pts.values()))
    if len(sparse)!=len(sparse_tracks):
        raise ValueError("Sparse PLY point count differs from COLMAP points3D.txt")
    sparse_match_error=cKDTree(sparse).query(sparse_tracks,k=1,workers=1)[0]
    if float(np.max(sparse_match_error))>1e-5:
        raise ValueError("Sparse PLY is not the geometry stored in COLMAP points3D.txt")
    sparse_track_verification={"points":len(sparse_tracks),"max_nearest_distance_source_units":float(np.max(sparse_match_error)),
            "status":"sparse_ply_matches_points3D_within_1e-5_units", "is_evidence_of_depth_fusion_registration":False}
    proximity=nearest_diagnostics(depth,sparse)
    # Source depth fusion has no verified per-depth pixel→COLMAP track/pose/transform receipt.
    status="UNREGISTERED"
    frame_verified=False
    features=[]
    for kind,key in (("plane","planes"),("edge","edges"),("corner","corners")):
        for f in files[key]:
            fid=f.get("plane_id",f.get("edge_id",f.get("corner_id")))
            supports=f.get("support_indices",[])
            assert isinstance(supports,list)
            if kind=="plane":
                supp=np.asarray(supports,dtype=int)
                if len(supp) and (supp.min()<0 or supp.max()>=len(depth)):
                    raise ValueError("Out of bounds feature support index")
                sample=[{"point_index":int(i),"xyz":depth[i].tolist()} for i in supp[:min(12,len(supp))]]
                support_total=len(supp)
            elif kind=="edge":
                arrays=f.get("support_indices_per_plane",[])
                supp=np.array([i for a in arrays for i in a],dtype=int)
                if len(supp) and (supp.min()<0 or supp.max()>=len(depth)):
                    raise ValueError("Invalid edge support index")
                sample=[{"point_index":int(i),"xyz":depth[i].tolist()} for i in supp[:min(12,len(supp))]]
                support_total=len(supp)
            else:
                sample=[];support_total=min(f.get("support_near_corner_per_plane",[0]))
            features.append({"feature_id":fid,"kind":kind,"geometry_source_sha256":civil["outputs"][key+".json"]["sha256"],
                "frame":civil["coordinate_frame"],"units":civil["units"],
                "semantic_label":"unknown","human_review_state":"pending",
                "field_confirmed":False,"confidence_calibrated":False,
                "support_count":int(support_total),"sampled_source_support_points":sample,
                "residual_p95_source_units":f.get("residual_p95_units"),
                "occupied_fraction_of_hull":f.get("observed_support_qa",{}).get("occupied_fraction_of_hull"),
                "image_candidates":[{"view_id":v["view_id"],"name":v["name"],"source_sha256":v["source_image_sha256"],
                                     "feature_projection":"NOT_ALLOWED_UNREGISTERED","image_support_confirmed":False} for v in review_views],
                "image_support_confirmed_count":0,"reason":"features from real-derived depth point support, camera linkage unproven; no physical identity promoted"})
    # Unclassified points are a SET of observations, not one continuous physical region.
    allocated=np.zeros(len(depth),dtype=bool)
    for plane in files["planes"]:
        ids=np.asarray(plane["support_indices"],dtype=int)
        if len(ids) and (ids.min()<0 or ids.max()>=len(depth)):
            raise ValueError("Plane support outside the point cloud")
        allocated[ids]=True
    unknown_idx=np.flatnonzero(~allocated)
    out.mkdir(parents=True,exist_ok=False)
    np.save(out/"unknown_point_indices.npy",unknown_idx)
    unknown={"schema_version":SCHEMA,"run_id":run_id,
             "meaning":"unassigned, sparse observations; NOT one connected surface or a verified missing wall",
             "point_count":int(len(unknown_idx)),
             "point_indices_sha256":digest(out/"unknown_point_indices.npy"),
             "source_geometry_frame":civil["coordinate_frame"],"units":civil["units"],
             "semantic_label":"unknown","field_confirmed":False,
             "support_sample":[{"point_index":int(i),"xyz":depth[i].tolist()} for i in unknown_idx[:50]],
             "no_inferred_occluded_surface":True}
    (out/"unknown_regions.json").write_text(json.dumps(unknown,indent=2),encoding="utf8")
    for i,p in enumerate(sorted(photos)):
        name=p.name
        shutil.copyfile(p,out/name)
        next(v for v in review_views if v["name"]==name)["review_photo_path"]=name
    reg={"schema_version":SCHEMA,"run_id":run_id,"status":status,
         "registration_verified":frame_verified,"metric_scale_verified":False,"can_project_depth_features_to_images":False,
         "source_depth_geometry_frame":civil["coordinate_frame"],"camera_pose_frame":"COLMAP_sfm_world_unscaled",
         "depth_fusion_source_sha256":civil.get("provenance",{}).get("original_source_sha256"),
         "civil_manifest_sha256":digest(src[0]),"source_file_sha256":{p.name:digest(p) for p in src},
         "camera_extrinsic_convention":"COLMAP WORLD_TO_CAMERA; numeric inverse CAMERA_TO_WORLD",
         "camera_intrinsics_model":"SIMPLE_RADIAL including radial k1", "registered_colmap_view_count":num_registered,
         "reviewed_photo_count":len(review_views),"reviewed_photo_models":review_views,
         "reprojection_diagnostic":"Internal SfM reprojection only; does not certify Depth Fusion correspondence to these images",
         "sparse_ply_to_track_consistency":sparse_track_verification,
         "sparse_depth_geometric_proximity":proximity,
         "failed_registration_gates":["no source depth pixel->frame ID matching receipt/hash ledger",
               "no independently verified depth-fusion to COLMAP transform/correspondences",
               "no heldout cross-modal 3D/2D residual validation",
               "no independent metric survey/gravity axis"],
         "unknown_semantics":["walls","floor","ceiling","occlusion","construction_status"],
         "feature_counts":{"planes":len(files["planes"]),"edges":len(files["edges"]),
                          "corners":len(files["corners"]),"field_verified":0,
                          "unclassified_points":int(len(unknown_idx))},
         "user_visible_warning":"UNREGISTERED: Photographs corroborate their own COLMAP sparse tracks, NOT the depth candidate planes. No 2D overlay or structural acceptance."}
    (out/"registration_qa.json").write_text(json.dumps(reg,indent=2),encoding="utf8")
    (out/"feature_evidence.json").write_text(json.dumps({"schema_version":SCHEMA,"run_id":run_id,
                                    "registration_verified":False,"features":features},indent=2),encoding="utf8")
    with (out/"feature_review.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f);w.writerow(["feature_id","kind","support_points","residual_p95_source_units",
            "occupied_fraction_of_hull","image_support_confirmed_count","registration","semantic","reviewer","decision","note"])
        for item in features:w.writerow([item["feature_id"],item["kind"],item["support_count"],
            item["residual_p95_source_units"],item["occupied_fraction_of_hull"],0,"UNREGISTERED","unknown","","pending",""])
    # Real registered photos on the LEFT, separate 3D depth candidate cloud on the RIGHT.
    # Never paint Depth Fusion features on photo pixels without co-registration.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sampled=depth[np.linspace(0,len(depth)-1,min(14000,len(depth)),dtype=int)]
    from PIL import Image
    for idx,view in enumerate(review_views[:3],1):
        fig=plt.figure(figsize=(12,5.8),facecolor="white")
        left=fig.add_subplot(121)
        with Image.open(out/view["name"]) as im:left.imshow(im)
        pixel=np.asarray(view["sparse_reprojection_observed_pixels"])
        if len(pixel):left.scatter(pixel[:,0],pixel[:,1],s=5,c="#00dd99",alpha=.5)
        left.set_title("REAL source photo + COLMAP sparse tracks\n"+view["name"]);left.axis("off")
        right=fig.add_subplot(122,projection="3d")
        right.scatter(*sampled.T,s=.2,c="#576c80",alpha=.35)
        right.view_init(elev=20+idx*15,azim=70+idx*45)
        right.set_title("Depth-fused candidate cloud — SEPARATE\nNo proven camera/depth registration")
        right.set_xlabel("X, unscaled");right.set_ylabel("Y, unscaled");right.set_zlabel("Z, unscaled")
        fig.suptitle("MyCON Shape3 camera evidence — UNREGISTERED / NO FIELD CERTIFICATION",fontsize=12)
        fig.tight_layout();fig.savefig(out/f"real_photo_shape3_review_{idx:02}.png",dpi=125);plt.close(fig)
    # HTML references existing isolated Round-2 clickable geometry viewer and this run's hashed photographs.
    rel=Path(os.path.relpath(files["viewer_overlay.json"].parent/"geometry_review.html",out)).as_posix()
    photos_markup="".join(f'<figure><img src="{v["review_photo_path"]}" alt="verified copied source photo" style="max-height:320px;max-width:100%"><figcaption>{v["name"]} — COLMAP internal reprojection only, SHA256 {v["source_image_sha256"][:16]}…</figcaption></figure>' for v in review_views)
    html=f'''<!doctype html><meta charset="UTF-8"><title>MyCON Civil Round3 Unregistered</title>
    <style>body{{font:14px system-ui;background:#16212e;color:#f3f4f6;margin:12px}}.warn{{background:#4c2a28;padding:10px}}iframe{{width:100%;height:72vh;border:1px solid #555}}.pics{{display:flex;overflow-x:auto;gap:12px}}figure{{margin:3px;min-width:190px}}</style>
    <h2>MyCON Civil Shape3 — registered camera evidence NOT established</h2>
    <div class="warn">UNREGISTERED: these source photos have valid internal COLMAP sparse reprojections. No 3D Depth Fusion feature has verified photo support. All structural labels UNKNOWN; units nonmetric.</div>
    <h3>Click features in existing Shape3 below (geometric evidence only)</h3><iframe src="{rel}" title="Uncertified geometric feature inspection"></iframe>
    <h3>Independent original site photographs — NOT projected Depth geometry</h3><div class="pics">{photos_markup}</div>
    <p>Machine-readable records: registration_qa.json, feature_evidence.json, feature_review.csv. No walls/floors/corners field-confirmed.</p>'''
    (out/"round3_shape3_photo_review.html").write_text(html,encoding="utf8")
    outputs={p.name:{"path_relative_to_project":p.relative_to(root).as_posix(),"sha256":digest(p),"bytes":p.stat().st_size}
             for p in sorted(out.iterdir()) if p.is_file()}
    bundle={"schema_version":BRIDGE,"run_id":run_id,"registration_status":status,
            "camera_and_depth_co_registered":False,"metric_scale_verified":False,
            "coordinate_frame_depth":civil["coordinate_frame"],"coordinate_frame_colmap":"COLMAP_sfm_world_unscaled",
            "units_depth":civil["units"],"units_camera":"arbitrary_COLMAP_units",
            "source_civil_run_id":civil["run_id"],"source_civil_manifest_sha256":digest(src[0]),
            "registration_qa":outputs["registration_qa.json"],"feature_evidence":outputs["feature_evidence.json"],
            "unknown_regions":outputs["unknown_regions.json"],"unknown_point_indices":outputs["unknown_point_indices.npy"],
            "review_html":outputs["round3_shape3_photo_review.html"],
            "geometry_viewer_overlay":{"path_relative_to_project":files["viewer_overlay.json"].relative_to(root).as_posix(),
                    "sha256":civil["outputs"]["viewer_overlay.json"]["sha256"]},
            "source_camera_photos":[{"name":v["name"],"view_id":v["view_id"],
                  "file":outputs[v["review_photo_path"]],"sparse_reprojection":v["sparse_reprojection"]} for v in review_views],
            "feature_photo_projection_allowed":False,"heatmap_enabled":False,
            "integration_policy":"Viewer must display camera and geometry in SEPARATE unregistered layers, not project geometry or assert physical labels",
            "outputs":outputs,"created_at_utc":datetime.now(timezone.utc).isoformat()}
    (out/"viewer_handoff.json").write_text(json.dumps(bundle,indent=2),encoding="utf8")
    return bundle


def main(argv=None):
    ap=argparse.ArgumentParser(description="Non-destructive real COLMAP internal QA + unregistered civil depth handoff")
    for a in ("project-root","civil-manifest","sparse-ply","cameras-txt","images-txt","points3d-txt","run-id"):
        ap.add_argument("--"+a,required=True)
    ap.add_argument("--photo",action="append",required=True)
    a=ap.parse_args(argv)
    try:
        z=make_review(Path(a.project_root),Path(a.civil_manifest),Path(a.sparse_ply),Path(a.cameras_txt),
                      Path(a.images_txt),Path(a.points3d_txt),[Path(p) for p in a.photo],a.run_id)
    except (ValueError,TypeError,KeyError,OSError) as err:
        print(json.dumps({"status":"blocked","reason":str(err)}));return 2
    print(json.dumps({"status":"generated","run_id":z["run_id"],"registration_status":z["registration_status"],
                      "photos":len(z["source_camera_photos"]),"metric_scale_verified":False}));return 0

if __name__=="__main__":raise SystemExit(main())