"""Owner A: finite nonmetric geometry FROM one SHA-linked COLMAP sparse-camera frame.

This never aligns a different Depth Fusion cloud or reuses historical poses for a new
R run. Reprojection is same-model evidence, NOT an independent site measurement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import camera_geometry_bridge as cam
from . import research_dense_candidates as candidates

SCHEMA = 'mycon.civil_sfm_feature_evidence.v1'
FRAME = 'colmap_sfm_world_nonmetric'


def _sha(p: Path) -> str:
    return cam.digest(p)


def _read_tracks(path: Path, *, error_px: float, min_track_length: int):
    ids, xyz, rgb, errors, ntrack, rejected = [], [], [], [], [], 0
    for line in path.open(encoding='utf-8'):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        a=line.split()
        if len(a) < 8 or (len(a)-8)%2:
            raise ValueError('Malformed COLMAP points3D track record')
        pid=int(a[0]); v=np.asarray([float(x) for x in a[1:4]])
        color=[int(x) for x in a[4:7]]; err=float(a[7]); length=(len(a)-8)//2
        if not np.isfinite(v).all() or not np.isfinite(err) or err<0 or any(c<0 or c>255 for c in color):
            raise ValueError('Invalid COLMAP 3D point/quality record')
        if err>error_px or length<min_track_length:
            rejected+=1; continue
        ids.append(pid);xyz.append(v);rgb.append(color);errors.append(err);ntrack.append(length)
    if len(ids)!=len(set(ids)) or len(ids)<100:
        raise ValueError('Insufficient unique quality-gated COLMAP sparse tracks')
    return (np.asarray(ids,dtype=np.int64), np.asarray(xyz,np.float64),
            np.asarray(rgb,np.uint8),np.asarray(errors,np.float64),
            np.asarray(ntrack,np.int32),rejected)


def reconcile_r_model(root: Path, cameras: Path, images: Path, points: Path,
                      r_dataset: Path | None, r_poses: Path | None) -> dict:
    """Reconcile the frozen train model with its source image and camera receipts."""
    if r_dataset is None and r_poses is None:
        return {"status": "R_OUTPUT_NOT_SUPPLIED", "shared_world_frame_verified": False}
    if (r_dataset is None) != (r_poses is None):
        raise ValueError("R dataset_manifest and camera_poses must be supplied together")
    from reconstruction.workflow import COLMAP_FRAME, IMAGE_KEYS, parse_colmap_text

    rp = cam.confined(root, r_poses)
    rd = cam.confined(root, r_dataset)
    poses = json.loads(rp.read_text(encoding="utf-8"))
    dataset = json.loads(rd.read_text(encoding="utf-8"))
    if (poses.get("schema_version") != "mycon.reconstruction.camera_poses.v1"
            or dataset.get("schema_version") != "mycon.reconstruction.dataset.v1"):
        raise ValueError("Unsupported Team R dataset/pose contract version")
    run = poses.get("run_id")
    if run != dataset.get("run_id") or not isinstance(run, str) or not run.startswith("recon-"):
        raise ValueError("Team R run mismatch")

    images_sha = _sha(images)
    is_train = poses.get("train_colmap_model_sha256") == images_sha
    is_registered = poses.get("registered_colmap_model_sha256") == images_sha
    evidence = {"r_run_id": run, "r_pose_sha256": _sha(rp), "r_dataset_sha256": _sha(rd),
                "shared_world_frame_verified": False}
    if not (is_train or is_registered):
        return {**evidence, "status": "UNREGISTERED_DIFFERENT_R_COLMAP_MODEL"}

    frames = {f["view_id"]: f for f in dataset["frames"]}
    by_id = {p["view_id"]: p for p in poses["poses"]}
    if (not frames or len(frames) != len(dataset["frames"]) or len(by_id) != len(poses["poses"])
            or frames.keys() != by_id.keys()):
        raise ValueError("R poses must cover exactly the unique dataset views")
    names = {}
    for view, f in frames.items():
        p = by_id[view]
        rel = Path(f["image_relpath"])
        if (rel.is_absolute() or ".." in rel.parts or "\\" in str(rel) or rel.parts[:1] != ("images",)
                or f["split"] not in ("train", "test")
                or any(p.get(k) != f.get(k) for k in ("image_sha256", "capture_session_id", "split", "image_relpath"))):
            raise ValueError("R pose image/session/split/hash identity mismatch")
        name = rel.relative_to("images").as_posix()
        if name in names:
            raise ValueError("Duplicate R source image name")
        names[name] = (f, p)
        if _sha(cam.confined(rd.parent, rel)) != f["image_sha256"]:
            raise ValueError("Team R source image SHA mismatch")

    digests = dict(zip(IMAGE_KEYS, map(_sha, (cameras, images, points))))
    if not is_train or poses.get("train_colmap_text_sha256") is None:
        return {**evidence, "status": "R_MODEL_NOT_BOUND_TO_FROZEN_TRAIN_TRIPLET"}
    if (poses.get("train_colmap_text_sha256") != digests
            or poses.get("dataset_manifest_sha256") != evidence["r_dataset_sha256"]
            or poses.get("test_pixels_used_only_for_pose_localization") is not True
            or dataset.get("metric_scale_verified") is not False
            or poses.get("metric_scale_verified") is True):
        raise ValueError("R train text model or dataset receipt mismatch")
    if tuple(p.name for p in (cameras, images, points)) != IMAGE_KEYS or len(
            {p.parent for p in (cameras, images, points)}) != 1:
        raise ValueError("R train text model must be one COLMAP directory")
    intrinsics, model_views = parse_colmap_text(images.parent)
    training = {name for name, (f, p) in names.items()
                if f["split"] == "train" and p.get("registered") is True}
    if len(training) < 3 or model_views.keys() != training:
        raise ValueError("R frozen train model has held-out, unknown or missing registered views")
    for name, model_view in model_views.items():
        frame, pose = names[name]
        camera = intrinsics.get(model_view["camera_id"])
        matrix = np.asarray(pose.get("world_to_camera_4x4"), dtype=float)
        if (camera is None or pose.get("camera") != camera
                or (camera["width"], camera["height"]) != (frame.get("width"), frame.get("height"))
                or pose.get("camera_id") != model_view["camera_id"]
                or pose.get("coordinate_frame") != COLMAP_FRAME
                or pose.get("units") != "reconstruction_units_unscaled"
                or pose.get("metric_scale_verified") is not False
                or matrix.shape != (4, 4)
                or not np.allclose(matrix, model_view["world_to_camera_4x4"],
                                   atol=1e-8, rtol=0)):
            raise ValueError("R registered train camera parameters or world-to-camera pose mismatch")
    return {**evidence, "status": "R_FROZEN_TRAIN_MODEL_INTERNALLY_RECONCILED",
            "model_role": "frozen_train", "source_model_sha256": digests,
            "registered_train_views": len(training), "shared_world_frame_verified": True}


def generate(root:Path, *, cameras:Path, images:Path, points:Path, photos:list[Path],
             run_id:str, max_error_px:float=2., min_track_length:int=4,
             threshold:float=.075, min_points:int=240, max_planes:int=9,
             trials:int=120, seed:int=17,
             r_dataset:Path|None=None, r_poses:Path|None=None)->dict:
    root=root.resolve()
    if not run_id.startswith('geom-civil-r-frame-') or not run_id[19:].replace('-','').replace('_','').isalnum():
        raise ValueError('Unique geom-civil-r-frame-* run ID required')
    dest=root/'runs'/run_id/'geometry'
    if dest.parent.exists():raise FileExistsError('Refusing to overwrite an existing research run')
    cams_path=cam.confined(root,cameras);images_path=cam.confined(root,images);points_path=cam.confined(root,points)
    pictures=[cam.confined(root,p) for p in photos]
    r_status = reconcile_r_model(root, cams_path, images_path, points_path, r_dataset, r_poses)
    shared = r_status["shared_world_frame_verified"]
    if shared:
        dataset_path = cam.confined(root, r_dataset)
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        allowed = {cam.confined(dataset_path.parent, Path(f["image_relpath"]))
                   for f in dataset["frames"] if f["split"] == "train"}
        if not set(pictures) <= allowed:
            raise ValueError("A source photo is not in the frozen R training split")
        picture_names = {p: p.relative_to(dataset_path.parent / "images").as_posix() for p in pictures}
    else:
        picture_names = {p: p.name for p in pictures}
    if len(pictures)<3 or len(set(picture_names.values()))!=len(pictures):
        raise ValueError('Need 3+ distinct source photos with registered COLMAP names')
    c=cam.parse_cameras(cams_path);p3=cam.parse_points(points_path)
    selected,number=cam.parse_images(images_path,set(picture_names.values()))
    from PIL import Image
    frame_evidence=[]
    observed={}
    for picture in sorted(pictures):
        name = picture_names[picture]
        im=selected[name];intr=c.get(im['camera_id'])
        if not intr:raise ValueError('Image camera not available')
        with Image.open(picture) as photo:
            if photo.size!=(intr['width'],intr['height']) or photo.format not in ('JPEG','PNG'):
                raise ValueError('Image camera dimensions or format mismatch')
            photo.verify()
        repro,dots=cam.reprojection(im,intr,p3)
        if repro['median_px']>2. or repro['p95_px']>5.:
            raise ValueError('COLMAP sparse-camera reprojection quality gate failed')
        obs={int(pid):(float(u),float(v)) for u,v,pid in im['observations']}
        observed[name]=obs
        frame_evidence.append({'name':name,'view_id':im['view_id'],
              'image_sha256':_sha(picture),'image_path_relative':picture.relative_to(root).as_posix(),
              'intrinsics':intr,'w2c':im['world_to_camera_4x4'].tolist(),
              'c2w':im['camera_to_world_4x4'].tolist(),
              'same_model_reprojection':repro,'is_independent_GT':False})
    ids,xyz,rgb,errors,ntracks,rejected=_read_tracks(points_path,error_px=max_error_px,min_track_length=min_track_length)
    # Guaranteed same source model by 3D point IDs; no geometric data from Depth Fusion are mixed in.
    pts=np.asarray([p3[int(pid)] for pid in ids])
    if not np.allclose(pts,xyz,rtol=0,atol=1e-10):
        raise ValueError('3D point ID/source coordinate mismatch')
    planes,edges,corners,qa=candidates.extract(xyz,threshold=threshold,min_points=min_points,
                      max_planes=max_planes,trials=trials,seed=seed,max_sample=min(len(xyz),16000))
    allotted=np.zeros(len(ids),bool)
    for plane in planes:
        supp=np.asarray(plane['support_indices'],int)
        if len(supp):allotted[supp]=True
    features={}
    image_row_index={f['name']:f for f in frame_evidence}
    def source_photo_evidence(indices):
        samples=[]
        unique=np.unique(np.asarray(indices,dtype=int))
        if len(unique) and (unique[0]<0 or unique[-1]>=len(ids)):
            raise ValueError('Feature support outside quality-gated source')
        source_pid=ids[unique]
        for row in frame_evidence:
            view=row['name'];obs=observed[view];found=[(int(i),int(pid),obs[int(pid)]) for i,pid in zip(unique,source_pid) if int(pid) in obs]
            if not found:continue
            source_samples=[{'source_point3D_id':pid,'filtered_point_index':i,
                            'observed_pixel_xy':list(uv),'xyz_source_units':xyz[i].tolist()}
                            for i,pid,uv in found[:50]]
            samples.append({'view_id':row['view_id'],'image_name':view,'image_sha256':row['image_sha256'],
                            'tracked_support_count':len(found),'observed_tracks':source_samples,
                            'meaning':'COLMAP_internal_2D_3D_correspondence_NOT_independent_wall_or_edge_validation'})
        return samples
    for kind,arr,key in [('plane',planes,'plane_id'),('edge',edges,'edge_id'),('corner',corners,'corner_id')]:
        out=[]
        for item in arr:
            a=dict(item)
            if kind=='plane':support=a['support_indices']
            elif kind=='edge':support=[j for group in a['support_indices_per_plane'] for j in group]
            else:support=[]
            photo_evidence=source_photo_evidence(support)
            sample_idx=np.unique(np.asarray(support,dtype=int))[:100]
            a.update({'geometry_frame':FRAME,'units':'arbitrary_COLMAP_units',
                  'semantic_label':'unknown','field_confirmed':False,
                  'image_validated_physical_feature':False,'confidence_calibrated':False,
                  'confidence_probability':None,'engineer_review':'pending',
                  'original_point3D_ids_sample':[int(i) for i in ids[sample_idx]],
                  'original_3d_point_sha256':_sha(points_path),
                  'source_image_track_evidence':photo_evidence,
                  'number_of_photo_views_with_tracked_inliers':len(photo_evidence),
                  'evidence_class':'registered_source_SfM_tracks_only',
                  'reasons':['finite source-point geometric fit, not independently surveyed physical surface',
                             'semantic gravity/height reference not certified',
                             'observations reused from same SfM fit, not independent validation']})
            if kind=='corner':
                a['state']='candidate_extrapolated'
                a['physical_corner_confirmed']=False
                a['hypothetical_intersection_only']=True
                a['support_to_actual_corner_independent']=False
            out.append(a)
        features[kind+'s']=out
    # Output is created only AFTER all source hashes, parser/pose/photo/reprojection and R gates passed.
    dest.mkdir(parents=True,exist_ok=False)
    for name in ('planes','edges','corners'):
        (dest/(name+'.json')).write_text(json.dumps({'schema_version':SCHEMA,'run_id':run_id,
             'coordinate_frame':FRAME,'units':'arbitrary_COLMAP_units',name:features[name]},indent=2,allow_nan=False),encoding='utf-8')
    np.save(dest/'cloud_xyz.npy',xyz,allow_pickle=False)
    np.save(dest/'source_point3d_ids.npy',ids,allow_pickle=False)
    np.save(dest/'unknown_filtered_point_indices.npy',np.flatnonzero(~allotted),allow_pickle=False)
    import trimesh
    col=np.full_like(rgb,190)
    palette=np.array([[80,220,210],[230,145,86],[158,141,235],[226,96,142],
                      [145,210,90],[250,194,88],[80,160,240],[170,205,185],[220,90,90]],dtype=np.uint8)
    for i,pl in enumerate(planes):col[np.asarray(pl['support_indices'],int)]=palette[i%len(palette)]
    ply=trimesh.points.PointCloud(xyz,colors=np.column_stack([col,np.full(len(col),255,dtype=np.uint8)]))
    (dest/'planes_colored.ply').write_bytes(trimesh.exchange.ply.export_ply(ply,encoding='binary_little_endian'))
    (dest/'source_rgb_reference.ply').write_bytes(trimesh.exchange.ply.export_ply(trimesh.points.PointCloud(xyz,colors=np.column_stack([rgb,np.full(len(rgb),255,dtype=np.uint8)])),encoding='binary_little_endian'))
    with (dest/'components.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['feature_id','type','point_support','num_source_photos','residual_p95_nonmetric','status','approved','reviewer','review_notes'])
        for kind in ('planes','edges','corners'):
            for feat in features[kind]:
                sp=feat.get('support_indices',feat.get('support_indices_per_plane',[]))
                if kind=='edges':sp=[i for group in sp for i in group]
                if kind=='corners':sp=[]
                w.writerow([feat.get({'planes':'plane_id','edges':'edge_id','corners':'corner_id'}[kind]),kind,len(set(sp)),feat['number_of_photo_views_with_tracked_inliers'],
                            feat.get('residual_p95_units'),'candidate_unverified',False,'',''])
    stats={'schema_version':SCHEMA,'run_id':run_id,'source_model_frame_verified_internal':True,
          'r_new_run_frame_verified':shared,'r_compatibility':r_status,
          'source_kind':('fresh_R' if shared else 'historical') + '_COLMAP_sparse_reconstruction_NOT_DENSE',
          'source_point_count':len(p3),'quality_gated_point_count':len(ids),
          'quality_rejected_count':rejected,'source_model_registered_camera_count':number,
          'queried_photo_count':len(frame_evidence),'coordinate_frame':FRAME,
          'metric_scale_verified':False,'gravity_verified':False,
          'semantic_classes_verified':False,'field_geometry_verified':False,
          'plane_count':len(planes),'edge_count':len(edges),'hypothetical_corner_count':len(corners),
          'independently_confirmed_corner_count':0,'unassigned_filtered_point_count':int((~allotted).sum()),
          'fitted_plane_threshold_source_units':threshold,'min_track_length':min_track_length,
          'max_per_point_COLMAP_reprojection_error_px':max_error_px,
          'source_reconstruction_diagnostics':qa,
          'photos':frame_evidence,
          'known_limitations':[
            'COLMAP sparse tracks are not a dense structural surface or independent metric GT',
            'Plane normals have unsigned/unknown gravity; floor, wall, ceiling all unknown',
            'Bounded convex hull can enclose unobserved holes; candidates not building elements',
            'No independent holdout geometry/edge/corner precision/recall, no physical measurements',
            'No Depth Fusion alignment; old depth model must remain a SEPARATE unregistered layer']}
    (dest/'geometry_qa.json').write_text(json.dumps(stats,indent=2,allow_nan=False),encoding='utf-8')
    (dest/'camera_views.json').write_text(json.dumps({'schema_version':SCHEMA,
        'coordinate_frame':FRAME,'images':frame_evidence},indent=2),encoding='utf-8')
    # Real source images with only observed COLMAP sparse tracks, explicitly not independent validation.
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for j,row in enumerate(frame_evidence[:3],1):
        with Image.open(cam.confined(root,Path(row['image_path_relative']))) as f: photo=np.array(f.convert('RGB'))
        fig=plt.figure(figsize=(11,6)); ax=fig.add_subplot(121)
        ax.imshow(photo);ax.set_title('Actual source photo: '+row['name']);ax.axis('off')
        for i,pl in enumerate(planes):
            idx=pl['support_indices'];pidset=set(map(int,ids[np.asarray(idx,int)]))
            xy=np.asarray([observed[row['name']][pid] for pid in pidset if pid in observed[row['name']]])
            if len(xy):ax.scatter(xy[:,0],xy[:,1],s=3,c=[palette[i%len(palette)]/255.],alpha=.45)
        ax=fig.add_subplot(122,projection='3d');ii=np.linspace(0,len(xyz)-1,min(7500,len(xyz)),dtype=int)
        ax.scatter(*xyz[ii].T,c=col[ii]/255.,s=.35,alpha=.55)
        ax.view_init(elev=25,azim=20+55*j);ax.set_title('COLMAP sparse cloud (SAME source world)')
        fig.suptitle('INTERNAL SfM TRACKS — NOT FIELD-CONFIRMED WALLS / CORNERS / METRIC')
        fig.tight_layout();fig.savefig(dest/f'real_sfm_shape3_view_{j:02}.png',dpi=115);plt.close(fig)
    # Tiny offline clickable/rotatable sampling Viewer, explicit support point IDs and candidate flags.
    sample=np.linspace(0,len(xyz)-1,min(10000,len(xyz)),dtype=int)
    viewdata={'points':xyz[sample].round(5).tolist(),'colors':col[sample].tolist(),
              'planes':[{'id':p['plane_id'],'count':p['point_count'],'normal':p['normal'],
                  'residual_p95_units':p['residual_p95_units'],'photo_views':p['number_of_photo_views_with_tracked_inliers'],
                  'boundary':p['finite_boundary_xyz']} for p in features['planes']],
              'edges':[{'id':e['edge_id'],'ends':e['endpoints_xyz']} for e in features['edges']],
              'corners':[{'id':f['corner_id'],'xyz':f['xyz'],'state':'candidate_extrapolated'} for f in features['corners']]}
    (dest/'viewer_overlay.json').write_text(json.dumps({'schema_version':SCHEMA,'frame':FRAME,
         'status':'frozen_train_SfM_internal_only' if shared else 'historical_SfM_same_camera_frame_only','metric_scale_verified':False,
         'feature_physical_confirmation':False,'render_data':viewdata},separators=(',',':')),encoding='utf-8')
    (dest/'shape3_sfm_review.html').write_text(_html(viewdata),encoding='utf-8')
    outs={p.name:{'relative_path':p.relative_to(root).as_posix(),'sha256':_sha(p),'bytes':p.stat().st_size}
          for p in sorted(dest.iterdir()) if p.is_file()}
    receipt={'schema_version':SCHEMA,'run_id':run_id,'created_at_utc':datetime.now(timezone.utc).isoformat(),
             'coordinate_frame':FRAME,'units':'arbitrary_COLMAP_units','metric_scale_verified':False,
             'r_run_shared_frame_verified':shared,'r_compatibility':r_status,
             ('source_colmap_model_sha256' if shared else 'source_historical_colmap_model_sha256'):{k:_sha(v) for k,v in
               [('cameras.txt',cams_path),('images.txt',images_path),('points3D.txt',points_path)]},
             'source_photo_sha256':{picture_names[p]:_sha(p) for p in pictures},
             'source_model_physical_surface_verified':False,
             'registration_state':'SFM_R_TRAIN_MODEL_INTERNAL' if shared else 'SFM_SAME_MODEL_ONLY_R_UNREGISTERED',
             'quality_gate':{'min_track_length':min_track_length,'max_error_px':max_error_px,
                 'plane_threshold_unscaled':threshold},
             'outputs':outs}
    (dest/'civil_r_frame_manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return {'status':'generated_frozen_train_sparse_same_camera_frame' if shared else 'generated_historical_sparse_same_camera_frame','run_id':run_id,'planes':len(planes),
        'edges':len(edges),'hypothetical_corners':len(corners),'confirmed_corners':0,
        'quality_points':len(ids),'R_shared_world_frame':shared,'output_dir':str(dest)}


def _html(data:dict)->str:
    # All data are generated from local numeric arrays. Do not fetch external JS/CDN.
    return '''<!doctype html><meta charset="utf-8"><title>MyCON sparse Shape3</title>
<style>body{background:#121c2c;color:#f0f3f8;font:14px system-ui;margin:16px}canvas{border:1px solid #536176;width:100%;height:68vh;touch-action:none}pre{max-height:180px;overflow:auto;white-space:pre-wrap}b{color:#f3bd72}</style>
<h3>MyCON COLMAP — inspectable Shape3</h3><p><b>UNSCALED / NO FIELD CERTIFIED FEATURES.</b> Only SAME-MODEL COLMAP sparse 3D→2D tracks. External depth and independent physical measurements require separate validation. Drag to rotate; scroll to zoom; click plane boundary/edge to inspect.</p>
<canvas id="v" width="1100" height="650"></canvas><pre id="ins">Select a drawn feature...</pre>
<script>const d='''+json.dumps(data,separators=(',',':'))+''';let a=0.8,b=0.4,z=2.6,drag=false,px=0,py=0,targets=[];
const cv=document.getElementById('v'),ctx=cv.getContext('2d'),info=document.getElementById('ins');
let P=d.points,cn=P.length,center=[0,0,0];for(let q of P){for(let i=0;i<3;i++)center[i]+=q[i]/cn}let radius=0;for(let q of P){let h=Math.hypot(...q.map((v,i)=>v-center[i]));radius=Math.max(radius,h)}radius=Math.max(radius,1);
function proj(q){let x=(q[0]-center[0])/radius,y=(q[1]-center[1])/radius,t=(q[2]-center[2])/radius,ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);let xx=x*ca-t*sa, zz=x*sa+t*ca,yy=y*cb-zz*sb,depth=y*sb+zz*cb,k=270*z/(3+depth);return [550+xx*k,325-yy*k]}
function redraw(){ctx.fillStyle='#132237';ctx.fillRect(0,0,1100,650);targets=[];
 for(let i=0;i<P.length;i++){let p=proj(P[i]);ctx.fillStyle=`rgb(${d.colors[i].join(',')})`;ctx.globalAlpha=.6;ctx.fillRect(p[0],p[1],1.6,1.6)}ctx.globalAlpha=1;
 for(let t of d.planes){let pts=t.boundary.map(proj);ctx.strokeStyle='#ffcb69';ctx.lineWidth=2;ctx.beginPath();for(let i=0;i<pts.length;i++){let p=pts[i];if(i)ctx.lineTo(...p);else ctx.moveTo(...p)}ctx.closePath();ctx.stroke();for(let p of pts)targets.push({p,type:'plane',value:t})}
 for(let e of d.edges){let q=e.ends.map(proj);ctx.strokeStyle='#67e8f9';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...q[0]);ctx.lineTo(...q[1]);ctx.stroke();for(let p of q)targets.push({p,type:'edge',value:e})}
 for(let c of d.corners){let p=proj(c.xyz);ctx.strokeStyle='#f87171';ctx.beginPath();ctx.arc(...p,5,0,2*Math.PI);ctx.stroke();targets.push({p,type:'hypothetical corner',value:c})}}
cv.onpointerdown=e=>{drag=true;px=e.clientX;py=e.clientY;cv.setPointerCapture(e.pointerId)};
cv.onpointermove=e=>{if(!drag)return;a+=(e.clientX-px)*.008;b+=(e.clientY-py)*.008;px=e.clientX;py=e.clientY;redraw()};cv.onpointerup=e=>{drag=false};cv.onwheel=e=>{e.preventDefault();z=Math.max(.4,Math.min(8,z*(e.deltaY>0?.90:1.10)));redraw()};
cv.onclick=e=>{let r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*cv.width/r.width,y=(e.clientY-r.top)*cv.height/r.height;let selected=targets.map(t=>({...t,dist:Math.hypot(t.p[0]-x,t.p[1]-y)})).sort((a,b)=>a.dist-b.dist)[0];if(selected&&selected.dist<22)info.textContent=selected.type+' / NOT FIELD CONFIRMED\\n'+JSON.stringify(selected.value,null,2)};redraw();</script>'''


def main(argv=None):
    p=argparse.ArgumentParser(description='Owner A: historical COLMAP same-frame sparse geometry; no metric/physical certification')
    for name in ('project-root','cameras-txt','images-txt','points3d-txt','run-id'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--photo',action='append',required=True)
    p.add_argument('--r-dataset');p.add_argument('--r-camera-poses')
    p.add_argument('--threshold',type=float,default=.075);p.add_argument('--max-planes',type=int,default=9)
    p.add_argument('--trials',type=int,default=120)
    a=p.parse_args(argv)
    try:
        result=generate(Path(a.project_root),cameras=Path(a.cameras_txt),images=Path(a.images_txt),points=Path(a.points3d_txt),
           photos=[Path(x) for x in a.photo],run_id=a.run_id,threshold=a.threshold,max_planes=a.max_planes,trials=a.trials,
           r_dataset=Path(a.r_dataset) if a.r_dataset else None,r_poses=Path(a.r_camera_poses) if a.r_camera_poses else None)
    except (ValueError,TypeError,KeyError,OSError) as exc:
        print(json.dumps({'status':'blocked','reason':str(exc)}));return 2
    print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
