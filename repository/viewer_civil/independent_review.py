"""Read-only A historical camera + Shape3 workspace; never co-register by appearance.

Uses the existing V bundle and preview renderer. Fresh R output is an optional
third independent view, never substituted by historical cameras. No source
files are modified; only a caller-selected new output directory is written.
"""
from __future__ import annotations
import argparse
import html
import json
import math
import re
import shutil
from pathlib import Path
from .core import ViewerInputError, RUN_ID, read_path, sha256, load_bundle, build_preview
from .producer_bridge import invert_colmap_world_to_camera, _intrinsics, convert_r_run, _xyz

HANDOFF = 'mycon.civil_geometry_viewer_handoff.v1'
QA = 'mycon.civil_camera_geometry_bridge.v1'
REVIEW = 'mycon.civil_geometry_review.v1'
WORKSPACE = 'mycon.viewer.independent_review.v1'
IDENTITY = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]


def _require(condition, message):
    if not condition:
        raise ViewerInputError(message)


def _load(path):
    obj=json.loads(path.read_text(encoding='utf-8'))
    _require(isinstance(obj,dict),'expected JSON object')
    return obj


def _checked(root, rec):
    _require(isinstance(rec,dict),'missing source receipt')
    path=read_path(root,rec.get('path_relative_to_project'))
    _require(sha256(path)==rec.get('sha256'),'source SHA256 mismatch: '+path.name)
    if 'bytes' in rec:
        _require(path.stat().st_size==rec['bytes'],'source byte count mismatch')
    return path


def _same_transform(a,b):
    return isinstance(b,list) and len(b)==4 and all(isinstance(r,list) and len(r)==4 for r in b) and all(
        isinstance(b[i][j],(int,float)) and math.isfinite(b[i][j]) and abs(a[i][j]-b[i][j])<1e-6
        for i in range(4) for j in range(4))


def _native_colmap(root):
    """Strict text model parser; source files have already been hash checked."""
    native={}
    for line in (root/'inputs/colmap_text/cameras.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        a=line.split(); cid=int(a[0]); _require(cid not in native,'duplicate native camera ID')
        native[cid]=_intrinsics({'model':a[1],'width':int(a[2]),'height':int(a[3]),'parameters':list(map(float,a[4:]))})
    poses={}; names=set()
    with (root/'inputs/colmap_text/images.txt').open() as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):continue
            a=line.split(maxsplit=9); _require(len(a)==10,'malformed native image row')
            vid=int(a[0]); name=a[9].strip(); _require(vid>0 and vid not in poses and name not in names,'duplicate native image ID/name')
            names.add(name); w,x,y,z=map(float,a[1:5]); t=list(map(float,a[5:8])); n=math.sqrt(w*w+x*x+y*y+z*z)
            _require(math.isfinite(n) and abs(n-1)<1e-5,'invalid native quaternion')
            w,x,y,z=[v/n for v in (w,x,y,z)]
            R=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
            W=[R[i]+[t[i]] for i in range(3)]+[[0,0,0,1]]
            poses[vid]=(name,native[int(a[8])],W,invert_colmap_world_to_camera(W))
            _require(next(f,None) is not None,'missing native observation row')
    points=[]; ids=set()
    with (root/'inputs/colmap_text/points3D.txt').open() as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):continue
            a=line.split();pid=int(a[0]); _require(pid not in ids,'duplicate native point ID');ids.add(pid)
            points.append(_xyz(IDENTITY,list(map(float,a[1:4]))))
    _require(bool(poses) and bool(points),'empty historical COLMAP model')
    n=len(points); step=max(1,math.ceil(n/25000))
    return poses,points[::step],n


def _inspect(root, handoff_path, expected_sha):
    """Validate producer hashes, original photo bytes, pose/intrinsics and frame labels."""
    hp=read_path(root,handoff_path)
    _require(re.fullmatch(r'[a-f0-9]{64}',expected_sha or '') and sha256(hp)==expected_sha,
             'A handoff must match externally supplied SHA256')
    h=_load(hp)
    _require(h.get('schema_version')==HANDOFF and RUN_ID.fullmatch(str(h.get('run_id',''))), 'A handoff schema/run mismatch')
    # This adapter only handles the explicitly separate, nonmetric A handoff.
    _require(h.get('registration_status')=='UNREGISTERED' and all(h.get(k) is False for k in
        ('camera_and_depth_co_registered','metric_scale_verified','feature_photo_projection_allowed','heatmap_enabled')),
        'A handoff must explicitly prohibit co-registration and metric coloring')
    qp=_checked(root,h.get('registration_qa')); q=_load(qp)
    _require(q.get('schema_version')==QA and q.get('run_id')==h['run_id'] and q.get('status')=='UNREGISTERED', 'A QA schema/run/status mismatch')
    _require(all(q.get(k) is False for k in ('registration_verified','metric_scale_verified','can_project_depth_features_to_images')), 'A QA flags conflict')
    _require(q.get('camera_pose_frame')==h.get('coordinate_frame_colmap')=='COLMAP_sfm_world_unscaled' and
             h.get('units_camera')=='arbitrary_COLMAP_units' and q.get('source_depth_geometry_frame')==h.get('coordinate_frame_depth'), 'A frame/units mismatch')
    op=_checked(root,h.get('geometry_viewer_overlay')); o=_load(op)
    _require(o.get('schema_version')==REVIEW and o.get('run_id')==h.get('source_civil_run_id') and
             o.get('coordinate_frame')==h.get('coordinate_frame_depth') and o.get('units')==h.get('units_depth') and
             o.get('metric_scale_verified') is False, 'A geometry overlay schema/frame/units mismatch')
    _require(o.get('units') in ('arbitrary_reconstruction_units','reconstruction_units','unknown_reconstruction_units'), 'A unsupported geometry units')
    cp=read_path(root,(op.parent/'civil_geometry_manifest.json').relative_to(root).as_posix())
    c=_load(cp)
    _require(sha256(cp)==h.get('source_civil_manifest_sha256')==q.get('civil_manifest_sha256') and
             c.get('schema_version')==REVIEW and c.get('run_id')==o['run_id'] and
             c.get('coordinate_frame')==o['coordinate_frame'] and c.get('units')==o['units'] and
             c.get('metric_scale_verified') is False and c.get('registration_verified') is False and
             c.get('outputs',{}).get('viewer_overlay.json',{}).get('sha256')==sha256(op), 'A civil source manifest/overlay mismatch')
    # Check all three native geometry role receipts and feature evidence, not just the overlay.
    for role in ('planes','edges','corners'):
        rec=c['outputs'][role+'.json']; p=read_path(root,rec['relative_path'])
        _require(sha256(p)==rec['sha256'],'A feature role SHA mismatch: '+role)
    for key in ('feature_evidence','unknown_regions','unknown_point_indices'):
        _checked(root,h.get(key))
    # Independently pin actual historical native model files against the producer QA ledger.
    for name in ('cameras.txt','images.txt','points3D.txt','colmap_sparse_unscaled.ply'):
        rel='inputs/'+('colmap_text/' if name.endswith('.txt') else '')+name
        _require(sha256(read_path(root,rel))==q.get('source_file_sha256',{}).get(name),'historical COLMAP source SHA mismatch: '+name)
    native,sparse,sparse_count=_native_colmap(root)
    _require(len(native)==q['registered_colmap_view_count'] and sparse_count==q.get('sparse_ply_to_track_consistency',{}).get('points'),'historical model count mismatch')
    models={x['view_id']:x for x in q['reviewed_photo_models']}
    _require(len(models)==len(q['reviewed_photo_models'])==q['reviewed_photo_count']==len(h['source_camera_photos']) and len(models)>0,'duplicate/missing A photo models')
    photos=[]; seen=set()
    from PIL import Image
    for rec in h['source_camera_photos']:
        vid=rec['view_id']; model=models.get(vid)
        _require(vid not in seen and model is not None and model['name']==rec['name'] and model['depth_fusion_projection_status']=='WITHHELD_UNREGISTERED','A photo/pose identity conflict')
        seen.add(vid); image=_checked(root,rec['file']); original=read_path(root,model['original_source_path'])
        _require(sha256(image)==sha256(original)==model['source_image_sha256'],'A original/preview image SHA mismatch')
        k=model['camera']
        _require(k['model']=='SIMPLE_RADIAL','unsupported A intrinsics model')
        _require(k['fx']==k['fy'],'A SIMPLE_RADIAL focal lengths disagree')
        intr=_intrinsics({'model':k['model'],'parameters':[k['fx'],k['cx'],k['cy'],k['k1']], 'width':k['width'],'height':k['height']})
        with Image.open(image) as im:
            _require(im.size==(intr['width'],intr['height']) and im.format in ('JPEG','PNG','WEBP'),'A image decode/dimensions mismatch')
            im.verify()
        inverse=invert_colmap_world_to_camera(model['world_to_camera_4x4'])
        _require(_same_transform(inverse,model['camera_to_world_4x4']),'A supplied inverse differs from COLMAP SE3 inverse')
        native_name,native_intr,native_w2c,_=native[vid]
        _require(native_name==model['name'] and native_intr==intr and _same_transform(native_w2c,model['world_to_camera_4x4']),'A photo pose differs from native COLMAP row')
        photos.append((vid,image,intr,inverse,model))
    _require(q['registered_colmap_view_count']>=len(photos),'invalid registered camera count')
    return h,q,o,photos,native,sparse,{'a_run_id':h['run_id'],'a_handoff_sha256':expected_sha,'a_qa_sha256':sha256(qp),'a_overlay_sha256':sha256(op),'a_civil_manifest_sha256':sha256(cp)}


def _bundle(dest, frame, units, entries, copies, provenance, source_label):
    common={'coordinate_frame':frame,'units':units,'quality':{'state':'source_candidate_not_field_validated'},'provenance':{**provenance,'run_id':dest.name}}
    base={'camera_poses':{'cameras':[]},'frame_manifest':{'frames':[]},'planes':{'planes':[]},'edges':{'edges':[]},'corners':{'corners':[]},'mesh':{'vertices':[],'faces':[]},'alignment_result':{'metric_scale_verified':False,'registration_verified':False,'independent_control_verified':False}}
    base.update(entries)
    versions={'camera_poses':'camera_poses','frame_manifest':'frames','planes':'planes','edges':'edges','corners':'corners','mesh':'mesh','alignment_result':'alignment','point_cloud':'point_cloud'}
    dest.mkdir(parents=True,exist_ok=False);(dest/'frames').mkdir()
    for source,rel in copies: shutil.copyfile(source,dest/rel)
    receipts={}
    for role, body in base.items():
        path=dest/(role+'.json'); meta={**common,'schema_version':'mycon.viewer.'+versions[role]+'.v1'}
        path.write_text(json.dumps({**meta,**body},allow_nan=False),encoding='utf-8')
        receipts[role]={**meta,'path':path.name,'sha256':sha256(path)}
    manifest={**common,'schema_version':'mycon.viewer_bundle.v1','run_id':dest.name,'synthetic':None,'metric_scale_verified':False,'registration_verified':False,
        'geometry_alignment_status':'UNREGISTERED — independent viewport; no camera/geometry correspondence', 'source_label':source_label,'source_run_id':provenance.get('a_run_id'),'camera_photo_policy':'partial_native_model' if dest.name=='historical_cameras' else 'all_linked','files':receipts}
    (dest/'viewer_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    _require(load_bundle(dest)['deviation_enabled'] is False,'unexpected metric gate')


def create_workspace(project_root, handoff_path, expected_sha, output, *, r_run_dir=None, preview=True):
    root=Path(project_root).resolve(); out=Path(output).resolve()
    _require(RUN_ID.fullmatch(out.name) and not out.exists() and not out.is_relative_to(root),'output must be new and outside source tree')
    if r_run_dir is not None:
        _require(not out.is_relative_to(Path(r_run_dir).resolve()),'output must be outside R source')
    h,q,o,photos,native,sparse,prov=_inspect(root,handoff_path,expected_sha)
    # Keep the original A coordinate frame, do not rename into a camera frame or transform.
    planes=[{'id':str(p['plane_id']),'vertices':[_xyz(IDENTITY,v) for v in p['finite_boundary_xyz']], 'state':'unknown','confidence':None,'coverage':None,'source_confidence':p.get('confidence'),'human_review_state':'pending'} for p in o['planes'] if len(p['finite_boundary_xyz'])>=3]
    edges=[{'id':str(e['edge_id']),'start':_xyz(IDENTITY,e['endpoints_xyz'][0]),'end':_xyz(IDENTITY,e['endpoints_xyz'][1]),'state':'unknown','confidence':None,'source_state':e.get('state')} for e in o['edges']]
    corners=[{'id':str(c['corner_id']),'xyz':_xyz(IDENTITY,c['xyz']),'state':'unknown','confidence':None,'source_state':c.get('state')} for c in o['corners']]
    points=[_xyz(IDENTITY,p) for p in o['points']]
    _require(len(points)<=25000,'A point preview exceeds producer sample cap')
    for items in (planes,edges,corners): _require(len({x['id'] for x in items})==len(items),'duplicate feature ID')
    cams=[];frames=[];copies=[]
    photo_by_id={v[0]:v for v in photos}
    for vid,(name,intr,w2c,inv) in sorted(native.items(),key=lambda item:item[1][0]):
        cid=f'CAM-{vid:04d}';fid=None; photo=photo_by_id.get(vid)
        if photo is not None:
            _,image,_,_,model=photo;fid=f'FRAME-{vid:04d}';rel=f'frames/{fid}{image.suffix.lower()}'
            frames.append({'frame_id':fid,'path':rel,'sha256':sha256(image),'source_view_id':name,'quality':{'state':'A_original_photo_hash_checked'}})
            copies.append((image,rel))
        cams.append({'camera_id':cid,'frame_id':fid,'source_view_id':name,'camera_to_world':inv,'extrinsic_convention':'camera_to_world',**intr,'quality':{'state':'historical_COLMAP_nonmetric' if photo else 'registered_pose_photo_not_supplied'}})
    # Data files are copied only after all A input checks succeed.
    out.mkdir(parents=True,exist_ok=False)
    _bundle(out/'historical_cameras',h['coordinate_frame_colmap'],'reconstruction_units',{'camera_poses':{'cameras':cams},'frame_manifest':{'frames':frames},'point_cloud':{'points':sparse}},copies,prov,'Historical COLMAP — original photos, not fresh R')
    _bundle(out/'shape3_depth',h['coordinate_frame_depth'],'reconstruction_units',{'planes':{'planes':planes},'edges':{'edges':edges},'corners':{'corners':corners},'point_cloud':{'points':points}},[],prov,'A Shape 3 — UNREGISTERED depth candidates; no camera overlay')
    views=[('historical_cameras','Historical COLMAP / source photos'),('shape3_depth','A Shape 3 / separate depth geometry')]
    r_status='not_supplied'
    if r_run_dir is not None:
        if not (Path(r_run_dir)/'camera_poses.json').is_file():r_status='blocked_missing_registered_R_poses'
        else:
            convert_r_run(r_run_dir,out/'fresh_r_cameras');views.append(('fresh_r_cameras','Fresh R / independent camera world'));r_status='converted_nonmetric'
    if preview:
        for name,_ in views:build_preview(out/name,out/(name+'_preview'))
        links=''.join(f'<a href="{name}_preview/index.html" target="_blank">{html.escape(label)}</a>' for name,label in views)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MyCON independent review</title><style>body{font:18px system-ui;max-width:900px;margin:60px auto;background:#101c29;color:#e5edf5}a{display:block;color:#a6eee0;padding:16px;border:1px solid #446477;margin:14px 0}strong{color:#f6cb82}</style><h1>MyCON Civil Viewer — independent scenes</h1><p><strong>UNREGISTERED: no camera-to-depth transform or metric BIM evidence.</strong> Each view has its own coordinates; no shared overlay or synchronized spatial navigation.</p><p>Historical photos available: '+str(len(photos))+' of '+str(q['registered_colmap_view_count'])+' registered COLMAP cameras. Fresh R: '+html.escape(r_status)+'.</p>'+links,encoding='utf-8')
    receipt={'schema_version':WORKSPACE,'run_id':out.name,'registration_status':'UNREGISTERED','metric_scale_verified':False,'feature_photo_projection_allowed':False,'source_receipts':prov,'source_frames':{'camera':h['coordinate_frame_colmap'],'geometry':h['coordinate_frame_depth']},'source_units':{'camera':h['units_camera'],'geometry':h['units_depth']},'fresh_r_status':r_status,'historical_registered_camera_count':q['registered_colmap_view_count'],'displayed_camera_count':len(cams),'displayed_photo_camera_count':len(photos),'historical_sparse_preview_points':len(sparse),'trajectory_order':'historical_image_filename_order_not_verified_timestamps','shape3_counts':{'points':len(points),'planes':len(planes),'edges':len(edges),'corners':len(corners),'field_confirmed':0},'views':{name:{'path':name+'/viewer_manifest.json','sha256':sha256(out/name/'viewer_manifest.json')} for name,_ in views}}
    (out/'workspace_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--a-project-root',type=Path,required=True);p.add_argument('--a-handoff',required=True);p.add_argument('--a-handoff-sha256',required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--r-run-dir',type=Path);p.add_argument('--bundles-only',action='store_true');a=p.parse_args(argv)
    try: result=create_workspace(a.a_project_root,a.a_handoff,a.a_handoff_sha256,a.out_dir,r_run_dir=a.r_run_dir,preview=not a.bundles_only)
    except (ValueError,OSError,KeyError,TypeError) as e: print(json.dumps({'status':'blocked','reason':str(e)}));return 2
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__': raise SystemExit(main())
