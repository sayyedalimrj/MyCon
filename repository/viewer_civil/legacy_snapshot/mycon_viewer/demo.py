"""Synthetic-only room/poses/frames fixture; never derive field measurements."""
from pathlib import Path
from datetime import datetime, timezone
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .io import sha256, SCHEMA, SAFE_ID

PREFIX = 'mycon.'

def _pose(pos, target):
    p=np.array(pos,float); f=np.array(target,float)-p; f/=np.linalg.norm(f)
    right=np.cross(f,[0,0,1]); right/=np.linalg.norm(right)
    up=np.cross(right,f); up/=np.linalg.norm(up)
    m=np.eye(4); m[:3,0]=right; m[:3,1]=-up; m[:3,2]=f; m[:3,3]=p
    # right, image-down, forward (pinhole +z forward)
    return np.round(m,7).tolist()

def make_demo(root: Path, run_id='viewer-synthetic-20261113') -> Path:
    root=Path(root)
    if not SAFE_ID.fullmatch(run_id): raise ValueError('invalid run id')
    root.mkdir(parents=True,exist_ok=False)
    (root/'frames').mkdir()
    when=datetime.now(timezone.utc).isoformat()
    common={'units':'arbitrary', 'coordinate_frame':'synthetic_room_local',
            'quality':{'status':'synthetic_illustration', 'calibrated':False, 'confidence_type':'demo_only'},
            'provenance':{'generator':'mycon_viewer.demo', 'dataset':'synthetic', 'run_id':run_id,'generated_at_utc':when}}
    room=[[-4,-3,0],[4,-3,0],[4,3,0],[-4,3,0],[-4,-3,3],[4,-3,3],[4,3,3],[-4,3,3]]
    surfaces=[('floor',[0,1,2,3],.9,.88),('wall_north',[3,2,6,7],.92,.82),('wall_south',[0,1,5,4],.55,.74),('wall_west',[0,3,7,4],.83,.78),('wall_east',[1,2,6,5],None,None),('ceiling',[4,5,6,7],.35,.61)]
    faces=[]; qualities=[]
    for name,ids,cov,conf in surfaces:
        faces.extend([[ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]])
        qualities.extend([{'surface_id':name,'coverage':cov,'confidence':conf,'state':'unknown' if cov is None else 'synthetic_estimate','provenance':'demo_only'}]*2)
    mesh={'vertices':room,'faces':faces,'face_attributes':qualities,
          'bim_overlay':{'status':'illustrative_unregistered','vertices':[[x+.15,y-.1,z] for x,y,z in room], 'faces':faces,'reference_id':'shape4_demo_placeholder'},
          'deviations':None}
    planes={'planes':[{'id':name,'vertices':[room[i] for i in ids], 'confidence':conf,'coverage':cov,
                       'state':'unknown' if cov is None else 'synthetic_estimate', 'provenance':{'kind':'synthetic'}} for name,ids,cov,conf in surfaces]}
    edges={'edges':[{'id':f'edge_{a}_{b}','start':room[a],'end':room[b],
                     'confidence':(.91 if i%4==0 else .65 if i%4==1 else .38),
                     'state':'candidate_extrapolated' if i%4==3 else 'synthetic_estimate', 'provenance':{'kind':'synthetic'}} for i,(a,b) in enumerate([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)])]}
    corners={'corners':[{'id':f'corner_{i}','xyz':p, 'confidence':(.84 if i<4 else None),
                         'state':'candidate_extrapolated' if i>=4 else 'synthetic_estimate', 'provenance':{'kind':'synthetic'}} for i,p in enumerate(room)]}
    cams=[]; frames=[]
    poses=[([-2.8,-1.5,1.55],[0,0,1.5]),([-1.8,1.3,1.5],[2,-1,1.5]),([.2,1.5,1.6],[0,-2,1.6]),([2.5,.4,1.5],[-2,0,1.6]),([.8,-1.7,1.6],[2,2,1.5]),([-2.2,-.2,1.55],[-3,2,1.5])]
    for i,(p,t) in enumerate(poses):
        cid=f'CAM-{i+1:03d}'; fid=f'FRAME-{i+1:03d}'; rel=f'frames/frame_{i+1:03d}.png'
        im=Image.new('RGB',(960,540),(20+i*5,31+i*6,47+i*4)); d=ImageDraw.Draw(im)
        d.polygon([(50,110),(910,110),(790,420),(160,420)],fill=(37+i*3,52+i*5,66+i*3),outline='#8CA5AC',width=5)
        d.polygon([(50,110),(200,40),(200,340),(160,420)],fill=(26,43,58))
        d.polygon([(910,110),(760,40),(760,340),(790,420)],fill=(48,63,69))
        d.line([(200,40),(760,40),(910,110)],fill='#BCB59C',width=5)
        d.line([(200,340),(760,340)],fill='#C5BFA5',width=6)
        d.rectangle((345,145,595,330),outline='#D7B97E',width=5)
        d.rectangle((388,173,552,318),fill=(67+i*3,100+i*4,106+i*2))
        d.rectangle((0,447,960,540),fill=(9,19,31))
        d.text((33,18),f'{cid}  /  {fid}',fill='white',stroke_width=0)
        d.text((33,468),'SYNTHETIC FRAME — NOT FIELD EVIDENCE',fill='#EDD18D')
        d.text((33,504),f'MyCON Viewer  |  illustrative pose {i+1}  |  arbitrary units',fill='#ACC3CE')
        im.save(root/rel)
        cams.append({'camera_id':cid,'frame_id':fid,'camera_to_world':_pose(p,t),'extrinsic_convention':'camera_to_world',
                     'fx':770,'fy':770,'cx':480,'cy':270,'width':960,'height':540,'quality':{'state':'synthetic'},'provenance':{'kind':'synthetic'}})
        frames.append({'frame_id':fid,'path':rel,'sha256':sha256(root/rel),'quality':{'state':'synthetic'},'provenance':{'kind':'synthetic'}})
    alignment={'metric_scale_verified':False,'registration_verified':False,'independent_control_verified':False,
               'reference_shape':'shape4','status':'unregistered_placeholder','transform':None}
    data={'camera_poses':{'cameras':cams},'frame_manifest':{'frames':frames},'planes':planes,'edges':edges,'corners':corners,'mesh':mesh,'alignment_result':alignment}
    receipts={}
    for role,item in data.items():
        schema_version=f'mycon.viewer.{role}.v1'
        doc={**common, 'schema_version':schema_version,**item}
        path=f'{role}.json'
        (root/path).write_text(json.dumps(doc,indent=2,ensure_ascii=False),encoding='utf-8')
        receipts[role]={**common,'path':path,'sha256':sha256(root/path),'schema_version':schema_version}
    manifest={**common,'schema_version':SCHEMA,'run_id':run_id,'synthetic':True,
              'metric_scale_verified':False,'registration_verified':False,'files':receipts,
              'notes':'All values are synthetic. Coverage and confidence are NOT calibrated field measurements.'}
    (root/'viewer_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return root
