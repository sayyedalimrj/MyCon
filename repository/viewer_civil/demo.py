"""Deterministic synthetic room and six synthetic camera/frame receipts; NEVER field evidence."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
import re
import struct
import zlib
from pathlib import Path
from .core import SCHEMA, RUN_ID, sha256


def _png(path: Path, index: int, width=480, height=270):
    """Write a real PNG with Python stdlib; source frames are intentionally synthetic."""
    def chunk(tag, raw):
        return struct.pack(">I", len(raw)) + tag + raw + struct.pack(">I", zlib.crc32(tag + raw) & 0xffffffff)
    scan = bytearray()
    for y in range(height):
        scan.append(0)
        for x in range(width):
            if x < 55 or x > 425 or y < 45 or y > 210:
                rgb = (20 + index*4, 40 + index*2, 55 + index*3)
            elif 150 < x < 330 and 70 < y < 177:
                rgb = (54 + index*5, 100 + index*4, 118 + index*3)
            elif x < 80 or x > 400 or y > 185:
                rgb = (71, 88, 96)
            else:
                rgb = (110 + index*3, 121 + index*2, 117)
            scan.extend(rgb)
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(bytes(scan), 8))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)


def _camera_to_world(pos, target):
    # +Z optical forward, +Y image down. Rigid, right-handed, row-major c2w.
    f = [target[i]-pos[i] for i in range(3)]
    norm = math.sqrt(sum(x*x for x in f)); f = [x/norm for x in f]
    r = [f[1]*1-f[2]*0, f[2]*0-f[0]*1, f[0]*0-f[1]*0]
    n = math.sqrt(sum(x*x for x in r)); r = [x/n for x in r]
    u = [r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]]
    return [[r[i], -u[i], f[i], pos[i]] for i in range(3)] + [[0,0,0,1]]


def make_demo(destination, run_id="viewer-synthetic-20261114"):
    """Create a new bundle, never overwriting any directory."""
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run id")
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    (dest/"frames").mkdir()
    when=datetime.now(timezone.utc).isoformat()
    common={"units":"arbitrary","coordinate_frame":"synthetic_room_local",
            "quality":{"status":"synthetic_illustration", "calibrated":False},
            "provenance":{"generator":"viewer_civil.demo","dataset":"synthetic",
                          "run_id":run_id,"generated_at_utc":when}}
    v=[[-4,-3,0],[4,-3,0],[4,3,0],[-4,3,0],
       [-4,-3,3],[4,-3,3],[4,3,3],[-4,3,3]]
    surfaces=[("floor",[0,1,2,3],.92,.85),("north_wall",[3,2,6,7],.85,.80),
              ("south_wall",[0,1,5,4],.52,.60),("west_wall",[0,3,7,4],.70,.72),
              ("east_wall",[1,2,6,5],None,None),("ceiling",[4,5,6,7],.30,.50)]
    faces=[];attrs=[]; planes=[]
    for name, ids, coverage, confidence in surfaces:
        faces.extend([[ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]])
        state="unknown" if coverage is None else "synthetic_estimate"
        attrs.extend([{"surface_id":name,"coverage":coverage,"confidence":confidence,"state":state}]*2)
        planes.append({"id":name,"vertices":[v[k] for k in ids],"confidence":confidence,
                       "coverage":coverage,"state":state,"provenance":{"source":"synthetic"}})
    pairs=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    edges=[{"id":f"edge-{a}-{b}","start":v[a],"end":v[b],
            "confidence":.8 if i%3 else None,
            "state":"unknown" if i%3==0 else "synthetic_estimate"} for i,(a,b) in enumerate(pairs)]
    corners=[{"id":f"corner-{i}","xyz":xyz,"confidence":.82 if i<4 else None,
              "state":"synthetic_estimate" if i<4 else "unknown"} for i,xyz in enumerate(v)]
    poses=[([-2.8,-1.5,1.5],[0,0,1.5]),([-1.8,1.3,1.5],[2,-1,1.5]),
           ([.2,1.5,1.6],[0,-2,1.6]),([2.5,.4,1.5],[-2,0,1.6]),
           ([.8,-1.7,1.6],[2,2,1.5]),([-2.2,-.2,1.55],[-3,2,1.5])]
    frames=[];cameras=[]
    for i,(pos,target) in enumerate(poses):
        cid=f"CAM-{i+1:03d}"; fid=f"FRAME-{i+1:03d}"; rel=f"frames/frame_{i+1:03d}.png"
        _png(dest/rel,i)
        frames.append({"frame_id":fid,"path":rel,"sha256":sha256(dest/rel),
                       "quality":{"state":"synthetic"},"provenance":{"run_id":run_id}})
        cameras.append({"camera_id":cid,"frame_id":fid,"camera_to_world":_camera_to_world(pos,target),
                        "extrinsic_convention":"camera_to_world","fx":395,"fy":395,
                        "cx":240,"cy":135,"width":480,"height":270,
                        "quality":{"state":"synthetic"},"provenance":{"run_id":run_id}})
    data={
        "camera_poses":{"cameras":cameras},
        "frame_manifest":{"frames":frames},
        "planes":{"planes":planes},
        "edges":{"edges":edges},
        "corners":{"corners":corners},
        "mesh":{"vertices":v,"faces":faces,"face_attributes":attrs,
                "bim_overlay":{"status":"synthetic_unregistered","vertices":[[x+.15,y-.10,z] for x,y,z in v],
                               "faces":faces},"deviations":None},
        "alignment_result":{"metric_scale_verified":False,"registration_verified":False,
                            "independent_control_verified":False,"transform":None}}
    receipt={}
    for role, payload in data.items():
        version="mycon.viewer."+role+".v1"
        content={**common,"schema_version":version,**payload}
        name=role+".json"
        (dest/name).write_text(json.dumps(content,indent=2),encoding="utf-8")
        receipt[role]={**common,"schema_version":version,"path":name,"sha256":sha256(dest/name)}
    manifest={**common,"schema_version":SCHEMA,"run_id":run_id,"synthetic":True,
              "metric_scale_verified":False,"registration_verified":False,"files":receipt,
              "disclaimer":"Synthetic fixture only; not real source photographs, field dimensions or progress"}
    (dest/"viewer_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return dest
