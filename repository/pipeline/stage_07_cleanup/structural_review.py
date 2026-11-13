"""Read-only post Stage-7 structural review: never certifies survey/metric dimensions.

Consumes mycon.comparison_ready_geometry.v1. Produces separate create-only
mycon.civil_geometry_review.v1 artifacts with image-independent uncertainty.
No BIM, IFC, MAX or Gaussian-center surface assumption is made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

SCHEMA = "mycon.civil_geometry_review.v1"
BASE_SCHEMA = "mycon.comparison_ready_geometry.v1"
PALETTE = np.array([[233, 103, 72], [55, 149, 186], [113, 173, 101],
                    [181, 118, 201], [224, 178, 65], [66, 190, 168],
                    [211, 113, 147], [108, 128, 213]], dtype=np.uint8)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def safe_input(root: Path, value: str) -> Path:
    p = (root / value).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError("Missing or out-of-project source: " + value)
    return p


def validated_outputs(root: Path, manifest: dict) -> dict[str, Path]:
    if manifest.get("schema_version") != BASE_SCHEMA or manifest.get("metric_scale_verified") is not False:
        raise ValueError("Incorrect base schema or unsupported metric-scale assertion")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Missing output manifest")
    checked = {}
    for name in ("clean_cloud.ply", "clean_cloud.npy", "planes.json", "edges.json", "corners.json"):
        record = outputs.get(name)
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ValueError("Missing source hash: " + name)
        p = safe_input(root, record["path"])
        if sha(p) != record["sha256"]:
            raise ValueError("Source hash mismatch: " + name)
        checked[name] = p
    return checked


def _fit(pts: np.ndarray) -> tuple[np.ndarray, float] | None:
    if len(pts) < 6:
        return None
    ctr = np.mean(pts, axis=0)
    _, singular, vt = np.linalg.svd(pts - ctr, full_matrices=False)
    if singular[1] < 1e-12:
        return None
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    return normal, float(-normal @ ctr)


def _patch_support(pts: np.ndarray, plane: dict, cell_limit: int = 72) -> dict:
    """Report occupied observed cells, never mistake the outer hull for filled surface."""
    idx = np.asarray(plane.get("support_indices", []), dtype=np.int64)
    if idx.ndim != 1 or not len(idx) or idx.min() < 0 or idx.max() >= len(pts):
        raise ValueError("Invalid plane point support indices")
    local_pts = pts[idx]
    center = np.asarray(plane["centroid"], dtype=float)
    u, v = np.asarray(plane["basis_u"], float), np.asarray(plane["basis_v"], float)
    projected = np.column_stack(((local_pts-center) @ u, (local_pts-center) @ v))
    ext = np.ptp(projected, axis=0)
    if not np.isfinite(ext).all() or min(ext) <= 1e-9:
        return {"status": "degenerate", "occupied_fraction_of_hull": None, "island_count": 0}
    # Capped occupancy grid for bounded memory independent of cloud size.
    cell = max(float(np.max(ext))/cell_limit, 1e-9)
    lo = projected.min(axis=0)
    dims = np.minimum(np.ceil(ext/cell).astype(int)+1, cell_limit+2)
    cells = np.floor((projected-lo)/cell).astype(int)
    cells = np.minimum(cells, dims-1)
    occupied = np.zeros(tuple(dims), dtype=bool)
    occupied[cells[:, 0], cells[:, 1]] = True
    # Dilation only for diagnostic reachability; does not create surface geometry.
    near = binary_dilation(occupied, iterations=1)
    components, count = label(near)
    try:
        hull = ConvexHull(projected)
        grid = np.indices(occupied.shape).reshape(2, -1).T
        query = lo + (grid + .5)*cell
        inside = Delaunay(projected[hull.vertices]).find_simplex(query) >= 0
        inside = inside.reshape(occupied.shape)
        denominator = int(inside.sum())
        coverage = float((near & inside).sum()/denominator) if denominator else None
    except QhullError:
        coverage = None
    return {"status": "observed_grid_diagnostic", "occupied_fraction_of_hull": coverage,
            "island_count": int(count), "support_count": int(len(idx)),
            "grid_cell_source_units": cell, "grid_shape": dims.tolist(),
            "outer_hull_may_bridge_occlusion": True}


def _stability(pts: np.ndarray, idx: np.ndarray, plane: dict, sigma: float, seed: int) -> dict:
    """Perturb *existing support* only; sensitivity diagnostic, not detection recall."""
    rng = np.random.default_rng(seed)
    original = np.asarray(plane["normal"], float)
    results = {}
    points = pts[idx]
    selection = {
        "density_half": points[::2],
        "noise_sigma_source_units": points + rng.normal(0., sigma, points.shape),
        "spatial_occlusion_quadrant": points[~((points[:, 0] > np.median(points[:, 0])) &
                                                (points[:, 1] > np.median(points[:, 1])))]
    }
    for label_, sample in selection.items():
        fitted = _fit(sample)
        if fitted is None:
            results[label_] = {"status": "insufficient_support", "sample_count": int(len(sample))}
            continue
        n, d = fitted
        angle = float(np.degrees(np.arccos(np.clip(abs(n @ original), 0, 1))))
        res = np.abs(sample @ n + d)
        results[label_] = {"status": "fitted", "sample_count": int(len(sample)),
                           "normal_angle_delta_degrees": angle,
                           "residual_p95_source_units": float(np.quantile(res, .95))}
    return results


def _write_ply(path: Path, xyz: np.ndarray, colors: np.ndarray) -> None:
    if not (len(xyz) == len(colors) and colors.shape == (len(xyz), 3)):
        raise ValueError("Invalid PLY RGB")
    import trimesh
    cloud = trimesh.points.PointCloud(xyz, colors=np.column_stack((colors, np.full(len(xyz), 255, dtype=np.uint8))))
    path.write_bytes(trimesh.exchange.ply.export_ply(cloud, encoding="binary_little_endian"))


def _images(out: Path, xyz: np.ndarray, planes: list, edges: list, corners: list, label_: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cap = min(len(xyz), 25000)
    ids = np.linspace(0, len(xyz)-1, cap, dtype=np.int64)
    for j, (elev, azim) in enumerate(((20, 35), (5, 125), (75, -80)), 1):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(*xyz[ids].T, s=.2, alpha=.25, color="#7d8490")
        for i, p in enumerate(planes):
            boundary = np.asarray(p["finite_boundary_xyz"], float)
            if len(boundary) < 3: continue
            loop = np.vstack((boundary, boundary[0]))
            ax.plot(*loop.T, linewidth=1.5, color=PALETTE[i%len(PALETTE)]/255.)
        for e in edges:
            ax.plot(*np.asarray(e["endpoints_xyz"]).T, color="#e5ad2d", linewidth=2)
        for c in corners:
            if c.get("state") == "observed_supported":
                ax.scatter(*c["xyz"], marker="x", color="#d21f2a", s=35)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("Shape 3 / observed features — " + label_ + " — NON METRIC")
        fig.tight_layout(); fig.savefig(out/f"shape3_view_{j:02d}.png", dpi=135); plt.close(fig)


def _html(overlay: dict) -> str:
    """Local canvas viewer, no CDN or hosted dependencies."""
    embedded = json.dumps(overlay, ensure_ascii=False).replace("</", "<\\/")
    return '''<!doctype html><html><head><meta charset="utf-8"><title>MyCON civil geometry review</title>
<style>body{font:14px system-ui;margin:0;background:#101820;color:#e3e6eb} header{padding:12px;background:#1c2837} canvas{width:100%;height:76vh;cursor:grab;display:block} label{margin:0 10px}aside{padding:10px;white-space:pre-wrap}</style></head>
<body><header><b>MyCON Shape 3 — review / unverified scale</b>
<label><input id="planes" type="checkbox" checked>Planes</label><label><input id="edges" type="checkbox" checked>Edges</label>
<label><input id="corners" type="checkbox" checked>Corners</label><label><input id="points" type="checkbox" checked>Points</label>
Rotate: drag · Zoom: mouse wheel · Click a feature for ID; no BIM/progress certification</header>
<canvas id="view"></canvas><aside id="detail">Select a feature. Human approval must be recorded externally.</aside>
<script id="data" type="application/json">''' + embedded + '''</script><script>
const data=JSON.parse(document.getElementById('data').textContent), cv=document.getElementById('view'), cx=cv.getContext('2d');
let a=.45,b=.25,zoom=1,down=false,px=0,py=0,hit=[];
const features=[...data.planes.map(p=>[p,'plane']),...data.edges.map(e=>[e,'edge']),...data.corners.map(c=>[c,'corner'])];
function size(){cv.width=cv.clientWidth*devicePixelRatio;cv.height=cv.clientHeight*devicePixelRatio;draw()};
function proj(p){let q=p.map((v,i)=>(v-data.center[i])*data.scale),x=q[0]*Math.cos(a)-q[1]*Math.sin(a),
y=q[0]*Math.sin(a)+q[1]*Math.cos(a),z=q[2];let t=y*Math.cos(b)-z*Math.sin(b),h=y*Math.sin(b)+z*Math.cos(b);
return [cv.width/2+x*cv.height*.32*zoom,cv.height/2-h*cv.height*.32*zoom,t]}
function draw(){cx.fillStyle='#101820';cx.fillRect(0,0,cv.width,cv.height);hit=[];
if(document.getElementById('points').checked){cx.fillStyle='#657788';data.points.forEach(p=>{let q=proj(p);cx.fillRect(q[0],q[1],1,1)})}
if(document.getElementById('planes').checked)data.planes.forEach((p,i)=>{let arr=p.finite_boundary_xyz;if(arr.length<3)return;
cx.beginPath();arr.forEach((v,j)=>{let q=proj(v);j?cx.lineTo(q[0],q[1]):cx.moveTo(q[0],q[1])});cx.closePath();
cx.strokeStyle=data.colors[i%data.colors.length];cx.lineWidth=2;cx.stroke();let q=proj(p.centroid);hit.push([q,p,'plane'])});
if(document.getElementById('edges').checked)data.edges.forEach(e=>{let p=proj(e.endpoints_xyz[0]),q=proj(e.endpoints_xyz[1]);cx.beginPath();cx.moveTo(...p.slice(0,2));cx.lineTo(...q.slice(0,2));cx.strokeStyle='#ffca3a';cx.lineWidth=3;cx.stroke();hit.push([[(p[0]+q[0])/2,(p[1]+q[1])/2],e,'edge'])});
if(document.getElementById('corners').checked)data.corners.forEach(c=>{let q=proj(c.xyz);cx.fillStyle=c.state==='observed_supported'?'#ff6379':'#b0bac5';cx.beginPath();cx.arc(q[0],q[1],5,0,Math.PI*2);cx.fill();hit.push([q,c,'corner'])})}
cv.addEventListener('pointerdown',e=>{down=true;px=e.clientX;py=e.clientY});window.addEventListener('pointerup',()=>down=false);
cv.addEventListener('pointermove',e=>{if(!down)return;a+=(e.clientX-px)/160;b=Math.max(-1.5,Math.min(1.5,b+(e.clientY-py)/160));px=e.clientX;py=e.clientY;draw()});
cv.addEventListener('wheel',e=>{e.preventDefault();zoom*=e.deltaY>0?.9:1.1;draw()},{passive:false});
cv.addEventListener('click',e=>{let x=e.offsetX*devicePixelRatio,y=e.offsetY*devicePixelRatio;let arr=hit.map(h=>[Math.hypot(h[0][0]-x,h[0][1]-y),h]).sort((p,q)=>p[0]-q[0]);if(arr.length&&arr[0][0]<40)document.getElementById('detail').textContent=JSON.stringify({kind:arr[0][1][2],feature:arr[0][1][1]},null,2)});
['points','planes','edges','corners'].forEach(id=>document.getElementById(id).addEventListener('change',draw));addEventListener('resize',size);size();
</script></body></html>'''


def generate(root: Path, base_manifest: Path, run_id: str, *, seed: int = 17,
             noise_sigma: float | None = None, input_kind: str = "unknown") -> dict:
    if not run_id.startswith("geom-civil-") or not run_id[11:].replace("-", "").replace("_", "").isalnum():
        raise ValueError("Run ID must use geom-civil-* with alphanumeric suffix")
    root = root.resolve(); base_manifest = base_manifest.resolve()
    if not base_manifest.is_relative_to(root) or not base_manifest.is_file():
        raise ValueError("Base manifest must be an existing project-relative file")
    base = json.loads(base_manifest.read_text())
    original = validated_outputs(root, base)  # hash gate BEFORE any output creation
    xyz = np.load(original["clean_cloud.npy"], allow_pickle=False)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Invalid clean cloud array")
    plane_data = json.loads(original["planes.json"].read_text()); edge_data = json.loads(original["edges.json"].read_text())
    corner_data = json.loads(original["corners.json"].read_text())
    planes, edges, corners = plane_data["planes"], edge_data["edges"], corner_data["corners"]
    if not all(isinstance(x, list) for x in (planes, edges, corners)):
        raise ValueError("Invalid feature lists")
    threshold = float(base.get("parameters", {}).get("threshold_reconstruction_units", 0.))
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("Missing valid source-unit RANSAC threshold")
    sigma = float(noise_sigma) if noise_sigma is not None else threshold*.5
    if not np.isfinite(sigma) or sigma < 0: raise ValueError("Invalid perturbation sigma")
    target = root/"runs"/run_id/"geometry"
    if target.parent.exists():
        raise FileExistsError("Refuse existing output Run ID: "+str(target.parent))
    plane_ids = set()
    covered = np.zeros(len(xyz), dtype=bool)
    stress = {}
    counts = {"planes": len(planes), "edges": len(edges), "corners": len(corners)}
    for i, p in enumerate(planes):
        pid=p["plane_id"]
        if pid in plane_ids: raise ValueError("Duplicate plane ID")
        plane_ids.add(pid)
        idx = np.asarray(p.get("support_indices", []), dtype=np.int64)
        if idx.ndim != 1 or not len(idx) or idx.min() < 0 or idx.max() >= len(xyz): raise ValueError("Bad plane support index")
        if covered[idx].any(): raise ValueError("Overlapping plane support indices")
        covered[idx] = True
        occupancy = _patch_support(xyz, p)
        p["observed_support_qa"] = occupancy
        p["semantic_label"] = "unknown"  # gravity/camera is not a signed, verified floor/ceiling classifier
        p["human_review_state"] = "pending"
        res=float(p.get("residual_p95_units", threshold))
        coverage=occupancy.get("occupied_fraction_of_hull")
        quality=float(np.clip(1-res/max(threshold,1e-12),0,1))
        p["confidence"] = {"calibrated_probability": None, "status": "not_calibrated",
                           "uncalibrated_fit_quality_score": quality,
                           "occupancy_fraction_observed_grid": coverage}
        p["physical_surface_confirmed"] = False
        stress[pid] = _stability(xyz, idx, p, sigma, seed+i)
    for e in edges:
        if not set(e.get("plane_ids", [])).issubset(plane_ids): raise ValueError("Edge has nonexistent plane")
        e["physical_edge_confirmed"] = False; e["human_review_state"] = "pending"
        e["confidence"] = {"calibrated_probability": None, "status": "not_calibrated",
                           "uncalibrated_support_score": float(np.clip(min(e.get("support_count_per_plane",[0]))/20,0,1))}
    edge_ids={e["edge_id"] for e in edges}
    for c in corners:
        supported = (c.get("state")=="observed_supported" and
                     c.get("multiscale_confirmed") is True and
                     len(c.get("edge_ids",[]))>=3 and
                     min(c.get("support_near_corner_per_plane",[0]))>=2 and
                     all(c.get("inside_observed_patch_per_plane",[])))
        if not set(c.get("plane_ids",[])).issubset(plane_ids) or not set(c.get("edge_ids",[])).issubset(edge_ids):
            raise ValueError("Corner has invalid feature references")
        if c.get("state")=="observed_supported" and not supported:
            c["state"]="unknown"; c["review_reason"]="support_or_multiscale_gate_failed"
        c["field_confirmed"] = False; c["human_review_state"] = "pending"
        c["confidence"] = {"calibrated_probability": None, "status": "not_calibrated",
                           "uncalibrated_local_support_score": float(np.clip(min(c.get("support_near_corner_per_plane",[0]))/10,0,1))}
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(original["clean_cloud.ply"], target/"clean_cloud.ply")
    # Preserve exact points/colors of cleaned source; recolor only an isolated review copy.
    shutil.copyfile(original["clean_cloud.npy"], target/"clean_cloud.npy")
    colors=np.tile(np.array([112,119,130],dtype=np.uint8),(len(xyz),1))
    for i,p in enumerate(planes): colors[np.asarray(p["support_indices"],int)] = PALETTE[i%len(PALETTE)]
    _write_ply(target/"planes_colored.ply",xyz,colors)
    frame=str(base.get("coordinate_frame", "unknown")); units=str(base.get("units", "unknown_reconstruction_units"))
    for name,features in (("planes",planes),("edges",edges),("corners",corners)):
        data={"schema_version":SCHEMA, "run_id":run_id, "coordinate_frame":frame, "units":units,
              "source_sha256":sha(original[name+".json"]),"provenance":{"source_run_id":base.get("run_id"),
              "source_manifest_sha256":sha(base_manifest)}, name:features}
        (target/(name+".json")).write_text(json.dumps(data,indent=2),encoding="utf-8")
    qa={"schema_version":SCHEMA,"run_id":run_id,"input_kind":input_kind,
        "coordinate_frame":frame,"units":units,"metric_scale_verified":False,"registration_verified":False,
        "confidence_calibrated":False,"camera_visibility_verified":False,
        "original_source_sha256":base.get("source_sha256"), "base_manifest_sha256":sha(base_manifest),
        "quality_assessment":"synthetic_algorithm_diagnostic_only" if input_kind=="synthetic" else "not_determined_without_independent_ground_truth",
        "counts":{**counts,"observed_supported_corner_candidates":sum(c.get("state")=="observed_supported" for c in corners),
                  "human_confirmed_corners":0,"unassigned_points":int((~covered).sum())},
        "sensitivity":{"scope":"fixed_plane_support_refit_only; no global repeat-detection or recall", "noise_sigma_source_units":sigma,
                        "seed":seed,"by_plane":stress},
        "unknowns":["floor_or_ceiling_identity","independent_survey_scale","occluded_surface","independent_feature_gt",
                    "material_identity","construction_status","calibrated_confidence"],
        "limitations":["Outer convex hull may bridge a data gap; inspect occupancy and source images",
                       "Numerical plane intersections never constitute field-verified corners",
                       "Sensitivity diagnostics are not independent accuracy estimates"],
        "software":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()}}
    (target/"geometry_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    with (target/"components.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["type","feature_id","state","support_count","residual_p95_source_units",
                                  "confidence_calibrated","manual_review","units","frame"])
        for p in planes:w.writerow(["plane",p["plane_id"],p.get("fit_state"),p.get("point_count"),p.get("residual_p95_units"),"no","pending",units,frame])
        for e in edges:w.writerow(["edge",e["edge_id"],e.get("fit_state"),sum(e.get("support_count_per_plane",[])),"","no","pending",units,frame])
        for c in corners:w.writerow(["corner",c["corner_id"],c.get("state"),min(c.get("support_near_corner_per_plane",[0])),"","no","pending",units,frame])
    with (target/"manual_review_template.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["feature_id","reviewer","review_date_utc","decision:accept/reject/unknown","reason","photo_or_survey_sha256"])
        for item in planes+edges+corners:w.writerow([item.get("plane_id",item.get("edge_id",item.get("corner_id"))),"","","unknown","",""])
    lo=xyz.min(axis=0);hi=xyz.max(axis=0)
    ids=np.linspace(0,len(xyz)-1,min(25000,len(xyz)),dtype=np.int64)
    # Viewer payload omits full support index arrays: original hashed JSON retains them.
    view_planes=[{k:v for k,v in p.items() if k not in ("support_indices", "halfspaces_uv")} for p in planes]
    view_edges=[{k:v for k,v in e.items() if k!="support_indices_per_plane"} for e in edges]
    overlay={"schema_version":SCHEMA,"run_id":run_id,"coordinate_frame":frame,"units":units,
             "metric_scale_verified":False,"center":((lo+hi)/2).tolist(),"scale":float(2/max(float(np.max(hi-lo)),1e-8)),
             "colors":["#%02x%02x%02x"%tuple(v) for v in PALETTE],
             "points":xyz[ids].tolist(),"planes":view_planes,"edges":view_edges,"corners":corners,
             "selection_policy":"uniform_index_subsample", "visible_points":len(ids),
             "visibility_provenance":"none; camera FOV is not occlusion proof"}
    (target/"viewer_overlay.json").write_text(json.dumps(overlay,separators=(",", ":")),encoding="utf-8")
    (target/"geometry_review.html").write_text(_html(overlay),encoding="utf-8")
    _images(target, xyz, planes, edges, corners, "SYNTHETIC" if input_kind=="synthetic" else "RECONSTRUCTION / UNVERIFIED")
    outputs={}
    for p in sorted(target.iterdir()):
        if p.is_file():outputs[p.name]={"relative_path":p.relative_to(root).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
    manifest={"schema_version":SCHEMA,"run_id":run_id,"created_at_utc":datetime.now(timezone.utc).isoformat(),
              "coordinate_frame":frame,"units":units,"metric_scale_verified":False,"registration_verified":False,
              "source_manifest_relative_path":base_manifest.relative_to(root).as_posix(),
              "source_manifest_sha256":sha(base_manifest),"source_run_id":base.get("run_id"),
              "source_camera_sha256":base.get("source_camera_sha256"),
              "source_stage7_planes_sha256":base.get("source_stage7_planes_sha256"),
              "input_kind":input_kind,"software":qa["software"],"seed":seed,"outputs":outputs,
              "provenance":{"method":"read_only_feature_review; reuse_existing_stage7_cloud_and_features",
                            "original_source_sha256":base.get("source_sha256"),"camera_evidence":"not independently verified"}}
    (target/"civil_geometry_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description="Isolated geometry post-review; all outputs create-only and nonmetric")
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--base-manifest",type=Path,required=True)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--input-kind",choices=("unknown","synthetic","real_reconstruction"),default="unknown")
    parser.add_argument("--noise-sigma",type=float)
    a=parser.parse_args(argv)
    try: result=generate(a.project_root,a.base_manifest,a.run_id,seed=a.seed,
                         input_kind=a.input_kind,noise_sigma=a.noise_sigma)
    except (OSError,ValueError,KeyError,TypeError,ImportError) as ex:
        print(json.dumps({"status":"blocked","reason":str(ex)}));return 2
    print(json.dumps({"status":"generated","run_id":result["run_id"],"metric_scale_verified":False}));return 0

if __name__=="__main__":
    raise SystemExit(main())
