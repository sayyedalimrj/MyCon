"""Round 4: same-run SfM/MVS and train-only GS staging; no trainer invoked."""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
from pathlib import Path
from .workflow import Blocked, checked_run, sha256, emit_json, parse_colmap_text, _run_colmap

def _read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def _files(root, paths):
    result = {}
    for p in paths:
        p = p.resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file():
            raise Blocked(f"Missing or outside run: {p}")
        result[p.relative_to(root.resolve()).as_posix()] = {
            "sha256": sha256(p), "bytes": p.stat().st_size}
    return result

def _ply_vertices(path):
    """Validate all XYZ rows of a COLMAP point-cloud PLY, not just its declared count."""
    import numpy as np
    types={"float":"f4","float32":"f4","double":"f8","float64":"f8",
           "uchar":"u1","uint8":"u1","char":"i1","int8":"i1","short":"i2",
           "ushort":"u2","int":"i4","uint":"u4","int32":"i4","uint32":"u4"}
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise Blocked("Invalid PLY header")
        n=None; fmt=None; vertex=False; props=[]
        for _ in range(1000):
            line=f.readline()
            if not line: break
            parts=line.decode("ascii").strip().split()
            if not parts: continue
            if parts[0]=="format": fmt=parts[1]
            elif parts[0]=="element":
                vertex=parts[1]=="vertex"
                if vertex: n=int(parts[2])
            elif parts[0]=="property" and vertex:
                if len(parts)!=3 or parts[1] not in types:
                    raise Blocked("Unsupported PLY vertex layout")
                props.append((parts[2],types[parts[1]]))
            elif parts[0]=="end_header":
                if not n or n<1 or not {"x","y","z"}.issubset({k for k,_ in props}):
                    raise Blocked("PLY contains no XYZ vertices")
                if fmt in ("binary_little_endian","binary_big_endian"):
                    endian="<" if fmt=="binary_little_endian" else ">"
                    dt=np.dtype([(k,endian+t) for k,t in props])
                    remaining=n
                    while remaining:
                        a=np.fromfile(f,dtype=dt,count=min(remaining,100000))
                        if len(a)==0 or not all(np.isfinite(a[k]).all() for k in ("x","y","z")):
                            raise Blocked("Truncated or nonfinite point cloud")
                        remaining-=len(a)
                elif fmt=="ascii":
                    inds=[next(i for i,(k,_) in enumerate(props) if k==v) for v in ("x","y","z")]
                    for _ in range(n):
                        p=f.readline().split()
                        if len(p)!=len(props) or not np.isfinite([float(p[i]) for i in inds]).all():
                            raise Blocked("Truncated or nonfinite point cloud")
                else:
                    raise Blocked("Unknown PLY format")
                return n
    raise Blocked("Invalid or truncated PLY header")

def stage(run_dir: Path, executable="colmap", max_image_size=1600, run_dense=True, dense_backend="pycolmap_cuda12"):
    """Call only after fresh workflow.run_colmap. Never use historical model or Test RGB."""
    root = run_dir.resolve()
    m = checked_run(root)
    audit = _read(root/"colmap/colmap_execution.json")
    if audit.get("status") != "colmap_executed" or audit.get("run_id") != root.name:
        raise Blocked("Fresh COLMAP execution receipt missing")
    if audit.get("input_sha256") != sha256(root/"dataset_manifest.json"):
        raise Blocked("Dataset identity mismatch")
    model = (root/audit["train_model_relpath"]).resolve()
    if not model.is_relative_to(root):
        raise Blocked("Model outside new run")
    names = ("cameras.bin","images.bin","points3D.bin")
    for n in names:
        if sha256(model/n) != audit["train_model_sha256"][n]:
            raise Blocked("Frozen train model hash mismatch")
    cameras, known = parse_colmap_text(root/"colmap/train_text")
    train = {Path(f["image_relpath"]).relative_to("images").as_posix(): f
             for f in m["frames"] if f["split"]=="train"}
    if not set(known).issubset(train) or len(known)<3:
        raise Blocked("Train sparse registration insufficient or contains non-train views")
    pose = _read(root/"camera_poses.json")
    pose_by_id = {p["view_id"]:p for p in pose["poses"]}
    if pose.get("run_id") != root.name or pose.get("train_colmap_model_sha256") != sha256(root/"colmap/train_text/images.txt"):
        raise Blocked("Pose provenance mismatch")
    for name in known:
        f=train[name]; p=pose_by_id[f["view_id"]]
        if not p["registered"] or p["image_sha256"]!=f["image_sha256"] or p["split"]!="train":
            raise Blocked("Pose/registered train identity mismatch")
    if not 320<=int(max_image_size)<=4096:
        raise Blocked("Invalid dense resolution")
    out=root/"same_run_products"
    out.mkdir(exist_ok=False)
    log=out/"colmap_products.log"
    source=out/"registered_train_images"
    source.mkdir()
    for name in sorted(known):
        src=root/train[name]["image_relpath"]
        dst=source/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        if sha256(dst)!=train[name]["image_sha256"]:
            raise Blocked("Copied train RGB hash mismatch")
    _run_colmap([executable,"model_analyzer","--path",str(model)],log)
    sparse=out/"sparse_train.ply"
    _run_colmap([executable,"model_converter","--input_path",str(model),"--output_path",str(sparse),"--output_type","PLY"],log)
    sparse_n=_ply_vertices(sparse)
    dense=out/"dense_workspace"
    dense.mkdir()
    _run_colmap([executable,"image_undistorter","--image_path",str(source),"--input_path",str(model),
                 "--output_path",str(dense),"--output_type","COLMAP","--max_image_size",str(max_image_size)],log)
    txt=out/"undistorted_text"
    txt.mkdir()
    _run_colmap([executable,"model_converter","--input_path",str(dense/"sparse"),"--output_path",str(txt),"--output_type","TXT"],log)
    uc, ui = parse_colmap_text(txt)
    if set(ui)!=set(known):
        raise Blocked("Undistorted registered image set changed")
    import numpy as np
    if any(not np.allclose(ui[n]["world_to_camera_4x4"],known[n]["world_to_camera_4x4"],atol=1e-6,rtol=0) for n in known):
        raise Blocked("Undistortion changed the source world transform")
    # Staging uses only registered TRAIN RGB, with corresponding undistorted intrinsics.
    gs=out/"gs_train_dataset"
    (gs/"images").mkdir(parents=True)
    (gs/"sparse/0").mkdir(parents=True)
    import cv2
    gs_images=[]
    for name in sorted(ui):
        src=dense/"images"/name
        dst=gs/"images"/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        cam=uc[ui[name]["camera_id"]]
        rgb=cv2.imread(str(dst),cv2.IMREAD_COLOR)
        if rgb is None or (rgb.shape[1],rgb.shape[0])!=(cam["width"],cam["height"]):
            raise Blocked("Undistorted RGB/intrinsics size mismatch")
        if cam["model"] not in ("PINHOLE","SIMPLE_PINHOLE") or not np.isfinite(cam["parameters"]).all() or cam["parameters"][0]<=0:
            raise Blocked("Invalid undistorted pinhole intrinsics")
        gs_images.append({"view_id":train[name]["view_id"],"source_image_sha256":train[name]["image_sha256"],
                          "undistorted_image_sha256":sha256(dst),"path":dst.relative_to(root).as_posix(),
                          "camera":cam,"world_to_camera_4x4":ui[name]["world_to_camera_4x4"]})
    for n in names:
        shutil.copy2(dense/"sparse"/n,gs/"sparse/0"/n)
    gs_receipt={"schema_version":"mycon.reconstruction.gs_staging.v1","run_id":root.name,
                "source_dataset_sha256":sha256(root/"dataset_manifest.json"),
                "camera_poses_sha256":sha256(root/"camera_poses.json"),
                "train_registered":len(known),"test_views_copied":0,"images":gs_images,
                "sparse_files":_files(root,[gs/"sparse/0"/n for n in names]),
                "coordinate_frame":root.name+":colmap_world","units":"reconstruction_units_unscaled",
                "training":"not_executed_requires_new_explicit_approval"}
    emit_json(gs/"staging_receipt.json",gs_receipt)
    status="not_requested"; dense_n=None
    fused=out/"dense_train_fused.ply"
    if run_dense:
        try:
            if not shutil.which("nvidia-smi"):
                raise Blocked("No nvidia-smi; Dense PatchMatch needs a compatible GPU build")
            subprocess.run(["nvidia-smi","-L"],check=True,capture_output=True,text=True)
            if dense_backend == "pycolmap_cuda12":
                # Official pinned CUDA wheel, separate process keeps a GPU crash out of the notebook kernel.
                script = ("import sys,pycolmap as p; "
                          "assert p.__version__=='3.13.0', p.__version__; "
                          "assert p.has_cuda, 'pycolmap is not CUDA-enabled'; "
                          "o=p.PatchMatchOptions(max_image_size=int(sys.argv[3]),geom_consistency=True,cache_size=4.0); "
                          "p.patch_match_stereo(sys.argv[1],options=o); "
                          "p.stereo_fusion(sys.argv[2],sys.argv[1],input_type='geometric')")
                _run_colmap([sys.executable,"-c",script,str(dense),str(fused),str(max_image_size)],log)
            elif dense_backend == "cli_cuda":
                _run_colmap([executable,"patch_match_stereo","--workspace_path",str(dense),"--workspace_format","COLMAP",
                         "--PatchMatchStereo.geom_consistency","true","--PatchMatchStereo.max_image_size",str(max_image_size)],log)
                _run_colmap([executable,"stereo_fusion","--workspace_path",str(dense),"--workspace_format","COLMAP",
                         "--input_type","geometric","--output_path",str(fused)],log)
            else:
                raise Blocked("Unsupported dense backend")
            dense_n=_ply_vertices(fused)
            status="executed_nonempty_unscaled_candidate"
        except (Blocked,OSError,ValueError,subprocess.SubprocessError) as exc:
            status="blocked_or_failed: "+str(exc)
    for n in names:
        if sha256(model/n)!=audit["train_model_sha256"][n]:
            raise Blocked("Frozen train model changed")
    assets=[root/"dataset_manifest.json",root/"camera_poses.json",root/"registration_quality.json",
            root/"coverage_report.json",root/"colmap/colmap_execution.json",sparse,
            root/"colmap/train_text/cameras.txt",root/"colmap/train_text/images.txt",
            root/"colmap/train_text/points3D.txt",gs/"staging_receipt.json",log]
    if dense_n is not None:
        assets.append(fused)
    result={"schema_version":"mycon.reconstruction.same_run_handoff.v1","run_id":root.name,
            "coordinate_frame":root.name+":colmap_world","pose_convention":"colmap_opencv_world_to_camera",
            "units":"reconstruction_units_unscaled","metric_scale_verified":False,
            "independent_registration_verified":False,"field_surface_validated":False,
            "registered_train":len(known),"sparse_vertices":sparse_n,
            "dense_status":status,"dense_backend":dense_backend,"dense_vertices_validated":dense_n,
            "dense_candidate_path":fused.relative_to(root).as_posix() if dense_n else None,
            "same_world_basis":"COLMAP undistorter of frozen TRAIN model; W2C numerical consistency checked",
            "gs_training":"not_executed_requires_new_explicit_approval",
            "for_A":"Use same-run sparse/optional fused PLY + original train camera text; candidate geometry only.",
            "for_V":"Use original dataset_manifest.json + camera_poses.json with producer_bridge.py; no external A cloud without independent frame bridge.",
            "files":_files(root,assets)}
    emit_json(out/"av_handoff.json",result)
    return result
