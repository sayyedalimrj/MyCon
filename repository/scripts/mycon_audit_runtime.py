"""MyCON execution integrity primitives; safe for unprivileged Colab CPU+GPU.

Derived artifacts are atomic and versioned. Original Recorder, prior models and
NVIDIA experiments are never modified. All source identities are content hashes,
not moving Git branches.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path

CORE_FILES=("mycon_colmap_gsplat.py","mycon_colmap_gsplat_env.py",
            "mycon_gsplat_resume_patch.py","mycon_gsplat_finish.py",
            "mycon_audit_runtime.py","mycon_capture_safe.py",
            "mycon_thesis_one_run.py","mycon_longsplat_autoenv.py")


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    """Replace derived receipt atomically; preserve any interrupted partial."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".partial."+uuid.uuid4().hex)
    try:
        with tmp.open("w",encoding="utf-8") as out:
            json.dump(payload,out,indent=2,ensure_ascii=False,sort_keys=True,
                      allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def immutable_json(path: Path, data: dict) -> None:
    if path.is_file():
        if json.loads(path.read_text()) != data:
            raise RuntimeError("MYCON_IMMUTABLE_RECEIPT_CONFLICT:"+str(path))
        return
    atomic_json(path,data)


def source_identity(script_dir: Path) -> dict:
    base=Path(script_dir)
    hashes={}
    for item in CORE_FILES:
        p=base/item
        if not p.is_file():
            raise RuntimeError("MYCON_SOURCE_FILE_MISSING:"+str(p))
        hashes[item]=sha256(p)
    return {"schema":"mycon.source.sha256.v1","files":hashes}


def require_source_lock(root: Path, script_dir: Path, *, create: bool) -> dict:
    path=Path(root)/"source_identity.json"
    current=source_identity(script_dir)
    if create:
        immutable_json(path,current)
    elif not path.is_file():
        raise RuntimeError("MYCON_CPU_SOURCE_LOCK_MISSING")
    saved=json.loads(path.read_text())
    if saved != current:
        raise RuntimeError("MYCON_SOURCE_CHANGED_BETWEEN_CPU_GPU_OR_RESUME: "+
                           str(path))
    return current


def quarantine_derived(path: Path, reason: str) -> Path:
    """Never delete partial derived data; move to unique sibling before retry."""
    path=Path(path)
    if not path.exists():
        return path
    destination=path.with_name(path.name+".preserved_"+reason+"_"+str(int(time.time()))+
                               "_"+uuid.uuid4().hex[:7])
    if destination.exists():
        raise RuntimeError("MYCON_QUARANTINE_COLLISION")
    path.rename(destination)
    return destination


def collect_frames(images: Path) -> list[Path]:
    paths=sorted(images.glob("frame_*.jpg"))
    if not 80<=len(paths)<=240 or any(p.stat().st_size<1000 for p in paths):
        raise RuntimeError("MYCON_FRAMES_COUNT_OR_SIZE_INVALID")
    return paths


def frame_manifest(images: Path) -> dict:
    frames=collect_frames(images)
    return {"schema":"mycon.all_frames.sha256.v1",
            "count":len(frames),
            "files":{p.name:{"size":p.stat().st_size,"sha256":sha256(p)}
                     for p in frames}}


def validate_frame_manifest(images: Path, manifest: dict) -> int:
    frames=collect_frames(images)
    expected=manifest.get("files")
    if not isinstance(expected,dict) or set(expected)!={x.name for x in frames}:
        raise RuntimeError("MYCON_FRAME_MANIFEST_NAMES_MISMATCH")
    for p in frames:
        row=expected[p.name]
        if row.get("size")!=p.stat().st_size or row.get("sha256")!=sha256(p):
            raise RuntimeError("MYCON_FRAME_SHA_MISMATCH:"+p.name)
    if len(frames)!=manifest.get("count"):
        raise RuntimeError("MYCON_FRAME_MANIFEST_COUNT_MISMATCH")
    return len(frames)


def metrics_valid(measured: dict) -> bool:
    if not isinstance(measured,dict):
        return False
    for name in ("psnr","ssim","lpips"):
        x=measured.get(name)
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
            return False
    return (measured["psnr"]>0 and 0<=measured["ssim"]<=1
            and measured["lpips"]>=0)


def validate_artifact_file(path: Path, min_bytes: int=1024) -> bool:
    return path.is_file() and path.stat().st_size>=min_bytes


def ensure_free_space(path: Path, minimum_bytes: int, stage: str) -> None:
    where=path if path.is_dir() else path.parent
    free=shutil.disk_usage(where).free
    if free<minimum_bytes:
        raise RuntimeError(f"MYCON_DISK_INSUFFICIENT_{stage}: "
                           f"{free/2**30:.2f} GiB free")
