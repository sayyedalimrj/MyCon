#!/usr/bin/env python3
"""Extract the verified ORIGINAL MyCON Recorder ZIP without altering any prior R4 data.

Only one verified recording is accepted; original frame+pose metadata stay in the
research experiment. The video is extracted without automatic rotation. The
LongSplat bridge applies the previously VERIFIED clockwise rotation to frames.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import uuid
import time
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def safe_name(s: str) -> str:
    if not s or '\\' in s or s.startswith('/') or '\x00' in s or ':' in s:
        raise ValueError('Unsafe ZIP member path: '+repr(s))
    parts=s.split('/')
    if any(p in ('','.', '..') for p in parts) or s.startswith('../'):
        raise ValueError('Unsafe ZIP member traversal: '+repr(s))
    normalized=posixpath.normpath(s)
    if normalized=='..' or normalized.startswith('../') or normalized.startswith('/'):
        raise ValueError('Unsafe ZIP member traversal: '+repr(s))
    return normalized


def manifest_rows(meta: dict) -> dict:
    rows=meta.get('files')
    if not isinstance(rows,list):raise ValueError('Recorder integrity_sha256.json lacks file manifest')
    out={}
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Bad Recorder manifest entry')
        name=safe_name(str(row.get('path','')))
        if name in out:raise ValueError('Duplicate manifest entry '+name)
        out[name]=row
    return out


def source_members(z: zipfile.ZipFile) -> tuple[str,dict,dict,dict]:
    entries={}
    for info in z.infolist():
        if info.is_dir():continue
        name=safe_name(info.filename)
        if name in entries:raise ValueError('Duplicate ZIP member: '+name)
        if ((info.external_attr>>16)&stat.S_IFMT(0o170000)) == stat.S_IFLNK:
            raise ValueError('ZIP symlink forbidden: '+name)
        entries[name]=info
    manifests=[k for k in entries if k=='integrity_sha256.json' or k.endswith('/integrity_sha256.json')]
    if len(manifests)!=1:raise ValueError('Require exactly one integrity_sha256.json')
    prefix=manifests[0][:-len('integrity_sha256.json')]
    session_name=prefix+'session.json'
    if session_name not in entries:raise ValueError('Missing original Recorder session.json')
    session=json.loads(z.read(session_name))
    meta=json.loads(z.read(manifests[0]))
    if meta.get('schema')!='mycon.capture.integrity.v1':
        raise ValueError('Unsupported Recorder integrity schema')
    mapping=manifest_rows(meta)
    expected={name[len(prefix):] for name in entries if name.startswith(prefix) and name!=manifests[0]}
    if set(mapping)!=expected:
        raise ValueError('Recorder integrity manifest misses or adds file entries')
    required={'session.json','pose.csv','imu.csv','qr_events.jsonl',
              'r4_camera.json','r4_controls.json','r4_compatibility.json',
              'R4_IMPORT_README.txt','tools/mycon_r4_bridge.py'}
    if not required.issubset(mapping):
        raise ValueError('Original Recorder has missing required sidecars: '+str(sorted(required-set(mapping))))
    video_name=safe_name(session.get('video_dataset','arcore_recording.mp4'))
    full=prefix+video_name
    if full not in entries:raise ValueError('Recorder video from session.json missing: '+full)
    info=entries[full]; row=mapping.get(video_name)
    if not row or type(row.get('size_bytes')) is not int or row['size_bytes']!=info.file_size:
        raise ValueError('Recorder video size differs from manifest or size missing')
    if info.file_size < 1000000:raise ValueError('Recorder video unexpectedly small')
    return prefix,entries,mapping,{'session':session,'video_name':video_name,'video_full':full,'manifest':meta}


def _extract_verified(z: zipfile.ZipFile, name: str, info: zipfile.ZipInfo, entry: dict,
                      target: Path) -> str:
    supplied=entry.get('sha256')
    expected=str(supplied).lower() if supplied is not None else None
    policy=entry.get('hash_policy','SHA256' if supplied else '')
    if policy=='SHA256':
        if not expected or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('Missing SHA-256 for '+name)
    elif policy=='DEFERRED_LARGE_BINARY' and supplied is None and Path(name).suffix.lower() in ('.mp4','.d16','.u8','.f32'):
        # Exactly the policy of the original MyCON Recorder native importer:
        # compute SHA256 now, validate ZIP CRC and size, and persist its receipt.
        expected=None
    else:
        raise ValueError('Unsupported native Recorder integrity policy for '+name)
    if entry.get('size_bytes')!=info.file_size:
        raise ValueError('Bad recorded byte size for '+name)
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+'.partial-'+uuid.uuid4().hex[:8])
    h=hashlib.sha256();written=0
    try:
        with z.open(info) as source, tmp.open('wb') as dst:
            for chunk in iter(lambda:source.read(4<<20),b''):
                written+=len(chunk)
                h.update(chunk)
                dst.write(chunk)
        if written!=info.file_size or (expected is not None and h.hexdigest()!=expected):
            raise RuntimeError('Recorder SHA256/length mismatch on '+name)
        os.replace(tmp,target)
        return h.hexdigest()
    finally:
        tmp.unlink(missing_ok=True)


def prepare(zip_path: Path, out: Path, verified_rotation_cw: int=90) -> dict:
    if verified_rotation_cw not in (-90,0,90,180):
        raise ValueError('Invalid physical pixel rotation')
    zip_path=zip_path.expanduser().resolve()
    if not zip_path.is_file():raise FileNotFoundError('Existing Recorder archive missing: '+str(zip_path))
    if zip_path.stat().st_size<2_000_000:raise ValueError('Source Recorder archive suspiciously small')
    out=out.expanduser().resolve()
    if zip_path==out or zip_path in out.parents:raise ValueError('Cannot write inside capture archive path')
    receipt=out/'capture_receipt.json';archive_sha=sha256(zip_path)
    if receipt.is_file():
        row=json.loads(receipt.read_text())
        video=out/'raw_capture.mp4'
        if row.get('archive_sha256')!=archive_sha or row.get('pixel_rotation_cw')!=verified_rotation_cw:
            raise RuntimeError('Capture identity/rotation changed; REFUSING to overwrite existing experiment')
        if video.is_file() and sha256(video)==row.get('video_sha256'):
            # Verify all separately imported Recorder evidence too; a partial
            # Drive-to-Colab restore must never silently lose camera/IMU/pose.
            sidecars=row.get('sidecars',{})
            sidecars_ok=(bool(sidecars) and all(
                (out/name).is_file() and (out/name).stat().st_size==int(meta['bytes'])
                and sha256(out/name)==meta['sha256']
                for name,meta in sidecars.items()))
            if sidecars_ok:
                print('REUSED_VERIFIED_CAPTURE_AND_SIDECARS:',video)
                return row
            print('Sidecar receipt mismatch: derived local capture will be restored from immutable ZIP',flush=True)
        # Derived LOCAL extracted inputs are recoverable from the immutable
        # original Drive ZIP. Quarantine this copy (never delete the original).
        quarantine=out.with_name(out.name+'.invalid_'+time.strftime('%Y%m%d_%H%M%S'))
        if quarantine.exists():raise RuntimeError('Capture recovery collision: '+str(quarantine))
        out.rename(quarantine)
        print('RECOVERING_CAPTURE_FROM_IMMUTABLE_ZIP; preserved broken local copy:',quarantine,flush=True)
    if out.exists() and any(out.iterdir()):
        # Only our local derived input area is moved, never user Drive sources.
        quarantine=out.with_name(out.name+'.partial_'+time.strftime('%Y%m%d_%H%M%S'))
        if quarantine.exists():raise RuntimeError('Capture recovery collision: '+str(quarantine))
        out.rename(quarantine)
        print('RECOVERING_INTERRUPTED_CAPTURE_EXTRACTION; previous files quarantined:',quarantine,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    # No whole-archive ZIP extraction: unpack only the media and Recorder evidence
    # needed by Stage 4.5, with manifest SHA256 validation.
    with zipfile.ZipFile(zip_path) as z:
        prefix,infos,rows,record=source_members(z)
        source_video=record['video_name'];dest=out/'raw_capture.mp4'
        vid_sha=_extract_verified(z,source_video,infos[record['video_full']],rows[source_video],dest)
        sidecar={}
        for relative in ('session.json','pose.csv','imu.csv','qr_events.jsonl','r4_camera.json',
                         'r4_controls.json','r4_compatibility.json','R4_IMPORT_README.txt'):
            full=prefix+relative
            if full in infos and relative in rows:
                actual=_extract_verified(z,relative,infos[full],rows[relative],out/relative)
                sidecar[relative]={'sha256':actual,'bytes':infos[full].file_size}
        mandatory=('session.json','pose.csv','imu.csv','r4_camera.json','r4_controls.json',
                   'r4_compatibility.json','qr_events.jsonl')
        missing=[name for name in mandatory if name not in sidecar]
        if missing:raise RuntimeError('Missing native verified sidecars: '+str(missing))
    if shutil.which('ffprobe') is None:
        raise RuntimeError('FFprobe required for full-video frame and rotation QA')
    q=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                      'stream=width,height,avg_frame_rate,nb_frames,duration:format=duration',
                      '-of','json',str(dest)],text=True,capture_output=True,timeout=60)
    if q.returncode:raise RuntimeError('Original video unreadable by ffprobe: '+q.stderr[-300:])
    data=json.loads(q.stdout);streams=data.get('streams',[])
    if not streams:raise RuntimeError('Original video lacks a video stream')
    duration=float(data.get('format',{}).get('duration') or streams[0].get('duration') or 0)
    if duration < 10:raise RuntimeError('Recorder video too short to match required complete worksite capture')
    row={'schema':'mycon.native_zip.extract.v2','status':'ORIGINAL_CAPTURE_SHA256_VERIFIED',
         'archive_name':zip_path.name,'archive_sha256':archive_sha,
         'raw_video':str(dest),'video_sha256':vid_sha,'video_bytes':dest.stat().st_size,
         'recorded_video_member':source_video,'pixel_rotation_cw':verified_rotation_cw,
         'full_video_duration_s':duration,'frames_source':streams[0].get('nb_frames'),
         'coded_width':int(streams[0].get('width',0)),'coded_height':int(streams[0].get('height',0)),
         'sidecars':sidecar,'survey_metric_validated':False,
         'notes':'Rotate decoded frame pixels exactly once (ffmpeg -noautorotate), not the original video/pose'}
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(row,ensure_ascii=False,indent=2))
    os.replace(temp,receipt)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rotation',type=int,choices=[-90,0,90,180],default=90)
    args=parser.parse_args()
    print(json.dumps(prepare(args.capture,args.output,args.rotation),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
