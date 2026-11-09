#!/usr/bin/env python3
"""Fetch/cache the *official* pretrained MASt3R checkpoint before GPU training.

The original NVIDIA LongSplat utility otherwise downloads this multi-GB weight
inside training. This module prevents partially transferred model weights from
being accepted as valid and caches them on Drive for future Colab runtimes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
URL='https://download.europe.naverlabs.com/ComputerVision/MASt3R/'+MODEL
MIN_BYTES=1_000_000_000


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def atomic_json(path: Path,value: dict):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temp,path)


def valid_cached(path: Path,meta: Path) -> dict | None:
    if not path.is_file() and not meta.is_file():return None
    if not path.is_file() or not meta.is_file():
        raise RuntimeError('MASt3R weight cache or SHA256 receipt is missing; DO NOT trust an incomplete file')
    row=json.loads(meta.read_text())
    if row.get('upstream')!=URL or row.get('size')!=path.stat().st_size or path.stat().st_size<MIN_BYTES:
        raise RuntimeError('MASt3R cache provenance or size mismatch')
    if sha(path)!=row.get('sha256'):
        raise RuntimeError('MASt3R cached model SHA256 mismatch')
    return row


def resumable_download(url: str,destination: Path, min_bytes: int=MIN_BYTES):
    destination.parent.mkdir(parents=True,exist_ok=True)
    incomplete=destination.with_suffix(destination.suffix+'.part')
    errors=[]
    for attempt in range(5):
        pos=incomplete.stat().st_size if incomplete.exists() else 0
        headers={'User-Agent':'MyCON-Thesis-LongSplat/1.0'}
        if pos:headers['Range']=f'bytes={pos}-'
        req=urllib.request.Request(url,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=90) as response:
                status=response.status
                if pos and status!=206:
                    # Server ignored Range; cannot safely append to old bytes.
                    incomplete.rename(incomplete.with_suffix('.rejected_range'))
                    pos=0
                if pos and status==206:
                    contentrange=response.headers.get('Content-Range','')
                    if not contentrange.startswith(f'bytes {pos}-'):
                        raise RuntimeError('Range response did not start at requested byte')
                mode='ab' if pos else 'wb'
                with incomplete.open(mode) as out:
                    while True:
                        data=response.read(4<<20)
                        if not data:break
                        out.write(data)
                length=incomplete.stat().st_size
                declared=response.headers.get('Content-Length')
                if declared and length != pos+int(declared):
                    raise IOError('Truncated MASt3R checkpoint transfer; will retry with Range')
                if length<min_bytes:
                    raise IOError(f'MASt3R pretrained model size too small: {length} bytes')
                os.replace(incomplete,destination)
                return
        except (urllib.error.URLError,TimeoutError,IOError,OSError,RuntimeError) as e:
            errors.append(f'{type(e).__name__}: {e}')
            if attempt<4:time.sleep(min(2+attempt,6))
    raise RuntimeError('MASt3R_WEIGHT_DOWNLOAD_FAILED: '+' | '.join(errors[-2:]))


def prepare_weights(nvidia_root: Path,drive_cache: Path, source_url: str=URL) -> dict:
    target=nvidia_root/'submodules'/'mast3r'/'checkpoints'/MODEL
    if not (nvidia_root/'utils'/'mast3r_utils.py').is_file():
        raise FileNotFoundError('Official NVIDIA checkout incomplete (missing MASt3R utility)')
    drive_cache.mkdir(parents=True,exist_ok=True)
    cached=drive_cache/MODEL
    receipt=drive_cache/'mast3r_checkpoint_receipt.json'
    if target.is_file():
        if target.stat().st_size<MIN_BYTES:
            raise RuntimeError('A partial local MASt3R weight file exists; refuse to overwrite it silently')
        # If verified Drive cache exists, compare two hashes. If no Drive cache,
        # this source-local weight was previously downloaded by the official
        # utility and we must first create an independent immutable receipt.
        old=valid_cached(cached,receipt)
        if old and sha(target)!=old['sha256']:
            raise RuntimeError('Existing local MASt3R weights disagree with verified Drive cache')
        if old:return {'status':'REUSED_LOCAL_WEIGHT','sha256':old['sha256'],'bytes':old['size']}
        raise RuntimeError('MASt3R local checkpoint has no verified cache receipt; refuse trusting an arbitrary multi-GB file')
    cached_report=valid_cached(cached,receipt)
    if cached_report is None:
        temp=drive_cache/MODEL
        if not target.is_file():
            local_download=nvidia_root.parent/(MODEL+'.verified_download')
            if local_download.exists():raise RuntimeError('Existing unreceipted weight download, preserve for diagnosis')
            resumable_download(source_url,local_download)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(local_download,target)
            if sha(local_download)!=sha(target):raise RuntimeError('Weight copy-to-NVIDIA hash mismatch')
            local_download.unlink(missing_ok=True)
        checksum=sha(target)
        tempcache=cached.with_suffix(cached.suffix+'.copying')
        if tempcache.exists():raise RuntimeError('Another or interrupted Drive weight cache copy exists; refuse overwrite')
        try:
            shutil.copy2(target,tempcache)
            if sha(tempcache)!=checksum:raise RuntimeError('Weight Drive cache hash mismatch')
            os.replace(tempcache,cached)
        finally:tempcache.unlink(missing_ok=True)
        cached_report={'schema':'mycon.mast3r.checkpoint.v1','upstream':source_url,
                       'name':MODEL,'size':cached.stat().st_size,'sha256':checksum,
                       'source':'official naverlabs MASt3R pretrained checkpoint'}
        atomic_json(receipt,cached_report)
    if not target.is_file():
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.restore')
        try:
            shutil.copy2(cached,tmp)
            if sha(tmp)!=cached_report['sha256']:raise RuntimeError('MASt3R restore SHA mismatch')
            os.replace(tmp,target)
        finally:tmp.unlink(missing_ok=True)
    if sha(target)!=cached_report['sha256']:
        raise RuntimeError('Final pretrained MASt3R model is corrupted')
    print('VERIFIED OFFICIAL MAST3R WEIGHTS:',target,'bytes',target.stat().st_size)
    return {'status':'WEIGHT_SHA256_VERIFIED','path':str(target),
            'bytes':target.stat().st_size,'sha256':cached_report['sha256'],
            'licensed_for_research':True,'model_inference_executed':False}


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--nvidia-root',type=Path,required=True)
    a.add_argument('--drive-cache',type=Path,required=True)
    opts=a.parse_args()
    print(json.dumps(prepare_weights(opts.nvidia_root,opts.drive_cache),indent=2))

if __name__=='__main__':main()
