#!/usr/bin/env python3
"""Download and verify the *actual* NVIDIA LongSplat research source and submodules.

This script intentionally does not vendor/relicense upstream NVIDIA code. The
original checkout is downloaded from the official repository under its original
research-only licence. No MyCON checkpoint, capture or result is modified.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

UPSTREAM = 'https://github.com/NVlabs/LongSplat.git'
# Verified from official NVlabs/LongSplat main on 2026-11-03.
PINNED_SHA = '19750775a9d19f30aa05a8333c4c6c231b2d5f4a'
SUBMODULES = (
    'submodules/mast3r',
    'submodules/diff-gaussian-rasterization',
    'submodules/simple-knn',
    'submodules/fused-ssim',
)
REQUIRED = ('train.py', 'render.py', 'convert_3dgs.py', 'LICENSE.md',
            'requirements.txt', '.gitmodules')


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(['git', *args], cwd=cwd, check=True,
                            capture_output=True, text=True)
    return result.stdout.rstrip(chr(13)+chr(10))


def verify(root: Path, expected: str = PINNED_SHA) -> dict:
    if not (root / '.git').exists():
        raise RuntimeError(f'Not a Git checkout: {root}')
    origin = _git('remote', 'get-url', 'origin', cwd=root)
    if origin.rstrip('/') not in (UPSTREAM, 'git@github.com:NVlabs/LongSplat.git'):
        raise RuntimeError(f'Unexpected upstream origin: {origin}')
    commit = _git('rev-parse', 'HEAD', cwd=root)
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or commit != expected:
        raise RuntimeError(f'Wrong LongSplat revision: {commit} expected {expected}')
    for rel in REQUIRED:
        if not (root/rel).is_file():
            raise RuntimeError(f'Incomplete NVIDIA checkout: missing {rel}')
    license_text = (root/'LICENSE.md').read_text(encoding='utf-8').lower()
    if 'non-commercially' not in license_text or 'research or evaluation' not in license_text:
        raise RuntimeError('Cannot establish original NVIDIA noncommercial licence')
    statuses = _git('submodule', 'status', '--recursive', cwd=root).splitlines()
    active = {}
    for line in statuses:
        if not line:
            continue
        if line[0] != ' ':
            raise RuntimeError('Submodule unavailable or at unexpected commit: '+line)
        fields = line.strip().split()
        if len(fields) >= 2:
            active[fields[1]] = fields[0]
    for rel in SUBMODULES:
        if rel not in active or not (root/rel).is_dir() or not any((root/rel).iterdir()):
            raise RuntimeError('Missing mandatory NVIDIA dependency checkout: '+rel)
    return {'schema':'mycon.nvidia_longsplat.checkout.v1',
            'upstream':UPSTREAM, 'commit':commit,
            'submodules':active, 'license':'NVIDIA research/evaluation noncommercial',
            'status':'CHECKOUT_VERIFIED_ONLY_NOT_CUDA_TESTED'}


def install(destination: Path, expected: str = PINNED_SHA) -> dict:
    destination = destination.expanduser().resolve()
    if destination.exists():
        result = verify(destination, expected)
        print('REUSED VERIFIED OFFICIAL NVIDIA SOURCE: '+str(destination))
        return result
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = parent / ('.'+destination.name+'.partial-'+uuid.uuid4().hex[:8])
    try:
        # Checkout a known 2026-11-03 upstream revision, not an unpinned release.
        _git('clone', '--recursive', UPSTREAM, str(temp))
        _git('checkout', '--detach', expected, cwd=temp)
        _git('submodule', 'sync', '--recursive', cwd=temp)
        _git('submodule', 'update', '--init', '--recursive', cwd=temp)
        result = verify(temp, expected)
        # No overwrite: safe on failure, preserve all working state.
        if destination.exists():
            raise RuntimeError(f'Destination appeared while cloning; keep both intact: {destination} {temp}')
        temp.rename(destination)
        print('OFFICIAL NVIDIA SOURCE INSTALLED: '+str(destination))
        return result
    except Exception:
        # Keep a failed checkout for forensic diagnosis, never silently delete.
        if temp.exists():
            print('PARTIAL_CHECKOUT_PRESERVED: '+str(temp), file=sys.stderr)
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest', type=Path, required=True,
                   help='Separate external source checkout; DO NOT point into Stage 3/5 checkpoint folders')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--write-report',type=Path)
    a=p.parse_args(argv)
    try:
        result = verify(a.dest.expanduser().resolve()) if a.verify_only else install(a.dest)
        if a.write_report:
            output=a.write_report.expanduser().resolve()
            output.parent.mkdir(parents=True,exist_ok=True)
            tmp=output.with_name(output.name+'.tmp')
            tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
            os.replace(tmp,output)
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0
    except (RuntimeError,ValueError,subprocess.CalledProcessError,OSError) as exc:
        print(f'NVIDIA_SOURCE_NOT_READY: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
