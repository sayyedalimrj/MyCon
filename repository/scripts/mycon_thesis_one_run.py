#!/usr/bin/env python3
"""Single supported Colab entrypoint for the real-data MyCON LongSplat acceptance run.

Default 'preflight' never trains. 'train' and 'all' require explicit selection.
Original Recorder, model/checkpoint folders and prior experiments are immutable.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from mycon_gpu_autoprofile import detect_hardware, profile_for
from mycon_longsplat_autoenv import effective_env, verify_cuda

DEFAULT_DRIVE = Path('/content/drive/MyDrive/MyCon_R4')
DEFAULT_LOCAL = Path('/content/MyCON_NVIDIA_Research_Work')
KNOWN_ENV = Path('/content/MyCON_agent_cuda121_20261105')
PINNED_ATTEMPT = 'stabilization_20261105'
MODES = ('preflight', 'train', 'all', 'status')


def validate_attempt(name: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', name):
        raise ValueError('ATTEMPT_ID_INVALID: ASCII letters, digits, _ and - only (max 48)')
    return name


def find_capture(drive: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        archive = supplied.expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError('CAPTURE_NOT_FOUND: ' + str(archive))
        return archive
    env_path = os.environ.get('MYCON_CAPTURE_ZIP')
    if env_path:
        return find_capture(drive, Path(env_path))
    # Accept only a unique large Recorder ZIP; never guess between captures.
    candidates: list[Path] = []
    for sub in ('captures', 'input_cache'):
        base = drive / sub
        if not base.is_dir():
            continue
        for archive in base.glob('**/*.zip'):
            if archive.is_file() and archive.stat().st_size >= 100_000_000:
                candidates.append(archive.resolve())
                if len(candidates) > 1:
                    break
        if len(candidates) > 1:
            break
    if len(candidates) != 1:
        raise RuntimeError('CAPTURE_PATH_REQUIRED: found ' + str(len(candidates)) +
                           ' potential Recorder ZIPs. Supply --capture /content/drive/MyDrive/.../original.zip')
    return candidates[0]


def choose_env(local: Path, profile: dict, diagnostics: Path,
               explicit: Path | None = None) -> tuple[Path, str]:
    if explicit is not None:
        return explicit.expanduser().resolve(), 'USER_SELECTED_VALIDATED_BY_CONTROLLER'
    h = profile['gpu']
    new_env = local / f"mycon_stabilized_sm{h['sm']}_{profile['torch_cuda_wheel']}"
    legacy = local / f"longsplat_py310_sm{h['sm']}_{profile['torch_cuda_wheel']}"
    diagnostics.mkdir(parents=True, exist_ok=True)
    for root in (KNOWN_ENV, legacy, new_env):
        py = root / 'bin/python'
        if not py.is_file():
            continue
        try:
            # Imports alone cannot certify binary CUDA compatibility.
            verify_cuda(py, profile, diagnostics / 'cuda_reuse_smoke.log',
                        effective_env(profile, root))
            print('MYCON_REUSED_LIVE_CUDA_VERIFIED:', root, flush=True)
            return root, 'REUSED_VERIFIED_CUDA_KERNELS'
        except Exception as exc:
            print('MYCON_OLD_ENV_NOT_VERIFIED:', root, type(exc).__name__,
                  str(exc)[-260:], flush=True)
    if new_env.exists():
        raise RuntimeError('NO_VERIFIED_ENV_AND_NEW_PREFIX_ALREADY_EXISTS: preserve ' +
                           str(new_env) + '; select a different empty --env-root explicitly')
    return new_env, 'NEW_ISOLATED_ENV_NO_OLD_DELETION'


def build_command(controller: Path, *, archive: Path, drive: Path, local: Path,
                  nvidia: Path, envroot: Path, mode: str, attempt: str,
                  quality: str) -> list[str]:
    if mode not in MODES or mode == 'status':
        raise ValueError('Invalid run mode')
    cmd = [sys.executable, str(controller), '--capture', str(archive),
           '--drive-root', str(drive), '--local-root', str(local),
           '--nvidia-root', str(nvidia), '--env-root', str(envroot),
           '--quality', quality, '--attempt-id', attempt]
    if mode == 'preflight':
        cmd += ['--stop-after', 'prepare']
    elif mode == 'train':
        cmd += ['--stop-after', 'train', '--approve-new-training']
    else:
        cmd += ['--approve-new-training']
    return cmd


def status(drive: Path, attempt: str) -> int:
    root = drive / 'research_longsplat_auto'
    if not root.is_dir():
        print('MYCON_STATUS: no research folder yet; no training claimed')
        return 2
    reports = sorted(root.glob('*/*/attempts/' + attempt + '/run_summary.json'),
                     key=lambda f: f.stat().st_mtime, reverse=True)
    if not reports:
        print('MYCON_STATUS: no run summary for attempt ' + attempt)
        return 2
    report = json.loads(reports[0].read_text())
    print(json.dumps({'file':str(reports[0]), 'status':report.get('status'),
                      'stages':report.get('stages', {}),
                      'bim_metric_validated':report.get('bim_metric_validated', False)},
                     ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=MODES, default='preflight')
    ap.add_argument('--capture', type=Path, help='Original FULL Recorder ZIP on Drive')
    ap.add_argument('--drive-root', type=Path, default=DEFAULT_DRIVE)
    ap.add_argument('--local-root', type=Path, default=DEFAULT_LOCAL)
    ap.add_argument('--nvidia-root', type=Path)
    ap.add_argument('--env-root', type=Path)
    ap.add_argument('--quality', choices=('quality','conservative'), default='quality')
    ap.add_argument('--attempt', default=PINNED_ATTEMPT)
    ap.add_argument('--accept-research-license', action='store_true',
                    help='Acknowledge official NVIDIA noncommercial research licence')
    a = ap.parse_args(argv)
    attempt = validate_attempt(a.attempt)
    drive = a.drive_root.expanduser().resolve()
    local = a.local_root.expanduser().resolve()
    if a.mode == 'status':
        return status(drive, attempt)
    if not a.accept_research_license:
        ap.error('Acknowledge NVIDIA research-only licence with --accept-research-license')
    if not drive.is_dir():
        raise RuntimeError('DRIVE_NOT_MOUNTED: mount Google Drive first')
    archive = find_capture(drive, a.capture)
    hardware = detect_hardware()
    profile = profile_for(hardware, a.quality)
    controller = Path(__file__).resolve().with_name('mycon_longsplat_auto_run.py')
    if not controller.is_file():
        raise RuntimeError('CANONICAL_CONTROLLER_NOT_FOUND: use this complete PR checkout')
    nvidia = (a.nvidia_root or local / 'NVIDIA_LongSplat').expanduser().resolve()
    diagnostics = drive / 'research_longsplat_auto' / 'diagnostics' / 'stabilization'
    envroot, source = choose_env(local, profile, diagnostics, a.env_root)
    cmd = build_command(controller, archive=archive, drive=drive, local=local,
                        nvidia=nvidia, envroot=envroot, mode=a.mode,
                        attempt=attempt, quality=a.quality)
    print('MYCON_ONE_RUN_SELECTION:',json.dumps({
          'mode':a.mode, 'capture':str(archive), 'gpu':hardware.model,
          'attempt':attempt, 'env':str(envroot), 'env_selection':source,
          'starts_training':a.mode in ('train','all'), 'rewrites_recorder':False
          },ensure_ascii=False),flush=True)
    proc = subprocess.run(cmd,check=False)
    if proc.returncode:
        print('MYCON_ONE_RUN_FAILED: inspect research_longsplat_auto/diagnostics/last_failure.json and logs',
              file=sys.stderr)
        return proc.returncode
    print('MYCON_ONE_RUN_PHASE_RETURNED_ZERO: inspect run_summary.json; metric BIM validation NOT certified',
          flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('MYCON_ONE_RUN_STOPPED_SAFELY:',type(exc).__name__,str(exc),file=sys.stderr)
        raise SystemExit(2)
