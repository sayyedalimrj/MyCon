#!/usr/bin/env python3
"""Research-only LongSplat backend for MyCON Stage 4.5.

This module is a non-destructive side branch. It does not change the canonical
Stage 1-11 outcomes or assert BIM-equivalent geometry. NVIDIA source remains in
an independent clone with its own non-commercial license.

Phases: plan, prepare, train, render, export, geometry, audit, status.

LongSplat has *not* been GPU-run on the MyCON worksite clip by this module's
maintainers. Every COMPLETE state below requires real output artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

VERSION = 'mycon.longsplat.research.v2'


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.new')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def capture_key(video: Path, sha: str) -> str:
    stem = re.sub(r'[^A-Za-z0-9_-]', '_', video.stem).strip('_')[:36] or 'worksite'
    return f'{stem}_{sha[:12]}'


def verify_longsplat_checkout(root: Path) -> None:
    for name in ('train.py', 'render.py', 'convert_3dgs.py', 'LICENSE.md'):
        if not (root / name).is_file():
            raise FileNotFoundError(f'LongSplat source missing: {root/name}')
    license_text = (root / 'LICENSE.md').read_text(encoding='utf-8').lower()
    if 'non-commercially' not in license_text:
        raise RuntimeError('Cannot establish original NVIDIA research license; refuse external training')


def commands(video: Path, source: Path, scene: Path, model: Path,
             fps: float, width: int, python: str, prune_ratio: float, post_iter: int) -> dict:
    return {
        # Re-encode only a reviewed/upright MP4. ffmpeg honors rotation metadata
        # unless the input was previously physically rotated; do not rotate twice.
        'prepare': ['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'error', '-i', str(video),
                    '-vf', f'fps={fps:g},scale={width}:-2:flags=lanczos', '-q:v', '2',
                    str(scene/'images'/'frame_%06d.jpg')],
        'train': [python, str(source/'train.py'), '--eval', '-s', str(scene),
                  '-m', str(model), '--images', 'images', '--mode', 'custom',
                  # Frames are already resized here; -r 4 would downscale twice.
                  '-r', '1', '--pose_iteration', '100', '--local_iter', '200',
                  '--global_iter', '500', '--post_iter', str(post_iter)],
        'render': [python, str(source/'render.py'), '-m', str(model)],
        'export': [python, str(source/'convert_3dgs.py'), '-m', str(model),
                   '--prune_ratio', str(prune_ratio)],
    }


def invoke(cmd: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as handle:
        handle.write('\n'+'='*60+'\n'+time.strftime('%FT%TZ', time.gmtime())+'\n')
        handle.write(json.dumps(cmd)+'\n')
        handle.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        raise RuntimeError(f'Command failed with code {proc.returncode}; see {log}')


def gaussian_ply(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 512:
        return False
    with path.open('rb') as f:
        header = f.read(8192).split(b'end_header', 1)[0]
    return all(token in header for token in (b'opacity', b'element vertex')) and (b'f_dc_0' in header or b'scale_0' in header)


def model_iteration(model: Path) -> int | None:
    values = []
    for folder in (model/'point_cloud').glob('iteration_*'):
        if folder.is_dir() and folder.name[10:].isdigit() and (folder/'point_cloud.ply').is_file():
            values.append(int(folder.name[10:]))
    return max(values) if values else None


def outputs(model: Path) -> dict:
    iteration = model_iteration(model)
    train_render = model/'train'/f'ours_{iteration}' if iteration is not None else model/'train'/'NO_ITERATION'
    return {
        'iteration': iteration,
        'novel_view_video': train_render/'videos'/'nvs_rgb.mp4',
        'novel_view_depth_video': train_render/'videos'/'nvs_depth.mp4',
        'rendered_train_depths': train_render/'depths',
        'rendered_train_images': train_render/'renders',
        'train_camera_poses': model/'cameras_all_train.json',
        'gaussian_ply': model/'converted_3dgs'/'point_cloud.ply',
    }


def manifest_base(video: Path, sha: str, repo: Path, fps: float, width: int,
                  prune_ratio: float, post_iter: int) -> dict:
    return {
        'schema': VERSION,
        'video': video.name,
        'video_sha256': sha,
        'longsplat_checkout': str(repo),
        'fps': fps, 'width': width, 'prune_ratio': prune_ratio,
        'post_iter': post_iter,
        'use': 'NONCOMMERCIAL_ACADEMIC_THESIS',
        'novel_views_supported_when_rendered': True,
        'geometry_is_metric_truth': False,
        'bim_progress_authorized': False,
        'explanation': '3DGS is a view synthesis representation; BIM comparisons require external metric registration and held-out controls',
    }


def require_matching_params(old: dict, fresh: dict) -> None:
    if not old:
        return
    for key in ('schema', 'video_sha256', 'fps', 'width', 'post_iter', 'prune_ratio'):
        if old.get(key) != fresh.get(key):
            raise RuntimeError(f'Existing experiment has different {key}; no overwrite or accidental resume')


def run(args: argparse.Namespace) -> int:
    video = args.video.expanduser().resolve()
    repo = args.longsplat_root.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    if not video.is_file() or video.suffix.lower() not in {'.mp4', '.mov', '.m4v'} or video.stat().st_size < 1024:
        raise ValueError('Provide the existing full-length upright video file, NOT Recorder ZIP')
    if not (1 <= args.fps <= 30 and 256 <= args.width <= 1920 and 0 < args.prune_ratio < 1 and 500 <= args.post_iter <= 100_000):
        raise ValueError('Invalid fps, width, prune ratio, or iteration count')
    verify_longsplat_checkout(repo)
    sha = digest_file(video)
    experiment = workspace/capture_key(video, sha)
    model = experiment/'longsplat_model'
    scene = experiment/'dataset'
    state_path = experiment/'research_state.json'
    fresh = manifest_base(video, sha, repo, args.fps, args.width, args.prune_ratio, args.post_iter)
    state = read_json(state_path)
    require_matching_params(state, fresh)
    cmd = commands(video, repo, scene, model, args.fps, args.width, args.python, args.prune_ratio, args.post_iter)
    observed = outputs(model)
    if args.phase == 'plan':
        print(json.dumps({**fresh, 'experiment': str(experiment), 'commands': cmd,
                          'note': 'dry-run, no stage mutations'}, indent=2))
        return 0
    if args.phase == 'status':
        report = {**fresh, 'phase_receipts': {key: bool(state.get(key)) for key in ('prepared','trained','rendered','exported','geometry_candidate')},
                  'files': {k: (str(v) if isinstance(v, Path) else v) for k, v in observed.items()}}
        print(json.dumps(report, indent=2))
        return 0
    if not args.accept_research_license:
        raise RuntimeError('Explicit --accept-research-license required; NVIDIA LongSplat is non-commercial research only')

    if args.phase == 'prepare':
        saved = state.get('prepared')
        frames = list((scene/'images').glob('frame_*.jpg'))
        if saved:
            if len(frames) != state['frame_count'] or len(frames) < 30:
                raise RuntimeError('Saved frame receipt disagrees with actual files; investigate, do not overwrite')
            print('REUSED', len(frames), 'verified-frame-count receipt')
            return 0
        if frames:
            raise RuntimeError('Partial extracted frames exist; refuse overwrite without diagnosis')
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('ffmpeg is required for frame extraction')
        (scene/'images').mkdir(parents=True, exist_ok=True)
        invoke(cmd['prepare'], cwd=repo, log=experiment/'logs'/'prepare.log')
        frames = sorted((scene/'images').glob('frame_*.jpg'))
        if len(frames) < 30 or any(f.stat().st_size == 0 for f in frames):
            raise RuntimeError('Frame extraction incomplete; no prepared receipt issued')
        state = {**fresh, 'prepared': True, 'frame_count': len(frames),
                 'trained': False, 'rendered': False, 'exported': False, 'geometry_candidate': False}
        atomic_json(state_path, state)
        print('PREPARED', len(frames), 'upright frames. Source unchanged.')
        return 0

    if not state.get('prepared') or not (scene/'images').is_dir():
        raise RuntimeError('Prepare phase must be complete before GPU phases')
    if args.phase == 'train':
        if state.get('trained'):
            if not observed['train_camera_poses'].is_file() or observed['iteration'] is None:
                raise RuntimeError('Training receipt exists but model artifacts are missing')
            print('REUSED COMPLETE LONGSPLAT TRAINING')
            return 0
        if model.exists() and any(model.iterdir()):
            raise RuntimeError('Partial training data found. The upstream trainer does not provide verified interruption resume; do not delete or overwrite.')
        model.mkdir(parents=True, exist_ok=True)
        invoke(cmd['train'], cwd=repo, log=experiment/'logs'/'train.log')
        observed = outputs(model)
        if observed['iteration'] is None or not observed['train_camera_poses'].is_file():
            raise RuntimeError('Trainer returned but no model + camera poses; refusing COMPLETE')
        state.update({'trained': True, 'iteration': observed['iteration']})
        atomic_json(state_path, state)
        print('TRAINING COMPLETE: photorealistic model, not metric certified')
        return 0
    if not state.get('trained'):
        raise RuntimeError('LongSplat training must complete first')

    if args.phase == 'render':
        if state.get('rendered') and observed['novel_view_video'].is_file():
            print('REUSED NOVEL-VIEW RENDER', observed['novel_view_video'])
            return 0
        invoke(cmd['render'], cwd=repo, log=experiment/'logs'/'render.log')
        observed = outputs(model)
        if (not observed['novel_view_video'].is_file() or observed['novel_view_video'].stat().st_size < 1024
            or not observed['train_camera_poses'].is_file()
            or not observed['rendered_train_depths'].is_dir()
            or len(list(observed['rendered_train_depths'].glob('*.npy'))) < 3):
            raise RuntimeError('Novel-view or raw-depth render missing. Do not issue success receipt')
        state.update({'rendered': True, 'novel_view_video': str(observed['novel_view_video']),
                      'raw_depth_dir': str(observed['rendered_train_depths']),
                      'camera_json': str(observed['train_camera_poses'])})
        atomic_json(state_path, state)
        print('NOVEL VIEW READY', observed['novel_view_video'])
        return 0

    if args.phase == 'export':
        if state.get('exported') and gaussian_ply(observed['gaussian_ply']):
            print('REUSED GAUSSIAN PLY', observed['gaussian_ply'])
            return 0
        invoke(cmd['export'], cwd=repo, log=experiment/'logs'/'export.log')
        if not gaussian_ply(observed['gaussian_ply']):
            raise RuntimeError('No valid Gaussian .ply (opacity + SH channels); refuse export receipt')
        state.update({'exported': True, 'gaussian_ply': str(observed['gaussian_ply'])})
        atomic_json(state_path, state)
        print('INTERACTIVE NOVEL-VIEW PLY READY:', observed['gaussian_ply'])
        return 0

    if args.phase == 'audit':
        # Held-out camera views test generalization. They do NOT validate
        # arbitrary unseen viewpoints or metric scale.
        from .longsplat_visual_audit import evaluate_holdout_views
        result=evaluate_holdout_views(model, observed['iteration'], experiment/'novel_view_qa.json')
        state.update({'visual_qa_status': result['status'],
                      'visual_qa_report': str(experiment/'novel_view_qa.json')})
        atomic_json(state_path, state)
        print('NOVEL-VIEW HOLDOUT QA:', result['status'], 'PSNR',result.get('psnr_mean_db'))
        return 0

    if args.phase == 'geometry':
        if not state.get('rendered'):
            raise RuntimeError('Raw depth and camera poses must first be rendered')
        from .longsplat_depth_geometry import extract_depth_candidate
        dest = experiment/'geometry_candidate'
        out = extract_depth_candidate(observed['rendered_train_depths'], observed['train_camera_poses'],
                                      scene/'images', dest, max_points=args.max_geometry_points)
        state.update({'geometry_candidate': True, 'geometry_report': str(out['report_path']),
                      'geometry_ply': str(out['ply_path'])})
        atomic_json(state_path, state)
        print('GEOMETRY CANDIDATE (ARBITRARY SCALE, NOT BIM-AUTHORIZED)', out['ply_path'])
        return 0
    raise ValueError('Unknown phase')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=('plan','prepare','train','render','export','geometry','audit','status'))
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--longsplat-root', required=True, type=Path)
    p.add_argument('--workspace', required=True, type=Path)
    p.add_argument('--python', default=sys.executable, help='Interpreter with all LongSplat CUDA deps')
    p.add_argument('--fps', default=10., type=float)
    p.add_argument('--width', default=512, type=int)
    p.add_argument('--post-iter', default=20000, type=int)
    p.add_argument('--prune-ratio', default=.6, type=float)
    p.add_argument('--max-geometry-points', default=1_500_000, type=int)
    p.add_argument('--accept-research-license', action='store_true')
    return run(p.parse_args(argv))

if __name__ == '__main__':
    raise SystemExit(main())