"""Read-only fail-closed MyCON Viewer v1 bundle loader. No reconstruction dependency."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = 'mycon.viewer_bundle.v1'
REQUIRED = ('camera_poses', 'frame_manifest', 'planes', 'edges', 'corners', 'mesh', 'alignment_result')
SAFE_ID = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$')

class ViewerInputError(ValueError):
    pass

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def safe_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or '\\' in relative or Path(relative).is_absolute() or ':' in relative:
        raise ViewerInputError(f'Unsafe relative path: {relative!r}')
    if any(x in ('', '.', '..') for x in relative.split('/')):
        raise ViewerInputError(f'Unsafe relative path: {relative!r}')
    base = root.resolve()
    p = (base / relative).resolve()
    if not p.is_relative_to(base):
        raise ViewerInputError(f'Path escapes bundle: {relative}')
    if not p.is_file():
        raise ViewerInputError(f'Missing input file: {relative}')
    return p

def _json(path: Path) -> dict:
    if path.stat().st_size > 25_000_000:
        raise ViewerInputError(f'JSON too large for standalone preview: {path.name}')
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ViewerInputError(f'Invalid JSON {path}: {exc}') from exc
    if not isinstance(obj, dict):
        raise ViewerInputError(f'Expected JSON object: {path}')
    return obj

def _gate(alignment: dict, manifest: dict) -> tuple[bool, str]:
    """Never infer metric deviation from visually aligned or synthetic data."""
    if manifest.get('synthetic'):
        return False, 'synthetic_dataset'
    for key in ('metric_scale_verified', 'registration_verified'):
        if manifest.get(key) is not True or alignment.get(key) is not True:
            return False, f'{key}_not_verified'
    if alignment.get('independent_control_verified') is not True:
        return False, 'independent_control_not_verified'
    if alignment.get('coordinate_frame') != manifest.get('coordinate_frame'):
        return False, 'alignment_frame_mismatch'
    if manifest.get('units') != 'm' or alignment.get('units') != 'm':
        return False, 'requires_verified_meters'
    return True, 'verified_metric_alignment'

def load_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest = _json(safe_relative(root, 'viewer_manifest.json'))
    if manifest.get('schema_version') != SCHEMA:
        raise ViewerInputError(f'Unsupported manifest schema: {manifest.get("schema_version")!r}; need {SCHEMA}')
    if not SAFE_ID.fullmatch(str(manifest.get('run_id', ''))):
        raise ViewerInputError('Unsafe or missing run_id')
    if not manifest.get('coordinate_frame') or not manifest.get('units'):
        raise ViewerInputError('Missing units/coordinate_frame')
    records = manifest.get('files')
    if not isinstance(records, dict):
        raise ViewerInputError('Missing files map')
    missing = sorted(set(REQUIRED) - records.keys())
    if missing:
        raise ViewerInputError(f'Missing required adapters: {missing}')
    assets: dict[str, dict] = {}
    for role, receipt in records.items():
        if not isinstance(receipt, dict):
            raise ViewerInputError(f'Malformed {role} receipt')
        for key in ('path','sha256','schema_version','units','coordinate_frame','quality','provenance'):
            if key not in receipt:
                raise ViewerInputError(f'{role} missing receipt {key}')
        p = safe_relative(root, receipt['path'])
        if not re.fullmatch('[0-9a-f]{64}', str(receipt['sha256'])) or sha256(p) != receipt['sha256']:
            raise ViewerInputError(f'SHA256 mismatch: {role}:{receipt["path"]}')
        if receipt['coordinate_frame'] != manifest['coordinate_frame'] or receipt['units'] != manifest['units']:
            raise ViewerInputError(f'Frame/units mismatch: {role}')
        obj = _json(p)
        for key in ('schema_version','units','coordinate_frame','quality','provenance'):
            if key not in obj:
                raise ViewerInputError(f'{role} content missing {key}')
        if obj['schema_version'] != receipt['schema_version'] or obj['coordinate_frame'] != manifest['coordinate_frame'] or obj['units'] != manifest['units']:
            raise ViewerInputError(f'Schema/frame/units conflict: {role}')
        assets[role] = obj
    frames = assets['frame_manifest'].get('frames', [])
    cameras = assets['camera_poses'].get('cameras', [])
    if not isinstance(frames, list) or not isinstance(cameras, list):
        raise ViewerInputError('Invalid camera or frame list')
    frame_ids = set()
    for frame in frames:
        fid = frame.get('frame_id')
        if not isinstance(fid, str) or fid in frame_ids:
            raise ViewerInputError(f'Non-unique frame_id {fid!r}')
        frame_ids.add(fid)
        file = safe_relative(root, frame['path'])
        if sha256(file) != frame.get('sha256'):
            raise ViewerInputError(f'Image SHA256 mismatch: {frame["path"]}')
    camera_ids = set()
    for cam in cameras:
        cid = cam.get('camera_id')
        if not isinstance(cid,str) or cid in camera_ids or cam.get('frame_id') not in frame_ids:
            raise ViewerInputError(f'Camera ID or frame link invalid: {cid!r}')
        camera_ids.add(cid)
        m = cam.get('camera_to_world')
        if not (isinstance(m, list) and len(m) == 4 and all(isinstance(row, list) and len(row) == 4 for row in m)):
            raise ViewerInputError(f'Camera transform not 4x4: {cid}')
        if cam.get('extrinsic_convention') != 'camera_to_world':
            raise ViewerInputError(f'Camera convention not explicit c2w: {cid}')
        for k in ('fx','fy','cx','cy','width','height'):
            if not isinstance(cam.get(k), (int, float)):
                raise ViewerInputError(f'Missing camera intrinsics {cid}:{k}')
    verified, reason = _gate(assets['alignment_result'], manifest)
    deviations = assets['mesh'].get('deviations')
    if verified and isinstance(deviations, dict):
        if deviations.get('units') != 'm' or deviations.get('independently_verified') is not True or not deviations.get('provenance') or len(deviations.get('per_face_m', [])) != len(assets['mesh'].get('faces', [])):
            verified = False
            reason = 'deviation_evidence_invalid'
    else:
        verified = False
        reason = reason if reason != 'verified_metric_alignment' else 'no_verified_deviation_samples'
    return {'manifest':manifest,'assets':assets, 'deviation_enabled': verified, 'deviation_reason':reason, 'root':root}
