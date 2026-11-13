"""Create-only SHA-audited archival of an explicitly approved NEW-run checkpoint.

No training, restart, resume, source mutation or historical 6999 compatibility is performed.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from .workflow import Blocked, checked_run, emit_json, sha256

def preserve_new_checkpoint(*, run_dir: Path, checkpoint: Path, archive_root: Path,
                            training_approved: bool = False) -> dict:
    checked_run(run_dir)
    if not training_approved:
        raise Blocked('Explicit new heavy-training approval absent: checkpoint archival disabled')
    root = run_dir.resolve()
    src = checkpoint.resolve()
    new_training = (root / 'gs_training_NEW_OPT_IN').resolve()
    if not src.is_relative_to(new_training) or not src.is_file() or src.is_symlink():
        raise Blocked('Only NEW-run training checkpoints are allowed (not historical R4/recovery)')
    if src.suffix.lower() not in ('.pt','.ckpt') or re.search(r'6999|006999',src.name):
        raise Blocked('Unexpected or historical 6999 checkpoint identity')
    if src.stat().st_size <= 0:
        raise Blocked('Checkpoint empty')
    plan = root / 'gs_training_plan.json'
    if not plan.is_file():
        raise Blocked('Missing new run GS training provenance plan')
    data = json.loads(plan.read_text())
    if data.get('run_id') != run_dir.name or data.get('schema_version') != 'mycon.reconstruction.gs_training_plan.v1':
        raise Blocked('GS training plan/run mismatch')
    if data.get('dataset_manifest_sha256') != sha256(root / 'dataset_manifest.json'):
        raise Blocked('Training dataset provenance changed')
    dest_dir = archive_root.resolve() / run_dir.name / 'approved_new_checkpoints'
    if dest_dir.exists():
        raise Blocked('Archive path exists. Never overwrite: choose another unique run/snapshot')
    dest_dir.mkdir(parents=True,exist_ok=False)
    dest = dest_dir / src.name
    with src.open('rb') as inp, dest.open('xb') as out:
        for chunk in iter(lambda: inp.read(4*1024*1024),b''):
            out.write(chunk)
        out.flush()
    source_hash = sha256(src)
    if sha256(dest)!=source_hash:
        raise Blocked('Checkpoint copy SHA mismatch; leave artifacts for forensic review')
    receipt = {'schema_version':'mycon.reconstruction.new_checkpoint_archive.v1',
               'run_id':run_dir.name,'new_training_checkpoint_only':True,
               'training_authorized_for_this_function':True,
               'source_path_relative_to_new_training_root':src.relative_to(new_training).as_posix(),
               'checkpoint_sha256':source_hash,'checkpoint_bytes':src.stat().st_size,
               'archived_file':dest.name,'dataset_manifest_sha256':data['dataset_manifest_sha256'],
               'historical_6999_or_recovery_touched':False}
    emit_json(dest_dir/'receipt.json',receipt)
    return receipt
