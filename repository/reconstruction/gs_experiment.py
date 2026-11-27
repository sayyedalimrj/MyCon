"""Build an independent train-only COLMAP/GS dataset; never launch training.

The historical 6999 checkpoint is intentionally not accepted by this tool.
A dataset for gsplat must never contain holdout RGB files, even if gsplat's
own internal validation split uses a subset of the new training partition.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .workflow import Blocked, checked_run, emit_json, rel, sha256, parse_colmap_text


def prepare_gs_dataset(run_dir:Path) -> dict:
    m=checked_run(run_dir)
    audit=run_dir/'colmap'/'colmap_execution.json'
    if not audit.is_file():
        raise Blocked('Fresh COLMAP must be completed before GS dataset creation')
    info=json.loads(audit.read_text(encoding='utf-8'))
    if info.get('schema_version') != 'mycon.reconstruction.colmap_plan.v1' or info.get('run_id') != run_dir.name or info.get('status')!='colmap_executed':
        raise Blocked('Fresh train-only COLMAP mapping not verified')
    model=(run_dir/info['train_model_relpath']).resolve()
    if not model.is_relative_to(run_dir.resolve()):
        raise Blocked('Train COLMAP model points outside the new run')
    expected=info['train_model_sha256']
    for name in ('cameras.bin','images.bin','points3D.bin'):
        if not (model/name).is_file() or sha256(model/name) != expected[name]:
            raise Blocked('New train-only COLMAP sparse model SHA mismatch')
    cams,known=parse_colmap_text(run_dir/'colmap'/'train_text')
    train=[f for f in m['frames'] if f['split']=='train']
    test_names={str(Path(f['image_relpath']).relative_to('images')).replace('\\','/') for f in m['frames'] if f['split']=='test'}
    if test_names.intersection(known):
        raise Blocked('Test image registered in new train-only COLMAP model; leak')
    if not known:
        raise Blocked('COLMAP train model did not register any frames')
    target=run_dir/'gs_train_dataset'
    if target.exists():
        raise Blocked('GS training dataset exists; no overwrite or partial repair')
    target.mkdir(exist_ok=False)
    (target/'images').mkdir()
    (target/'sparse').mkdir()
    (target/'sparse'/'0').mkdir()
    receipts=[]
    for f in train:
        name=Path(f['image_relpath']).relative_to('images')
        src=(run_dir/f['image_relpath']).resolve()
        if sha256(src) != f['image_sha256']:
            raise Blocked('Source train frame changed since selection')
        dest=target/'images'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        # physical copy, not symlink to mutable source
        shutil.copy2(src,dest)
        receipts.append({'view_id':f['view_id'],'image_sha256':sha256(dest),'relpath':rel(dest,run_dir)})
    for name in ('cameras.bin','images.bin','points3D.bin'):
        shutil.copy2(model/name,target/'sparse'/'0'/name)
    report={'schema_version':'mycon.reconstruction.gs_train_dataset.v1','run_id':run_dir.name,
            'source_dataset_sha256':sha256(run_dir/'dataset_manifest.json'),
            'colmap_execution_sha256':sha256(audit),'train_sparse_model_sha256':expected,
            'coordinate_frame':'colmap_opencv_world_to_camera','units':'reconstruction_units_unscaled',
            'metric_scale_verified':False,'train_view_count':len(train),'test_views_copied':0,
            'test_capture_sessions_excluded':sorted({f['capture_session_id'] for f in m['frames'] if f['split']=='test'}),
            'image_receipts':receipts,
            'gsplat_example_command_template':[
                'python','examples/simple_trainer.py','default',
                '--data_dir',str(target.resolve()), '--result_dir',str((run_dir/'gs_training_NEW_OPT_IN').resolve()),
                '--disable_viewer','--save_ply'],
            'command_execution':'NOT_EXECUTED_fresh_training_requires_explicit_opt_in',
            'resumption':'NOT_IMPLEMENTED_no_automatic_retrain_or_cross_dataset_checkpoint_restore',
            'notice':'gsplat internal eval split is not the protected independent MyCON test set'}
    emit_json(target/'gs_train_dataset.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    a=parser.parse_args(argv)
    try:
        r=prepare_gs_dataset(a.run_dir)
        print(json.dumps({'status':'prepared_training_dataset_ONLY','train':r['train_view_count'],'test_copied':0}))
        return 0
    except (Blocked,OSError,ValueError,KeyError) as e:
        print(json.dumps({'status':'blocked','reason':str(e)}))
        return 2

if __name__=='__main__':
    raise SystemExit(main())
