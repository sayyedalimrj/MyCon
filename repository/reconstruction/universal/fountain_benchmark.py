"""Independent RGBD comparison using held-out reference poses for alignment verification."""
import argparse
from pathlib import Path
from zipfile import ZipFile
import json
import re

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from reconstruction.da3_replay import load_ply

POSE_POSITION_P95_MAX = .12
POSE_ROTATION_P95_MAX = 8.
THRESHOLDS = (.02, .05, .10)


def evaluate(root, reference, output):
    source=json.loads((root/'Camera_Poses_and_Intrinsics.json').read_text())['cameras']
    source={int(re.search(r'f(\d+)\.jpg$',x['name']).group(1)):x for x in source}

    with ZipFile(reference) as archive:
        lines=archive.read('scene/key.log').decode().splitlines()
        ref = {int(lines[i].split()[0]):np.array([list(map(float,j.split())) for j in lines[i+1:i+5]])
               for i in range(0,len(lines),5)}
        color=sorted(i for i in archive.namelist() if i.startswith('image/') and i.endswith('.jpg'))
        depth=sorted(i for i in archive.namelist() if i.startswith('depth/') and i.endswith('.png'))
        if not len(color)==len(depth)==len(ref)==33:
            raise ValueError('Expected the 33-image Fountain benchmark release')
        timestamps=lambda a:np.array([int(re.search(r'-(\d+)\.',x).group(1)) for x in a])
        synchronization_s=abs(timestamps(color)-timestamps(depth))/1e6
        if synchronization_s.max()>=.04:
            raise ValueError('Corresponding image/depth samples differ by more than 40 ms')
        indices=sorted(ref.keys() & source.keys())
        a=np.array([source[i]['eye'] for i in indices])
        b=np.array([ref[i][:3,3] for i in indices])
        fit=np.array([i%5!=4 for i in indices])
        af,bf=a[fit]-a[fit].mean(axis=0),b[fit]-b[fit].mean(axis=0)
        u,_,v=np.linalg.svd(af.T@bf)
        sign=np.sign(np.linalg.det(v.T@u.T))
        Q=(v.T*np.array([1,1,sign]))@u.T
        scale=(bf*(af@Q.T)).sum()/np.square(af).sum()
        offset=b[fit].mean(axis=0)-scale*(a[fit].mean(axis=0)@Q.T)
        position=np.linalg.norm(scale*(a@Q.T)+offset-b,axis=1)
        target=np.array([ref[i][:3,:3] for i in indices])
        predicted=Q@np.array([source[i]['axes'] for i in indices]).transpose(0,2,1)
        rotation=Rotation.from_matrix(predicted.transpose(0,2,1)@target).magnitude()*180/np.pi
        p95=float(np.quantile(position[~fit],.95))
        r95=float(np.quantile(rotation[~fit],.95))
        accepted=bool(p95<=POSE_POSITION_P95_MAX and r95<=POSE_ROTATION_P95_MAX and scale>0)
        report={'status':'SCORED' if accepted else 'REFERENCE_ALIGNMENT_WITHHELD',
                'reference':'Open3D SampleFountainRGBDImages scene/integrated.ply and scene/key.log',
                'reconstruction_inputs':'All 33 original RGB frames. Reference geometry and poses were excluded from reconstruction.',
                'timestamp_color_depth_pair_max_s':float(synchronization_s.max()),
                'reference_pose_indices_reserved_for_alignment_check':[i for i in indices if i%5==4],
                'camera_position_p95_reference_units':p95,'camera_rotation_p95_degrees':r95,
                'predefined_alignment_limits':[POSE_POSITION_P95_MAX,POSE_ROTATION_P95_MAX],
                'similarity_scale':float(scale),'rotation':Q.tolist(),'translation':offset.tolist(),
                'scope':'Single scene comparison to sensor-derived reference in its coordinates. No survey accuracy.',
                'scoring':None}
        if accepted:
            with archive.open('scene/integrated.ply') as ply:
                head=ply.readline()
                while head.strip()!=b'end_header':
                    if head.startswith(b'element vertex '): n=int(head.split()[-1])
                    head=ply.readline()
                    if not head:
                        raise ValueError('Invalid reference PLY')
                cloud=np.frombuffer(ply.read(n*12),dtype='<f4').reshape(n,3)
            ref_tree=cKDTree(cloud)
            summary={}
            for name in ('Sparse','Dense'):
                xyz,_=load_ply(root/(name+'.ply'))
                points=scale*(xyz@Q.T)+offset
                distance_to_ref=ref_tree.query(points,workers=2)[0]
                distance_to_model=cKDTree(points).query(cloud,workers=2)[0]
                metrics={}
                for t in THRESHOLDS:
                    precision=float(np.mean(distance_to_ref<=t))
                    recall=float(np.mean(distance_to_model<=t))
                    metrics[str(t)]={'accuracy_fraction':precision,'completeness_fraction':recall,
                                     'f1':(2*precision*recall/max(precision+recall,1e-12))}
                summary[name]={'accuracy_median':float(np.median(distance_to_ref)),
                               'accuracy_p95':float(np.quantile(distance_to_ref,.95)),
                               'completeness_median':float(np.median(distance_to_model)),
                               'symmetric_chamfer_l2_mean':float((distance_to_ref.mean()+distance_to_model.mean())/2),
                               'threshold_metrics':metrics}
            report['scoring']=summary
            report['distance_note']='Euclidean nearest vertex distances over the complete supplied RGBD mesh vertices.'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],p95,r95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("reference_zip", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    evaluate(args.export_dir, args.reference_zip, args.output_json)


if __name__ == "__main__":
    main()
