#!/usr/bin/env python3
"""Safely summarize recorded ARCore pose/IMU WITHOUT pretending they are surveyed truth.

Read native recorder sidecars for input QA, not pose injection into unposed
LongSplat. Pure rotation/low parallax should be explicitly visible in the report.
"""
from __future__ import annotations
import csv
import json
import math
import statistics
from pathlib import Path


def audit(path: Path) -> dict:
    src=Path(path)
    pose=src/'pose.csv'
    session=src/'session.json'
    camera=src/'r4_camera.json'
    imu=src/'imu.csv'
    for f in (pose,session,camera,imu):
        if not f.is_file():raise FileNotFoundError('Missing native Recorder evidence: '+str(f))
    identity=json.loads(session.read_text(encoding='utf-8'))
    cal=json.loads(camera.read_text(encoding='utf-8'))
    times=[];locations=[];quats=[];tracked=0;total=0
    with pose.open(newline='',encoding='utf-8-sig') as stream:
        rows=csv.DictReader(stream)
        if not rows.fieldnames or not {'timestamp_ns','tx_m','ty_m','tz_m'}.issubset(set(rows.fieldnames)):
            raise ValueError('Recorder pose.csv missing required timestamp/XYZ fields')
        for row in rows:
            total+=1
            try:
                t=int(row['timestamp_ns'])
                xyz=tuple(float(row[k]) for k in ('tx_m','ty_m','tz_m'))
                if t<=0 or not all(math.isfinite(v) for v in xyz):continue
                if row.get('tracking','TRACKING').upper() not in ('TRACKING',''):
                    continue
                times.append(t);locations.append(xyz);tracked+=1
                if all(k in row for k in ('qx','qy','qz','qw')):
                    q=tuple(float(row[k]) for k in ('qx','qy','qz','qw'))
                    quats.append(q if all(math.isfinite(v) for v in q) else None)
                else:quats.append(None)
            except (ValueError,TypeError,KeyError):continue
    if len(times)<20:
        raise RuntimeError('RECORDER_POSE_INSUFFICIENT: fewer than 20 valid ARCore tracking rows')
    if any(x>=y for x,y in zip(times,times[1:])):
        raise RuntimeError('RECORDER_POSE_TIME_NONMONOTONIC: cannot safely reason about motion')
    span=(times[-1]-times[0])*1e-9
    distances=[];angles=[];rotation_low_translation=0
    for i in range(1,len(times)):
        a=locations[i-1];b=locations[i]
        d=math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
        distances.append(d)
        qa,qb=quats[i-1],quats[i]
        if qa is not None and qb is not None:
            dot=sum(x*y for x,y in zip(qa,qb))
            na=math.sqrt(sum(v*v for v in qa));nb=math.sqrt(sum(v*v for v in qb))
            cos=abs(dot/max(na*nb,1e-12))
            a_deg=2*math.degrees(math.acos(max(0.,min(1.,cos))))
            angles.append(a_deg)
            if d<0.025 and a_deg>3.:rotation_low_translation+=1
    report={'schema':'mycon.recorder.motion_qa.v1','status':'VALID_NATIVE_POSE_INPUT_NOT_SURVEY_METRIC',
            'all_rows':total,'tracked_rows':tracked,'tracking_fraction':tracked/max(total,1),
            'time_span_s':span,'median_step_translation_m':statistics.median(distances),
            'max_step_translation_m':max(distances),'median_step_rotation_deg':statistics.median(angles) if angles else None,
            'rotation_dominant_small_baseline_steps':rotation_low_translation,
            'session_video_member':identity.get('video_dataset'),
            'camera_metadata_present':bool(cal),'imu_bytes':imu.stat().st_size,
            'pose_used_as_training_truth':False,'surveyed_metric_validation':False,
            'interpretation':'Recorded pose is a session-local motion prior. Small-room rotation may have insufficient triangulation baseline.'}
    return report
