"""Recorder motion audit. Gyro channels are cross-checked in clock and plausible unit space."""
import csv

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


def audit_imu(root):
    with (root/"pose.csv").open(newline="") as f:
        poses={int(r["timestamp_ns"]):[float(r[k]) for k in ("qx","qy","qz","qw")]
               for r in csv.DictReader(f) if r["tracking"]=="TRACKING"}
    with (root/"imu.csv").open(newline="") as f:
        gyro={int(r["timestamp_ns"]):[float(r[f"v{i}"]) for i in range(3)]
              for r in csv.DictReader(f) if r["sensor"]=="GYRO"}
    pt=sorted(poses);gt=sorted(gyro)
    if len(pt)<10 or len(gt)<10:
        raise ValueError("Insufficient tracked gyro/pose samples")
    offset=min(pt[0],gt[0])
    tp=(np.asarray(pt,dtype=np.int64)-offset)*1e-9
    tg=(np.asarray(gt,dtype=np.int64)-offset)*1e-9
    q=np.asarray([poses[x] for x in pt]);g=np.asarray([gyro[x] for x in gt])
    if not np.isfinite(q).all() or not np.isfinite(g).all() or (abs(np.linalg.norm(q,axis=1)-1)>.02).any():
        raise ValueError("Nonfinite or nonunit pose/gyro data")
    raw=Rotation.from_quat(q)
    jumps=(raw[:-1].inv()*raw[1:]).magnitude()
    resets=int(np.sum((np.diff(tp)>.2)|((jumps>1)&(np.diff(tp)<.12))))
    t=np.arange(max(tp[0],tg[0])+.6,min(tp[-1],tg[-1])-.6,.05)
    if len(t)<80:
        raise ValueError("Too little continuous motion for clock comparison")
    r=Slerp(tp,raw)(t)
    angular=(r[:-1].inv()*r[1:]).as_rotvec()/.05
    norms=np.linalg.norm(g,axis=1)
    estimates=[]
    for lag in np.arange(-.5,.501,.05):
        a=gaussian_filter1d(np.linalg.norm(angular,axis=1),4)
        b=gaussian_filter1d(np.interp(t[1:]+lag,tg,norms),4)
        if min(a.std(),b.std())<.015:
            continue
        estimates.append((float(np.corrcoef(a,b)[0,1]),float(lag),
                          float(np.median(a)/(np.median(b)+1e-6))))
    if not estimates:
        raise ValueError("Too little rotation to establish time correspondence")
    corr,lag,ratio=max(estimates)
    return {"regular_grid_peak_correlation":corr,"best_clock_offset_s":lag,
            "gyro_vs_pose_median_rate_ratio":ratio,"possible_tracking_resets":resets,
            "motion_clock_gate":bool(corr>=.6 and .4<=ratio<=2.5 and resets==0),
            "axis_note":"Norm comparison is invariant to fixed axis signs, handedness and permutation.",
            "translation_prior_requires_visual_alignment":True}
