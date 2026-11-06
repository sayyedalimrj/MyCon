import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from mycon_recorder_motion_audit import audit


def test_small_room_rotation_dominant_is_detected_without_claiming_survey(tmp_path):
    import math
    root=tmp_path
    (root/'session.json').write_text(json.dumps({'video_dataset':'original.mp4'}))
    (root/'r4_camera.json').write_text('{"K": [1,2,3]}')
    (root/'imu.csv').write_text('timestamp_ns,ax\n1,0')
    lines=['timestamp_ns,tx_m,ty_m,tz_m,qx,qy,qz,qw,tracking']
    for i in range(60):
        theta=i*math.radians(8)
        lines.append(f'{1_000_000_000+i*33333333},{.002*i},0,0,0,0,{math.sin(theta/2)},{math.cos(theta/2)},TRACKING')
    (root/'pose.csv').write_text('\n'.join(lines))
    report=audit(root)
    assert report['tracked_rows']==60
    assert report['rotation_dominant_small_baseline_steps']>=50
    assert report['surveyed_metric_validation'] is False
    assert report['pose_used_as_training_truth'] is False


def test_invalid_pose_timestamp_fails_before_gpu(tmp_path):
    (tmp_path/'session.json').write_text('{}')
    (tmp_path/'r4_camera.json').write_text('{}')
    (tmp_path/'imu.csv').write_text('a')
    (tmp_path/'pose.csv').write_text('timestamp_ns,tx_m,ty_m,tz_m\n'+'\n'.join('0,0,0,0' for _ in range(30)))
    with pytest.raises(RuntimeError,match='INSUFFICIENT'):
        audit(tmp_path)
