#!/usr/bin/env python3
"""Runtime-only shim for V4 native Recorder motion-audit timestamp disorder.

Load the original NVIDIA/MyCON runner *unchanged*. When only the optional
motion QA reports non-monotonic pose timestamps, compute QA from a sorted,
unique, temporary pose CSV. Neither source sidecars nor package SHA manifest
nor GPU environment is changed. Pose is never injected as metric truth.
"""
from __future__ import annotations
import csv
import json
import math
import runpy
import sys
import tempfile
from collections import Counter
from pathlib import Path


def _tracking_good(r):
    try:
        if str(r.get('tracking', 'TRACKING')).upper() not in ('TRACKING', ''):
            return False
        x = [float(r[k]) for k in ('tx_m', 'ty_m', 'tz_m')]
        return all(math.isfinite(z) for z in x)
    except (ValueError, KeyError, TypeError):
        return False


def _recovered_pose_report(original_audit, path: Path, reason: RuntimeError):
    source = Path(path)
    pose = source / 'pose.csv'
    with pose.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or 'timestamp_ns' not in fields:
            raise ValueError('Cannot recover native pose QA without timestamp_ns') from reason
        rows = list(reader)
    # Preserve every original sidecar on disk. This temporary copy is only for
    # optional QA; the actual LongSplat path remains unposed (see runner).
    stamped, malformed = [], []
    for sequence, row in enumerate(rows):
        try:
            stamp = int(row['timestamp_ns'])
            if stamp <= 0:
                malformed.append(row)
                continue
            stamped.append((stamp, sequence, row))
        except (ValueError, TypeError, KeyError):
            malformed.append(row)
    if not stamped:
        raise RuntimeError('No usable native recorder timestamps') from reason
    ordered_original = [t for t, _, r in stamped if _tracking_good(r)]
    original_inversions = sum(a >= b for a, b in zip(ordered_original, ordered_original[1:]))
    # Equal timestamps often come from duplicated asynchronous ARCore callbacks.
    # Select a valid tracking row first, then the earliest original row.
    best = {}
    for stamp, index, row in stamped:
        if stamp not in best or (_tracking_good(row) and not _tracking_good(best[stamp][1])):
            best[stamp] = (index, row)
    ordered = [best[t][1] for t in sorted(best)]
    # Malformed rows are retained for the audit's total-row denominator; its
    # existing validator skips them for motion measurements.
    ordered.extend(malformed)
    with tempfile.TemporaryDirectory(prefix='mycon_motion_qa_only_') as temporary:
        folder = Path(temporary)
        for name in ('session.json', 'r4_camera.json', 'imu.csv'):
            item = source / name
            if not item.is_file():
                raise FileNotFoundError(str(item)) from reason
            (folder / name).symlink_to(item)
        with (folder / 'pose.csv').open('w', newline='', encoding='utf-8') as dest:
            writer = csv.DictWriter(dest, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(ordered)
        report = original_audit(folder)
    report.update({
        'status': 'VALID_NATIVE_POSE_REORDERED_FOR_MOTION_QA_ONLY',
        'timestamp_disorder_recovered_for_qa': True,
        'native_pose_csv_unchanged': True,
        'original_pose_rows': len(rows),
        'original_tracking_time_inversions': original_inversions,
        'duplicate_positive_timestamps_dropped_for_qa': len(stamped) - len(best),
        'malformed_timestamp_rows': len(malformed),
        'pose_used_as_training_truth': False,
        'surveyed_metric_validation': False,
        'warning': 'Native ARCore callbacks were out of order or duplicated; '
                   'QA statistics derive from a temporary time-sorted copy. '
                   'Recorder source remains unchanged; not surveyed metric truth.',
    })
    print('POSE_TIMESTAMP_QA_RECOVERED:',
          f"reversed={original_inversions}, duplicates={len(stamped)-len(best)},"
          f" valid_tracking={report['tracked_rows']}", flush=True)
    return report


def audit_with_recovery(path):
    from mycon_recorder_motion_audit import audit
    try:
        return audit(path)
    except RuntimeError as exc:
        if 'RECORDER_POSE_TIME_NONMONOTONIC' not in str(exc):
            raise
        return _recovered_pose_report(audit, Path(path), exc)


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: bootstrap.py ORIGINAL_RUNNER.py [runner arguments...]')
    runner = Path(sys.argv[1]).resolve()
    if not runner.is_file() or runner.name != 'mycon_longsplat_auto_run.py':
        raise RuntimeError('Refusing unknown runner: ' + str(runner))
    argv = sys.argv[2:]
    sys.path.insert(0, str(runner.parent))
    import mycon_recorder_motion_audit as original_module
    old = original_module.audit

    def patched_audit(path):
        try:
            return old(path)
        except RuntimeError as exc:
            if 'RECORDER_POSE_TIME_NONMONOTONIC' not in str(exc):
                raise
            return _recovered_pose_report(old, Path(path), exc)

    original_module.audit = patched_audit
    sys.argv = [str(runner), *argv]
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()
