"""Offline tests for multi-GB MASt3R weight caching; never download model in tests."""
from pathlib import Path
from unittest.mock import patch
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import mycon_longsplat_weights as weights


def roots(tmp_path):
    source=tmp_path/'NVIDIA_LongSplat';(source/'utils').mkdir(parents=True)
    (source/'utils'/'mast3r_utils.py').write_text('# upstream fixture')
    return source,tmp_path/'DriveModelCache'


def test_official_weights_are_cached_and_verified_without_retraining(tmp_path):
    source,cache=roots(tmp_path)
    with patch.object(weights,'MIN_BYTES',100),patch.object(weights,'resumable_download',side_effect=lambda url,path:path.write_bytes(b'Z'*200)) as download:
        first=weights.prepare_weights(source,cache)
        assert first['status']=='WEIGHT_SHA256_VERIFIED' and first['bytes']==200
        target=source/'submodules'/'mast3r'/'checkpoints'/weights.MODEL
        target.unlink()
        second=weights.prepare_weights(source,cache)
        assert second['status']=='WEIGHT_SHA256_VERIFIED'
        assert target.read_bytes()==b'Z'*200
        assert download.call_count==1


def test_tampered_drive_cache_must_be_rejected(tmp_path):
    source,cache=roots(tmp_path)
    with patch.object(weights,'MIN_BYTES',100),patch.object(weights,'resumable_download',side_effect=lambda url,path:path.write_bytes(b'A'*300)):
        weights.prepare_weights(source,cache)
        target=source/'submodules'/'mast3r'/'checkpoints'/weights.MODEL
        target.unlink()
        (cache/weights.MODEL).write_bytes(b'B'*300)
        with pytest.raises(RuntimeError,match='SHA256 mismatch'):
            weights.prepare_weights(source,cache)


def test_unreceipted_local_weights_are_never_silently_trusted(tmp_path):
    source,cache=roots(tmp_path)
    target=source/'submodules'/'mast3r'/'checkpoints'/weights.MODEL
    target.parent.mkdir(parents=True);target.write_bytes(b'X'*200)
    with patch.object(weights,'MIN_BYTES',100):
        with pytest.raises(RuntimeError,match='no verified cache receipt'):
            weights.prepare_weights(source,cache)
