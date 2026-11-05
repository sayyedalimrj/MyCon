"""Offline guardrail tests for the single supported LongSplat entrypoint."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import mycon_thesis_one_run as m
import mycon_longsplat_auto_run as runner


class TestOneRun(unittest.TestCase):
    def test_attempt_id_cannot_traverse(self):
        for s in ('../old','a/b','x'*49,'a b',''):
            with self.assertRaises(ValueError):
                m.validate_attempt(s)
        self.assertEqual(m.validate_attempt('stabilization_20261105'),
                         'stabilization_20261105')

    def test_preflight_does_not_approve_training(self):
        kwargs=dict(archive=Path('/input.zip'),drive=Path('/drive'),local=Path('/tmp/a'),
                    nvidia=Path('/tmp/LongSplat'),envroot=Path('/tmp/env'),
                    attempt='safe1',quality='quality')
        cmd=m.build_command(Path('/controller.py'),mode='preflight',**kwargs)
        self.assertIn('prepare',cmd)
        self.assertNotIn('--approve-new-training',cmd)
        self.assertIn('--attempt-id',cmd)
        cmd2=m.build_command(Path('/controller.py'),mode='train',**kwargs)
        self.assertIn('--approve-new-training',cmd2)
        self.assertIn('--stop-after',cmd2)
        cmd3=m.build_command(Path('/controller.py'),mode='all',**kwargs)
        self.assertIn('--approve-new-training',cmd3)
        self.assertNotIn('--stop-after',cmd3)

    def test_capture_ambiguity_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'captures').mkdir()
            for name in ('a.zip','b.zip'):
                with (root/'captures'/name).open('wb') as f:
                    f.truncate(100_000_001)
            with self.assertRaisesRegex(RuntimeError,'CAPTURE_PATH_REQUIRED'):
                m.find_capture(root,None)

    def test_missing_original_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                m.find_capture(Path(td),Path(td)/'original.zip')

    def test_live_verified_env_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            known=root/'old'
            (known/'bin').mkdir(parents=True)
            (known/'bin/python').write_text('python')
            previous=m.KNOWN_ENV
            m.KNOWN_ENV=known
            try:
                with patch.object(m,'verify_cuda') as check:
                    env,reason=m.choose_env(
                        root,{'gpu':{'sm':75},'torch_cuda_wheel':'cu121','torch_cuda_arch':'7.5'},root/'logs')
                self.assertEqual(env,known)
                self.assertEqual(reason,'REUSED_VERIFIED_CUDA_KERNELS')
                check.assert_called_once()
            finally:
                m.KNOWN_ENV=previous

    def test_broken_environment_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            local=root/'work'
            bad=local/'mycon_stabilized_sm75_cu121'
            (bad/'bin').mkdir(parents=True)
            (bad/'bin/python').write_text('python')
            with patch.object(m,'verify_cuda',side_effect=RuntimeError('cuda broken')):
                with self.assertRaisesRegex(RuntimeError,'NO_VERIFIED_ENV_AND_NEW_PREFIX_ALREADY_EXISTS'):
                    m.choose_env(local,{'gpu':{'sm':75},'torch_cuda_wheel':'cu121','torch_cuda_arch':'7.5'},
                                 root/'logs')
            self.assertTrue((bad/'bin/python').exists())


    def test_attempt_isolated_without_changing_legacy_default(self):
        root=Path('/experiments/quality_8fps')
        self.assertEqual(runner.isolate_attempt(root,None),root)
        self.assertEqual(runner.isolate_attempt(root,'trial_1'),
                         root/'attempts/trial_1')

    def test_attempt_cannot_escape_checkpoint_root(self):
        root=Path('/experiments/quality_8fps')
        for attack in ('../checkpoints','a/b','..','a b'):
            with self.assertRaises(ValueError):
                runner.isolate_attempt(root,attack)

    def test_read_only_status_has_no_fabricated_success(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(m.status(Path(td),'attempt1'),2)


if __name__=='__main__':
    unittest.main()
