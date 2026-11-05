"""Offline safety and provenance tests for the selected SpatialScene3D-style baseline."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/"scripts"))
import mycon_colmap_gsplat as m
import mycon_colmap_gsplat_env as env


class BaselineGuards(unittest.TestCase):
    def test_registered_image_count_handles_blank_observation_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"images.txt"
            p.write_text("# camera model\n"
                         "1 1 0 0 0 0 0 0 1 frame_000001.jpg\n"
                         "\n"
                         "2 1 0 0 0 0 0 0 1 frame_000002.jpg\n"
                         "10.0 11.0 -1\n")
            self.assertEqual(m.count_registered_images(p),2)

    def test_colmap_registration_gate_rejects_bad_pose_coverage(self):
        for registered,frames in ((0,190),(30,190),(70,190)):
            with self.assertRaisesRegex(RuntimeError,"COLMAP_REGISTRATION_INSUFFICIENT"):
                m.colmap_gate(registered,frames)
        self.assertEqual(m.colmap_gate(150,190)["status"],"COLMAP_REGISTERED")
        self.assertFalse(m.colmap_gate(150,190)["bim_progress_authorized"])

    def test_frames_below_minimum_refused(self):
        with self.assertRaisesRegex(RuntimeError,"TOO_FEW"):
            m.colmap_gate(50,50)

    def test_colmap_commands_use_cpu_SIFT_and_own_cache(self):
        commands=m.commands_for_scene(Path("/scene"),Path("/db"),Path("/sparse"))
        self.assertEqual([c[1] for c in commands],
                         ["feature_extractor","sequential_matcher","mapper"])
        self.assertIn("SIMPLE_RADIAL",commands[0])
        self.assertIn("--SiftExtraction.use_gpu",commands[0])
        self.assertIn("--SiftMatching.use_gpu",commands[1])

    def test_train_command_runs_official_gsplat_and_saves_real_model(self):
        cmd=m.gsplat_command(Path("/env/bin/python"),Path("/gsplat"),
                             Path("/scene"),Path("/result"))
        self.assertIn("/gsplat/examples/simple_trainer.py",cmd)
        self.assertIn("--save_ply",cmd)
        self.assertIn("--disable_viewer",cmd)
        self.assertIn("--packed",cmd)
        self.assertNotIn("--disable_video",cmd)
        self.assertEqual(cmd[cmd.index("--max_steps")+1],str(m.STEPS))

    def test_output_receipt_is_write_once(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"input.json"
            m.save_once(p,{"source_sha256":"abc"})
            m.save_once(p,{"source_sha256":"abc"})
            with self.assertRaisesRegex(RuntimeError,"IMMUTABLE_RECEIPT_CONFLICT"):
                m.save_once(p,{"source_sha256":"different"})
            self.assertEqual(json.loads(p.read_text())["source_sha256"],"abc")

    def test_live_output_detection_never_certifies_metric_bim(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertFalse(m.evidence(root)["ply_present"])
            for filename in ("ckpts/ckpt_6999_rank0.pt",
                             "ply/point_cloud_6999.ply","videos/traj_6999.mp4"):
                p=root/filename
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b"dummy artifact")
            actual=m.evidence(root)
            self.assertTrue(actual["ply_present"])
            self.assertTrue(actual["checkpoints_present"])
            self.assertTrue(actual["novel_view_video_present"])
            self.assertFalse(actual["metric_BIM_validated"])

    def test_status_mode_is_read_only_and_never_claims_run(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(mode="status",drive_root=Path(d),
                                 attempt="baseline_20261105")
            self.assertEqual(m.run(args)["status"],"NO_ATTEMPT_YET")

    def test_environment_never_overwrites_partial_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            prefix=root/"env"
            prefix.mkdir()
            (prefix/"protected_checkpoints.txt").write_text("not to delete")
            with self.assertRaisesRegex(RuntimeError,"PARTIAL_ENV_PRESERVED"):
                env.setup(prefix,root,root/"gsplat",root/"log")
            self.assertTrue((prefix/"protected_checkpoints.txt").exists())

    def test_unsafe_attempt_cannot_change_write_root(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(mode="status",drive_root=Path(d),
                                 attempt="../../previous")
            with self.assertRaises(ValueError):
                m.run(args)

if __name__=="__main__":unittest.main()
