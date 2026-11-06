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
            self.assertFalse(actual["ply_present"])
            self.assertFalse(actual["checkpoints_present"])
            self.assertFalse(actual["novel_view_video_present"])
            self.assertFalse(actual["canonical_artifact_hashes_verified"])
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
            with (patch.object(env.subprocess,"run",
                               return_value=SimpleNamespace(returncode=0,stdout="7.5")),
                  patch.object(env.shutil,"disk_usage",
                               return_value=SimpleNamespace(free=40*(1024**3)))):
                with self.assertRaisesRegex(RuntimeError,"PARTIAL_ENV_PRESERVED"):
                    env.setup(prefix,root,root/"gsplat",root/"log")
            self.assertTrue((prefix/"protected_checkpoints.txt").exists())

    def test_cpu_geometry_does_not_install_gpu_libs_or_training(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            capture=root/"original.zip"
            capture.write_bytes(b"not opened in mocked test")
            a=SimpleNamespace(mode="geometry",attempt="trial_01",
                drive_root=root,local_root=root/"scratch",capture=capture,
                approve_training=False)
            native={"archive_sha256":"a"*64,"full_video_duration_s":96}
            with (patch.object(m,"find_capture",return_value=capture),
                  patch.object(m,"prepare_capture",return_value=native),
                  patch.object(m,"prepare_frames",return_value=180),
                  patch.object(m,"prepare_colmap",return_value={"registered":170}),
                  patch.object(m,"ensure_source",side_effect=AssertionError("GPU SOURCE FETCHED")),
                  patch.object(m,"setup_gsplat",side_effect=AssertionError("GPU ENV INSTALLED"))):
                status=m.run(a)
            self.assertEqual(status["status"],"GEOMETRY_PREPARED_COLMAP_REGISTERED_NO_GPU")
            self.assertFalse(status["training_executed"])

    def test_gpu_all_without_preflight_cannot_start_costly_steps(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            capture=root/"original.zip"
            capture.write_bytes(b"not opened in mocked test")
            a=SimpleNamespace(mode="all",attempt="trial_01",
                drive_root=root,local_root=root/"scratch",capture=capture,
                approve_training=True)
            native={"archive_sha256":"a"*64,"full_video_duration_s":96}
            folder=root/"research_colmap_gsplat"/("a"*16)/"trial_01"
            folder.mkdir(parents=True)
            (folder/"verified_capture_cpu.json").write_text(json.dumps(native))
            (folder/"colmap_receipt.json").write_text(json.dumps({"registered":170}))
            candidate=folder/"bim_candidate"/"colmap_sparse_unscaled.ply"
            candidate.parent.mkdir()
            candidate.write_bytes(b"not-real-cloud")
            with (patch.object(m,"find_capture",return_value=capture),
                  patch.object(m,"digest",return_value="a"*64),
                  patch.object(m,"prepare_capture",side_effect=AssertionError("ZIP EXTRACTED")),
                  patch.object(m,"prepare_frames",side_effect=AssertionError("FRAMES PROCESSED")),
                  patch.object(m,"ensure_source",side_effect=AssertionError("SOURCE FETCHED")),
                  patch.object(m,"setup_gsplat",side_effect=AssertionError("CUDA INSTALLED"))):
                with self.assertRaisesRegex(RuntimeError,"(MYCON_CPU_SOURCE_LOCK_MISSING|PREFLIGHT_MUST_PASS)"):
                    m.run(a)

    def test_preflight_without_verified_cpu_receipt_fails_without_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            capture=root/"original.zip"
            capture.write_bytes(b"dummy")
            a=SimpleNamespace(mode="preflight",attempt="trial_01",
                drive_root=root,local_root=root/"scratch",capture=capture,
                approve_training=False)
            with (patch.object(m,"find_capture",return_value=capture),
                  patch.object(m,"digest",return_value="a"*64),
                  patch.object(m,"setup_gsplat",side_effect=AssertionError("CUDA INSTALLED"))):
                with self.assertRaisesRegex(RuntimeError,"CPU_SOURCE_AND_GEOMETRY_RECEIPT_MISSING"):
                    m.run(a)

    def test_cpu_verify_is_separate_from_all_native_packages(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            capture=root/"original.zip"
            capture.write_bytes(b"fake")
            a=SimpleNamespace(mode="verify",attempt="trial_01",
                drive_root=root,local_root=root/"scratch",capture=capture,
                approve_training=False)
            with (patch.object(m,"find_capture",return_value=capture),
                  patch.object(m,"verify_original_archive",
                               return_value={"status":"CPU_SOURCE_AND_RECORDER_SCHEMA_PASS_NOT_GPU_OR_TRAINING"}),
                  patch.object(m,"prepare_capture",side_effect=AssertionError("ZIP EXTRACTED")),
                  patch.object(m,"setup_gsplat",side_effect=AssertionError("GPU INSTALLED"))):
                result=m.run(a)
            self.assertEqual(result["status"],"CPU_SOURCE_AND_RECORDER_SCHEMA_PASS_NOT_GPU_OR_TRAINING")

    def test_not_yet_verified_runtime_refuses_training_without_pip(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch.object(env,"run_logged",side_effect=AssertionError("PIP INVOKED")):
                with self.assertRaisesRegex(RuntimeError,"NOT_ALREADY_PREFLIGHT_VERIFIED"):
                    env.verify_existing(root/"missing_env",root/"gsplat",root/"log")

    def test_unsafe_attempt_cannot_change_write_root(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(mode="status",drive_root=Path(d),
                                 attempt="../../previous")
            with self.assertRaises(ValueError):
                m.run(args)

if __name__=="__main__":unittest.main()
