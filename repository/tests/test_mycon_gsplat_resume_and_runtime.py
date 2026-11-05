"""Regression guards for full-state gsplat v1.5.3 recovery and CPU->GPU."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import mycon_colmap_gsplat as c
import mycon_gsplat_resume_patch as resume
import mycon_gsplat_finish as finish


class OneRuntimeNoWastedCredits(unittest.TestCase):
    def test_registered_images_staged_without_local_source_video(self):
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td)
            exp=tmp/"drive"/"0123456789abcdef"/"trial"
            scene=exp/"scene"
            images=scene/"images"
            model=scene/"sparse/0"
            images.mkdir(parents=True)
            model.mkdir(parents=True)
            names=[f"frame_{i:06d}.jpg" for i in range(80)]
            for name in names:(images/name).write_bytes(b"safe image content"*100)
            lines=["# COLMAP images.txt\n"]
            for i,name in enumerate(names):
                lines.extend([f"{i+1} 1 0 0 0 0 0 0 1 {name}\n","\n"])
            colmap=exp/"colmap_text"
            colmap.mkdir()
            (colmap/"images.txt").write_text("".join(lines))
            for n in ("cameras.bin","images.bin","points3D.bin"):
                (model/n).write_bytes(n.encode()+b"non-empty")
            (exp/"frames_receipt.json").write_text(json.dumps({
                "count":80,
                "first_sha256":c.digest(images/names[0]),
                "last_sha256":c.digest(images/names[-1]),
                "manifest":c.frame_manifest(images),
            }))
            self.assertEqual(c.verify_prepared_frames(scene,exp),80)
            local=tmp/"gpu"
            local.mkdir()
            with patch.object(c.shutil,"disk_usage",
                              return_value=SimpleNamespace(free=40*1024**3)):
                staged=c.local_training_scene(scene,exp,local)
            self.assertEqual(len(list((staged/"images").glob("*.jpg"))),80)
            self.assertEqual(
                c.digest(staged/"sparse/0"/"points3D.bin"),
                c.digest(model/"points3D.bin"))
            self.assertFalse((local/"verified_recorder_input/raw_capture.mp4").exists())
            # Same registered images are correctly reused and never overwritten.
            with patch.object(c.shutil,"disk_usage",
                              return_value=SimpleNamespace(free=40*1024**3)):
                self.assertEqual(staged,c.local_training_scene(scene,exp,local))

    def test_video_sha_verification_rejects_corruption_after_switch(self):
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td)
            scene=tmp/"scene"
            images=scene/"images"
            images.mkdir(parents=True)
            for i in range(80):
                (images/f"frame_{i:06d}.jpg").write_bytes(b"A"*1200)
            bad=c.frame_manifest(images)
            bad["files"]["frame_000040.jpg"]["sha256"]="bad"
            (tmp/"frames_receipt.json").write_text(json.dumps({
                "count":80,"manifest":bad}))
            with self.assertRaisesRegex(RuntimeError,"SHA_MISMATCH"):
                c.verify_prepared_frames(scene,tmp)

    def test_gpu_preflight_only_reuses_persisted_video_not_vanished_mp4(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            archive=root/"original.zip"
            archive.write_bytes(b"dummy")
            sha="f"*64
            exp=root/"research_colmap_gsplat"/sha[:16]/"trial"
            exp.mkdir(parents=True)
            (exp/"verified_capture_cpu.json").write_text(json.dumps({
                "archive_sha256":sha,"full_video_duration_s":96.0}))
            (exp/"colmap_receipt.json").write_text(json.dumps({"registered":160}))
            output=exp/"bim_candidate"/"colmap_sparse_unscaled.ply"
            output.parent.mkdir()
            output.write_bytes(b"ply\n"*200)
            args=SimpleNamespace(mode="preflight",capture=archive,
                 drive_root=root,local_root=root/"fastgpu",attempt="trial",
                 approve_training=False)
            with (patch.object(c,"find_capture",return_value=archive),
                  patch.object(c,"require_source_lock"),
                  patch.object(c,"digest",return_value=sha),
                  patch.object(c,"prepare_capture",side_effect=AssertionError("ZIP EXTRACTED")),
                  patch.object(c,"prepare_frames",side_effect=AssertionError("LOCAL VIDEO USED")),
                  patch.object(c,"prepare_colmap",side_effect=AssertionError("COLMAP REPEATED")),
                  patch.object(c,"verify_prepared_frames",return_value=180),
                  patch.object(c,"local_training_scene",return_value=root/"fastgpu/scene"),
                  patch.object(c,"ensure_source"),
                  patch.object(c,"setup_gsplat",return_value=root/"env/bin/python"),
                  patch.object(c,"ensure_trainer",return_value=root/"gsplat/mycon_trainer.py"),
                  patch.object(c,"real_video_cuda_smoke",return_value={
                      "status":"REAL_20_STEP_VIDEO_T4_PASS","steps":20})):
                result=c.run(args)
            self.assertEqual(result["status"],
                             "PREFLIGHT_REAL_VIDEO_GPU_SMOKE_PASSED_NO_FULL_TRAINING")
            self.assertEqual(result["registered"],160)

    def test_patched_official_source_signatures_are_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,"PATCH_SIGNATURE_CHANGED"):
            resume.patch_source("some unrelated source")
        self.assertEqual(resume.PATCH_VERSION,"mycon-resume-v1")

    def test_recovery_export_refuses_when_no_real_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError,"MYCON_NO_COMPLETE_FINAL_MODEL_CHECKPOINT"):
                finish.make_final_model(Path("/env/python"),Path(td))

    def test_command_uses_copied_resume_trainer_and_correct_steps(self):
        cmd=c.gsplat_command(Path("/env/python"),Path("/gsplat"),
                             Path("/scene"),Path("/result"),
                             Path("/gsplat/examples/mycon_resumable_trainer.py"))
        self.assertIn("mycon_resumable_trainer.py",cmd[1])
        self.assertEqual(cmd[cmd.index("--max_steps")+1],"7000")
        self.assertIn("--save_ply",cmd)
        self.assertIn("--packed",cmd)

    def test_incomplete_old_checkpoint_does_not_fake_bim(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"ply"/"point_cloud_6999.ply"
            p.parent.mkdir(parents=True)
            p.write_bytes(b"dummy")
            outcome=c.evidence(Path(td))
            self.assertFalse(outcome["metric_BIM_validated"])
            self.assertFalse(outcome["novel_view_video_present"])
            self.assertFalse(outcome["heldout_evaluation_present"])


if __name__=="__main__":
    unittest.main()
