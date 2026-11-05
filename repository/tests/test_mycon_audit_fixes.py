"""Audit regressions that prove CPU replay, durability, model identity and BIM safety."""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import mycon_colmap_gsplat as c
import mycon_audit_runtime as audit
import mycon_gsplat_finish as finish
import mycon_gsplat_resume_patch as resume


class AuditFixes(unittest.TestCase):
    def test_atomic_status_and_source_lock(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            audit.atomic_json(root/"status.json",{"ok":True})
            self.assertTrue(json.loads((root/"status.json").read_text())["ok"])
            self.assertFalse(list(root.glob("*.partial.*")))
            manifest=audit.require_source_lock(root,ROOT/"scripts",create=True)
            self.assertEqual(manifest,audit.require_source_lock(root,ROOT/"scripts",create=False))
            self.assertEqual(len(manifest["files"]),8)
            with patch.object(audit,"source_identity",
                              return_value={"schema":"changed","files":{}}):
                with self.assertRaisesRegex(RuntimeError,"SOURCE_CHANGED"):
                    audit.require_source_lock(root,ROOT/"scripts",create=False)

    def test_all_frame_hashes_catch_middle_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            for i in range(80):
                (folder/f"frame_{i:06d}.jpg").write_bytes((str(i)*1500).encode())
            manifest=audit.frame_manifest(folder)
            self.assertEqual(audit.validate_frame_manifest(folder,manifest),80)
            (folder/"frame_000037.jpg").write_bytes(b"B"*2000)
            with self.assertRaisesRegex(RuntimeError,"FRAME_SHA_MISMATCH"):
                audit.validate_frame_manifest(folder,manifest)

    def test_broken_ffmpeg_run_recovers_automatically_without_deleting_old_images(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            scene=root/"scene";images=scene/"images";images.mkdir(parents=True)
            (images/"frame_000001.jpg").write_bytes(b"partial")
            video=root/"video.mp4";video.write_bytes(b"original video")
            exp=root/"experiment";exp.mkdir()
            def produce_frames(cmd,log,*args,**kwargs):
                for i in range(1,101):
                    (images/f"frame_{i:06d}.jpg").write_bytes((str(i)*1500).encode())
            with patch.object(c,"run_logged",side_effect=produce_frames):
                count=c.prepare_frames(video,scene,exp,root/"ffmpeg.log")
            self.assertEqual(count,100)
            self.assertEqual(c.verify_prepared_frames(scene,exp),100)
            old=list(scene.glob("images.preserved_interrupted_ffmpeg_*"))
            self.assertEqual(len(old),1)
            self.assertEqual((old[0]/"frame_000001.jpg").read_bytes(),b"partial")

    def test_existing_colmap_ply_can_recover_receipt_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/"scene";exp=root/"experiment"
            model=scene/"sparse"/"0";model.mkdir(parents=True)
            for name in ("cameras.bin","images.bin","points3D.bin"):
                (model/name).write_bytes(b"binary"+name.encode())
            candidate=exp/"bim_candidate"/"colmap_sparse_unscaled.ply"
            candidate.parent.mkdir(parents=True)
            payload=b"ply"+b"x"*2048
            candidate.write_bytes(payload)
            def convert(cmd,log,*args,**kwargs):
                if "model_converter" in cmd:
                    if cmd[cmd.index("--output_type")+1]=="TXT":
                        target=Path(cmd[cmd.index("--output_path")+1])
                        target.mkdir(parents=True,exist_ok=True)
                        (target/"images.txt").write_text(
                            "".join(f"{i} 1 0 0 0 0 0 0 1 frame_{i:06d}.jpg\n\n"
                                    for i in range(1,151)))
                        (target/"cameras.txt").write_text("test\n")
                        (target/"points3D.txt").write_text("test\n")
                    else:
                        # A CPU receipt is not trusted until a fresh conversion
                        # confirms the cloud belongs to the selected model.
                        Path(cmd[cmd.index("--output_path")+1]).write_bytes(payload)
            with (patch.object(c,"run_logged",side_effect=convert),
                  patch.object(c.shutil,"which",return_value="/usr/bin/colmap")):
                result=c.prepare_colmap(scene,exp,180,root,root/"colmap.log")
            self.assertEqual(result["registered"],150)
            self.assertEqual(result["selected_submodel"],"0")
            self.assertEqual(candidate.read_bytes(),payload)
            self.assertTrue((exp/"colmap_receipt.json").is_file())

    def test_wrong_existing_ply_is_quarantined_before_new_cpu_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/"scene";exp=root/"exp"
            model=scene/"sparse"/"1"
            model.mkdir(parents=True)
            for name in ("cameras.bin","images.bin","points3D.bin"):
                (model/name).write_bytes(b"selected COLMAP model"+name.encode())
            candidate=exp/"bim_candidate"/"colmap_sparse_unscaled.ply"
            candidate.parent.mkdir(parents=True)
            old=b"ply"+b"OLD_OTHER_SUBMODEL"*80
            corrected=b"ply"+b"TRUE_SELECTED_SUBMODEL"*80
            candidate.write_bytes(old)
            def convert(cmd,log,*args,**kwargs):
                if cmd[1]!="model_converter":
                    raise AssertionError("MAPPER SHOULD BE REUSED")
                dest=Path(cmd[cmd.index("--output_path")+1])
                typ=cmd[cmd.index("--output_type")+1]
                if typ=="TXT":
                    dest.mkdir(parents=True,exist_ok=True)
                    (dest/"images.txt").write_text(
                        "".join(f"{i} 1 0 0 0 0 0 0 1 frame_{i:06d}.jpg\n\n"
                                for i in range(1,151)))
                    (dest/"cameras.txt").write_text("CAMS")
                    (dest/"points3D.txt").write_text("POINTS")
                elif typ=="PLY":
                    dest.write_bytes(corrected)
                else:
                    raise AssertionError("WRONG_COLMAP_CONVERSION")
            with (patch.object(c,"run_logged",side_effect=convert),
                  patch.object(c.shutil,"which",return_value="/usr/bin/colmap")):
                result=c.prepare_colmap(scene,exp,180,root,root/"log.txt")
            self.assertEqual(result["selected_submodel"],"1")
            self.assertEqual(candidate.read_bytes(),corrected)
            backups=list(candidate.parent.glob(
                "colmap_sparse_unscaled.ply.preserved_different_or_incomplete_colmap_ply_*"))
            self.assertEqual(len(backups),1)
            self.assertEqual(backups[0].read_bytes(),old)
            self.assertEqual(
                json.loads((exp/"colmap_receipt.json").read_text())[
                    "sparse_candidate_ply_sha256"],c.digest(candidate))

    def test_verified_20_step_gpu_smoke_reused_across_colab_disconnect(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);exp=root/"exp";exp.mkdir()
            (exp/"colmap_receipt.json").write_text(json.dumps({"registered":154}))
            (exp/"frames_receipt.json").write_text(json.dumps({"count":190}))
            identity={"source_sha256":c.digest(exp/"colmap_receipt.json"),
                      "frame_manifest_sha256":c.digest(exp/"frames_receipt.json"),
                      "gsplat_tag":"v1.5.3","torch_cuda":"2.4.1+cu121",
                      "sm":"7.5","smoke_steps":20}
            audit.atomic_json(exp/"real_video_gpu_smoke_receipt.json",{
                **identity,"status":"REAL_20_STEP_VIDEO_T4_PASS",
                "steps":20,"max_gpu_allocated_gib":2.1})
            with patch.object(c,"run_logged",side_effect=AssertionError("GPU_SMOKE_REPEATED")):
                reused=c.real_video_cuda_smoke(
                    root/"env/bin/python",root/"source",root/"scene",exp,root/"trainer")
            self.assertTrue(reused["reused_after_runtime_reset"])
            self.assertEqual(reused["steps"],20)

    def test_canonical_checkpoint_and_ply_share_one_source_hash(self):
        with tempfile.TemporaryDirectory() as td:
            result=Path(td)
            recovery=result/"recovery"/"resume_step_006999.pt"
            recovery.parent.mkdir()
            recovery.write_bytes(b"original checkpoint contents not modified")
            audit.atomic_json(Path(str(recovery)+".sha256.json"),{
                "size":recovery.stat().st_size,"sha256":audit.sha256(recovery)})
            def fake_export(cmd,log,**kwargs):
                canonical=Path(cmd[cmd.index("6999")-2]) if False else (
                    result/"canonical"/"ckpts"/"ckpt_6999_rank0.pt")
                ply=result/"canonical"/"ply"/"point_cloud_6999.ply"
                canonical.parent.mkdir(parents=True,exist_ok=True)
                ply.parent.mkdir(parents=True,exist_ok=True)
                canonical.write_bytes(b"same-source-checkpoint"*300)
                ply.write_bytes(b"same-source-ply"*300)
            with patch.object(finish,"run_logged",side_effect=fake_export):
                path=finish.make_final_model(Path("/env/bin/python"),result)
                self.assertTrue(path.is_file())
                self.assertEqual(path,finish.make_final_model(Path("/env/bin/python"),result))
            model=json.loads((result/"canonical"/"canonical_model.json").read_text())
            self.assertEqual(model["source_sha256"],audit.sha256(recovery))
            self.assertEqual(model["source_kind"],"post_optimizer_recovery")
            self.assertEqual(model["colmap_to_gaussian_4x4"][0],[1,0,0,0])

    def test_interrupted_immutable_receipt_partial_does_not_block_retry(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)
            stuck=directory/"geometry.json.partial"
            stuck.write_text("unfinished write from disconnected CPU")
            target=directory/"geometry.json"
            c.save_once(target,{"source_sha256":"abc"})
            self.assertEqual(json.loads(target.read_text()),{"source_sha256":"abc"})
            self.assertEqual(stuck.read_text(),"unfinished write from disconnected CPU")
            with self.assertRaisesRegex(RuntimeError,"IMMUTABLE_RECEIPT_CONFLICT"):
                c.save_once(target,{"source_sha256":"different"})

    def test_interrupted_canonical_ply_is_preserved_and_regenerated(self):
        with tempfile.TemporaryDirectory() as td:
            result=Path(td)
            recovery=result/"recovery"/"resume_step_006999.pt"
            recovery.parent.mkdir()
            recovery.write_bytes(b"valid snapshot supplied by real trainer"*200)
            audit.atomic_json(Path(str(recovery)+".sha256.json"),{
                "size":recovery.stat().st_size,
                "sha256":audit.sha256(recovery)})
            old=result/"canonical"/"ckpts"/"ckpt_6999_rank0.pt"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"interrupted model write")
            calls=[]
            def export(cmd, log, **kwargs):
                calls.append(cmd)
                target=result/"canonical"/"ckpts"/"ckpt_6999_rank0.pt"
                ply=result/"canonical"/"ply"/"point_cloud_6999.ply"
                target.parent.mkdir(parents=True,exist_ok=True)
                ply.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(b"valid checkpoint"*400)
                ply.write_bytes(b"ply result"*300)
            with patch.object(finish,"run_logged",side_effect=export):
                outcome=finish.make_final_model(Path("/env/bin/python"),result)
            self.assertTrue(outcome.is_file())
            self.assertEqual(len(calls),1)
            preserved=list(result.glob("canonical.preserved_unreceipted_canonical_model_*"))
            self.assertEqual(len(preserved),1)
            self.assertEqual(
                (preserved[0]/"ckpts"/"ckpt_6999_rank0.pt").read_bytes(),
                b"interrupted model write")
            self.assertTrue((result/"canonical"/"canonical_model.json").is_file())

    def test_corrupt_resume_checkpoint_has_preservation_and_fallback_guard(self):
        import inspect
        patch_source=inspect.getsource(resume.patch_source)
        self.assertIn("invalid_candidates.append((candidate,stamp",patch_source)
        self.assertIn("MYCON_INVALID_RECOVERY_PRESERVED",patch_source)
        self.assertIn("NO_VALID_HASHED_RECOVERY_SNAPSHOT",patch_source)

    def test_nan_inf_not_accepted_as_valid_reconstruction(self):
        self.assertFalse(audit.metrics_valid({"psnr":float("nan"),"ssim":0.8,"lpips":0.1}))
        self.assertFalse(audit.metrics_valid({"psnr":30,"ssim":float("inf"),"lpips":0.1}))
        self.assertTrue(audit.metrics_valid({"psnr":25.0,"ssim":0.81,"lpips":0.14}))

    def test_patch_has_resumption_and_explicit_same_coordinate_frame(self):
        import inspect
        contents=inspect.getsource(resume.patch_source)
        self.assertIn('"normalize_world_space: bool = False"',contents)
        self.assertIn("MYCON_RECOVERY_VERIFIED",contents)
        self.assertIn("MYCON_GAUSSIAN_GROWTH_GUARD_EXCEEDED",contents)


if __name__=="__main__":
    unittest.main()
