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
            self.assertEqual(len(manifest["files"]),5)
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
                        raise AssertionError("PLY WAS UNNECESSARILY OVERWRITTEN")
            with patch.object(c,"run_logged",side_effect=convert):
                result=c.prepare_colmap(scene,exp,180,root,root/"colmap.log")
            self.assertEqual(result["registered"],150)
            self.assertEqual(result["selected_submodel"],"0")
            self.assertEqual(candidate.read_bytes(),payload)
            self.assertTrue((exp/"colmap_receipt.json").is_file())

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

    def test_nan_inf_not_accepted_as_valid_reconstruction(self):
        self.assertFalse(audit.metrics_valid({"psnr":float("nan"),"ssim":0.8,"lpips":0.1}))
        self.assertFalse(audit.metrics_valid({"psnr":30,"ssim":float("inf"),"lpips":0.1}))
        self.assertTrue(audit.metrics_valid({"psnr":25.0,"ssim":0.81,"lpips":0.14}))

    def test_patch_keeps_colmap_geometry_frame(self):
        source="""import time
normalize_world_space: bool = True
        trainloader = torch.utils.data.DataLoader(
            # eval the full set
"""
        patched=resume.patch_source(source)
        self.assertIn("normalize_world_space: bool = False",patched)
        self.assertIn("MYCON_RECOVERY_VERIFIED",patched)
        self.assertIn("MYCON_GAUSSIAN_GROWTH_GUARD_EXCEEDED",patched)
        self.assertNotIn("checkpoint = torch.load(\n                mycon_saved[-1]",patched)


if __name__=="__main__":
    unittest.main()
