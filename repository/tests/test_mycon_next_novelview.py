"""MyCON next LongSplat experiment: CPU regression and preservation only."""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import mycon_next_novelview as m


def put_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj),encoding="utf-8")


class NovelViewSafety(unittest.TestCase):
    def test_run_name_rejects_traversal(self):
        with self.assertRaises(ValueError):
            m.work_paths(Path("/tmp/abc"),"../baseline")

    def test_model_training_is_opt_in_before_any_read_or_write(self):
        with tempfile.TemporaryDirectory() as td:
            paths=m.work_paths(Path(td),"longsplat_test_20261109")
            with self.assertRaisesRegex(RuntimeError,"NOT_AUTHORIZED"):
                m.do_train(paths,"",2,20000)
            self.assertFalse(paths["new"].exists())

    def test_manifest_is_idempotent_and_conflicts_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"receipt.json"
            m.save_once(path,{"status":"verified","value":1})
            m.save_once(path,{"value":1,"status":"verified"})
            with self.assertRaisesRegex(RuntimeError,"CONFLICT"):
                m.save_once(path,{"value":2})
            self.assertEqual(json.loads(path.read_text())["value"],1)

    def test_real_data_identity_contract_on_synthetic_fixtures(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            paths=m.work_paths(root,"longsplat_test_20261109")
            canonical=paths["baseline"]/"gsplat_result"/"canonical"
            put_json(canonical/"canonical_evaluation.json",{
                "schema":"mycon.canonical_evaluation.v1",
                "ply_sha256":"afd378d8849b31c1455c1f73e4faaac9cb5188b5d4502355bd2b2fdbcbaa64ec",
                "canonical_checkpoint_sha256":"a"*64,
                "metrics":{"psnr":20.7359,"ssim":.6176,"lpips":.522,"num_GS":1477725}})
            put_json(canonical/"canonical_model.json",{"canonical_sha256":"a"*64})
            image_dir=paths["dataset"]/"images"
            image_dir.mkdir(parents=True)
            files={}
            for i in range(1,701):
                name=f"frame_{i:06d}.jpg"
                blob=bytes([i%256])*1300
                (image_dir/name).write_bytes(blob)
                files[name]={"bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest()}
            put_json(paths["dataset"]/"dataset_receipt.json",{
                "fps":8,"width":512,"training_executed":False,
                "source_recorder_archive_sha256":
                  "811c5a1323e45e2a4ef635eeda9f4335be06ef01ccb74ed3b431328400f862b6",
                "manifest":{"files":files}})
            before=(canonical/"canonical_evaluation.json").read_bytes()
            result=m.prepare(paths)
            self.assertEqual(result["frames"],700)
            self.assertFalse(result["site_metric_validated"])
            self.assertEqual(before,(canonical/"canonical_evaluation.json").read_bytes())
            m.prepare(paths)
            self.assertEqual(len(list(paths["new"].glob("*.json"))),3)
            (image_dir/"frame_000001.jpg").write_bytes(b"mismatched")
            with self.assertRaisesRegex(RuntimeError,"HASH_MISMATCH"):
                m.verify(paths)

    def test_training_output_cannot_be_reused_implicitly(self):
        with tempfile.TemporaryDirectory() as td:
            paths=m.work_paths(Path(td),"longsplat_test_20261109")
            paths["new"].mkdir(parents=True)
            put_json(paths["new"]/"training_started.json",{"status":"old_ongoing"})
            with patch.object(m,"prepare",return_value={"frames":769}),patch.object(m,"pinned_source"),\
                 patch.object(m,"cuda_environment"),patch.object(m,"stage_copy"):
                with self.assertRaisesRegex(RuntimeError,"PARTIAL_TRAINING_PRESERVED"):
                    m.do_train(paths,m.CONFIRM,2,20000)

if __name__=="__main__":
    unittest.main()
