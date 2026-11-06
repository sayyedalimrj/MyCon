"""Regression: existing T4 CUDA environment must not require another 18GiB
just to reuse the already-verified registered COLMAP scene.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import mycon_colmap_gsplat as m


class RegisteredSceneCacheDisk(unittest.TestCase):
    def source(self, base):
        experiment=base/"research_colmap_gsplat"/("a"*16)/"trial"
        scene=experiment/"scene"
        model=scene/"sparse"/"0"
        images=scene/"images"
        model.mkdir(parents=True)
        images.mkdir(parents=True)
        for name in ("cameras.bin","images.bin","points3D.bin"):
            (model/name).write_bytes((name+"-verified") .encode()*15)
        names=[]
        for i in range(1,61):
            name=f"frame_{i:06d}.jpg"
            (images/name).write_bytes((name+"-derived").encode()*20)
            names.append(name)
        txt=experiment/"colmap_text/images.txt"
        txt.parent.mkdir()
        txt.write_text("".join(
            f"{i} 1 0 0 0 0 0 0 1 {name}\n\n"
            for i,name in enumerate(names,1)))
        cloud=experiment/"bim_candidate/colmap_sparse_unscaled.ply"
        cloud.parent.mkdir()
        cloud.write_bytes(b"ply\n" * 250)
        row={
            "selected_submodel":"0",
            "selected_model_sha256":{n:m.digest(model/n) for n in
                 ("cameras.bin","images.bin","points3D.bin")},
            "registered_images_txt_sha256":m.digest(txt),
            "sparse_candidate_ply_sha256":m.digest(cloud)}
        (experiment/"colmap_receipt.json").write_text(json.dumps(row))
        return scene,experiment,names

    def test_reuse_with_less_than_eighteen_gib(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            scene,exp,names=self.source(base)
            local=base/"local"
            local.mkdir()
            with patch.object(m.shutil,"disk_usage",
                              return_value=SimpleNamespace(free=700*(1024**2))):
                dest=m.local_training_scene(scene,exp,local)
            self.assertEqual(set(p.name for p in (dest/"images").glob("*.jpg")),
                             set(names))
            self.assertTrue((dest/"sparse/0/images.bin").is_file())
            # Rerunning on a T4 with less free disk after installation still
            # accepts the identical, hashed cache without replacing files.
            with patch.object(m.shutil,"disk_usage",
                              return_value=SimpleNamespace(free=600*(1024**2))):
                self.assertEqual(m.local_training_scene(scene,exp,local),dest)

    def test_missing_cache_still_rejects_insufficient_actual_space(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            scene,exp,_=self.source(base)
            local=base/"local"
            local.mkdir()
            with patch.object(m.shutil,"disk_usage",
                              return_value=SimpleNamespace(free=400*(1024**2))):
                with self.assertRaisesRegex(RuntimeError,
                        "INSUFFICIENT_LOCAL_DISK_FOR_MISSING_SCENE_CACHE"):
                    m.local_training_scene(scene,exp,local)
            self.assertFalse((local/"scene_cache").exists())


if __name__=="__main__":
    unittest.main()
