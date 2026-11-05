"""Real metric-anchor BIM handoff tests; run after numpy/plyfile/PyYAML installed."""
from __future__ import annotations
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from mycon_colmap_bim_adapter import aligned_colmap_cloud


class BimGeometryHandoff(unittest.TestCase):
    def test_real_ifc_non_coplanar_fit_and_independent_holdout(self):
        import numpy as np
        from plyfile import PlyData,PlyElement
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            ifc=root/"as-designed.ifc"
            ifc.write_bytes(b"ISO-10303-21;\n"+b"DATA;\n"+b"0"*2048+
                            b"\nENDSEC;\nEND-ISO-10303-21;\n")
            cloud=root/"source_colmap.ply"
            rec=np.empty(200,dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                    ("red","u1"),("green","u1"),("blue","u1")])
            gen=np.random.default_rng(12)
            raw=gen.uniform(-1,1,(200,3))
            for k,column in enumerate(("x","y","z")):rec[column]=raw[:,k]
            for color in ("red","green","blue"):rec[color]=120
            PlyData([PlyElement.describe(rec,"vertex")],text=False).write(str(cloud))
            points=[(0,0,0),(1,0,0),(0,2,0),(0,0,3),
                    (2,3,4),(0.2,0.5,1.7)]
            anchors=root/"anchors.csv"
            with anchors.open("w",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(["id","source_x","source_y","source_z",
                                 "target_x_m","target_y_m","target_z_m","split"])
                for i,(x,y,z) in enumerate(points):
                    writer.writerow([f"a{i}",x,y,z,2*x+5,2*y+10,2*z+12,
                                     "fit" if i<4 else "holdout"])
            cfg=root/"base_site.yaml"
            cfg.write_text("project:\n  root: .\nbim:\n  allow_synthetic_ifc_fallback_for_tests: true\n")
            before=cloud.read_bytes()
            out=root/"metric_review"
            result=aligned_colmap_cloud(cloud,anchors,ifc,out,
                                        max_holdout_m=.05,base_config=cfg)
            self.assertLess(result["holdout_max_m"],1e-5)
            self.assertEqual(result["sim3_scale"],2.0)
            self.assertFalse(result["stage9_progress_authorized"])
            self.assertFalse(result["stage11_schedule_authorized"])
            self.assertTrue((out/"stage8_review_config.yaml").is_file())
            self.assertEqual(before,cloud.read_bytes())
            self.assertEqual(result,aligned_colmap_cloud(cloud,anchors,ifc,out,
                                                        .05,cfg))

    def test_missing_ifc_and_synthetic_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            fake=root/"fake.ifc";fake.write_bytes(b"fake"*400)
            cloud=root/"source.ply";cloud.write_bytes(b"ply"*500)
            anchors=root/"anchors.csv";anchors.write_text("none\n")
            with self.assertRaisesRegex(RuntimeError,"REAL_IFC_STEP"):
                aligned_colmap_cloud(cloud,anchors,fake,root/"out")


if __name__=="__main__":
    unittest.main()
