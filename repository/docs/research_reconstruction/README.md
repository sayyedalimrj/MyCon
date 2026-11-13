# MyCON Reconstruction R (2026-11-13)

Dedicated branch \`research/mycon-reconstruction-20261113\`, base Integration only. Per approved six-stream amendment, **do not edit other workstreams, historical captures, COLMAP or gsplat 6999**.

The local Round-2 source bundle distributed with the coordinator chat preserves the original 10 tests and extends them to 19 synthetic-only CPU passes. This GitHub draft initially includes an independent **fail-closed audit adapter** and the reviewed research documentation; full multiview source still needs native transfer to this branch before merge. Do not interpret a draft PR as G2 real-data completion.

Native audit:
\`\`\`bash
PYTHONPATH=. python -m reconstruction.safety_gates --run-dir /new/recon-unique-run
PYTHONPATH=. pytest -q tests/test_research_reconstruction_safety_gates.py
\`\`\`
Versioned inputs: \`mycon.reconstruction.dataset.v1\` with exact run-relative image SHA, sessions and disjoint train/test; optional \`mycon.reconstruction.camera_poses.v1\` with matching view/SHA and new calibrated poses. Flags for physical metric scale and field validation are conservative false; no IFC/global_id or physical progress inferred.

See [evidence review](../docs/research_reconstruction/evidence_review.md). Scope missing in this repository draft: complete frame-selection / fresh train-only COLMAP / coverage / GS staging / scoring / human-review pipeline, independently audited COLMAP/MVS/GS real room artifacts and a verified matched 6999 baseline.
