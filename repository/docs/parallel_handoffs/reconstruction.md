# Reconstruction R — handoff to Integration coordinator

Integration base read: \`c81962332cb4220f52eeb95acfcda61e48d39d24\`. Owned paths only under approved \`parallel_6_streams_approval_20261113.md\`. No main merge, no source/checkpoint/legacy data write. Implementation SHA: see PR HEAD (self-reference cannot be embedded without changing commit hash).

## Delivery and testing

- Local standalone complete source archive (available in the original contributor chat) contains initial \`workflow.py\`, original 10 tests and 9 new tests, \`coverage.py\`, \`preflight.py\`, \`visual_review.py\`, \`gs_experiment.py\`, operating documentation.
- Local actual command \`PYTHONPATH=. pytest -q tests/test_research_reconstruction_*.py\` -> **19 passed**, synthetic CPU ONLY (source archive, not this incomplete GitHub PR checkout).
- Local Python 3.13.5, torch 2.10.0+cpu; CUDA unavailable; COLMAP, gsplat and LPIPS unavailable. No notebook actual GPU runtime.
- Native GitHub draft presently adds \`reconstruction/safety_gates.py\`, \`tests/test_research_reconstruction_safety_gates.py\`, this handoff, paper/code/notebook review and README. Dedicated tests should be run on exact PR SHA; never substitute 19 local source-archive results for branch CI.

## Synthetic-only receipt (NOT MyCON site)

Two generated AVI sessions: train 35 / test 35; mock poses registered 35 train / 2 test (mocked COLMAP text only), no COLMAP executable. Synthetic manifest SHA256 \`7dff74715e7b9d1b1b310b889c1830fe3b4aea5b76401955649937342e295110\`; camera poses SHA256 \`e7fc2261331b7fcf27dad3beec5400ffa84cb9d3fa622e89c0c728ef0f1f81c6\`. Not a scientifically useful accuracy metric.

## Outstanding / blockers

1. Full native source and original 19 tests MUST be committed in this owner-only path before draft ready-for-review. This draft is deliberately INCOMPLETE and must not merge.
2. Real source capture and fresh COLMAP cameras not locally accessible; full room selection/pose MVS and structural wall/corner QA not executed.
3. No opt-in fresh GS GPU training executed; no new real checkpoint/PLY, matched held-out renders, trajectory, MP4, PSNR/SSIM/LPIPS or fair 6999 comparison. Historical old reported log metrics do not certify held-out quality.
4. No independently measured metric scale, ground truth, IFC mapping or physical progress. Keep scale false and unknown conditions explicit.
5. Viewer/geometry interface request: consume only SHA-pinned verified test cameras and physically extracted reviewed dense surfaces, not Gaussian centers as BIM structural truth; coordinator-owned Colab changes deferred.

R branch/PR target: \`research/mycon-integration-20261112\`. Coordinator alone merges after exact HEAD checks, structural real-video/surface evidence and scientific gates.
