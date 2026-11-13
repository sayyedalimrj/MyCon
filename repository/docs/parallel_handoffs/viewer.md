# Civil Viewer V — handoff, round 2 (2026-11-14)

## Source and ownership
- Repo: https://github.com/sayyedalimrj/MyCon
- Approved governance: `repository/docs/integration/parallel_6_streams_approval_20261113.md`; issue #48.
- Branch: `research/mycon-viewer-20261113`, based on Integration head `c81962332cb4220f52eeb95acfcda61e48d39d24`; Draft PR target `research/mycon-integration-20261112`.
- Last full-source/docs commit before this handoff: `ceb8d11d5b0a4ace379d108f94756e348172f8be`. The PR head SHA after the handoff is recorded in GitHub metadata; this file cannot self-reference its own Git SHA.
- Additions only: `repository/viewer_civil/{core.py,demo.py,__init__.py,__main__.py,browser.py,static/index.html,static/viewer.css,static/viewer.js,README.md}`, `repository/requirements-research-viewer.txt`, `repository/tests/test_research_viewer_bundle.py`, `repository/docs/research_viewer/{evidence_review.md,thesis_media_manifest.md}`, and this handoff.
- No writes to `pipeline/**`, `colab/**`, `gui/**`, original MyCon_R4/video/GS pt checkpoints/COLMAP or shared defaults. No force, merge or train.

## Interface and contract
- Read-only `load_bundle(root)` returns `{manifest,assets,deviation_enabled,deviation_reason,root}`.
- Required `mycon.viewer_bundle.v1` roles: `camera_poses`, `frame_manifest`, `planes`, `edges`, `corners`, `mesh`, `alignment_result`; optional JSON `point_cloud`.
- Every asset must have path relative to bundle, SHA256, version, units, coordinate_frame, quality and provenance, matching a parent run_id. Images are independently SHA256 checked. Camera optical +Z forward, image +Y down; rigid row-major c2w with intrinsics.
- Preview outputs `mycon.viewer_preview.v1`; `viewer_build.json` includes original manifest SHA256. All output targets must not exist.
- Alignment and independent surveyed-scale gates prohibit numeric BIM deviation until independently verified. Unknown/occluded is not zero/failed/0% built. Synthetic mode force-disables deviation. Source GS Gaussian centers do not become verified civil planes.
- Example synthetic run id: `viewer-synthetic-20261114`, units `arbitrary`, frame `synthetic_room_local`, six cameras/frames. Provenance generator `viewer_civil.demo`. The generator is deterministic in geometry but timestamps/hash values vary between runs; each output is independently hashed.

## Test proof and limits
- **Prior standalone baseline, actually executed in this session**: `cd /mnt/data/viewer_work && PYTHONPATH=. pytest tests/test_viewer.py -q --disable-warnings --maxfail=1` -> **7 passed, exit 0** in 3.80 s, including restricted Chromium/Canvas selection and layers. This test is of the preserved original ZIP, **not** of the newly committed GitHub source.
- **Current GitHub branch dedicated tests:** `cd repository && python -m pytest -q tests/test_research_viewer_bundle.py`; code written but **NOT_EXECUTED on checked-out Git branch in the local sandbox** because `git ls-remote ...` returns `Could not resolve host: github.com`. Await a real GitHub Actions branch-head specialized run; green generic workflow does not qualify. Report exact result when available.
- **Upstream nerfview official 00_dummy_rendering.py** inspected at `bd802d765fe3ccef6c66538927bf3ce69dd69632`, license Apache-2.0; **not executed** due network/torch dependency blocker.
- **No** GPU 3DGS rendering test, Google Drive/Colab runtime test, real MyCON room source frame, ground-truth survey, verified BIM match, or paper-ready numeric evaluation.

## Evidence links / hashes
- Verified papers and reproducible source comparison: `repository/docs/research_viewer/evidence_review.md` (3 SIGGRAPH DOI papers + ISARC civil study, 2 pinned upstream repos).
- Previous standalone media exact SHA256 and regeneration steps: `repository/docs/research_viewer/thesis_media_manifest.md`. Media preserved in user conversation archive, **not uploaded to GitHub**; avoid suggesting otherwise.
- Input source SHA256 of immutable original MyCon_R4 / checkpoint is **not read/available**; do not invent. Synthetic created source has a new hash per run; validate local `viewer_manifest.json` and every receipt before displaying.
- Real site cameras, reconstructed GS outputs, mesh and independent scale controls: **blocked_missing_input**. CUDA/WebGL-on-GPU: **not_executed**.

## Coordinator integration needs
1. Independently run dedicated Viewer tests on the PR head; review CI logs and actual browser screenshot when environment allows.
2. Keep this Viewer branch separate and unmerged until coordinator gates; coordinator alone wires final Notebook or optional Viser backend.
3. Produce real verified camera poses/frames and structural primitives via read-only conversion into `mycon.viewer_bundle.v1`; do not patch A/R/D producer files in this PR.
4. Run real same-camera Shape3 review with human audit of Unknown/occlusion; only publish metric Shape4 after survey/registration/held-out evidence.
5. No unapproved merge to Integration/main and no source-data overwrite.
