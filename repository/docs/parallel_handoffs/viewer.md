# Civil Viewer V — handoff, round 2 (2026-11-14)

## Source and ownership
- Repo: https://github.com/sayyedalimrj/MyCon
- Approved governance: `repository/docs/integration/parallel_6_streams_approval_20261113.md`; issue #48.
- Branch: `research/mycon-viewer-20261113`, based on Integration head `c81962332cb4220f52eeb95acfcda61e48d39d24`; Draft PR target `research/mycon-integration-20261112`.
- Latest tested source/test commit: `0d40c08fc5e3cbbf03902f27364ce4310e3fe568`. Browser frontend promoted at `9178ea905635b8c7c2fa243bc38bb746980c904d`; media committed at `1d957be650e4c8410005a17541f40844604b35c0`. See Draft PR #50 for current head commit.
- Additions only: `repository/viewer_civil/{core.py,demo.py,__init__.py,__main__.py,browser.py,static/index.html,static/viewer.css,static/viewer.js,legacy_snapshot/**,media/*.jpg,media/*.mp4,README.md}`, `repository/requirements-research-viewer.txt`, `repository/tests/test_research_viewer_bundle.py`, `repository/docs/research_viewer/{evidence_review.md,thesis_media_manifest.md}`, and this handoff.
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
- **Actual GitHub branch dedicated CPU CI**: [run #43507387686](https://github.com/sayyedalimrj/MyCon/actions/runs/43507387686) on SHA `0d40c08fc5e3cbbf03902f27364ce4310e3fe568`. `cd repository && python -m pytest -q tests/test_research_viewer_bundle.py` via workstream driver: **13 passed, 2 skipped, 0 failed, exit 0 (3.13 s)**. The optional Plotly/Playwright browser cases were skipped because runner lacks those dependencies. The prior separate standalone browser test executed 7/7 locally; current branch browser code not tested on GPU/WebGL.
- **Upstream nerfview official 00_dummy_rendering.py** inspected at `bd802d765fe3ccef6c66538927bf3ce69dd69632`, license Apache-2.0; **not executed** due network/torch dependency blocker.
- **No** GPU 3DGS rendering test, Google Drive/Colab runtime test, real MyCON room source frame, ground-truth survey, verified BIM match, or paper-ready numeric evaluation.

## Evidence links / hashes
- Verified papers and reproducible source comparison: `repository/docs/research_viewer/evidence_review.md` (3 SIGGRAPH DOI papers + ISARC civil study, 2 pinned upstream repos).
- Previous standalone media exact SHA256 and regeneration steps: `repository/docs/research_viewer/thesis_media_manifest.md`. Original high-quality files preserved in user ZIP; two real synthetic-session browser JPEGs and one MP4 derivative **are uploaded to GitHub** under `repository/viewer_civil/media/` with verified sizes and SHA256 in media manifest.
- Input source SHA256 of immutable original MyCon_R4 / checkpoint is **not read/available**; do not invent. Synthetic created source has a new hash per run; validate local `viewer_manifest.json` and every receipt before displaying.
- Real site cameras, reconstructed GS outputs, mesh and independent scale controls: **blocked_missing_input**. CUDA/WebGL-on-GPU: **not_executed**.

## Coordinator integration needs
1. Independently run dedicated Viewer tests on the PR head; review CI logs and actual browser screenshot when environment allows.
2. Keep this Viewer branch separate and unmerged until coordinator gates; coordinator alone wires final Notebook or optional Viser backend.
3. Produce real verified camera poses/frames and structural primitives via read-only conversion into `mycon.viewer_bundle.v1`; do not patch A/R/D producer files in this PR.
4. Run real same-camera Shape3 review with human audit of Unknown/occlusion; only publish metric Shape4 after survey/registration/held-out evidence.
5. No unapproved merge to Integration/main and no source-data overwrite.

## Round-2 revision receipt
- PR: https://github.com/sayyedalimrj/MyCon/pull/50 — Draft, Integration target, unmerged.
- Original standalone source copied exactly into `repository/viewer_civil/legacy_snapshot/`; new primary frontend adopts the proven original browser UI with optional point-cloud display. Synthetic source frames are not field photographs.
- `viewer_civil/media/` contains two 1440×900 browser-capture JPEG images and compressed 960px H.264 rotation video from the original synthetic run; provenance and SHA256 recorded.
- New deviation gate requires independent held-out scale/registration/survey receipts with SHA256 plus reviewer and per-face evidence. Flags alone are insufficient. No real site or GPU data has been tested. Published synthetic JPG/MP4 files have SHA256 checked by new dedicated test; current source/test revision is `0d40c08fc5e3cbbf03902f27364ce4310e3fe568`, [workstream CI](https://github.com/sayyedalimrj/MyCon/actions/runs/43507387686): **13 passed, 2 skipped**, 0 failed. The 2 skips are Plotly/Playwright optional test cases, not real browser pass.
