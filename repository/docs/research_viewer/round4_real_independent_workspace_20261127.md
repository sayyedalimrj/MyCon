# Viewer R4 — real independent review (2026-11-27)

Code commit: `dd4420581ea2a8ebfc1a42e63be96c01e9177e47` (existing draft PR #50; no merge). Run ID: `viewer-r4-final-real-20261127`.

## Read-only inputs and provenance
- A #42 head `14a928e4d545e171c0f4dbc168446ed0596888bc`; A run `geom-civil-camera-20261114-real-depth-002`. Producer archive Drive ID `1pOZXRyCFNBRyxOY6jJ_ugUhbobdvrqnh`, 27,378,466 bytes, ZIP CRC checked; SHA256 `902e11df91b7c159f787b5730cd946ba220def57ed631882fd341ad9a26f996c`.
- A handoff SHA256 `0cf1896eb988197b1558c8c86f60aa951ec1c526ee9a9334dabc7c06e96e67aa`; registration QA `ce123618ea2a24fed85cf69d1475243a305485422471214747591eb2f74293ea`; overlay `021cce09edf43132302afde1c53e714347fa69b94b9fa6123dca2571d9f10ab0`; civil manifest `d9f5cc124a0b176b11473dd59407a44ac1cae862d2af1db252a003ccffc719b0`. Native COLMAP files and five original JPEGs checked against A's receipts. Native quaternion world-to-camera inverted and checked against photo QA. No original data modified.
- R #49 head `639810de055bac7e5986c35731423fcee78c737e`: fresh registered pose manifest absent, not a same-run substitute. Real workspace receipt fresh R status `not_supplied`. Optional R input without poses is reported blocked, never populated with historical poses.

## Independent views
Historical camera frame `COLMAP_sfm_world_unscaled` (arbitrary COLMAP units): 192 registered camera poses/frusta, 5 linked photos, 187 explicitly unavailable photos, 17,298 sampled points from the same historical sparse model (51,894 points). Path follows image filename order, not verified capture times. Example CAM-0121 opens original frame_000120.jpg (640 x 1138), SHA256 `1687d6004fdf9deb150575fda484da51c7c7fc54464a7ba1544923b7d155c447`.
Shape3 depth frame `colmap_world_unverified` (arbitrary reconstruction units): source depth cloud 104,161 points; displayed producer sample 25,000 points; 15 finite plane boundaries, 49 finite edge candidates, 7 corner candidates, 0 field-confirmed. No invented filled mesh, no camera overlay. All 71 features selectable and unknown/pending. Units normalized to reconstruction_units label without rescaling; original frame/unit labels in receipt. Registration UNREGISTERED, metric false, feature/photo projection false; no shared overlay or synchronized spatial navigation.

## Reproduce (new output path outside source roots)
From repository/ with viewer requirements installed:
```bash
python -m viewer_civil.independent_review \
  --a-project-root /path/to/extracted_A \
  --a-handoff runs/geom-civil-camera-20261114-real-depth-002/geometry/viewer_handoff.json \
  --a-handoff-sha256 0cf1896eb988197b1558c8c86f60aa951ec1c526ee9a9334dabc7c06e96e67aa \
  --out-dir /path/to/new/viewer-r4-review
```
Optional `--r-run-dir /path/to/new_R_run` invokes existing R converter only when registered pose file exists. Open generated index.html. Existing reviewed-transform adapter remains separate. No field registration inferred by this new independent review.

## Actual tests
- Local Python 3.13.5: `PYTHONPATH=. pytest -q tests/test_research_viewer_independent_review.py`: 8 passed (0.11s).
- GitHub code commit PR test-merge checkout d727265 (not an actual PR merge): [contract CPU](https://github.com/sayyedalimrj/MyCon/actions/runs/46037016428) 141 passed, 1 skipped (4.46s); [workstream CPU](https://github.com/sayyedalimrj/MyCon/actions/runs/46037016439) 36 passed, 3 skipped (2.45s); both successful.
- Actual local Chromium/Playwright on real archived A inputs: original JPEG byte hash and camera click, missing photo clears prior image, actual Canvas marker click, 71 features selectable, independent toggles, no projection, no JS errors: PASS. Requested WebGL with ANGLE/SwiftShader: no usable context, both views correctly fallback to Canvas, no JS errors. WebGL rendering/GPU **NOT VERIFIED**; no claims of a successful WebGL render.
- Exact local committed adapter/core/UI/test blobs: 206c5b43b10857c3a6132f482ce40f17b3eae48d, fcd52b983cab8d4547bc984205231f9ed41c5e9d, dcdd4bb1989754261055e7904819980841bca1e8, bdfd15b39eeeab94b40e420bc9e42a4ceee3664b. Browser helper unchanged and exact. Local mirror not full exact branch checkout (Git DNS blocked); earlier R helper with local compatibility aliases was used for local run only, not committed. CI used repository test-merge checkout.

## Private evidence
Offline ZIP MyCON_R4_Real_Viewer_Offline.zip SHA256 e5ec62911a3822bd3f9ed1ca98555aaa8722fdecd7dabf6e1cb8e56ceb61458f (9,770,629 bytes, CRC verified), with actual frames, previews, receipt, browser logs and media. Real screenshots COLMAP SHA256 03c2995ae4d102d681a1989c637fccca15dd7dfc304a6964f9eb3fb723382607 and Shape3 SHA256 8574c0f7a471759f6b1a819166840dbc899643a583b42e9d97f05e53f0751aa3. Four-second actual screenshot video SHA256 f2c7f5d9c793f69364502dfab56919ced9e9571c9c1b6df7f04d9af84114eab0. These are private session downloads, not public repo artifacts; no original photos posted to GitHub. Dedicated Drive folder created, upload permission returned 403; private ZIP provided to user instead.

## Remaining
Fresh R registered poses with SHA-verified images, independently reviewed A/R transform and held-out projection QA, field operator signoff and metric survey, real WebGL context/browser test. Do not equate historical sparse cameras with fresh R, or candidate boundaries with actual walls/completion. No PR creation or merge.
