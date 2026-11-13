# Civil Viewer evidence artifacts — synthetic browser execution

Previous independent standalone bundle (preserved, not a real-site result):
- ChatGPT conversation ZIP: `MyCON_Civil_Viewer_Standalone_20261113.zip`; SHA256 `97200bb676da4a1a0a1361a2c17e4dea8e9a8cf2fde6d39be0addde843556605`.
- Prior actual browser screenshot, `viewer_outputs/viewer-synthetic-20261113-final1/thesis_figure_01_civil_viewer.png`, SHA256 `68e9ba180a32149b42252336820f131fb0e1fc54e7683dc3d36f80be1111fc17`.
- Prior actual browser rotation video, `viewer_outputs/viewer-synthetic-20261113-final1/thesis_preview_rotation.mp4`, SHA256 `d0fd5c222e56a2e78d356f4a77a86b279de8c5e36b777aac0d361fe22d91e021`; H.264, 1440x900, 8 fps. 
- The **original full-quality PNG and MP4** remain in the retained ChatGPT ZIP. For reproducible GitHub review we also committed two **1440×900 JPEG** browser screenshots and a 960px-wide, 8fps H.264 MP4 derivative to `repository/viewer_civil/media/` at commit `1d957be650e4c8410005a17541f40844604b35c0`. They are still synthetic-fixture browser captures, **not field capture**.

GitHub media (verified file sizes by GitHub):
- [Civil viewer screenshot](../../viewer_civil/media/thesis_figure_01_civil_viewer.jpg): 182019 bytes; SHA256 `03dc5e903a88e2b074c4f8f89d30af08f098df337898c5c56f7e858b06e75ec3`.
- [Coverage / Shape4 screenshot](../../viewer_civil/media/thesis_figure_02_coverage_and_shape4_overlay.jpg): 207039 bytes; SHA256 `bf6201664a5795099912520a57b3413e685c2ad898f52840091b27c1cfa43759`.
- [Synthetic orbital preview video](../../viewer_civil/media/thesis_preview_rotation.mp4): 126196 bytes; SHA256 `b226934edb6a9ec6bd27ef87122fe1dc3e9f2fc5f485c732e23d0f5ffc0a18a7`.
The original source screenshots and video are larger/high-quality PNG/MP4 inside the user ZIP; the committed media are derived compression/resizing, with the same synthetic caveat.

Browser screenshot/video above depict **synthetic construction-like geometry and synthetic frame images**, not actual surveyed site. This repo contains a generator and capture CLI for newly produced independent artifacts. Re-create to a fresh output directory; archive copies with SHA256 and an explicit `SYNTHETIC` thesis caption. Without a valid MyCON camera-source bundle, no genuine site-frame screenshot or real-site video can be published.

To produce real browser-run evidence from this Git commit:
```bash
cd repository
python -m viewer_civil demo runs/viewer-synthetic-20261114/source --run-id viewer-synthetic-20261114
python -m viewer_civil render runs/viewer-synthetic-20261114/source runs/viewer-synthetic-20261114/preview
python -m viewer_civil capture runs/viewer-synthetic-20261114/preview/index.html runs/viewer-synthetic-20261114/preview.png
python -m viewer_civil video runs/viewer-synthetic-20261114/preview/index.html runs/viewer-synthetic-20261114/rotation.mp4 --seconds 4
sha256sum runs/viewer-synthetic-20261114/preview.png runs/viewer-synthetic-20261114/rotation.mp4
```

For actual field data, independently verify SHA256-protected source camera frame/pose and 3D surface provenance; never replace synthetic with an unlabeled placeholder. Unverified scale remains `false`.
