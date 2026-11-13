# Civil Viewer evidence artifacts — synthetic browser execution

Previous independent standalone bundle (preserved, not a real-site result):
- ChatGPT conversation ZIP: `MyCON_Civil_Viewer_Standalone_20261113.zip`; SHA256 `97200bb676da4a1a0a1361a2c17e4dea8e9a8cf2fde6d39be0addde843556605`.
- Prior actual browser screenshot, `viewer_outputs/viewer-synthetic-20261113-final1/thesis_figure_01_civil_viewer.png`, SHA256 `68e9ba180a32149b42252336820f131fb0e1fc54e7683dc3d36f80be1111fc17`.
- Prior actual browser rotation video, `viewer_outputs/viewer-synthetic-20261113-final1/thesis_preview_rotation.mp4`, SHA256 `d0fd5c222e56a2e78d356f4a77a86b279de8c5e36b777aac0d361fe22d91e021`; H.264, 1440x900, 8 fps. 
- The actual PNG/MP4 media are in the **retained ChatGPT downloadable ZIP**, not in this Git tree. Do not cite these as GitHub-hosted assets or pretend they are field capture.

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
