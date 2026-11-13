# Civil Viewer — evidence review (2026-11-14)

**Search sources and date:** ACM DOI records; dblp SIGGRAPH; official nerfstudio docs; upstream GitHub README, LICENSE, pinned commits and example source. Searches on 2026-11-14: `Nerfstudio viewer Viser frustum`, `3D Gaussian Splatting SIGGRAPH DOI`, `2DGS surfaces geometry DOI`, `BIM as-built Gaussian splatting ISARC DOI`, `nerfview dummy rendering examples`, `viser scene point clouds examples`. Original work is left read-only.

## Verified peer-reviewed primary papers

| Source | Identifiers | Technique and civil decision | Limitations |
| --- | --- | --- | --- |
| Tancik, M. et al., **Nerfstudio: A Modular Framework for Neural Radiance Field Development**, SIGGRAPH 2023 | https://doi.org/10.1145/3588432.3591516 | Interactive training-time viewer, camera/source review and exports; learn from viewer interaction but keep the MyCON Viewer independent of training | A rendered image does not certify a civil surface or metric geometry |
| Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G. **3D Gaussian Splatting for Real-Time Radiance Field Rendering**, ACM TOG 42(4), 2023 | https://doi.org/10.1145/3592433 | Fast anisotropic Gaussian radiance-field rendering; optional future visualization of an existing GS model, without retraining | Gaussian centres are **not** themselves surveyed plane/edge/corner evidence |
| Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S. **2D Gaussian Splatting for Geometrically Accurate Radiance Fields**, SIGGRAPH 2024 | https://doi.org/10.1145/3641519.3657428 | Surface-oriented splats and geometric normal/depth regularization; reason to show A-validated finite patches separately from appearance | 2DGS improves surface modeling but not independent scale, IFC identity or construction progress |
| Chen, G., Vlaminck, M., Allebosch, G., Philips, W., Luong, H. **BIM-Align: An Automatic Framework for BIM to As-Built Alignment with Multi-View Imagery and 3D Gaussian Splatting**, ISARC 2026 | https://doi.org/10.22260/ISARC2026/0192 | Construction-oriented BIM-to-as-built alignment; motivates separate, validated registration before overlays/deviation | Paper's experimental alignment errors cannot be reused as MyCON performance |

Primary records: https://docs.nerf.studio/developer_guides/viewer/index.html ; https://doi.org/10.1145/3592433 ; https://www.cvlibs.net/publications/Huang2024SIGGRAPH.pdf ; https://www.iaarc.org/publications/2026_proceedings_of_the_43rd_isarc_singapore/bim_align-an_automatic_framework_for_bim_to_as_built_alignment_with_multi_view_imagery_and_3d_gaussian_splatting.html .

## Inspected GitHub candidates — exact revisions

1. **nerfstudio-project/nerfview** https://github.com/nerfstudio-project/nerfview/tree/bd802d765fe3ccef6c66538927bf3ce69dd69632 — **Apache-2.0** (verified `LICENSE`). Official minimal viewer based on Viser; inspected `examples/00_dummy_rendering.py` and `examples/requirements.txt`: Python, NumPy, Viser, tyro; advanced `gsplat==1.0.0`, PyTorch, nvdiffrast and CUDA dependencies. **Adapt interface ideas, not import as hard requirement.** Very useful when a validated GS renderer is provided; the sample is not evidence of MyCON GPU execution.
2. **viser-project/viser** https://github.com/viser-project/viser/tree/448ca33ec61cec37876d29e48872dcd0ea112e5a — **Apache-2.0** (verified `LICENSE`). Inspected official `examples/01_scene/00_coordinate_frames.py` and `01_point_clouds.py`; web 3D scene, camera frames, point-cloud controls. **Optional future backend**, not forced because runtime/network/install may be unavailable on users' devices. Source viewer remains self-contained and has Canvas fallback.

Official viewer documentation: https://docs.nerf.studio/developer_guides/viewer/index.html ; https://github.com/nerfstudio-project/nerfview ; https://github.com/viser-project/viser .
No code from either upstream repository has been copied; only API and UX ideas.

## Actual upstream executable example / attempt

- **Source** https://github.com/nerfstudio-project/nerfview/blob/bd802d765fe3ccef6c66538927bf3ce69dd69632/examples/00_dummy_rendering.py
- **Instructions**: `python -m venv .venv; . .venv/bin/activate; pip install torch; pip install -e .; pip install -r examples/requirements.txt; python examples/00_dummy_rendering.py --port 8080`. Input is a synthetic dummy renderer, **no MyCON data**. Browser on `127.0.0.1:8080`.
- **Execution status**: **NOT_EXECUTED** in local tool environment: `git ls-remote https://github.com/sayyedalimrj/MyCon.git HEAD` fails `Could not resolve host: github.com`; obtaining compatible CUDA/torch/gsplat or exact upstream checkout via this environment is blocked. Source file, requirements and license were actually read via the GitHub connector. Do **not** label inspected code as run.
- Self-contained MyCON synthetic standalone viewer from prior package has separately executed browser/CPU tests and PNG/MP4, not an upstream nerfview demonstration; original package retained outside GitHub until ported by its owner.

## Comparison and decision

| Question | Plotly WebGL + Canvas fallback (existing standalone) | Viser / nerfview server |
| --- | --- | --- |
| Data access | Static read-only receipt-validated bundle, opens offline | Websocket server/client and optional CUDA renderer |
| Frustums/frame images | Built from calibrated camera-to-world and hashed frames | Rich built-in camera primitives; images still need provenance adapter |
| Point cloud/mesh/GS | Basic mesh, patch/line/marker view; GS radiance field **not** rendered | GS via optional heavy GPU renderer; also meshes and points |
| No WebGL | Lightweight Canvas drawing, explicitly marked reduced capability | Three.js/WebGL needed |
| Civil correctness | Does not measure or certify geometry; metric gate prohibits unsupported deviation | Same necessary gate; visualization cannot substitute for survey |
| Decision | **Adopt** small read-only portable default, iterate on proven compatibility | **Adapt** later, opt-in after independent backend and actual GPU smoke |

## Claim -> scientific source -> code -> experiment -> civil interpretation

| Claim | Paper | Implementation evidence | Experiment or gate | Civil meaning |
| --- | --- | --- | --- | --- |
| Interactive cameras/frustums are useful | Nerfstudio 2023 DOI above | Viser + nerfview pinned upstream | MyCON synthetic selection and layer tests; real capture pending | Show provenance of actual source images |
| Splat scene may look realistic | Kerbl 2023 DOI above | nerfview `03_gsplat_rendering.py` | **not_executed**: GPU/checkpoint and Colab not validated | Appearance must not certify structural corners |
| Surface primitives need independent QA | Huang 2024 DOI above | Viewer finite patches / confidence states | Synthetic-only with unknown handling | Review rather than assert surveyed planes |
| BIM deviation requires registration | Chen et al. 2026 DOI above | Adapter requires independent controls & metric gate | **blocked_missing_real_data** | Never show numerical as-built discrepancies for arbitrary-scale reconstruction |

**Risks:** WebGL disabled by sandbox/security, missing calibrated poses, inconsistent camera axes, hash mismatch, large mesh/Gaussian datasets, mixed units, unreliable confidence, occluded surfaces mislabeled as absent. Mitigate by explicit failure/unknown, no metric heatmap without survey, per-frame SHA256, versioned adapter, user-triggered renderer selection and reproducible immutable outputs. No real scientific quality, performance or field accuracy numbers are claimed.
