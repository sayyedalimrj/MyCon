# Workstream D — Scientific / implementation / notebook evidence review

**Search date:** 2026-11-14; owner: D Metric & BIM; scoped to approved six-workstream contract `parallel_6_streams_approval_20261113.md`. Project-room IFC / independent survey NOT provided. Publication-based outcomes below do not imply MyCON performance.

## Search process

Sources: publisher pages (IEEE Xplore, Elsevier/ScienceDirect), dblp bibliographic corroboration, Open3D official docs and pinned GitHub files/commits, IfcOpenShell project source/license. Queries: "Umeyama 1991 similarity 10.1109/34.88573", "10.1016/j.autcon.2014.05.014 scan-vs-BIM", "Besl McKay ICP 10.1109/34.121791", "Open3D 0.19 point-to-plane ICP notebook", "Open3D global registration notebook", "IfcOpenShell geometry IFC LGPL". Excluded uncertain Chen–Medioni DOI and unverified social writeups. Verified paper DOI and venue against publisher/DBLP. GitHub revision endpoints and notebook source inspected 2026-11-14 (read-only).

## 1. Peer-reviewed papers (DOIs verified)

| Paper | Verified publication | Method, useful engineering signal | Limitations for MyCON |
|---|---|---|---|
| **Shinji Umeyama (1991), "Least-squares estimation of transformation parameters between two point patterns"** | *IEEE Transactions on Pattern Analysis and Machine Intelligence* **13(4), 376–380**. https://doi.org/10.1109/34.88573 ; https://ieeexplore.ieee.org/abstract/document/88573/ | Closed-form proper similarity (uniform scale, orthogonal R, translation) with known pairs; avoids reflection. Initial estimate for observed corners with surveyed correspondences. | Ordinary least squares vulnerable to incorrectly matched points; no metric truth from self-fitting, unknown scale uncertainty without survey/holdouts. |
| **Frédéric Bosché, Mahmoud Ahmed, Yelda Turkan, Carl T. Haas, Ralph Haas (2015), "The value of integrating Scan-to-BIM and Scan-vs-BIM techniques for construction monitoring using laser scanning and BIM: The case of cylindrical MEP components"** | *Automation in Construction* **49, 201–213**; online article DOI https://doi.org/10.1016/j.autcon.2014.05.014 ; https://www.sciencedirect.com/science/article/pii/S0926580514001319 | Combines model-informed construction component retrieval and scan comparison, including cylinder/Hough shape reasoning. Supports component-specific tolerances, explicit evidence. | Investigates cylindrical MEP utility corridor and laser scan, not unscaled phone video room; construction progress and missing geometry do not transfer. |
| **Paul J. Besl, Neil D. McKay (1992), "A Method for Registration of 3-D Shapes"** | *IEEE Transactions on Pattern Analysis and Machine Intelligence* **14(2), 239–256**. https://doi.org/10.1109/34.121791 ; https://dblp.org/rec/journals/pami/BeslM92.html | Iterative closest point with good initial transform and geometric matches; use point-to-plane/ICP refinement AFTER scale and correspondence gates. | Local minimum, false correspondence on repeated walls, holes/occlusion and partial surfaces; cannot establish metric scale. |

Additional reviewed alternative: Rusinkiewicz & Levoy (2001), "Efficient Variants of the ICP Algorithm", 3DIM 2001 https://gfx.cs.princeton.edu/gfx/pubs/Rusinkiewicz_2001_EVO/ (no independently verified DOI in this review). Useful for normal-oriented, fast local refinement, not a substitute for survey ground truth.

## 2. Actual source repositories (revision + licensing checked)

| Repo / exact pin | Licensing and required dependencies | Decision |
|---|---|---|
| **Open3D** https://github.com/isl-org/Open3D/tree/v0.19.0 pinned annotated resolved commit `1e7b17438687a0b0c1e5a7187321ac7044afe275`. `LICENSE` at this tag is **MIT**. | Python 3.11 compatible, point-cloud registration, FPFH, RANSAC, ICP; compiled native wheels; CPU-heavy/GPU optional, Open3D 0.19 generally needs compatible NumPy/OS. Official examples at https://www.open3d.org/docs/release/python_example/pipelines/index.html | **Adapt optionally** for verification/refinement after reviewed scale and ROI; avoid mandatory dependency in present CPU NumPy adapter. Open3D geometry matching alone cannot validate completeness or metric scale. |
| **IfcOpenShell** https://github.com/IfcOpenShell/IfcOpenShell/tree/990d82e580ca0c1529596bc2fd8910cba61b9ec6 (inspected HEAD, `VERSION` = `0.9.1`). `COPYING.LESSER` = **LGPL-3.0**, other modules may have GPL components. | Python compiled wheel plus IFC geometry engine, IFC2x3/IFC4/IFC4x3 and `IfcConvert`. Official docs https://docs.ifcopenshell.org/ifcconvert.html ; license applicability differs by subproject. | **Adapt only as separately reviewed IFC-to-finite-BIM-planes producer**, with explicit IFC model revision, unit conversion/coordinate origin, geometry triangulation, per-element GlobalId and SHA. No implicit conversion from OBJ to semantically valid IFC; never copy GPL code blindly. |

The pinned IfcOpenShell revision is a moving-HEAD *snapshot hash* fixed at the review date, not a claimed stable release tag. No third-party code was copied into MyCON.

## 3. Real reproducible published notebooks

- **Official executable Jupyter notebook, ICP:** https://github.com/isl-org/Open3D/blob/v0.19.0/docs/jupyter/pipelines/icp_registration.ipynb at pinned notebook blob SHA `25c44d2f536f81fc2670b1985819381467c441bc` / Open3D commit `1e7b17438687a0b0c1e5a7187321ac7044afe275`. Its 17 cells import Open3D/NumPy, use `o3d.data.DemoICPPointClouds()` downloaded example clouds, compare initial and iterative point-to-point/point-to-plane solutions. Requires `pip install open3d==0.19.0 numpy jupyter`, assets accessible to download, and either display backend or headless visualization override; `docs/jupyter/open3d_tutorial.py` used. Run: `cd docs/jupyter/pipelines && jupyter nbconvert --to notebook --execute icp_registration.ipynb --ExecutePreprocessor.timeout=300 --output icp_registration.executed.ipynb`. **Not executed in this MyCON session:** Open3D not locally installed and GitHub/remote demo data DNS inaccessible to local runtime. The notebook is REAL source, not a simulation of a notebook.
- **Official executable global RANSAC notebook:** https://github.com/isl-org/Open3D/blob/v0.19.0/docs/jupyter/pipelines/global_registration.ipynb , blob `df86d7e9caa5cae674e879fb30c486b641828d11`; 27 cells showing voxel/FPFH+RANSAC then refinement. Same requirements and access blocker, **not executed**. Comparative algorithm only; no real MyCON data.

## 4. Method choices and evidence chain

| Claim | Paper / reference | Code evidence available | Required experiment | Civil interpretation |
|---|---|---|---|---|
| Metric scale from physical lengths | geometric similarity, Umeyama 1991 | MyCON `civil_metric_bim/core.py::length_scale` | >=3 independent high-confidence actual measured fit spans, >=1 independent heldout, propagated uncertainty and survey signoff | Nominal block/profile dimensions are proposals, not as-built measurements |
| Proper Sim(3) from reviewed corners | Umeyama 1991 | `umeyama`, `build_transform` | non-degenerate 3D geometry, controlled synthetic R,t,s, real heldout survey TRE | transformed BIM coords only; original captured source coordinates remain immutable |
| Local ICP improves fit | Besl & McKay 1992; Open3D ICP example | Present D adapter does NOT execute production ICP | compare fixed-scale initial alignment vs ICP using same fitted correspondences, independent controls, occlusion masks | lower fitted error does not establish external accuracy |
| Plane-to-plane/point-to-plane | ICP variants; Open3D global registration/plane methods | review finite plane and normal in BIM JSON, D `deviation` | matched finite actual surface regions, normal orientation QA, signed distance and sample coverage; exclude gaps | avoid false deviations on concealed/unseen faces |
| Scan-vs-BIM difference and element semantics | Bosché et al. 2015 | D optional IFC/BIM feature JSON, future IfcOpenShell producer; Stage8 A-owned | IFC revision/units/GlobalId and heldout physical field survey controls | geometric evidence `unknown` never implies `not_built` |

**Baseline A:** independent length-only robust scale and heldout; can run absent BIM, but no rotation/translation without correspondence.
**Baseline B:** BIM feature Umeyama (possibly RANSAC robustify) using >=3 independent fit references; BIM-derived scale provisional until independently surveyed. Optionally compare robust point-to-plane ICP against unrefined fit with heldouts.
**Failure risks:** collinear/coplanar or spatially clustered controls; arbitrary reconstruction units confused with mm/cm; repeated identical features/planes, flipped normals, false corner extensions; nominal vs actual steel shape variation; unknown mesh coverage; overconfident covariance under correlated survey repeats; fitting thresholds tuned on test anchors; false zero deviations from no-data. Retain source and hashes, don't move original cloud.

## 5. Reproducibility and status

Scientific comparisons require site/session-separated train/fit vs holdout; independent actual field survey distances, IFC revision/coordinate origin, evaluation threshold preregistration, signed normal convention, and field-approved uncertainty budget. This review is **research only**; no GPU or field accuracy was measured; notebook and external code were **inspected, not executed**. The subsequent D-specific CPU tests should be logged with exact branch SHA, command, counts and exit code. `metric_validation=pending`, `metric_scale_verified=false`, `registration_verified=false`, and `can_feed_stage8=false` until coordinator-approved independent control gate.
