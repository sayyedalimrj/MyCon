"""Synthetic 7-stage floor dataset generator.

This package builds a single, fixed building floor and then exports the
same floor at seven progressive construction stages. It is designed to
provide a turn-key, deterministic, dependency-light test asset for the
MyCon construction-progress pipeline.

Modules:

* ``scene_spec``        - load the YAML config and validate it.
* ``layout``            - compute the deterministic geometry of every
                          structural and architectural element.
* ``stage_controller``  - select which elements to emit at each stage
                          and what surface finish they should have.
* ``materials``         - PBR-style procedural texture/material library.
* ``ifc_builder``       - export an IFC4 BIM file per stage.
* ``mesh_builder``      - export OBJ / GLB triangle meshes per stage.
* ``camera_path``       - plan a believable handheld walk through the
                          building.
* ``smartphone_sim``    - apply post-processing that makes the renders
                          look like a real smartphone clip.
* ``renderer``          - pure-Python software rasterizer.
* ``video_exporter``    - encode frames into MP4 via ffmpeg.
* ``metadata_exporter`` - per-stage metadata + dataset manifest.
* ``validate``          - sanity checks on geometry and outputs.

The package has zero hard dependency on Blender, OpenCV, Open3D or a
GPU; it only needs ``numpy``, ``Pillow``, ``trimesh``, ``imageio``,
``imageio-ffmpeg`` and ``ifcopenshell``.
"""

from __future__ import annotations

__version__ = "1.0.0"
