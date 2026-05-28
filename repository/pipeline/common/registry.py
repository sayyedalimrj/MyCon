"""Stage registry: typed metadata for every pipeline stage.

This module is the single source of truth for *what stages exist*, *what each
stage reads and writes*, and *how to invoke each stage as a CLI*. It is
intentionally a **descriptive registry**, not an execution engine. The
canonical way to run a stage remains:

    python3 -m pipeline.stage_XX.<run_module> --config configs/<x>.yaml

The registry exposes the metadata that the future GUI, the future REST API,
and the existing :mod:`scripts.run_stage` launcher all need:

- Human-readable name, short description, and ordering position.
- Upstream stage dependencies (file-contract DAG edges).
- Required config keys (delegated to the typed
  :mod:`pipeline.common.schema` views so this list cannot drift from what
  the stage actually validates).
- Declared YAML keys whose *values* are stage inputs / outputs (paths or
  directories). The registry does not resolve those paths itself; resolution
  is :mod:`pipeline.common.paths`'s job.
- The conventional report basename a stage writes under
  ``runs/<run_id>/reports/`` so the artifact-aggregator and provenance layer
  can find it without each stage having to publish its own discovery hook.
- A *capability flag set*: ``heavy``, ``server_required``, ``optional``,
  ``stub_or_partial``. These are the same flags
  ``scripts/run_pipeline_plan.py`` already documents; surfacing them on the
  descriptor lets the GUI render appropriate badges and lets CI gate on
  them.

Why this design
---------------

The repository already has 14 stage entry points whose ``run_*`` functions
have non-uniform signatures (some take ``cfg, *, force, log_level``, others
take ``cfg, force=False, log_level="INFO"``). Wrapping all of them behind a
single typed callable would force a wide refactor for negligible benefit at
this phase. The registry instead records the *CLI module* and the *function
name* per stage. A caller (today: ``scripts/run_stage.py``; tomorrow: the
GUI's run-control backend) can either shell out to ``python3 -m
<module> --config <path>``, which is what every stage already supports, or
import the function and call it directly when running in-process.

Both paths are honored by the descriptor — the in-process path is opt-in
(``descriptor.callable()`` returns the loaded function) so importing the
registry never imports Open3D or COLMAP wrappers.

Public API
----------

- :class:`StageDescriptor` — the typed record per stage.
- :class:`StageRegistry` — keyed lookup, ordered iteration, dependency
  validation.
- :data:`STAGE_REGISTRY` — the canonical 14-entry registry, populated at
  import time. Treat it as read-only; mutate via
  :meth:`StageRegistry.register` only.
- :func:`build_default_registry` — pure function that constructs the
  canonical registry. Useful for tests that want a fresh instance.
"""

from __future__ import annotations

import importlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from pipeline.common.schema import (
    Stage01IngestSchema,
    Stage02KeyframesSchema,
    Stage03ColmapSchema,
    Stage04RefinementSchema,
    Stage05DenseSchema,
    Stage06DA3Schema,
    Stage07CleanupSchema,
    Stage08BimEvalSchema,
    Stage09ProgressSchema,
    Stage10CopilotSchema,
)

__all__ = [
    "StageCapability",
    "StageDescriptor",
    "StageRegistry",
    "STAGE_REGISTRY",
    "build_default_registry",
    "RegistryError",
]


class RegistryError(RuntimeError):
    """Raised when a registry operation has a structural problem."""


# ---------------------------------------------------------------------------
# StageCapability is a small string-enum-like set of tags the GUI / CI can
# read without importing the descriptor's internals. We use a frozenset of
# string constants rather than an Enum to keep the registry serializable to
# JSON without custom encoders.
# ---------------------------------------------------------------------------


class StageCapability:
    """String constants describing what a stage requires to run usefully."""

    HEAVY: str = "heavy"
    """Stage performs significant computation (MVS, COLMAP, ICP)."""

    SERVER_REQUIRED: str = "server_required"
    """Stage requires a server profile (GPU / large RAM / model cache)."""

    OPTIONAL: str = "optional"
    """Stage is not part of the metric-truth path (visualization, evidence)."""

    STUB_OR_PARTIAL: str = "stub_or_partial"
    """Stage has a real entry point but its core algorithm is not yet wired
    in this repository (DA3 model inference, 3DGS training, VLM offscreen
    rendering). Useful so the GUI can render an honest badge."""

    @classmethod
    def all(cls) -> frozenset[str]:
        return frozenset({cls.HEAVY, cls.SERVER_REQUIRED, cls.OPTIONAL, cls.STUB_OR_PARTIAL})


@dataclass(frozen=True)
class StageDescriptor:
    """Typed, immutable description of a single pipeline stage.

    The descriptor is the contract the GUI, the registry-driven CLI launcher,
    and the artifact aggregator all consume. It must not embed any heavy
    Python imports (Open3D, COLMAP wrappers); attribute access is always
    cheap.

    Attributes
    ----------
    name : str
        Stable machine identifier, e.g. ``stage_08_bim_registration``. Used
        as a dict key, as a directory name, and as the value of
        ``report["stage"]``.
    order : int
        Canonical execution order; ties broken by ``name``. Stages 8a and 8b
        share the bim-eval directory but have distinct orders 80 / 81.
    title : str
        Short human-readable title for GUI display, e.g. "BIM registration".
    description : str
        One-paragraph plain-English description of what the stage does.
    cli_module : str
        Dotted Python path of the runnable CLI module, e.g.
        ``pipeline.stage_08_bim_eval.run_registration``. Suitable for
        ``python3 -m <cli_module>``.
    callable_name : str
        Name of the in-process callable inside ``cli_module``, typically
        ``run_<short>``. Use :meth:`callable` to load it lazily.
    schema_class : type
        Class object for the typed schema view this stage uses (one of the
        ``StageNNSchema`` classes). The descriptor delegates
        :attr:`required_config_keys` to this class so the registry cannot
        drift from the stage's actual config surface.
    dependencies : tuple[str, ...]
        Names of stages whose outputs this stage reads. Forms the file-
        contract DAG edges used by the GUI run-graph.
    inputs : tuple[str, ...]
        Dotted YAML keys whose values are paths consumed by this stage.
    outputs : tuple[str, ...]
        Dotted YAML keys whose values are paths written by this stage.
    report_basename : str | None
        Filename of the JSON report this stage writes under
        ``runs/<run_id>/reports/``, when applicable. Used by the artifact
        aggregator to discover stage outputs.
    capabilities : frozenset[str]
        Subset of :class:`StageCapability` constants for this stage.
    """

    name: str
    order: int
    title: str
    description: str
    cli_module: str
    callable_name: str
    schema_class: type
    dependencies: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    report_basename: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)

    # ---- methods ---------------------------------------------------------

    def required_config_keys(self) -> tuple[str, ...]:
        """Forward to the typed schema view so the list cannot drift."""
        return tuple(self.schema_class.required_config_keys())

    def callable(self) -> Callable[..., Any]:
        """Lazily import and return the in-process entry-point callable.

        Importing ``cli_module`` only happens when the caller actually wants
        to run the stage in-process. The GUI run-control backend will use
        this; the existing ``python3 -m`` flow will not.
        """
        try:
            module = importlib.import_module(self.cli_module)
        except ImportError as exc:
            raise RegistryError(
                f"Stage {self.name!r} cli_module {self.cli_module!r} could not be imported: {exc}"
            ) from exc
        try:
            fn = getattr(module, self.callable_name)
        except AttributeError as exc:
            raise RegistryError(
                f"Stage {self.name!r} cli_module {self.cli_module!r} has no callable "
                f"named {self.callable_name!r}"
            ) from exc
        if not callable(fn):
            raise RegistryError(
                f"Stage {self.name!r} cli_module {self.cli_module!r}.{self.callable_name} is not callable"
            )
        return fn

    def cli_invocation(self, config_path: str) -> tuple[str, ...]:
        """Return the canonical ``python3 -m <module> --config <path>`` argv.

        Returned as a tuple so callers can pass it directly to
        :func:`subprocess.run` without ambiguity about quoting.
        """
        return ("python3", "-m", self.cli_module, "--config", str(config_path))

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view; the GUI / API layer can serve this directly."""
        return {
            "name": self.name,
            "order": self.order,
            "title": self.title,
            "description": self.description,
            "cli_module": self.cli_module,
            "callable_name": self.callable_name,
            "dependencies": list(self.dependencies),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "required_config_keys": list(self.required_config_keys()),
            "report_basename": self.report_basename,
            "capabilities": sorted(self.capabilities),
        }


# ---------------------------------------------------------------------------
# Registry container.
#
# Implemented as an ordered-dict wrapper rather than a bare dict so iteration
# order matches descriptor.order. This is the order the GUI's pipeline view
# will display, and it is the order documented in run_pipeline_plan.py.
# ---------------------------------------------------------------------------


class StageRegistry:
    """Ordered, queryable collection of :class:`StageDescriptor`."""

    def __init__(self, descriptors: Sequence[StageDescriptor] = ()) -> None:
        self._by_name: "OrderedDict[str, StageDescriptor]" = OrderedDict()
        for d in sorted(descriptors, key=lambda x: (x.order, x.name)):
            self.register(d)

    # ---- mutation -------------------------------------------------------

    def register(self, descriptor: StageDescriptor) -> None:
        """Add a descriptor to the registry. Raises on duplicate name."""
        if descriptor.name in self._by_name:
            raise RegistryError(f"Stage already registered: {descriptor.name!r}")
        self._validate_capabilities(descriptor)
        self._by_name[descriptor.name] = descriptor

    # ---- lookup --------------------------------------------------------

    def get(self, name: str) -> StageDescriptor:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise RegistryError(f"Unknown stage name: {name!r}") from exc

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __iter__(self) -> Iterator[StageDescriptor]:
        # Iteration is in registration / insertion order. Because
        # ``__init__`` sorts by ``(order, name)``, this is also the
        # canonical pipeline order.
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name.keys())

    def as_mapping(self) -> Mapping[str, StageDescriptor]:
        """Read-only view of the registry as a mapping."""
        return dict(self._by_name)

    # ---- structural checks ----------------------------------------------

    def validate_dependencies(self) -> None:
        """Ensure every declared dependency points at a registered stage.

        Raises :class:`RegistryError` listing all violations rather than
        bailing on the first, so a misconfigured registry surfaces every
        problem in one go.
        """
        violations: list[str] = []
        for d in self:
            for dep in d.dependencies:
                if dep not in self._by_name:
                    violations.append(f"{d.name!r} depends on unknown stage {dep!r}")
        if violations:
            raise RegistryError("Registry dependency check failed:\n  " + "\n  ".join(violations))

    def topological_order(self) -> tuple[StageDescriptor, ...]:
        """Return descriptors in a topologically-valid order.

        The canonical ``order`` field is the topological ordering of the
        file-contract DAG. This method sorts by ``(order, name)`` and then
        verifies that every dependency comes earlier in that sorted view.
        If a stage's dependency has a *higher or equal* ``order`` than the
        stage itself, :class:`RegistryError` is raised so the
        inconsistency is surfaced rather than silently masked by
        registration order.
        """
        ordered: tuple[StageDescriptor, ...] = tuple(sorted(self, key=lambda d: (d.order, d.name)))
        index_of = {d.name: i for i, d in enumerate(ordered)}
        for d in ordered:
            for dep in d.dependencies:
                if dep not in index_of:
                    raise RegistryError(
                        f"Stage {d.name!r} depends on unknown stage {dep!r}"
                    )
                if index_of[dep] >= index_of[d.name]:
                    raise RegistryError(
                        f"Stage order violates DAG: {d.name!r} (order={d.order}) depends on "
                        f"{dep!r} (order={self.get(dep).order})"
                    )
        return ordered

    def to_dict(self) -> list[dict[str, Any]]:
        """JSON-serializable list-of-dicts view of every descriptor."""
        return [d.to_dict() for d in self]

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _validate_capabilities(descriptor: StageDescriptor) -> None:
        unknown = descriptor.capabilities - StageCapability.all()
        if unknown:
            raise RegistryError(
                f"Stage {descriptor.name!r} has unknown capabilities: {sorted(unknown)}"
            )


# ---------------------------------------------------------------------------
# Canonical registry construction.
#
# Each descriptor below documents a real shipping stage. The fields are
# populated by direct inspection of the run_*.py source so the registry
# matches what the stages actually do. Order numbers leave room (multiples
# of 10) so future stages can be inserted without renumbering.
#
# Sources for inputs/outputs:
#   - docs/data_contracts.md (Stages 1, 2 — formal contract)
#   - configs/site01.yaml (paths.* and inputs.* keys each stage references)
#   - the run_*.py files themselves (cfg_get / output_path call sites)
# ---------------------------------------------------------------------------


def build_default_registry() -> StageRegistry:
    """Construct and return the canonical :class:`StageRegistry`."""
    descriptors: tuple[StageDescriptor, ...] = (
        StageDescriptor(
            name="stage_01_ingest",
            order=10,
            title="Video ingest and normalization",
            description=(
                "Decode the raw video, normalize FPS and pixel format, extract per-frame "
                "quality metrics (sharpness, exposure, motion, novelty, feature density)."
            ),
            cli_module="pipeline.stage_01_ingest.run_ingest",
            callable_name="run_ingest",
            schema_class=Stage01IngestSchema,
            dependencies=(),
            inputs=("inputs.video",),
            outputs=("paths.normalized_video", "paths.metadata_json", "paths.quality_csv"),
            report_basename="stage_01_ingest_report.json",
            capabilities=frozenset({StageCapability.HEAVY}),
        ),
        StageDescriptor(
            name="stage_02_keyframes",
            order=20,
            title="Adaptive keyframe selection",
            description=(
                "Segment the normalized video, score frames using Stage 1 quality metrics, "
                "and write a manifest of selected keyframes plus a contact-sheet preview."
            ),
            cli_module="pipeline.stage_02_keyframes.select_keyframes",
            callable_name="run_keyframe_selection",
            schema_class=Stage02KeyframesSchema,
            dependencies=("stage_01_ingest",),
            inputs=("paths.normalized_video", "paths.quality_csv"),
            outputs=("paths.keyframes_dir", "paths.manifest_csv", "paths.contact_sheet"),
            report_basename="keyframe_summary.json",
            capabilities=frozenset(),
        ),
        StageDescriptor(
            name="stage_03_colmap",
            order=30,
            title="COLMAP sparse SfM",
            description=(
                "Run COLMAP feature extraction, matching, and incremental SfM on the "
                "selected keyframes; produces the sparse model and database."
            ),
            cli_module="pipeline.stage_03_colmap.run_sparse",
            callable_name="run_sparse",
            schema_class=Stage03ColmapSchema,
            dependencies=("stage_02_keyframes",),
            inputs=("paths.keyframes_dir", "paths.manifest_csv"),
            outputs=("paths.colmap_db", "paths.sparse_dir"),
            report_basename="sparse_stats.json",
            capabilities=frozenset({StageCapability.HEAVY, StageCapability.SERVER_REQUIRED}),
        ),
        StageDescriptor(
            name="stage_04_refinement",
            order=40,
            title="Sparse refinement (bundle adjustment)",
            description=(
                "Run COLMAP final bundle adjustment on the Stage 3 sparse model; writes "
                "a refined sparse model used by Stage 5 and Stage 6."
            ),
            cli_module="pipeline.stage_04_refinement.run_refinement",
            callable_name="run_refinement",
            schema_class=Stage04RefinementSchema,
            dependencies=("stage_03_colmap",),
            inputs=("paths.sparse_dir",),
            outputs=("paths.sparse_refined_dir",),
            report_basename="refinement_stats.json",
            capabilities=frozenset({StageCapability.HEAVY, StageCapability.SERVER_REQUIRED}),
        ),
        StageDescriptor(
            name="stage_04_5_cams_gs",
            order=45,
            title="CAMS-GS / 3DGS dataset preparation",
            description=(
                "Optional: prepare a Nerfstudio/Splatfacto-compatible dataset from the "
                "refined sparse model. Visualization-only; not part of the metric-truth path."
            ),
            cli_module="pipeline.stage_04_5_cams_gs.run_cams_gs_prepare",
            callable_name="run_cams_gs_prepare",
            schema_class=Stage04RefinementSchema,
            dependencies=("stage_04_refinement",),
            inputs=("paths.sparse_refined_dir",),
            outputs=(),
            report_basename="cams_gs_prepare_summary.json",
            capabilities=frozenset({StageCapability.OPTIONAL, StageCapability.STUB_OR_PARTIAL}),
        ),
        StageDescriptor(
            name="stage_05_dense",
            order=50,
            title="COLMAP dense MVS",
            description=(
                "Run COLMAP image_undistorter, patch_match_stereo, and stereo_fusion on "
                "the refined sparse model; produces the fused dense point cloud."
            ),
            cli_module="pipeline.stage_05_dense.run_dense",
            callable_name="run_dense",
            schema_class=Stage05DenseSchema,
            dependencies=("stage_04_refinement",),
            inputs=("paths.sparse_refined_dir",),
            outputs=("paths.dense_workspace", "paths.fused_ply"),
            report_basename="dense_summary.json",
            capabilities=frozenset({StageCapability.HEAVY, StageCapability.SERVER_REQUIRED}),
        ),
        StageDescriptor(
            name="stage_06_da3_assist",
            order=60,
            title="DA3 monocular depth assistance",
            description=(
                "Conditional: when dense coverage is poor, ingest precomputed monocular "
                "depth maps, align them to the COLMAP frame with a scale-only RANSAC, "
                "and fuse them into a depth-assisted point cloud. The depth model itself "
                "is provided externally (precomputed npy files or external_command)."
            ),
            cli_module="pipeline.stage_06_da3_assist.run_da3_assist",
            callable_name="run_da3_assist",
            schema_class=Stage06DA3Schema,
            dependencies=("stage_05_dense",),
            inputs=("paths.fused_ply", "paths.sparse_refined_dir"),
            outputs=("paths.da3_dir",),
            report_basename="da3_summary.json",
            capabilities=frozenset({StageCapability.STUB_OR_PARTIAL}),
        ),
        StageDescriptor(
            name="stage_07_cleanup",
            order=70,
            title="Point cloud cleanup and meshing",
            description=(
                "Open3D-based cleanup: voxel downsample, statistical and radius outlier "
                "rejection, optional semantic HSV filter, plane RANSAC, and ball-pivoting "
                "or Poisson meshing."
            ),
            cli_module="pipeline.stage_07_cleanup.run_cleanup",
            callable_name="run_cleanup",
            schema_class=Stage07CleanupSchema,
            dependencies=("stage_05_dense",),
            inputs=("paths.fused_ply", "paths.da3_dir"),
            outputs=("paths.clean_dir",),
            report_basename="cleanup_summary.json",
            capabilities=frozenset({StageCapability.HEAVY, StageCapability.SERVER_REQUIRED}),
        ),
        StageDescriptor(
            name="stage_07_5_vlm_qa",
            order=75,
            title="Pre-BIM visual / VLM QA",
            description=(
                "Render evidence views of the cleaned cloud and feed them to a visual-QA "
                "step. Default provider is a deterministic mock; real VLM providers are "
                "swappable via the plugin layer."
            ),
            cli_module="pipeline.stage_07_5_vlm_qa.run_vlm_qa",
            callable_name="run_vlm_qa",
            schema_class=Stage07CleanupSchema,
            dependencies=("stage_07_cleanup",),
            inputs=("paths.clean_dir",),
            outputs=(),
            report_basename="vlm_qa_summary.json",
            capabilities=frozenset({StageCapability.OPTIONAL, StageCapability.STUB_OR_PARTIAL}),
        ),
        StageDescriptor(
            name="stage_07_6_viewer_export",
            order=76,
            title="Viewer artifact export",
            description=(
                "Bundle the artifact set produced by upstream stages into a static "
                "viewer index for download."
            ),
            cli_module="pipeline.stage_07_6_viewer_export.run_viewer_export",
            callable_name="run_viewer_export",
            schema_class=Stage07CleanupSchema,
            dependencies=("stage_07_cleanup",),
            inputs=("paths.clean_dir",),
            outputs=(),
            report_basename="viewer_export_summary.json",
            capabilities=frozenset({StageCapability.OPTIONAL}),
        ),
        StageDescriptor(
            name="stage_07_7_cams_gs_evidence",
            order=77,
            title="3DGS evidence package",
            description=(
                "Wrap the Stage 4.5 dataset/training status into a viewer-friendly "
                "evidence package. Visualization-only; not metric truth."
            ),
            cli_module="pipeline.stage_07_7_cams_gs_evidence.run_cams_gs_evidence",
            callable_name="run_cams_gs_evidence",
            schema_class=Stage07CleanupSchema,
            dependencies=("stage_04_5_cams_gs",),
            inputs=(),
            outputs=(),
            report_basename="cams_gs_evidence_summary.json",
            capabilities=frozenset({StageCapability.OPTIONAL, StageCapability.STUB_OR_PARTIAL}),
        ),
        StageDescriptor(
            name="stage_08_metric_alignment",
            order=80,
            title="Metric anchor alignment",
            description=(
                "Estimate a Sim(3) scan-to-BIM transform from CSV-supplied metric anchors "
                "and known distances using closed-form Umeyama plus a 3-of-N RANSAC. "
                "Produces the initial transform consumed by Stage 8 BIM registration."
            ),
            cli_module="pipeline.stage_08_bim_eval.run_metric_alignment",
            callable_name="main",
            schema_class=Stage08BimEvalSchema,
            dependencies=(),
            inputs=(),
            outputs=("paths.bim_aligned_dir",),
            report_basename="metric_alignment_report.json",
            capabilities=frozenset({StageCapability.SERVER_REQUIRED}),
        ),
        StageDescriptor(
            name="stage_08_bim_registration",
            order=81,
            title="BIM extraction and registration",
            description=(
                "Extract IFC geometry, FPFH+RANSAC coarse registration, staged "
                "point-to-point/point-to-plane ICP. Emits the scan_aligned point cloud "
                "and the registration_report.json consumed by Stage 9."
            ),
            cli_module="pipeline.stage_08_bim_eval.run_registration",
            callable_name="run_registration",
            schema_class=Stage08BimEvalSchema,
            dependencies=("stage_07_cleanup",),
            inputs=("paths.clean_dir", "inputs.ifc"),
            outputs=("paths.bim_aligned_dir",),
            report_basename="registration_report.json",
            capabilities=frozenset({StageCapability.HEAVY, StageCapability.SERVER_REQUIRED}),
        ),
        StageDescriptor(
            name="stage_09_progress",
            order=90,
            title="Progress and deviation metrics",
            description=(
                "Compute per-element accuracy, completeness, F-score @ tau (with Wilson "
                "and bootstrap confidence intervals), per-activity rollups, and the "
                "deviation map. Produces the canonical progress_summary.json."
            ),
            cli_module="pipeline.stage_09_progress.run_progress",
            callable_name="run_progress",
            schema_class=Stage09ProgressSchema,
            dependencies=("stage_08_bim_registration",),
            inputs=("paths.bim_aligned_dir", "inputs.schedule"),
            outputs=("paths.metrics_dir",),
            report_basename="progress_summary.json",
            capabilities=frozenset({StageCapability.SERVER_REQUIRED}),
        ),
        StageDescriptor(
            name="stage_10_copilot",
            order=100,
            title="Evidence-only VLM copilot",
            description=(
                "Build an evidence package over Stage 8 / Stage 9 artifacts and answer a "
                "question via a local VLM (Ollama / OpenAI-compatible) or a deterministic "
                "mock. Answers are post-validated against deterministic quality gates."
            ),
            cli_module="pipeline.stage_10_copilot.run_ask",
            callable_name="main",
            schema_class=Stage10CopilotSchema,
            dependencies=("stage_09_progress",),
            inputs=("paths.metrics_dir",),
            outputs=(),
            report_basename=None,  # Stage 10 writes copilot/<...>/latest_evidence_package.json, not a stage report.
            capabilities=frozenset({StageCapability.OPTIONAL}),
        ),
    )

    registry = StageRegistry(descriptors)
    registry.validate_dependencies()
    # Confirm the order numbers are themselves a topological ordering. This
    # is an internal consistency check; if it ever fires, the descriptor
    # tuple above has been edited inconsistently.
    registry.topological_order()
    return registry


# Module-level singleton. Build at import; if the build fails, importing
# pipeline.common.registry fails — which is what we want, because nothing
# downstream can use a half-built registry.
STAGE_REGISTRY: StageRegistry = build_default_registry()
