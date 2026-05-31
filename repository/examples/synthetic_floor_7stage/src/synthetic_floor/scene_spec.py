"""Scene specification loader.

Reads ``config/scene.yaml`` into a small set of typed dataclasses so the
rest of the pipeline can rely on a fixed shape. Keeping this layer
separate makes it trivial to swap the YAML file for a Python dict in
unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


# ---------------------------------------------------------------------
# Dataclasses (deliberately simple; not tied to YAML schema details)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    bays_x: int
    bays_y: int
    column_size_m: float


@dataclass(frozen=True)
class Opening:
    width_m: float
    height_m: float
    sill_height_m: float = 0.0  # only used by windows


@dataclass(frozen=True)
class RoomSpec:
    id: str
    name: str
    sw: tuple[float, float]
    ne: tuple[float, float]


@dataclass(frozen=True)
class DoorSpec:
    id: str
    from_: str
    to: str
    side: str
    offset_m: float


@dataclass(frozen=True)
class WindowSpec:
    id: str
    facade: str  # "south" | "north" | "west" | "east"
    offset_m: float


@dataclass(frozen=True)
class FloorSpec:
    length_m: float
    width_m: float
    height_m: float
    slab_thickness_m: float
    exterior_wall_thickness_m: float
    interior_wall_thickness_m: float
    grid: Grid
    door_opening: Opening
    window_opening: Opening
    rooms: Sequence[RoomSpec]
    doors: Sequence[DoorSpec]
    windows: Sequence[WindowSpec]
    # --- Optional detailed-construction features (default off so the
    #     original example is byte-for-byte unchanged) -----------------
    with_site: bool = False               # exterior ground plane around the building
    site_margin_m: float = 8.0            # how far the ground extends past the walls
    with_foundation: bool = False         # pad footings / pedestals under columns
    foundation_depth_m: float = 0.6       # footing thickness below the slab
    foundation_pad_m: float = 1.2         # square footing footprint
    with_beams: bool = False              # concrete beams spanning columns under the slab
    beam_depth_m: float = 0.45            # beam height
    beam_width_m: float = 0.30            # beam width
    with_window_frames: bool = False      # visible frame around each window pane
    window_frame_depth_m: float = 0.12    # frame profile depth/width
    with_floor_finish: bool = False       # final floor finish layer (tile/wood/epoxy)
    floor_finish_thickness_m: float = 0.03
    floor_finish_type: str = "tile"       # informational; drives default material


@dataclass(frozen=True)
class StageSpec:
    id: int
    name: str
    description: str
    elements: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class HandJitterSpec:
    translation_amplitude_m: float
    rotation_amplitude_deg: float
    breathing_period_s: float
    walking_period_s: float
    walking_pitch_amp_deg: float
    walking_yaw_amp_deg: float
    walking_z_amp_m: float


@dataclass(frozen=True)
class ExposureSpec:
    target_brightness: float
    adapt_speed: float
    max_ev_change_per_frame: float


@dataclass(frozen=True)
class NoiseSpec:
    iso_equivalent: float
    read_noise_sigma: float
    photon_scale: float


@dataclass(frozen=True)
class MotionBlurSpec:
    shutter_fraction: float
    samples: int


@dataclass(frozen=True)
class MotionSpec:
    """Procedural human-operator camera-motion parameters.

    Every value is read from ``camera.motion`` in the YAML (with the
    defaults below) so the trajectory can be tuned per scene without
    touching code.
    """
    # --- pathfinding / collision ---
    coverage_lane_spacing_m: float = 3.0     # serpentine lane spacing (full coverage)
    collision_margin_m: float = 0.65         # keep at least this far from walls
    column_clearance_m: float = 0.45         # extra radius kept around columns
    path_smoothness: float = 0.5             # 0=polyline, 1=very rounded (Catmull-Rom)
    # --- gaze / look-around (decoupled from translation) ---
    scan_yaw_amplitude_deg: float = 72.0     # how far the gaze sweeps left/right
    scan_period_sec: float = 8.5             # period of the slow horizontal scan
    turn_around_interval_sec: float = 13.0   # how often to look ~180 deg behind
    turn_around_duration_sec: float = 3.2    # eased duration of a look-behind
    gaze_inertia_tau_sec: float = 0.55       # smoothing time-constant (physical inertia)
    focus_distance_m: float = 4.5            # look-at distance along the gaze ray
    pitch_scan_amplitude_deg: float = 14.0   # gentle up/down gaze drift
    pitch_scan_period_sec: float = 6.0
    # --- 6-DOF verticality (crouch to floor / rise to ceiling) ---
    vertical_inspect_enabled: bool = True
    crouch_height_m: float = 0.80            # lowest eye height when inspecting floor
    rise_height_m: float = 2.15              # highest eye height when inspecting ceiling
    vertical_inspect_interval_sec: float = 11.0
    vertical_inspect_duration_sec: float = 4.2
    inspect_pitch_deg: float = 34.0          # look down/up during crouch/rise

    @classmethod
    def from_raw(cls, raw: "Mapping[str, Any] | None") -> "MotionSpec":
        raw = dict(raw or {})
        d = cls()  # defaults

        def g(k, default, cast=float):
            return cast(raw[k]) if k in raw and raw[k] is not None else default

        return cls(
            coverage_lane_spacing_m=g("coverage_lane_spacing_m", d.coverage_lane_spacing_m),
            collision_margin_m=g("collision_margin_m", d.collision_margin_m),
            column_clearance_m=g("column_clearance_m", d.column_clearance_m),
            path_smoothness=g("path_smoothness", d.path_smoothness),
            scan_yaw_amplitude_deg=g("scan_yaw_amplitude_deg", d.scan_yaw_amplitude_deg),
            scan_period_sec=g("scan_period_sec", d.scan_period_sec),
            turn_around_interval_sec=g("turn_around_interval_sec", d.turn_around_interval_sec),
            turn_around_duration_sec=g("turn_around_duration_sec", d.turn_around_duration_sec),
            gaze_inertia_tau_sec=g("gaze_inertia_tau_sec", d.gaze_inertia_tau_sec),
            focus_distance_m=g("focus_distance_m", d.focus_distance_m),
            pitch_scan_amplitude_deg=g("pitch_scan_amplitude_deg", d.pitch_scan_amplitude_deg),
            pitch_scan_period_sec=g("pitch_scan_period_sec", d.pitch_scan_period_sec),
            vertical_inspect_enabled=g("vertical_inspect_enabled", d.vertical_inspect_enabled, bool),
            crouch_height_m=g("crouch_height_m", d.crouch_height_m),
            rise_height_m=g("rise_height_m", d.rise_height_m),
            vertical_inspect_interval_sec=g("vertical_inspect_interval_sec", d.vertical_inspect_interval_sec),
            vertical_inspect_duration_sec=g("vertical_inspect_duration_sec", d.vertical_inspect_duration_sec),
            inspect_pitch_deg=g("inspect_pitch_deg", d.inspect_pitch_deg),
        )


@dataclass(frozen=True)
class CameraSpec:
    width_px: int
    height_px: int
    fps: int
    horizontal_fov_deg: float
    k1: float
    k2: float
    hold_height_m: float
    duration_per_stage_sec: float
    walk_speed_m_s: float
    look_at_height_m: float
    hand_jitter: HandJitterSpec
    exposure: ExposureSpec
    noise: NoiseSpec
    motion_blur: MotionBlurSpec
    rolling_shutter_row_delay_sec: float
    motion: MotionSpec = field(default_factory=MotionSpec)

    @property
    def aspect(self) -> float:
        return self.width_px / max(1, self.height_px)


@dataclass(frozen=True)
class RendererSpec:
    backend: str
    ambient_light: float
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_color: tuple[float, float, float]
    sky_color: tuple[float, float, float]
    shadow_softness: float
    vignette_strength: float
    emit_material_samples: bool
    # --- physically-based lighting / tone-mapping (GPU Blender path) ---
    world_strength: float = 0.35       # ambient sky fill; low => deep shadows, not flat
    sun_energy: float = 2.8            # key light intensity
    exposure: float = -0.20            # EV bias applied after Filmic
    view_transform: str = "Filmic"     # tone-mapping operator
    view_look: str = "Medium High Contrast"
    use_fast_gi: bool = True           # Cycles fast-GI / AO approximation
    ao_factor: float = 0.55            # ambient-occlusion strength (contact shadows)
    ao_distance: float = 1.2           # AO sampling distance (m)
    # --- parallel rendering ---
    parallel_workers_count: int = 1    # concurrent Blender processes per stage


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    bim: Path
    mesh: Path
    renders: Path
    video: Path
    depth: Path
    segmentation: Path
    camera: Path
    manifests: Path
    logs: Path

    @classmethod
    def resolve(cls, base: Path, raw: Mapping[str, str]) -> "OutputPaths":
        root = (base / raw["root"]).resolve()
        return cls(
            root=root,
            bim=root / raw["bim_dir"],
            mesh=root / raw["mesh_dir"],
            renders=root / raw["renders_dir"],
            video=root / raw["video_dir"],
            depth=root / raw["depth_dir"],
            segmentation=root / raw["segmentation_dir"],
            camera=root / raw["camera_dir"],
            manifests=root / raw["manifests_dir"],
            logs=root / raw["logs_dir"],
        )

    def ensure(self) -> None:
        for p in (
            self.root, self.bim, self.mesh, self.renders, self.video,
            self.depth, self.segmentation, self.camera, self.manifests,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SceneSpec:
    project_name: str
    run_id: str
    description: str
    random_seed: int
    floor: FloorSpec
    stages: Sequence[StageSpec]
    camera: CameraSpec
    renderer: RendererSpec
    output: OutputPaths
    config_path: Path
    config_raw: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------


def load_scene_spec(config_path: Path, *, base_dir: Path | None = None) -> SceneSpec:
    """Load and validate the master YAML configuration.

    Parameters
    ----------
    config_path:
        Path to the YAML file (typically
        ``examples/synthetic_floor_7stage/config/scene.yaml``).
    base_dir:
        Directory used to resolve relative output paths. Defaults to
        the parent of ``config_path`` (i.e. the example folder).
    """
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw)!r}")

    base = base_dir.resolve() if base_dir else config_path.parent.parent

    project = raw["project"]
    floor_raw = raw["floor"]
    grid = Grid(
        bays_x=int(floor_raw["grid"]["bays_x"]),
        bays_y=int(floor_raw["grid"]["bays_y"]),
        column_size_m=float(floor_raw["grid"]["column_size_m"]),
    )
    door_op = Opening(
        width_m=float(floor_raw["door_opening"]["width_m"]),
        height_m=float(floor_raw["door_opening"]["height_m"]),
    )
    win_op = Opening(
        width_m=float(floor_raw["window_opening"]["width_m"]),
        height_m=float(floor_raw["window_opening"]["height_m"]),
        sill_height_m=float(floor_raw["window_opening"]["sill_height_m"]),
    )
    rooms = tuple(
        RoomSpec(
            id=str(r["id"]),
            name=str(r["name"]),
            sw=(float(r["sw"][0]), float(r["sw"][1])),
            ne=(float(r["ne"][0]), float(r["ne"][1])),
        )
        for r in floor_raw["rooms"]
    )
    doors = tuple(
        DoorSpec(
            id=str(d["id"]),
            from_=str(d["from"]),
            to=str(d["to"]),
            side=str(d["side"]),
            offset_m=float(d["offset_m"]),
        )
        for d in floor_raw["doors"]
    )
    windows = tuple(
        WindowSpec(
            id=str(w["id"]),
            facade=str(w["facade"]),
            offset_m=float(w["offset_m"]),
        )
        for w in floor_raw["windows"]
    )
    floor = FloorSpec(
        length_m=float(floor_raw["length_m"]),
        width_m=float(floor_raw["width_m"]),
        height_m=float(floor_raw["height_m"]),
        slab_thickness_m=float(floor_raw["slab_thickness_m"]),
        exterior_wall_thickness_m=float(floor_raw["exterior_wall_thickness_m"]),
        interior_wall_thickness_m=float(floor_raw["interior_wall_thickness_m"]),
        grid=grid,
        door_opening=door_op,
        window_opening=win_op,
        rooms=rooms,
        doors=doors,
        windows=windows,
        with_site=bool(floor_raw.get("with_site", False)),
        site_margin_m=float(floor_raw.get("site_margin_m", 8.0)),
        with_foundation=bool(floor_raw.get("with_foundation", False)),
        foundation_depth_m=float(floor_raw.get("foundation_depth_m", 0.6)),
        foundation_pad_m=float(floor_raw.get("foundation_pad_m", 1.2)),
        with_beams=bool(floor_raw.get("with_beams", False)),
        beam_depth_m=float(floor_raw.get("beam_depth_m", 0.45)),
        beam_width_m=float(floor_raw.get("beam_width_m", 0.30)),
        with_window_frames=bool(floor_raw.get("with_window_frames", False)),
        window_frame_depth_m=float(floor_raw.get("window_frame_depth_m", 0.12)),
        with_floor_finish=bool(floor_raw.get("with_floor_finish", False)),
        floor_finish_thickness_m=float(floor_raw.get("floor_finish_thickness_m", 0.03)),
        floor_finish_type=str(floor_raw.get("floor_finish_type", "tile")),
    )

    stages_raw = raw["stages"]
    if len(stages_raw) != 7:
        raise ValueError(f"Expected exactly 7 stages, got {len(stages_raw)}")
    stages = tuple(
        StageSpec(
            id=int(s["id"]),
            name=str(s["name"]),
            description=str(s["description"]),
            elements=dict(s["elements"]),
        )
        for s in stages_raw
    )

    cam_raw = raw["camera"]
    hand = HandJitterSpec(**cam_raw["hand_jitter"])
    exp = ExposureSpec(**cam_raw["exposure"])
    noise = NoiseSpec(**cam_raw["noise"])
    blur = MotionBlurSpec(**cam_raw["motion_blur"])
    cam = CameraSpec(
        width_px=int(cam_raw["width_px"]),
        height_px=int(cam_raw["height_px"]),
        fps=int(cam_raw["fps"]),
        horizontal_fov_deg=float(cam_raw["horizontal_fov_deg"]),
        k1=float(cam_raw["k1"]),
        k2=float(cam_raw["k2"]),
        hold_height_m=float(cam_raw["hold_height_m"]),
        duration_per_stage_sec=float(cam_raw["duration_per_stage_sec"]),
        walk_speed_m_s=float(cam_raw["walk_speed_m_s"]),
        look_at_height_m=float(cam_raw["look_at_height_m"]),
        hand_jitter=hand,
        exposure=exp,
        noise=noise,
        motion_blur=blur,
        rolling_shutter_row_delay_sec=float(cam_raw["rolling_shutter"]["row_delay_sec"]),
        motion=MotionSpec.from_raw(cam_raw.get("motion")),
    )

    rend_raw = raw["renderer"]
    renderer = RendererSpec(
        backend=str(rend_raw["backend"]),
        ambient_light=float(rend_raw["ambient_light"]),
        sun_azimuth_deg=float(rend_raw["sun_azimuth_deg"]),
        sun_elevation_deg=float(rend_raw["sun_elevation_deg"]),
        sun_color=tuple(float(c) for c in rend_raw["sun_color"]),
        sky_color=tuple(float(c) for c in rend_raw["sky_color"]),
        shadow_softness=float(rend_raw["shadow_softness"]),
        vignette_strength=float(rend_raw["vignette_strength"]),
        emit_material_samples=bool(rend_raw["emit_material_samples"]),
        world_strength=float(rend_raw.get("world_strength", 0.35)),
        sun_energy=float(rend_raw.get("sun_energy", 2.8)),
        exposure=float(rend_raw.get("exposure", -0.20)),
        view_transform=str(rend_raw.get("view_transform", "Filmic")),
        view_look=str(rend_raw.get("view_look", "Medium High Contrast")),
        use_fast_gi=bool(rend_raw.get("use_fast_gi", True)),
        ao_factor=float(rend_raw.get("ao_factor", 0.55)),
        ao_distance=float(rend_raw.get("ao_distance", 1.2)),
        parallel_workers_count=int(rend_raw.get("parallel_workers_count", 1)),
    )

    out_paths = OutputPaths.resolve(base, raw["output"])

    return SceneSpec(
        project_name=str(project["name"]),
        run_id=str(project["run_id"]),
        description=str(project["description"]),
        random_seed=int(project["random_seed"]),
        floor=floor,
        stages=stages,
        camera=cam,
        renderer=renderer,
        output=out_paths,
        config_path=config_path,
        config_raw=raw,
    )
