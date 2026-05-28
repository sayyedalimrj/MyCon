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
