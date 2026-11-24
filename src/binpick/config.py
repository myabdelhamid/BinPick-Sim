"""Load and validate the single YAML config into frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _range(name: str, r) -> tuple[float, float]:
    lo, hi = (float(x) for x in r)
    if not 0 < lo <= hi:
        raise ValueError(f"{name}: need 0 < min <= max, got {r}")
    return lo, hi


def _positive(obj, *names: str) -> None:
    for n in names:
        if getattr(obj, n) <= 0:
            raise ValueError(f"{n}: must be > 0, got {getattr(obj, n)}")


@dataclass(frozen=True)
class MaterialConfig:
    prob: float
    density: float
    friction: float
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class PartsConfig:
    n_boxes: int
    n_cylinders: int
    box_footprint: tuple[float, float]
    box_height: tuple[float, float]
    cyl_diameter: tuple[float, float]
    cyl_length: tuple[float, float]
    mass_min: float
    mass_max: float
    materials: dict[str, MaterialConfig]

    def __post_init__(self):
        for n in ("box_footprint", "box_height", "cyl_diameter", "cyl_length"):
            object.__setattr__(self, n, _range(n, getattr(self, n)))
        if self.n_boxes < 0 or self.n_cylinders < 0:
            raise ValueError("n_boxes/n_cylinders: must be >= 0")
        if not 0 < self.mass_min < self.mass_max:
            raise ValueError(f"mass_min: need 0 < mass_min < mass_max, got {self.mass_min}, {self.mass_max}")
        total = sum(m.prob for m in self.materials.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"materials: probabilities must sum to 1, got {total}")


@dataclass(frozen=True)
class CupConfig:
    diameter: float
    length: float
    rated_hold_kg: float
    tool_length: float

    def __post_init__(self):
        _positive(self, "diameter", "length", "rated_hold_kg", "tool_length")


@dataclass(frozen=True)
class BinConfig:
    inner: tuple[float, float, float]
    wall: float

    def __post_init__(self):
        object.__setattr__(self, "inner", tuple(float(x) for x in self.inner))
        if len(self.inner) != 3 or min(self.inner) <= 0:
            raise ValueError(f"inner: need 3 positive dims, got {self.inner}")
        _positive(self, "wall")


@dataclass(frozen=True)
class TrayConfig:
    rows: int
    cols: int
    slot: float
    wall_height: float

    def __post_init__(self):
        _positive(self, "rows", "cols", "slot", "wall_height")


@dataclass(frozen=True)
class CamerasConfig:
    width: int
    height: int
    bin_fovy: float
    bin_height_above_rim: float
    tray_fovy: float
    tray_height: float

    def __post_init__(self):
        _positive(self, *(f.name for f in fields(self)))


@dataclass(frozen=True)
class ShakerConfig:
    amplitude: float
    freq_hz: float
    duration_s: float

    def __post_init__(self):
        _positive(self, "amplitude", "freq_hz", "duration_s")


@dataclass(frozen=True)
class SettleConfig:
    lin_tol: float
    ang_tol: float
    drop_timeout_s: float
    max_redrops: int
    drop_height_above_rim: float
    inset: float

    def __post_init__(self):
        _positive(self, "lin_tol", "ang_tol", "drop_timeout_s")
        if self.max_redrops < 0:
            raise ValueError("max_redrops: must be >= 0")


@dataclass(frozen=True)
class LayoutConfig:
    robot_base_pos: tuple[float, float, float]
    bin_center_xy: tuple[float, float]
    tray_center_xy: tuple[float, float]
    table_height: float

    def __post_init__(self):
        for n in ("robot_base_pos", "bin_center_xy", "tray_center_xy"):
            object.__setattr__(self, n, tuple(float(x) for x in getattr(self, n)))


@dataclass(frozen=True)
class SimConfig:
    timestep: float
    noslip_iterations: int
    part_solref_timeconst: float
    part_condim: int

    def __post_init__(self):
        _positive(self, "timestep")
        if self.part_solref_timeconst < 2 * self.timestep:
            raise ValueError("part_solref_timeconst: must be >= 2 * timestep for stable contacts")
        if self.part_condim not in (1, 3, 4, 6):
            raise ValueError(f"part_condim: must be 1, 3, 4 or 6, got {self.part_condim}")


@dataclass(frozen=True)
class MotionConfig:
    max_joint_vel: tuple[float, ...]
    max_joint_acc: float
    max_joint_acc_loaded: float
    max_cart_vel: float
    max_cart_acc: float
    max_cart_acc_empty: float
    cart_step: float
    pregrasp_dist: float
    approach_overtravel: float
    lift_dist: float
    clear_above_rim: float
    place_clearance: float
    retreat_dist: float
    settle_after_move_s: float

    def __post_init__(self):
        object.__setattr__(self, "max_joint_vel", tuple(float(v) for v in self.max_joint_vel))
        if len(self.max_joint_vel) != 7 or min(self.max_joint_vel) <= 0:
            raise ValueError("max_joint_vel: need 7 positive values")
        _positive(self, "max_joint_acc", "max_joint_acc_loaded", "max_cart_vel", "max_cart_acc",
                  "max_cart_acc_empty", "cart_step", "pregrasp_dist", "lift_dist", "retreat_dist")


@dataclass(frozen=True)
class TimingConfig:
    capture_s: float
    plan_s: float
    vacuum_build_s: float
    release_s: float
    verify_s: float


@dataclass(frozen=True)
class SuctionConfig:
    lip_points: int
    gap_ok: float
    gap_max: float
    contact_dist: float
    margin: float
    travel: float
    curve_radius_min: float
    curve_limit_q: float
    pull_force_n: float
    peel_lever_ratio: float
    shear_friction: float
    bellows_timeconst: float
    break_filter_s: float
    vacuum_kpa: float
    sealed_threshold_kpa: float
    lost_threshold_kpa: float
    lost_debounce_polls: int
    sensor_noise_kpa: float

    def __post_init__(self):
        _positive(self, "lip_points", "gap_max", "contact_dist", "travel", "curve_radius_min", "pull_force_n",
                  "vacuum_kpa", "sealed_threshold_kpa")
        if not 0 < self.curve_limit_q <= 1:
            raise ValueError("curve_limit_q: must be in (0, 1]")
        if not 0 <= self.gap_ok < self.gap_max:
            raise ValueError(f"gap_ok: need 0 <= gap_ok < gap_max, got {self.gap_ok}, {self.gap_max}")
        if self.sealed_threshold_kpa >= self.vacuum_kpa:
            raise ValueError("sealed_threshold_kpa: must be below vacuum_kpa")
        if not 0 < self.lost_threshold_kpa <= self.sealed_threshold_kpa:
            raise ValueError("lost_threshold_kpa: need 0 < lost_threshold_kpa <= sealed_threshold_kpa")


@dataclass(frozen=True)
class PerceptionConfig:
    cell: float
    min_part_height: float
    wall_margin: float
    max_slope: float
    crease_deg: float
    min_surface_mm2: float
    empty_mm2: float

    def __post_init__(self):
        _positive(self, "cell", "min_part_height", "max_slope", "crease_deg", "min_surface_mm2", "empty_mm2")


@dataclass(frozen=True)
class PlannerConfig:
    candidate_step: float
    edge_margin: float
    max_tilt_deg: float
    wall_clearance: float
    obstacle_height: float
    obstacle_clearance: float
    flat_drop: float
    per_surface: int
    w_height: float
    w_seal: float
    w_wall: float
    w_center: float
    wall_saturation: float

    def __post_init__(self):
        _positive(self, "candidate_step", "max_tilt_deg", "per_surface", "wall_saturation")


@dataclass(frozen=True)
class Config:
    seed: int
    parts: PartsConfig
    cup: CupConfig
    bin: BinConfig
    tray: TrayConfig
    cameras: CamerasConfig
    shaker: ShakerConfig
    settle: SettleConfig
    layout: LayoutConfig
    sim: SimConfig
    motion: MotionConfig
    timing: TimingConfig
    suction: SuctionConfig
    perception: PerceptionConfig
    planner: PlannerConfig


def load_config(path: str | Path | None = None) -> Config:
    raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text())
    p = dict(raw["parts"])
    p["materials"] = {
        k: MaterialConfig(prob=float(v["prob"]), density=float(v["density"]),
                          friction=float(v["friction"]), rgba=tuple(v["rgba"]))
        for k, v in p["materials"].items()
    }
    return Config(
        seed=int(raw["seed"]),
        parts=PartsConfig(**p),
        cup=CupConfig(**raw["cup"]),
        bin=BinConfig(**raw["bin"]),
        tray=TrayConfig(**raw["tray"]),
        cameras=CamerasConfig(**raw["cameras"]),
        shaker=ShakerConfig(**raw["shaker"]),
        settle=SettleConfig(**raw["settle"]),
        layout=LayoutConfig(**raw["layout"]),
        sim=SimConfig(**raw["sim"]),
        motion=MotionConfig(**raw["motion"]),
        timing=TimingConfig(**raw["timing"]),
        suction=SuctionConfig(**raw["suction"]),
        perception=PerceptionConfig(**raw["perception"]),
        planner=PlannerConfig(**raw["planner"]),
    )
