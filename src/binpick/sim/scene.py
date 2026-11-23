"""Build the cell with MjSpec: Panda + suction cup, bin on a shaker, tray, cameras, parts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from binpick.config import Config
from binpick.sim.parts import PartSpec

PANDA_XML = Path(__file__).resolve().parents[3] / "assets" / "franka_emika_panda" / "panda_nohand.xml"
PREFIX = "panda/"
G = 9.81
# Arm pose that keeps the arm clear of both cameras (joint1 turned toward -y).
PARK_Q = np.array([-1.5708, -0.3, 0.0, -2.2, 0.0, 1.9, 0.7854])
TRAY_BASE = 0.010      # tray plate thickness
DIVIDER_T = 0.003      # slot divider thickness
PARK_X0 = 3.0          # parts waiting to be dropped sit on the floor at x >= 3 m


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    cfg: Config
    parts: list[PartSpec]
    part_body_ids: list[int]
    part_qpos_adr: list[int]
    arm_joint_names: list[str]
    slot_centers: np.ndarray
    cup_site: str = PREFIX + "cup_tip"
    cup_geom: str = PREFIX + "cup"
    adhesion_actuator: str = "suction"
    shaker_actuators: tuple[str, str] = ("shaker_x", "shaker_y")
    bin_body: str = "bin"
    arm_actuator_names: list[str] = field(default_factory=lambda: [f"{PREFIX}actuator{i}" for i in range(1, 8)])

    def bin_inner_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame (min, max) corners of the bin's inner volume at the current shaker pose."""
        mujoco.mj_kinematics(self.model, self.data)
        c = self.data.body(self.bin_body).xpos
        ix, iy, iz = self.cfg.bin.inner
        w = self.cfg.bin.wall
        lo = np.array([c[0] - ix / 2, c[1] - iy / 2, c[2] + w])
        hi = np.array([c[0] + ix / 2, c[1] + iy / 2, c[2] + w + iz])
        return lo, hi

    def park_arm(self) -> None:
        for j, a, q in zip(self.arm_joint_names, self.arm_actuator_names, PARK_Q):
            self.data.joint(j).qpos[0] = q
            self.data.joint(j).qvel[0] = 0.0
            self.data.actuator(a).ctrl[0] = q
        mujoco.mj_forward(self.model, self.data)


def tray_slot_centers(cfg: Config) -> np.ndarray:
    """Slot centers on the tray surface, row-major (slot 0 = row 0, col 0). Cols run along x."""
    t, (cx, cy) = cfg.tray, cfg.layout.tray_center_xy
    z = cfg.layout.table_height + TRAY_BASE
    return np.array([
        [cx + (c - (t.cols - 1) / 2) * t.slot, cy + (r - (t.rows - 1) / 2) * t.slot, z]
        for r in range(t.rows) for c in range(t.cols)
    ])


def _look_at_xyaxes(pos, target) -> list[float]:
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return [*x, *y]


def _add_panda(spec: mujoco.MjSpec, cfg: Config) -> None:
    arm = mujoco.MjSpec.from_file(str(PANDA_XML))
    for k in list(arm.keys):
        arm.delete(k)
    for b in arm.bodies:  # the real Panda controller compensates the arm's own gravity
        b.gravcomp = 1.0
    flange = arm.body("attachment")
    tool = flange.add_body(name="suction_tool")
    tl, cl, r = cfg.cup.tool_length, cfg.cup.length, cfg.cup.diameter / 2
    tool.add_geom(name="tool_shaft", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, tl / 2, 0],
                  pos=[0, 0, tl / 2], rgba=[0.2, 0.2, 0.2, 1], mass=0.15)
    cup = tool.add_body(name="cup_body", pos=[0, 0, tl])
    cup.add_geom(name="cup", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, cl / 2, 0],
                 pos=[0, 0, cl / 2], rgba=[0.9, 0.5, 0.1, 1], mass=0.05)
    cup.add_site(name="cup_tip", pos=[0, 0, cl], size=[0.003, 0, 0])
    frame = spec.worldbody.add_frame(pos=list(cfg.layout.robot_base_pos))
    spec.attach(arm, prefix=PREFIX, frame=frame)
    act = spec.add_actuator(name="suction", target=PREFIX + "cup_body", trntype=mujoco.mjtTrn.mjTRN_BODY)
    act.set_to_adhesion(gain=cfg.cup.rated_hold_kg * G)
    act.ctrlrange = [0.0, 1.0]
    act.ctrllimited = 1


def _add_bin(spec: mujoco.MjSpec, cfg: Config) -> None:
    (bx, by), z0 = cfg.layout.bin_center_xy, cfg.layout.table_height
    ix, iy, iz = cfg.bin.inner
    w = cfg.bin.wall
    body = spec.worldbody.add_body(name="bin", pos=[bx, by, z0])
    rng = cfg.shaker.amplitude * 3
    for name, axis in (("shaker_x", [1, 0, 0]), ("shaker_y", [0, 1, 0])):
        body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis, range=[-rng, rng],
                       damping=50.0, armature=1.0)
    grey = [0.55, 0.57, 0.6, 1]
    ox, oy = ix / 2 + w / 2, iy / 2 + w / 2
    boxes = {
        "bin_floor": ([ix / 2 + w, iy / 2 + w, w / 2], [0, 0, w / 2]),
        "bin_wall_px": ([w / 2, iy / 2 + w, iz / 2], [ox, 0, w + iz / 2]),
        "bin_wall_nx": ([w / 2, iy / 2 + w, iz / 2], [-ox, 0, w + iz / 2]),
        "bin_wall_py": ([ix / 2, w / 2, iz / 2], [0, oy, w + iz / 2]),
        "bin_wall_ny": ([ix / 2, w / 2, iz / 2], [0, -oy, w + iz / 2]),
    }
    for name, (size, pos) in boxes.items():
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=grey, mass=2.0)
    for name in ("shaker_x", "shaker_y"):
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=20000.0, kv=1000.0)
        a.ctrlrange = [-rng, rng]
        a.ctrllimited = 1


def _add_tray(spec: mujoco.MjSpec, cfg: Config) -> None:
    t, (cx, cy), z0 = cfg.tray, cfg.layout.tray_center_xy, cfg.layout.table_height
    W, H = t.cols * t.slot, t.rows * t.slot
    body = spec.worldbody.add_body(name="tray", pos=[cx, cy, z0])
    body.add_geom(name="tray_base", type=mujoco.mjtGeom.mjGEOM_BOX, size=[W / 2, H / 2, TRAY_BASE / 2],
                  pos=[0, 0, TRAY_BASE / 2], rgba=[0.15, 0.35, 0.2, 1])
    hz, zc = t.wall_height / 2, TRAY_BASE + t.wall_height / 2
    for c in range(t.cols + 1):
        body.add_geom(name=f"tray_div_x{c}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[DIVIDER_T / 2, H / 2, hz],
                      pos=[-W / 2 + c * t.slot, 0, zc], rgba=[0.9, 0.9, 0.9, 1])
    for r in range(t.rows + 1):
        body.add_geom(name=f"tray_div_y{r}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[W / 2, DIVIDER_T / 2, hz],
                      pos=[0, -H / 2 + r * t.slot, zc], rgba=[0.9, 0.9, 0.9, 1])


def _add_cameras_and_lights(spec: mujoco.MjSpec, cfg: Config) -> None:
    (bx, by), z0 = cfg.layout.bin_center_xy, cfg.layout.table_height
    rim = z0 + cfg.bin.wall + cfg.bin.inner[2]
    cams = cfg.cameras
    spec.worldbody.add_camera(name="bin_cam", pos=[bx, by, rim + cams.bin_height_above_rim], fovy=cams.bin_fovy)
    tx, ty = cfg.layout.tray_center_xy
    spec.worldbody.add_camera(name="tray_cam", pos=[tx, ty, z0 + TRAY_BASE + cams.tray_height], fovy=cams.tray_fovy)
    eye, target = [1.45, -0.95, 1.15], [0.2, 0.15, 0.15]
    spec.worldbody.add_camera(name="overview_cam", pos=eye, xyaxes=_look_at_xyaxes(eye, target), fovy=50)
    spec.worldbody.add_light(name="key", pos=[0.5, -0.5, 2.5], dir=[-0.2, 0.2, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.6, 0.6, 0.6], castshadow=True)
    spec.worldbody.add_light(name="fill", pos=[-1, 1, 2.0], dir=[0.4, -0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.3, 0.3, 0.3], castshadow=False)


def _add_parts(spec: mujoco.MjSpec, cfg: Config, parts: list[PartSpec]) -> None:
    for p in parts:
        mat = cfg.parts.materials[p.material]
        if p.shape == "box":
            gtype, size, hz = mujoco.mjtGeom.mjGEOM_BOX, [d / 2 for d in p.dims], p.dims[2] / 2
        else:
            gtype, size, hz = mujoco.mjtGeom.mjGEOM_CYLINDER, [p.dims[0] / 2, p.dims[1] / 2, 0], p.dims[1] / 2
        body = spec.worldbody.add_body(name=f"part_{p.id}", pos=[PARK_X0 + 0.2 * p.id, 0, hz + 0.001])
        body.add_freejoint(name=f"part_{p.id}_free")
        body.add_geom(name=f"part_{p.id}_geom", type=gtype, size=size, mass=p.mass,
                      friction=[mat.friction, 0.005, 0.0001], condim=cfg.sim.part_condim,
                      solref=[cfg.sim.part_solref_timeconst, 1.0],
                      rgba=list(mat.rgba))


def build_scene(cfg: Config, parts: list[PartSpec]) -> Scene:
    spec = mujoco.MjSpec()
    spec.modelname = "binpick_cell"
    spec.option.timestep = cfg.sim.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = cfg.sim.noslip_iterations
    spec.visual.global_.offwidth = max(1280, cfg.cameras.width)
    spec.visual.global_.offheight = max(960, cfg.cameras.height)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            rgba=[0.82, 0.82, 0.8, 1])

    _add_panda(spec, cfg)
    _add_bin(spec, cfg)
    _add_tray(spec, cfg)
    _add_cameras_and_lights(spec, cfg)
    _add_parts(spec, cfg, parts)

    model = spec.compile()
    data = mujoco.MjData(model)
    arm_joints = [f"{PREFIX}joint{i}" for i in range(1, 8)]
    scene = Scene(
        model=model, data=data, cfg=cfg, parts=list(parts),
        part_body_ids=[model.body(f"part_{p.id}").id for p in parts],
        part_qpos_adr=[int(model.jnt_qposadr[model.joint(f"part_{p.id}_free").id]) for p in parts],
        arm_joint_names=arm_joints,
        slot_centers=tray_slot_centers(cfg),
    )
    scene.park_arm()
    return scene
