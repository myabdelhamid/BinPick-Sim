"""Test-only helpers: single-part scenes and arm teleporting."""

import mujoco
import numpy as np

from binpick.motion.ik import solve_cup_pose
from binpick.sim.parts import PartSpec
from binpick.sim.scene import build_scene


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def single_part_scene(cfg, spec: PartSpec, quat=(1, 0, 0, 0), xy_offset=(0.0, 0.0), z=None):
    """Scene with one part posed in the middle of the bin (resting on the floor unless z is given)."""
    s = build_scene(cfg, [spec])
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    hz = spec.dims[2] / 2 if spec.shape == "box" else spec.dims[1] / 2
    a = s.part_qpos_adr[0]
    s.data.qpos[a:a + 3] = [c[0] + xy_offset[0], c[1] + xy_offset[1], lo[2] + hz if z is None else z]
    s.data.qpos[a + 3:a + 7] = quat
    mujoco.mj_forward(s.model, s.data)
    return s


def teleport_cup(scene, pos, approach_dir=(0, 0, -1), yaw=0.0):
    r = solve_cup_pose(scene, np.asarray(pos, float), np.asarray(approach_dir, float), yaw)
    assert r.ok, r
    for j, a, q in zip(scene.arm_joint_names, scene.arm_actuator_names, r.q):
        scene.data.joint(j).qpos[0] = q
        scene.data.joint(j).qvel[0] = 0.0
        scene.data.actuator(a).ctrl[0] = q
    mujoco.mj_forward(scene.model, scene.data)
    return r.q


def part_top(scene, i=0):
    """World z of a box part's top face center (part lying flat)."""
    spec = scene.parts[i]
    p = scene.data.xpos[scene.part_body_ids[i]]
    return p + np.array([0, 0, spec.dims[2] / 2 if spec.shape == "box" else spec.dims[0] / 2])


def loose_parts_scene(cfg):
    """Three flat boxes lying apart in the bin: nothing pins them, so every pick should work."""
    specs = [PartSpec(i, "box", "plastic", (0.08, 0.06, h), 0.08 * 0.06 * h * 1200) for i, h in enumerate((0.05, 0.04, 0.03))]
    s = build_scene(cfg, specs)
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    for spec, a, dx in zip(specs, s.part_qpos_adr, (-0.12, 0.0, 0.12)):
        s.data.qpos[a:a + 3] = [c[0] + dx, c[1], lo[2] + spec.dims[2] / 2]
    for _ in range(200):
        mujoco.mj_step(s.model, s.data)
    return s
