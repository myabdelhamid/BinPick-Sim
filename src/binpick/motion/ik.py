"""Cup-pointing-down inverse kinematics with mink."""

from __future__ import annotations

from dataclasses import dataclass

import mink
import numpy as np

from binpick.sim.scene import Scene

# Elbow-up seed; joint1 is aimed at the target before solving.
READY_Q = np.array([0.0, -0.2, 0.0, -2.0, 0.0, 1.8, 0.7854])


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray        # 7 arm joint angles
    pos_err: float       # m
    rot_err: float       # rad
    ok: bool


def _cup_down_rotation(yaw: float) -> mink.SO3:
    """Cup axis (site z) along world -z, rotated by `yaw` about world z."""
    return mink.SO3.from_z_radians(yaw) @ mink.SO3.from_x_radians(np.pi)


def solve_cup_down(scene: Scene, target_pos: np.ndarray, yaw: float = 0.0, q0: np.ndarray | None = None,
                   pos_tol: float = 1e-3, rot_tol: float = 1e-2, max_iters: int = 500) -> IKResult:
    """Cup straight down at `target_pos` with wrist yaw `yaw`.

    Parts are placed by aligning their long side with the slot, which is symmetric under a
    180 deg turn, so yaw + pi is tried if yaw fails (keeps joint7 inside its limits).
    """
    best = None
    for y in (yaw, yaw + np.pi):
        r = _solve(scene, target_pos, y, q0, pos_tol, rot_tol, max_iters)
        if r.ok:
            return r
        if best is None or r.pos_err < best.pos_err:
            best = r
    return best


def _solve(scene: Scene, target_pos: np.ndarray, yaw: float, q0: np.ndarray | None,
           pos_tol: float, rot_tol: float, max_iters: int) -> IKResult:
    model = scene.model
    arm_adr = [int(model.jnt_qposadr[model.joint(j).id]) for j in scene.arm_joint_names]
    q_full = scene.data.qpos.copy()
    if q0 is None:
        base = np.asarray(scene.cfg.layout.robot_base_pos)
        q0 = READY_Q.copy()
        q0[0] = np.arctan2(target_pos[1] - base[1], target_pos[0] - base[0])
    q_full[arm_adr] = q0

    cfg = mink.Configuration(model, q_full)
    task = mink.FrameTask(scene.cup_site, "site", position_cost=1.0, orientation_cost=1.0)
    task.set_target(mink.SE3.from_rotation_and_translation(_cup_down_rotation(yaw), np.asarray(target_pos, float)))
    posture = mink.PostureTask(model, cost=1e-3)
    posture.set_target(q_full)
    limits = [mink.ConfigurationLimit(model)]

    dt = 0.1
    for _ in range(max_iters):
        err = task.compute_error(cfg)
        pos_err, rot_err = float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))
        if pos_err < pos_tol and rot_err < rot_tol:
            break
        vel = mink.solve_ik(cfg, [task, posture], dt, solver="daqp", damping=1e-3, limits=limits)
        cfg.integrate_inplace(vel, dt)
    err = task.compute_error(cfg)
    pos_err, rot_err = float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))
    return IKResult(q=cfg.q[arm_adr].copy(), pos_err=pos_err, rot_err=rot_err,
                    ok=pos_err < pos_tol and rot_err < rot_tol)
