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
    yaw: float = 0.0     # yaw actually used (may be requested yaw + pi)


DOWN = np.array([0.0, 0.0, -1.0])


def cup_rotation(approach_dir: np.ndarray, yaw: float) -> np.ndarray:
    """3x3 rotation whose z axis (cup axis) is `approach_dir`; x is world (cos yaw, sin yaw, 0)
    projected onto the plane normal to the axis. For a straight-down approach this is Rz(yaw) Rx(pi)."""
    z = np.asarray(approach_dir, float)
    z = z / np.linalg.norm(z)
    ref = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = ref - (ref @ z) * z
    if np.linalg.norm(x) < 1e-6:  # axis horizontal along ref: fall back to world z as reference
        x = np.array([0.0, 0.0, 1.0]) - z[2] * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def solve_cup_down(scene: Scene, target_pos: np.ndarray, yaw: float = 0.0, q0: np.ndarray | None = None,
                   pos_tol: float = 1e-3, rot_tol: float = 1e-2, max_iters: int = 500) -> IKResult:
    """Cup straight down at `target_pos` with wrist yaw `yaw`."""
    return solve_cup_pose(scene, target_pos, DOWN, yaw, q0, pos_tol, rot_tol, max_iters)


def solve_cup_pose(scene: Scene, target_pos: np.ndarray, approach_dir: np.ndarray, yaw: float = 0.0,
                   q0: np.ndarray | None = None, pos_tol: float = 1e-3, rot_tol: float = 1e-2,
                   max_iters: int = 500, flip: bool = True) -> IKResult:
    """Cup tip at `target_pos`, cup axis along `approach_dir` (the direction the cup moves toward the part).

    Parts are placed by aligning their long side with the slot, which is symmetric under a
    180 deg turn, so yaw + pi is solved too and the closer one used (keeps joint7 inside its limits).
    `flip=False` solves yaw only: for waypoints of a line, where a half turn would be a jump.
    """
    results = [_solve(scene, target_pos, approach_dir, y, q0, pos_tol, rot_tol, max_iters) for y in ((yaw, yaw + np.pi) if flip else (yaw,))]
    ok = [r for r in results if r.ok]
    if not ok:
        return min(results, key=lambda r: r.pos_err)
    if len(ok) == 1:
        return ok[0]
    ref = np.array([scene.data.joint(j).qpos[0] for j in scene.arm_joint_names]) if q0 is None else q0
    return min(ok, key=lambda r: float(np.abs(r.q - ref).max()))  # the equivalent yaw needing least motion


def _solve(scene: Scene, target_pos: np.ndarray, approach_dir: np.ndarray, yaw: float, q0: np.ndarray | None,
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
    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(cup_rotation(approach_dir, yaw)),
                                                        np.asarray(target_pos, float)))
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
                    ok=pos_err < pos_tol and rot_err < rot_tol, yaw=yaw)


def yaw_of(R: np.ndarray) -> float:
    """Inverse of `cup_rotation` for the yaw: the horizontal vector whose projection onto the plane
    normal to the cup axis is the cup's x axis is ref = x - (x_z / z_z) z."""
    x, z = R[:, 0], R[:, 2]
    if abs(z[2]) < 1e-6:
        return 0.0
    ref = x - (x[2] / z[2]) * z
    return float(np.arctan2(ref[1], ref[0]))
