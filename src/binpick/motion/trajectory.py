"""Time-parameterized arm trajectories and a tracking executor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from binpick.config import MotionConfig
from binpick.motion.ik import solve_cup_pose
from binpick.sim.scene import Scene

SAMPLE_DT = 0.01
MAX_WAYPOINT_JUMP = 0.3  # rad; larger means an IK branch flip between neighbouring waypoints
# Quintic s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5: peak ds/dtau = 1.875, peak d2s/dtau2 = 5.7735
_V, _A = 1.875, 5.7735


class IKError(RuntimeError):
    pass


@dataclass(frozen=True)
class JointTrajectory:
    t: np.ndarray   # (N,)
    q: np.ndarray   # (N, 7)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def sample(self, t: float) -> np.ndarray:
        t = min(max(t, 0.0), self.duration)
        return np.array([np.interp(t, self.t, self.q[:, j]) for j in range(self.q.shape[1])])

    def velocity(self, t: float) -> np.ndarray:
        h = SAMPLE_DT / 2
        return (self.sample(t + h) - self.sample(t - h)) / (2 * h) if self.duration > 0 else np.zeros(self.q.shape[1])


def _quintic(n: int) -> np.ndarray:
    tau = np.linspace(0, 1, n)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _quintic_time(dist: float, vmax: float, amax: float) -> float:
    return max(_V * dist / vmax, np.sqrt(_A * dist / amax), SAMPLE_DT)


def joint_move(q0: np.ndarray, q1: np.ndarray, m: MotionConfig) -> JointTrajectory:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    T = _quintic_time(float(np.abs(q1 - q0).max()), m.max_joint_vel, m.max_joint_acc)
    n = int(np.ceil(T / SAMPLE_DT)) + 1
    s = _quintic(n)[:, None]
    return JointTrajectory(np.linspace(0, T, n), q0 + s * (q1 - q0))


def cartesian_line(scene: Scene, q0: np.ndarray, p0: np.ndarray, p1: np.ndarray, approach_dir: np.ndarray,
                   yaw: float, m: MotionConfig) -> JointTrajectory:
    """Cup tip moves on the straight line p0 -> p1 with fixed orientation (axis `approach_dir`, `yaw`)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    k = max(2, int(np.ceil(L / m.cart_step)) + 1)
    qs, q = [], np.asarray(q0, float)
    for i, u in enumerate(np.linspace(0, 1, k)):
        r = solve_cup_pose(scene, p0 + u * (p1 - p0), approach_dir, yaw, q0=q)
        if not r.ok:
            raise IKError(f"waypoint {i}/{k - 1} unreachable (pos_err={r.pos_err:.4f})")
        if i and np.abs(r.q - q).max() > MAX_WAYPOINT_JUMP:
            raise IKError(f"IK branch jump at waypoint {i}")
        yaw, q = r.yaw, r.q
        qs.append(q)
    qs = np.array(qs)
    jdist = float(np.abs(np.diff(qs, axis=0)).sum(axis=0).max())
    T = max(_quintic_time(L, m.max_cart_vel, m.max_cart_acc), _quintic_time(jdist, m.max_joint_vel, m.max_joint_acc))
    n = int(np.ceil(T / SAMPLE_DT)) + 1
    s = _quintic(n) * (k - 1)  # fractional waypoint index along the path
    q = np.array([np.interp(s, np.arange(k), qs[:, j]) for j in range(qs.shape[1])]).T
    return JointTrajectory(np.linspace(0, T, n), q)


def concat(*trs: JointTrajectory) -> JointTrajectory:
    t, q, off = [], [], 0.0
    for tr in trs:
        t.append(tr.t + off)
        q.append(tr.q)
        off += tr.duration
    return JointTrajectory(np.concatenate(t), np.concatenate(q))


def execute(scene: Scene, traj: JointTrajectory, on_step: Callable[[Scene], bool] | None = None,
            hold_s: float | None = None) -> bool:
    """Track `traj` with the Panda position actuators (plus velocity feed-forward for their damping).

    `on_step` runs after every physics step; returning False aborts (the arm holds where it is).
    """
    m, d = scene.model, scene.data
    ids = [m.actuator(a).id for a in scene.arm_actuator_names]
    kv_over_kp = np.array([-m.actuator_biasprm[i, 2] / m.actuator_gainprm[i, 0] for i in ids])
    hold = scene.cfg.motion.settle_after_move_s if hold_s is None else hold_s
    t0 = d.time
    n = int(np.ceil((traj.duration + hold) / m.opt.timestep))
    for _ in range(n):
        t = d.time - t0 + m.opt.timestep
        d.ctrl[ids] = traj.sample(t) + kv_over_kp * (traj.velocity(t) if t < traj.duration else 0.0)
        mujoco.mj_step(m, d)
        if on_step is not None and on_step(scene) is False:
            d.ctrl[ids] = [d.joint(j).qpos[0] for j in scene.arm_joint_names]
            return False
    return True
