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
# Time scaling: trapezoidal velocity whose acceleration phases are "modified trapezoid" pulses
# (sine jerk ramps over the first and last quarter, constant in the middle): finite jerk, and the
# mean/peak acceleration ratio C = 0.5 + 0.5 * 2/pi keeps it close to time-optimal.
# Normalized path s in [0, 1]; ramp time Ta, total T, peak ds/dt = 1/(T - Ta), peak d2s/dt2 = 1/(C Ta (T - Ta)).
C = 0.5 + 1.0 / np.pi


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


def _optimal(D: float, V: float, A: float) -> float:
    """Shortest time for distance D under speed V and peak acceleration A."""
    if D <= 0:
        return 0.0
    if D >= V * V / (C * A):
        return D / V + V / (C * A)
    return 2 * np.sqrt(D / (C * A))


def _time_scale(limits: list[tuple[float, float, float]]) -> tuple[float, float]:
    """(T, Ta) satisfying every (distance, vmax, amax) with one shared normalized profile."""
    T = max([SAMPLE_DT] + [_optimal(D, V, A) for D, V, A in limits])
    K = max([0.0] + [D / (C * A) for D, _, A in limits])
    Ta = (T - np.sqrt(max(T * T - 4 * K, 0.0))) / 2
    return T, float(np.clip(Ta, min(SAMPLE_DT, T / 2), T / 2))


def _accel_shape(u: np.ndarray) -> np.ndarray:
    """Modified-trapezoid acceleration pulse on u in [0, 1], peak 1."""
    return np.where(u < 0.25, np.sin(2 * np.pi * u), np.where(u > 0.75, np.sin(2 * np.pi * (1 - u)), 1.0))


def _profile(T: float, Ta: float, n: int) -> np.ndarray:
    t = np.linspace(0, T, n)
    fine = np.linspace(0, T, max(20 * n, 2000))
    a = np.zeros_like(fine)
    up, down = fine < Ta, fine > T - Ta
    a[up] = _accel_shape(fine[up] / Ta)
    a[down] = -_accel_shape((T - fine[down]) / Ta)
    v = np.concatenate([[0.0], np.cumsum((a[1:] + a[:-1]) / 2 * np.diff(fine))])
    x = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(fine))])
    return np.clip(np.interp(t, fine, x / x[-1]), 0.0, 1.0)


def joint_move(q0: np.ndarray, q1: np.ndarray, m: MotionConfig, loaded: bool = False) -> JointTrajectory:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    acc = m.max_joint_acc_loaded if loaded else m.max_joint_acc
    T, Ta = _time_scale([(abs(d), v, acc) for d, v in zip(q1 - q0, m.max_joint_vel)])
    n = int(np.ceil(T / SAMPLE_DT)) + 1
    s = _profile(T, Ta, n)[:, None]
    return JointTrajectory(np.linspace(0, T, n), q0 + s * (q1 - q0))


def cartesian_line(scene: Scene, q0: np.ndarray, p0: np.ndarray, p1: np.ndarray, approach_dir: np.ndarray,
                   yaw: float, m: MotionConfig, loaded: bool = True) -> JointTrajectory:
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
    jdist = np.abs(np.diff(qs, axis=0)).sum(axis=0)
    cart_acc = m.max_cart_acc if loaded else m.max_cart_acc_empty
    joint_acc = m.max_joint_acc_loaded if loaded else m.max_joint_acc
    T, Ta = _time_scale([(L, m.max_cart_vel, cart_acc)]
                        + [(float(d), v, joint_acc) for d, v in zip(jdist, m.max_joint_vel)])
    n = int(np.ceil(T / SAMPLE_DT)) + 1
    s = _profile(T, Ta, n) * (k - 1)  # fractional waypoint index along the path
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
