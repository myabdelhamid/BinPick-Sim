"""Time-parameterized arm trajectories and a tracking executor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

from binpick.config import MotionConfig
from binpick.motion.ik import solve_cup_pose
from binpick.sim.scene import Scene

SAMPLE_DT = 0.01
LINE_POS_TOL, LINE_ROT_TOL = 1e-4, 1e-3  # waypoints of a straight line: a wavy line drags parts against their neighbours
MAX_POS_ERR, MAX_ROT_ERR = 1e-3, 1e-2          # a waypoint missed by more than this makes the line infeasible
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

    def acceleration(self, t: float) -> np.ndarray:
        h = SAMPLE_DT
        return (self.sample(t + h) - 2 * self.sample(t) + self.sample(t - h)) / h**2 if self.duration > 0 else np.zeros(self.q.shape[1])


def _time_scale(limits: list[tuple[float, float, float]]) -> tuple[float, float]:
    """(T, Ta) satisfying every (distance, vmax, amax) with one shared normalized profile: the peak speed is
    distance / (T - Ta), the peak acceleration distance / (C Ta (T - Ta))."""
    K = max([0.0] + [D / (C * A) for D, _, A in limits])
    S = max([SAMPLE_DT / 2, np.sqrt(K)] + [D / V for D, V, _ in limits])  # T - Ta
    Ta = max(K / S, SAMPLE_DT / 2)
    return float(S + Ta), float(Ta)


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


def _joint_acc(m: MotionConfig, loaded: bool) -> tuple[float, ...]:
    return tuple(min(a, m.max_joint_acc_loaded) for a in m.max_joint_acc) if loaded else m.max_joint_acc


def joint_move(q0: np.ndarray, q1: np.ndarray, m: MotionConfig, loaded: bool = False) -> JointTrajectory:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    T, Ta = _time_scale([(abs(d), v, a) for d, v, a in zip(q1 - q0, m.max_joint_vel, _joint_acc(m, loaded))])
    n = int(np.ceil(T / SAMPLE_DT)) + 1
    s = _profile(T, Ta, n)[:, None]
    return JointTrajectory(np.linspace(0, T, n), q0 + s * (q1 - q0))


def cartesian_line(scene: Scene, q0: np.ndarray, p0: np.ndarray, p1: np.ndarray, approach_dir: np.ndarray,
                   yaw: float, m: MotionConfig, loaded: bool = True) -> JointTrajectory:
    """Cup tip moves on the straight line p0 -> p1 with fixed orientation (axis `approach_dir`, `yaw`)."""
    return cartesian_path(scene, q0, [p0, p1], approach_dir, yaw, m, loaded)


def cartesian_path(scene: Scene, q0: np.ndarray, points: list, approach_dir: np.ndarray, yaw: float, m: MotionConfig,
                   loaded: bool = True) -> JointTrajectory:
    """Cup tip moves through `points` (smooth at the corners) with fixed orientation."""
    way = [np.asarray(points[0], float)]
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / m.cart_step)))
        way += [a + (b - a) * k / n for k in range(1, n + 1)]
    dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(way, axis=0), axis=1))]
    qs, q = [], np.asarray(q0, float)
    for i, p in enumerate(way):
        solve = lambda y: solve_cup_pose(scene, p, approach_dir, y, q0=q, pos_tol=LINE_POS_TOL, rot_tol=LINE_ROT_TOL, flip=False)  # noqa: E731
        r = solve(yaw)
        if i == 0 and np.abs(r.q - q).max() > MAX_WAYPOINT_JUMP:
            r = solve(yaw + np.pi)  # the cup may be turned half a turn from what was asked
        if not r.ok and (r.pos_err > MAX_POS_ERR or r.rot_err > MAX_ROT_ERR):
            raise IKError(f"waypoint {i}/{len(way) - 1} unreachable (pos_err={r.pos_err:.4f})")
        if np.abs(r.q - q).max() > MAX_WAYPOINT_JUMP:  # waypoint 0 is checked against q0 (no step command)
            raise IKError(f"IK branch jump at waypoint {i}")
        yaw, q = r.yaw, r.q
        qs.append(q)
    qs = np.array(qs)
    jdist = np.abs(np.diff(qs, axis=0)).sum(axis=0)
    cart_acc = m.max_cart_acc if loaded else m.max_cart_acc_empty
    T, Ta = _time_scale([(dist[-1], m.max_cart_vel, cart_acc)]
                        + [(float(d), v, a) for d, v, a in zip(jdist, m.max_joint_vel, _joint_acc(m, loaded))])
    if dist[-1] <= 0:
        return JointTrajectory(np.array([0.0, T]), np.repeat(qs[:1], 2, axis=0))
    spline = CubicSpline(dist, qs)
    for _ in range(6):  # near a singularity the joints move unevenly: slow down until the datasheet limits hold (the part's load is limited by the cup's own acceleration)
        n = int(np.ceil(T / SAMPLE_DT)) + 1
        tr = JointTrajectory(np.linspace(0, T, n), spline(_profile(T, Ta, n) * dist[-1]))
        dt = T / (n - 1)
        fast = (np.abs(np.diff(tr.q, axis=0)) / dt / np.array(m.max_joint_vel)).max()
        hard = (np.abs(np.diff(tr.q, n=2, axis=0)) / dt**2 / np.array(m.max_joint_acc)).max() if n > 2 else 0.0
        k = max(fast, np.sqrt(hard))
        if k <= 1.0:
            break
        k *= 1.005  # a little more, so that the next pass ends it
        T, Ta = T * k, Ta * k
    return tr


def reverse(tr: JointTrajectory) -> JointTrajectory:
    return JointTrajectory(tr.t, tr.q[::-1])


def concat(*trs: JointTrajectory) -> JointTrajectory:
    t, q, off = [], [], 0.0
    for tr in trs:
        t.append(tr.t + off)
        q.append(tr.q)
        off += tr.duration
    return JointTrajectory(np.concatenate(t), np.concatenate(q))


def execute(scene: Scene, traj: JointTrajectory, on_step: Callable[[Scene], bool] | None = None,
            hold_s: float | None = None, sideways_gain: float = 0.0) -> bool:
    """Track `traj` with the Panda position actuators, plus feed-forward for their damping and the arm's own inertia
    (as the real controller does; the part's weight is unknown and not included).

    `sideways_gain` adds stiffness against horizontal deviation of the cup, seen in the joint readings.
    `on_step` runs after every physics step; returning False aborts (the arm holds where it is).
    """
    m, d = scene.model, scene.data
    ids = [m.actuator(a).id for a in scene.arm_actuator_names]
    kp = np.array([m.actuator_gainprm[i, 0] for i in ids])
    kv_over_kp = np.array([-m.actuator_biasprm[i, 2] for i in ids]) / kp
    hold = scene.cfg.motion.settle_after_move_s if hold_s is None else hold_s
    t0 = d.time
    n = int(np.ceil((traj.duration + hold) / m.opt.timestep))
    tip, J = m.site(scene.cup_site).id, np.zeros((3, m.nv))
    qadr = [int(m.joint(j).qposadr[0]) for j in scene.arm_joint_names]
    dofs = [int(m.joint(j).dofadr[0]) for j in scene.arm_joint_names]
    for _ in range(n):
        t = d.time - t0 + m.opt.timestep
        q = traj.sample(t)
        if t < traj.duration:
            q = q + kv_over_kp * traj.velocity(t) + scene.arm_inertia(q) @ traj.acceleration(t) / kp
        if sideways_gain > 0:
            mujoco.mj_jacSite(m, d, J, None, tip)
            off = J[:2, dofs] @ (traj.sample(t - m.opt.timestep) - d.qpos[qadr])
            q = q + sideways_gain * np.linalg.pinv(J[:, dofs])[:, :2] @ off
        d.ctrl[ids] = q
        mujoco.mj_step(m, d)
        if on_step is not None and on_step(scene) is False:
            d.ctrl[ids] = [d.joint(j).qpos[0] for j in scene.arm_joint_names]
            return False
    return True
