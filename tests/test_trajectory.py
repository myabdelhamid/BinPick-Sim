import dataclasses

import mujoco
import numpy as np
import pytest

from binpick.motion.ik import DOWN, solve_cup_down, solve_cup_pose, yaw_of
from binpick.motion.trajectory import IKError, cartesian_line, cartesian_path, execute, joint_move
from binpick.sim.scene import PARK_Q, build_scene
from helpers import teleport_cup


@pytest.fixture()
def empty(cfg):
    return build_scene(cfg, [])


def _arm_q(scene):
    return np.array([scene.data.joint(j).qpos[0] for j in scene.arm_joint_names])


def test_joint_move_limits(cfg):
    q1 = PARK_Q + np.array([1.0, 0.3, 0.0, 0.5, 0.0, -0.4, 0.2])
    tr = joint_move(PARK_Q, q1, cfg.motion)
    assert np.allclose(tr.sample(0.0), PARK_Q) and np.allclose(tr.sample(tr.duration), q1)
    dt = np.diff(tr.t)[:, None]
    v = np.diff(tr.q, axis=0) / dt
    a = np.diff(v, axis=0) / dt[1:]
    assert np.all(np.abs(v).max(axis=0) <= np.array(cfg.motion.max_joint_vel) * 1.01)
    assert np.all(np.abs(a).max(axis=0) <= np.array(cfg.motion.max_joint_acc) * 1.05)


def test_cartesian_line_is_straight_and_limited(empty, cfg):
    lo, hi = empty.bin_inner_bounds()
    c = (lo + hi) / 2
    p0, p1 = c + [0, 0, 0.25], c + [0, 0, 0.05]
    q0 = solve_cup_down(empty, p0).q
    tr = cartesian_line(empty, q0, p0, p1, DOWN, 0.0, cfg.motion)
    d = mujoco.MjData(empty.model)
    pts = []
    for q in tr.q[:: max(1, len(tr.q) // 30)]:
        for j, v in zip(empty.arm_joint_names, q):
            d.joint(j).qpos[0] = v
        mujoco.mj_kinematics(empty.model, d)
        pts.append(d.site(empty.cup_site).xpos.copy())
    pts = np.array(pts)
    off_line = np.linalg.norm(np.cross(pts - p0, (p1 - p0) / np.linalg.norm(p1 - p0)), axis=1)
    assert off_line.max() < 2e-3
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.diff(tr.t[:: max(1, len(tr.q) // 30)])[: len(pts) - 1]
    assert speed.max() <= cfg.motion.max_cart_vel * 1.1


def test_execute_tracks_plan(empty, cfg):
    lo, hi = empty.bin_inner_bounds()
    target = (lo + hi) / 2 + [0, 0, 0.15]
    q1 = solve_cup_down(empty, target).q
    tr = joint_move(_arm_q(empty), q1, cfg.motion)
    errs = []

    def watch(s):
        errs.append(np.abs(_arm_q(s) - tr.sample(s.data.time - t0)).max())
        return True

    t0 = empty.data.time
    assert execute(empty, tr, watch)
    assert np.linalg.norm(empty.data.site(empty.cup_site).xpos - target) < 2e-3
    assert max(errs) < 0.02  # rad, during motion


def test_execute_abort(empty, cfg):
    tr = joint_move(_arm_q(empty), _arm_q(empty) + 0.3, cfg.motion)
    assert execute(empty, tr, lambda s: False) is False


def test_straight_line_does_not_waste_ik_on_a_flipped_wrist(cfg, monkeypatch):
    """Waypoints follow the previous one; the yaw + pi alternative would be a wrist jump and is not solved."""
    from binpick.motion import ik
    from binpick.motion.ik import solve_cup_down

    s = build_scene(cfg, [])
    p = np.array([0.55, 0.0, 0.35])
    r = solve_cup_down(s, p, yaw=0.3)
    calls, real = [], ik._solve
    monkeypatch.setattr(ik, "_solve", lambda *a: calls.append(1) or real(*a))
    tr = cartesian_line(s, r.q, p, p - [0, 0, 0.15], DOWN, r.yaw, cfg.motion)  # 16 waypoints
    assert np.abs(np.diff(tr.q, axis=0)).max() < 0.1
    assert len(calls) == 16  # one each


@pytest.mark.parametrize("at", [(0.0, 0.0), (0.13, 0.09), (-0.13, -0.09)])
def test_a_vertical_lift_is_really_vertical(empty, cfg, at):
    """A part touching a wall or a neighbour jams if the cup moves sideways while lifting."""
    lo, hi = empty.bin_inner_bounds()
    p0 = np.r_[(lo + hi)[:2] / 2 + at, lo[2] + 0.04]
    teleport_cup(empty, p0)
    tip0 = empty.data.site(empty.cup_site).xpos.copy()
    R0 = empty.data.site(empty.cup_site).xmat.reshape(3, 3).copy()
    tr = cartesian_line(empty, _arm_q(empty), tip0, tip0 + [0, 0, 0.15], R0[:, 2], yaw_of(R0), cfg.motion, loaded=True)
    off, rot = [], []

    def watch(s):
        st = s.data.site(s.cup_site)
        off.append(np.linalg.norm(st.xpos[:2] - tip0[:2]))
        rot.append(np.arccos(np.clip((np.trace(R0.T @ st.xmat.reshape(3, 3)) - 1) / 2, -1, 1)))
        return True

    assert execute(empty, tr, watch)
    assert max(off) < 4e-4, max(off)
    assert np.degrees(max(rot)) < 0.12


def test_path_with_a_corner_passes_its_points(empty, cfg):
    lo, hi = empty.bin_inner_bounds()
    p0 = np.r_[(lo + hi)[:2] / 2, lo[2] + 0.05]
    q0 = teleport_cup(empty, p0)
    pts = [p0, p0 + [0.0015, 0, 0.006], p0 + [0.0015, 0, 0.15]]
    tr = cartesian_path(empty, q0, pts, DOWN, 0.0, cfg.motion)
    d = mujoco.MjData(empty.model)
    tip = []
    for q in tr.q:
        for j, v in zip(empty.arm_joint_names, q):
            d.joint(j).qpos[0] = v
        mujoco.mj_kinematics(empty.model, d)
        tip.append(d.site(empty.cup_site).xpos - p0)
    tip = np.array(tip)
    assert np.allclose(tip[-1], pts[-1] - p0, atol=3e-4)
    at6 = tip[np.argmin(np.abs(tip[:, 2] - 0.006))]
    assert at6[0] == pytest.approx(0.0015, abs=4e-4)
    assert np.abs(tip[:, 0]).max() < 0.0022 and np.all(np.diff(tip[:, 2]) >= -1e-6)


@pytest.mark.parametrize("gain,worst", [(3.0, 0.45)])
def test_sideways_hold_reduces_the_sag_under_a_load(empty, cfg, gain, worst):
    """A 1.5 kg pull on the tool bends the arm sideways too; the joint readings show it."""
    lo, hi = empty.bin_inner_bounds()
    p0 = np.r_[(lo + hi)[:2] / 2, lo[2] + 0.2]
    sag = []
    for g in (0.0, gain):
        q0 = teleport_cup(empty, p0)
        body = empty.model.geom(empty.cup_geom).bodyid[0]
        empty.data.xfrc_applied[body, 2] = -1.5 * 9.81
        execute(empty, joint_move(q0, q0, cfg.motion), hold_s=0.5, sideways_gain=g)
        empty.data.xfrc_applied[body, 2] = 0.0
        sag.append(np.linalg.norm(empty.data.site(empty.cup_site).xpos[:2] - p0[:2]))
    assert sag[0] > 3e-4 and sag[1] < worst * sag[0]


def test_a_waypoint_a_little_off_the_line_is_better_than_no_line(empty, cfg, monkeypatch):
    """The tight tolerance is a goal; only a waypoint missed by more than the normal tolerance makes a line infeasible."""
    from binpick.motion import trajectory as tr
    real = tr.solve_cup_pose

    def slightly_off(*a, **k):
        r = real(*a, **k)
        return dataclasses.replace(r, ok=False, pos_err=5e-4)

    p = (lambda b: np.r_[(b[0] + b[1])[:2] / 2, b[0][2] + 0.1])(empty.bin_inner_bounds())
    q0 = teleport_cup(empty, p)
    monkeypatch.setattr(tr, "solve_cup_pose", slightly_off)
    assert cartesian_line(empty, q0, p, p + [0, 0, 0.1], DOWN, 0.0, cfg.motion).duration > 0
    monkeypatch.setattr(tr, "solve_cup_pose", lambda *a, **k: dataclasses.replace(real(*a, **k), ok=False, pos_err=5e-3))
    with pytest.raises(tr.IKError):
        cartesian_line(empty, q0, p, p + [0, 0, 0.1], DOWN, 0.0, cfg.motion)


def test_line_near_a_singularity_keeps_the_joint_acceleration_limit(empty, cfg):
    rng, checked = np.random.default_rng(3), 0
    lo, hi = empty.bin_inner_bounds()
    for _ in range(80):
        az, tilt = rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(25, 60))
        n = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
        p = np.r_[rng.uniform(lo[:2] + 0.03, hi[:2] - 0.03), lo[2] + 0.12]
        r = solve_cup_pose(empty, p + n * 0.25, -n)
        if not r.ok:
            continue
        try:
            tr = cartesian_line(empty, r.q, p + n * 0.25, p, -n, r.yaw, cfg.motion, loaded=False)
        except IKError:
            continue
        a = np.diff(tr.q, n=2, axis=0) / np.diff(tr.t).mean() ** 2
        assert np.all(np.abs(a).max(axis=0) <= np.array(cfg.motion.max_joint_acc) * 1.15)
        checked += 1
    assert checked >= 15


def test_fast_moves_stay_within_the_motors_torque_limits(empty, cfg):
    """Datasheet accelerations must not saturate the joint motors (87 / 12 Nm)."""
    lo, hi = empty.bin_inner_bounds()
    ids = [empty.model.actuator(a).id for a in empty.arm_actuator_names]
    limit, worst = empty.model.actuator_forcerange[ids, 1], np.zeros(7)

    def watch(s):
        worst[:] = np.maximum(worst, np.abs(s.data.actuator_force[ids]))
        return True

    for p in ((hi[0] - 0.05, lo[1] + 0.05), (lo[0] + 0.05, hi[1] - 0.05), (0.0, 0.5), (-0.24, 0.65)):
        execute(empty, joint_move(_arm_q(empty), solve_cup_down(empty, np.r_[p, 0.35]).q, cfg.motion), watch)
    assert np.all(worst < 0.9 * limit), worst / limit
