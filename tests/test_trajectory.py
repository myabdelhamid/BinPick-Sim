import mujoco
import numpy as np
import pytest

from binpick.motion.ik import DOWN, solve_cup_down
from binpick.motion.trajectory import cartesian_line, execute, joint_move
from binpick.sim.scene import PARK_Q, build_scene


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
    assert np.abs(a).max() <= cfg.motion.max_joint_acc * 1.05


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
    assert len(calls) == 17  # one each, plus the alternative at the first
