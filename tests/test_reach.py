import numpy as np
import pytest

from binpick.motion.ik import DOWN, solve_cup_down, solve_cup_pose
from binpick.sim.scene import build_scene


def reach_targets(scene):
    lo, hi = scene.bin_inner_bounds()
    m = scene.cfg.cup.diameter / 2 + 0.03  # cup radius + wall clearance
    z_floor = lo[2] + 0.01
    corners = [np.array([x, y, z_floor]) for x in (lo[0] + m, hi[0] - m) for y in (lo[1] + m, hi[1] - m)]
    slots = [c + np.array([0, 0, 0.10]) for c in scene.slot_centers]  # place height incl. tallest part
    return corners + slots


def test_all_targets_reachable(scene):
    failures = []
    for t in reach_targets(scene):
        for yaw in (0.0, np.pi / 2):
            r = solve_cup_down(scene, t, yaw=yaw)
            if not r.ok:
                failures.append((np.round(t, 3).tolist(), yaw, round(r.pos_err, 4), round(r.rot_err, 4)))
    assert not failures, failures


def test_ik_does_not_touch_scene_data(scene):
    before = scene.data.qpos.copy()
    solve_cup_down(scene, scene.slot_centers[0] + [0, 0, 0.1])
    assert np.array_equal(before, scene.data.qpos)


def test_solution_puts_cup_on_target(scene):
    import mujoco

    t = scene.slot_centers[7] + np.array([0, 0, 0.1])
    r = solve_cup_down(scene, t, yaw=0.3)
    d = mujoco.MjData(scene.model)
    d.qpos[:] = scene.data.qpos
    for j, q in zip(scene.arm_joint_names, r.q):
        d.joint(j).qpos[0] = q
    mujoco.mj_kinematics(scene.model, d)
    assert np.linalg.norm(d.site(scene.cup_site).xpos - t) < 1e-3
    assert d.site(scene.cup_site).xmat.reshape(3, 3)[:, 2] @ np.array([0, 0, -1.0]) > 0.9999


def _arm_hits(scene, q):
    import mujoco

    m, d = scene.model, mujoco.MjData(scene.model)
    d.qpos[:] = scene.data.qpos
    for j, v in zip(scene.arm_joint_names, q):
        d.joint(j).qpos[0] = v
    mujoco.mj_forward(m, d)
    hits = []
    for c in d.contact[: d.ncon]:
        n1, n2 = (m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
        if c.dist < 0 and n1.startswith("panda/") != n2.startswith("panda/"):
            hits.append((n1, n2))
    return hits


def test_reach_poses_are_collision_free(scene):
    bad = []
    for t in reach_targets(scene):
        for yaw in (0.0, np.pi / 2):
            r = solve_cup_down(scene, t, yaw=yaw)
            h = _arm_hits(scene, r.q)
            if h:
                bad.append((np.round(t, 3).tolist(), yaw, h[:2]))
    assert not bad, bad


def test_tilted_approach(scene):
    import mujoco

    from binpick.motion.ik import solve_cup_pose

    lo, hi = scene.bin_inner_bounds()
    t = (lo + hi) / 2
    tilt = np.deg2rad(20)
    a = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])  # cup moving toward the part, 20 deg off vertical
    r = solve_cup_pose(scene, t, a, yaw=0.4)
    assert r.ok
    d = mujoco.MjData(scene.model)
    d.qpos[:] = scene.data.qpos
    for j, q in zip(scene.arm_joint_names, r.q):
        d.joint(j).qpos[0] = q
    mujoco.mj_kinematics(scene.model, d)
    z = d.site(scene.cup_site).xmat.reshape(3, 3)[:, 2]
    assert np.arccos(np.clip(z @ a, -1, 1)) < 0.01
    assert np.linalg.norm(d.site(scene.cup_site).xpos - t) < 1e-3


def test_ik_is_fast(scene):
    import time

    targets = reach_targets(scene)
    solve_cup_down(scene, targets[0])  # warm the cache
    t0 = time.perf_counter()
    for t in targets:
        solve_cup_down(scene, t)
    # about 11 ms here, 33 ms on a slow CI machine; the limit only catches a real slowdown (robot time is not affected)
    assert (time.perf_counter() - t0) / len(targets) < 0.1


@pytest.mark.parametrize("depth", [0.02, 0.06, 0.11])  # thinnest to longest part hanging under the cup
def test_every_slot_is_reachable_at_transport_height(cfg, depth):
    s = build_scene(cfg, [])
    rim = s.bin_inner_bounds()[1][2]
    for n, c in enumerate(s.slot_centers):
        z = max(rim + depth + cfg.motion.clear_above_rim, c[2] + cfg.motion.place_clearance + depth + cfg.motion.pregrasp_dist)
        for yaw in (0.0, np.pi / 2):
            assert solve_cup_pose(s, np.r_[c[:2], z], DOWN, yaw).ok, (n, depth, yaw)


def test_ik_solver_failure_is_an_unreachable_pose_not_a_crash(cfg, monkeypatch):
    import mink

    def boom(*a, **k):
        raise mink.NoSolutionFound("daqp")

    s = build_scene(cfg, [])
    monkeypatch.setattr(mink, "solve_ik", boom)
    assert not solve_cup_pose(s, np.array([0.5, 0.0, 0.3]), DOWN, 0.0).ok
