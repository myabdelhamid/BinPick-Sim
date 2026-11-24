"""Perception-side placing: the robot does not know the part's height and grips off-center."""
import dataclasses

import mujoco
import numpy as np
import pytest

from binpick.perception.view import observe_bin
from binpick.planning.oracle import oracle_targets
from binpick.planning.rules import rules_targets
from binpick.planning.types import GraspTarget
from binpick.sim.parts import PartSpec
from binpick.sim.scene import build_scene
from binpick.supervisor.machine import Supervisor
from helpers import single_part_scene


def _settled(cfg, spec, **kw):
    s = single_part_scene(cfg, spec, **kw)
    for _ in range(150):
        mujoco.mj_step(s.model, s.data)
    return s


@pytest.mark.parametrize("h,kg", [(0.02, 0.1), (0.05, 1.0), (0.07, 1.5)])
def test_unknown_height_part_is_set_down_gently(cfg, h, kg):
    s = _settled(cfg, PartSpec(0, "box", "plastic", (0.09, 0.07, h), kg))
    (t,) = oracle_targets(s)
    t = GraspTarget(None, t.pos, t.normal, t.tilt_deg, 0.11, t.long_axis)  # height unknown: only the upper bound
    b, rel = s.part_body_ids[0], []

    def obs(sc, ph):
        if ph == "release" and not rel:
            rel.append(sc.data.xpos[b][2] - h / 2)

    r = Supervisor(s, np.random.default_rng(0), seed=0, observer=obs).attempt(t)
    assert r.outcome == "success", r
    assert abs(rel[0] - s.slot_centers[0][2]) < 0.004  # bottom within 4 mm of the tray at release


def test_off_center_grip_still_lands_in_the_slot_center(cfg):
    s = _settled(cfg, PartSpec(0, "box", "plastic", (0.10, 0.08, 0.04), 0.3))
    (t,) = oracle_targets(s)
    off = GraspTarget(None, t.pos + [0.013, 0.009, 0], t.normal, t.tilt_deg, t.depth, t.long_axis, center=t.pos)
    r = Supervisor(s, np.random.default_rng(0), seed=0).attempt(off)
    assert r.outcome == "success", r
    assert np.abs(s.data.xpos[s.part_body_ids[0]][:2] - s.slot_centers[0][:2]).max() < 0.006


def _arm_pixels(s, cam):
    m = s.model
    arm = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("panda/")]
    with mujoco.Renderer(m, height=240, width=320) as r:
        r.enable_segmentation_rendering()
        r.update_scene(s.data, camera=cam)
        x = r.render()
    return int((np.isin(x[..., 0], arm) & (x[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))).sum())


def test_view_pose_is_in_neither_camera(cfg):
    s = build_scene(cfg, [])
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    sup._to_view()
    assert _arm_pixels(s, "bin_cam") == 0 and _arm_pixels(s, "tray_cam") == 0


def test_arm_is_clear_of_the_bin_camera_after_a_failure(cfg):
    s = _settled(cfg, PartSpec(0, "box", "plastic", (0.09, 0.07, 0.03), 0.2))
    (t,) = oracle_targets(s)
    miss = GraspTarget(None, t.pos + [0.045, 0, 0], t.normal, 0.0, t.depth, t.long_axis)  # half over the edge
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    r = sup.attempt(miss)
    assert r.reason == "no_seal"
    assert _arm_pixels(s, "bin_cam") == 0


def test_camera_to_slot_without_any_part_pose(cfg):
    s = _settled(cfg, PartSpec(0, "box", "plastic", (0.10, 0.07, 0.045), 0.35))
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    r = sup.attempt(lambda: rules_targets(observe_bin(s), cfg))
    assert r.outcome == "success" and r.target_part == 0


def test_unreachable_targets_are_skipped_within_one_attempt(cfg):
    s = _settled(cfg, PartSpec(0, "box", "plastic", (0.10, 0.07, 0.045), 0.35))
    (t,) = oracle_targets(s)
    far = GraspTarget(None, np.array([2.0, 2.0, 0.1]), t.normal, 0.0, 0.05, t.long_axis)
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    r = sup.attempt(lambda: [far, t])
    assert r.outcome == "success" and r.attempt == 1
    assert [e["event"] for e in sup.events] == ["plan_rejected"]
    assert r.phase_times["plan"] == pytest.approx(cfg.timing.plan_s, abs=0.01)


@pytest.mark.parametrize("err_mm", [0.0, 2.0, -2.0])
def test_cup_touches_gently_even_if_the_height_estimate_is_off(cfg, err_mm):
    """The approach stops when contact is felt, instead of pressing to a fixed depth."""
    s = _settled(cfg, PartSpec(0, "box", "plastic", (0.10, 0.08, 0.04), 0.3))
    (t,) = oracle_targets(s)
    t = GraspTarget(None, t.pos + [0, 0, err_mm * 1e-3], t.normal, t.tilt_deg, t.depth, t.long_axis)
    cup, part, push = s.model.geom(s.cup_geom).bodyid[0], s.part_body_ids[0], []
    f = np.zeros(6)

    def obs(sc, ph):
        if ph == "grip":
            tot = 0.0
            for k in range(sc.data.ncon):
                c = sc.data.contact[k]
                if {sc.model.geom_bodyid[c.geom1], sc.model.geom_bodyid[c.geom2]} == {cup, part}:
                    mujoco.mj_contactForce(sc.model, sc.data, k, f)
                    tot += f[0]
            push.append(tot)

    r = Supervisor(s, np.random.default_rng(0), seed=0, observer=obs).attempt(t)
    assert r.outcome == "success", r
    assert push[0] < 12.0, push[0]  # N when the vacuum is switched on (a fixed 2 mm press gives 30 N)


def test_view_pose_resets_the_arm_posture(cfg):
    """Every cycle starts from the same joints, so IK seeding cannot drift into an awkward posture."""
    s = build_scene(cfg, [])
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    sup._to_view()
    q1 = sup._q()
    from binpick.motion.trajectory import joint_move
    sup._run(joint_move(sup._q(), sup._q() + [0.5, -0.4, 0.7, -0.5, 0.6, 0.4, -0.8], cfg.motion))
    sup._to_view()
    assert np.abs(sup._q() - q1).max() < 0.01


def test_ik_recovers_from_an_awkward_posture(cfg, monkeypatch):
    s = build_scene(cfg, [])
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    bad = np.array([2.5, 1.2, -2.5, -0.4, 2.5, 3.4, 2.5])  # wound-up arm: seeding IK from here does not converge
    monkeypatch.setattr(sup, "_q", lambda: bad.copy())
    from binpick.motion.ik import DOWN
    assert sup._ik_min_motion(np.array([0.55, 0.0, 0.35]), DOWN).ok
    assert sup._ik_min_motion(np.array([0.2, 0.5, 0.35]), DOWN, 0.4).ok


def test_a_slot_with_a_stray_part_is_not_used(cfg):
    """A part lying in slot 1 (dropped there earlier) must be seen by the tray camera and the slot skipped."""
    specs = [PartSpec(i, "box", "plastic", (0.08, 0.06, 0.04), 0.23) for i in range(3)]
    s = build_scene(cfg, specs)
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    for a, p in zip(s.part_qpos_adr, ([c[0] - 0.1, c[1], lo[2] + 0.02], [c[0] + 0.1, c[1], lo[2] + 0.02],
                                      s.slot_centers[1] + [0.02, 0.01, 0.02])):
        s.data.qpos[a:a + 3] = p
    for _ in range(150):
        mujoco.mj_step(s.model, s.data)
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    r = [sup.attempt(lambda: rules_targets(observe_bin(s), cfg)) for _ in range(2)]
    assert [x.outcome for x in r] == ["success"] * 2 and [x.slot for x in r] == [0, 2]


@pytest.mark.parametrize("bound", [0.04, 0.11])  # exact thickness (part lay on the floor) and the loosest bound
@pytest.mark.parametrize("kg", [0.15, 1.2])
def test_set_down_contact_is_felt_not_missed(cfg, kg, bound):
    """The probe must feel the tray (not be ended by the seal breaking), within 20 ms and at low speed."""
    s = _settled(cfg, PartSpec(0, "box", "steel", (0.09, 0.07, 0.04), kg))
    (t,) = oracle_targets(s)
    t = GraspTarget(None, t.pos, t.normal, t.tilt_deg, bound, t.long_axis)
    tray, b, log = s.model.body("tray").id, s.part_body_ids[0], []

    def obs(sc, ph):
        if ph == "place":
            hit = any({sc.model.geom_bodyid[c.geom1], sc.model.geom_bodyid[c.geom2]} == {tray, b} for c in sc.data.contact[:sc.data.ncon])
            log.append((sc.data.time, hit, sc.data.xpos[b][2]))

    sup = Supervisor(s, np.random.default_rng(0), seed=0, observer=obs)
    r = sup.attempt(t)
    assert r.outcome == "success" and sup.set_down_felt
    first = next(i for i, x in enumerate(log) if x[1])
    stop = log[first][0] + 0.021 + cfg.motion.settle_after_move_s
    assert log[-1][0] < stop  # the place phase ended (touch detected, then came to rest) right after first contact
    v = (log[first - 10][2] - log[first - 1][2]) / (log[first - 1][0] - log[first - 10][0])
    assert 0.03 < v < 0.12  # m/s on landing


def test_contact_on_the_fast_approach_stops_the_arm(cfg):
    """The camera was wrong by 40 mm: the cup meets the part while still moving fast."""
    s = single_part_scene(cfg, PartSpec(0, "box", "aluminium", (0.09, 0.09, 0.06), 1.3))
    for _ in range(100):
        mujoco.mj_step(s.model, s.data)
    t = oracle_targets(s)[0]
    top = t.pos[2]
    wrong = dataclasses.replace(t, pos=t.pos - [0, 0, 0.04])
    low = []
    sup = Supervisor(s, np.random.default_rng(0), seed=0, observer=lambda sc, ph: low.append(sc.data.site(sc.cup_site).xpos[2]))
    rec = sup.attempt(wrong)
    assert rec.sensed == "collision" and rec.outcome == "fail"
    assert min(low) > top - 0.006
    assert low[-1] > s.bin_inner_bounds()[1][2]  # backed out


def test_normal_approaches_are_not_taken_for_contact(cfg):
    for xy in ((0, 0), (0.12, 0.08), (-0.12, -0.08), (0.1, -0.09)):
        s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.04), 0.3), xy_offset=xy)
        for _ in range(100):
            mujoco.mj_step(s.model, s.data)
        assert Supervisor(s, np.random.default_rng(0), seed=0).attempt(oracle_targets(s)[0]).outcome == "success", xy
