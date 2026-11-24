"""Perception-side placing: the robot does not know the part's height and grips off-center."""
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
