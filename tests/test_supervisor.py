import dataclasses

import mujoco
import numpy as np
import pytest

from binpick.planning.oracle import GraspTarget, oracle_targets
from binpick.sim.parts import PartSpec
from binpick.supervisor.machine import Supervisor
from binpick.supervisor.records import write_csv
from helpers import loose_parts_scene, quat_axis_angle, single_part_scene


@pytest.fixture(scope="module")
def two_picks(cfg):
    scene = loose_parts_scene(cfg)
    trace = []
    sup = Supervisor(scene, np.random.default_rng(0), seed=1000,
                     observer=lambda s, phase: trace.append((phase, s.data.time, s.data.xpos[s.part_body_ids].copy())))
    recs = []
    for _ in range(2):
        t = oracle_targets(scene)[0]
        recs.append((t, sup.attempt(t)))
    return scene, sup, recs, trace


def test_first_pick_succeeds(two_picks):
    _, _, recs, _ = two_picks
    t, r = recs[0]
    assert r.outcome == "success", r
    assert r.reason == "" and r.referee_ok and r.verify_ok
    assert r.gripped_part == t.part_index and r.slot == 0
    assert r.cycle_time_s < 10.0  # M2 sanity bound; the 8 s target is measured in the M5 benchmark


def test_second_pick_uses_next_slot(two_picks):
    _, _, recs, _ = two_picks
    assert recs[1][1].slot == 1 and recs[1][1].attempt == 2


def test_part_clears_rim_during_transport(two_picks):
    scene, _, recs, trace = two_picks
    rim = scene.bin_inner_bounds()[1][2]
    i = recs[0][1].gripped_part
    hs = scene.parts[i].dims
    lo, hi = scene.bin_inner_bounds()
    r = max(hs)
    over_bin = [pos[i][2] for phase, _, pos in trace if phase in ("lift", "transport")
                and np.all(pos[i][:2] > lo[:2] - r) and np.all(pos[i][:2] < hi[:2] + r)]
    over_bin = over_bin[len(over_bin) // 2:]  # second half: the part is out of the pile and moving away
    assert over_bin and min(over_bin) - r / 2 > rim - 0.005  # conservative: half the largest dim below center


def test_phase_times_add_up(two_picks):
    _, _, recs, _ = two_picks
    r = recs[0][1]
    assert sum(r.phase_times.values()) == pytest.approx(r.cycle_time_s, abs=0.01)
    assert {"capture", "plan", "approach", "grip", "lift", "transport", "place", "verify"} <= r.phase_times.keys()


def test_bad_grasp_point_gives_no_seal(cfg):
    cyl = PartSpec(0, "cylinder", "plastic", (0.036, 0.100), 0.12)  # below the 40 mm curve limit: cannot seal
    s = single_part_scene(cfg, cyl, quat=quat_axis_angle([1, 0, 0], np.pi / 2))
    lo, _ = s.bin_inner_bounds()
    a = s.part_qpos_adr[0]
    s.data.qpos[a + 2] = lo[2] + 0.018
    import mujoco
    mujoco.mj_forward(s.model, s.data)
    top = s.data.qpos[a:a + 3] + [0, 0, 0.018]
    t = GraspTarget(0, top, np.array([0, 0, 1.0]), 0.0, 0.036, np.array([0, 1.0, 0]))
    r = Supervisor(s, np.random.default_rng(0), seed=0).attempt(t)
    assert r.outcome == "fail" and r.reason == "no_seal" and r.slot is None


def test_thin_cylinder_is_picked_from_its_side(cfg):
    import mujoco

    cyl = PartSpec(0, "cylinder", "steel", (0.042, 0.100), 1.09)
    s = single_part_scene(cfg, cyl, quat=quat_axis_angle([1, 0, 0], np.pi / 2))
    s.data.qpos[s.part_qpos_adr[0] + 2] = s.bin_inner_bounds()[0][2] + 0.021
    for _ in range(100):
        mujoco.mj_step(s.model, s.data)
    (t,) = oracle_targets(s)
    r = Supervisor(s, np.random.default_rng(0), seed=0).attempt(t)
    assert r.outcome == "success", r


@pytest.mark.parametrize("slot", [0, 4, 15, 17, 19])  # tray corners and the far row
def test_part_is_placed_in_far_slots(cfg, slot):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.09, 0.06, 0.05), 0.32))
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    for _ in range(slot):
        sup.slots.commit()
    r = sup.attempt(oracle_targets(s)[0])
    assert r.outcome == "success" and r.slot == slot, r


def test_unreachable_target_is_not_an_attempt(cfg):
    box = PartSpec(0, "box", "plastic", (0.09, 0.06, 0.03), 0.2)
    s = single_part_scene(cfg, box)
    t = GraspTarget(0, np.array([2.0, 2.0, 0.1]), np.array([0, 0, 1.0]), 0.0, 0.03, np.array([1.0, 0, 0]))
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    assert sup.attempt(t) is None
    assert sup.events[-1]["event"] == "plan_rejected"


def test_csv(two_picks, tmp_path):
    _, sup, _, _ = two_picks
    p = tmp_path / "a.csv"
    write_csv(sup.records, p)
    lines = p.read_text().splitlines()
    assert len(lines) == 3 and lines[0].startswith("seed,attempt,")


def test_lift_moves_away_from_the_wall_the_part_touches(cfg):
    ix = cfg.bin.inner[0]
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.04), 0.3), xy_offset=(-(ix / 2 - 0.0402), 0.0))
    for _ in range(100):
        mujoco.mj_step(s.model, s.data)
    from binpick.perception.view import observe_bin
    from binpick.planning.rules import rules_targets

    t = rules_targets(observe_bin(s), cfg)[0]
    xs = []
    sup = Supervisor(s, np.random.default_rng(0), seed=0,
                     observer=lambda sc, ph: xs.append(sc.data.site(sc.cup_site).xpos.copy()) if ph == "lift" else None)
    rec = sup.attempt(t)
    xs = np.array(xs)
    up = xs[np.argmin(np.abs(xs[:, 2] - xs[0, 2] - 0.012))]
    assert rec.outcome == "success" and up[0] - xs[0, 0] == pytest.approx(cfg.motion.lift_away, abs=6e-4)


def test_heavy_part_is_felt_and_moved_gently(cfg):
    times = {}
    for mass in (0.3, 1.6):
        s = single_part_scene(cfg, PartSpec(0, "box", "steel", (0.08, 0.08, 0.04), mass))
        for _ in range(200):
            mujoco.mj_step(s.model, s.data)
        sup = Supervisor(s, np.random.default_rng(0), seed=0)
        rec = sup.attempt(oracle_targets(s)[0])
        assert rec.outcome == "success"
        assert sup.felt_mass_kg == pytest.approx(mass, abs=0.15)
        times[mass] = rec.phase_times["transport"]
    assert times[1.6] > 1.3 * times[0.3]


def test_lift_at_the_far_bin_edge_goes_inward_when_straight_up_is_out_of_reach(cfg):
    ix, iy = cfg.bin.inner[:2]
    for xy in ((ix / 2 - 0.045, -0.09), (ix / 2 - 0.045, 0.10)):
        spec = PartSpec(0, "box", "plastic", (0.08, 0.08, 0.06), 0.4)
        s = single_part_scene(cfg, spec, xy_offset=xy)
        a = s.part_qpos_adr[0]
        s.data.qpos[a + 2] = s.bin_inner_bounds()[0][2] + 0.03
        for _ in range(100):
            mujoco.mj_step(s.model, s.data)
        t = oracle_targets(s)[0]
        t = dataclasses.replace(t, depth=0.11)  # a camera cannot know the part is thinner
        sup = Supervisor(s, np.random.default_rng(0), seed=0)
        rec = sup.attempt(t)
        assert rec.outcome == "success", (xy, rec.reason)


def test_light_parts_are_carried_faster_than_heavy_ones(cfg):
    acc = {}
    for kg in (0.1, 0.5, 1.0, 1.7):
        s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.04), 0.3))
        sup = Supervisor(s, np.random.default_rng(0), seed=0)
        sup.felt_mass_kg = kg
        acc[kg] = (sup._carry_acc(1.0), sup._carry_acc(0.8))
    assert acc[0.1][0] >= acc[0.5][0] > acc[1.0][0] > acc[1.7][0] >= cfg.motion.carry_acc_min
    assert acc[0.1][0] <= cfg.motion.carry_acc_max and acc[1.0][1] < acc[1.0][0]


@pytest.mark.parametrize("slot", [0, 10, 15])
def test_heaviest_cylinder_gripped_on_its_side_is_not_dropped(cfg, slot):
    s = single_part_scene(cfg, PartSpec(0, "cylinder", "steel", (0.0614, 0.0725), 1.69), quat=quat_axis_angle([1, 0, 0], np.pi / 2), xy_offset=(0.1, 0.05))
    s.data.qpos[s.part_qpos_adr[0] + 2] = s.bin_inner_bounds()[0][2] + 0.0614 / 2
    for _ in range(200):
        mujoco.mj_step(s.model, s.data)
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    for k in range(slot):
        sup.slots.mark_used(k)
    t = dataclasses.replace(oracle_targets(s)[0], seal=0.83)
    assert sup.attempt(t).outcome == "success"


def carry(cfg, slot, monkeypatch=None, depth=None):
    """Bottom height of a carried 60 mm box over the transport phase, with whether it was near the bin."""
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.06, 0.06, 0.06), 0.25), xy_offset=(-0.10, 0.08))
    for _ in range(100):
        mujoco.mj_step(s.model, s.data)
    b, (lo, hi), out = s.part_body_ids[0], s.bin_inner_bounds(), []

    def watch(sc, ph):
        if ph == "transport":
            p = sc.data.xpos[b]
            near = np.all(p[:2] > lo[:2] - 0.05) and np.all(p[:2] < hi[:2] + 0.05)
            out.append((near, p[2] - 0.03))

    sup = Supervisor(s, np.random.default_rng(0), seed=0, observer=watch)
    for k in range(slot):
        sup.slots.mark_used(k)
    t = oracle_targets(s)[0]
    rec = sup.attempt(t if depth is None else dataclasses.replace(t, depth=depth))
    return rec, out, s


@pytest.mark.parametrize("slot", [4, 9])
def test_carried_part_stays_above_the_rim_until_it_is_past_the_bin(cfg, slot):
    rec, out, s = carry(cfg, slot)
    assert rec.outcome == "success"
    assert min(z for near, z in out if near) > s.bin_inner_bounds()[1][2] + 0.005


def test_low_transport_is_not_used_if_it_would_cut_the_rim(cfg, monkeypatch):
    """With every height refused the arm must not carry the part at all."""
    from binpick.motion.collision import CellCheck

    monkeypatch.setattr(CellCheck, "clears_rim", lambda *a: False)
    rec, _, _ = carry(cfg, 4)
    assert rec.outcome == "fail" and rec.reason == "ik_fail"


@pytest.mark.parametrize("slot", [10, 15])
def test_carried_part_passes_over_the_tallest_part_that_can_stand_in_the_tray(cfg, slot):
    rec, out, s = carry(cfg, slot)
    assert rec.outcome == "success"
    assert min(z for _, z in out) > s.slot_centers[0][2] + 0.110 + 0.010


def test_transport_really_is_lower_than_the_rim_when_it_can_be(cfg):
    rec, out, s = carry(cfg, 15)
    assert out[-1][1] < s.bin_inner_bounds()[1][2]


def test_retreat_rises_above_the_tallest_part_before_going_sideways(cfg):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.02), 0.15))
    for _ in range(100):
        mujoco.mj_step(s.model, s.data)
    tips = []
    sup = Supervisor(s, np.random.default_rng(0), seed=0,
                     observer=lambda sc, ph: tips.append(sc.data.site(sc.cup_site).xpos.copy()) if ph == "retreat" else None)
    sup.attempt(oracle_targets(s)[0])
    tips = np.array(tips)
    low = tips[tips[:, 2] < s.slot_centers[0][2] + 0.110]
    assert np.linalg.norm(low[:, :2] - tips[0, :2], axis=1).max() < 0.004
