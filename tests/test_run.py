import dataclasses
import json

import mujoco
import numpy as np
import pytest

from binpick.planning.oracle import oracle_targets
from binpick.sim.parts import PartSpec
from binpick.sim.scene import build_scene
from binpick.supervisor.machine import Supervisor
from binpick.supervisor.run import run_bin
from helpers import loose_parts_scene, quat_axis_angle, single_part_scene


def test_rules_planner_empties_a_loose_bin(cfg):
    s = loose_parts_scene(cfg)
    r = run_bin(s, "rules", seed=0)
    assert r.stop == "empty" and r.emptied and r.placed == 3 and r.left_in_bin == 0
    assert len(r.records) == 3 and all(x.outcome == "success" for x in r.records)
    assert sorted(x.slot for x in r.records) == [0, 1, 2]


def test_run_is_repeatable(cfg):
    a, b = (run_bin(loose_parts_scene(cfg), "random", seed=7, max_attempts=3) for _ in range(2))
    assert [(x.target_part, x.outcome, x.reason, x.cycle_time_s) for x in a.records] == \
           [(x.target_part, x.outcome, x.reason, x.cycle_time_s) for x in b.records]


def test_attempt_limit(cfg):
    r = run_bin(loose_parts_scene(cfg), "rules", seed=0, max_attempts=1)
    assert r.stop == "attempt_limit" and len(r.records) == 1 and not r.emptied and r.left_in_bin == 2


def test_gives_up_after_three_shakes_in_a_row(cfg):
    stick = PartSpec(0, "box", "plastic", (0.10, 0.040, 0.040), 0.2)  # no face fits the cup
    r = run_bin(single_part_scene(cfg, stick), "rules", seed=0)
    assert r.stop == "no_candidate" and r.shakes == 3 and r.records == [] and r.left_in_bin == 1


def test_shake_moves_parts_and_returns_home(cfg):
    ball = PartSpec(0, "cylinder", "plastic", (0.06, 0.08), 0.27)
    s = single_part_scene(cfg, ball, quat=quat_axis_angle([1, 0, 0], np.pi / 2))
    s.data.qpos[s.part_qpos_adr[0] + 2] = s.bin_inner_bounds()[0][2] + 0.03
    lo0, _ = s.bin_inner_bounds()
    b, rel = s.part_body_ids[0], []
    sup = Supervisor(s, np.random.default_rng(0), seed=0,
                     observer=lambda sc, ph: rel.append((sc.data.xpos[b] - sc.data.body("bin").xpos)[:2]))
    sup.shake(np.random.default_rng(1))
    lo1, hi1 = s.bin_inner_bounds()
    p1 = s.data.qpos[s.part_qpos_adr[0]:s.part_qpos_adr[0] + 3]
    assert np.ptp(rel, axis=0).max() > 0.010  # the part really moved about inside the bin
    assert np.allclose(lo0, lo1, atol=1e-3)
    assert np.all(p1[:2] > lo1[:2]) and np.all(p1[:2] < hi1[:2])
    assert [e["event"] for e in sup.events] == ["shake"]


def test_run_command_writes_outputs(tmp_path, cfg, monkeypatch):
    from binpick import cli

    monkeypatch.setattr(cli, "make_settled_scene", lambda c, seed: (loose_parts_scene(c), None))
    assert cli.run_main(["--seed", "1000", "--planner", "rules", "--out", str(tmp_path)]) == 0
    out = tmp_path / "run_seed1000_rules"
    s = json.loads((out / "summary.json").read_text())
    assert s["placed"] == 3 and s["stop"] == "empty" and s["attempts"] == 3
    assert (out / "attempts.csv").read_text().count("\n") == 4 and (out / "events.json").exists()


def test_a_spot_that_just_failed_is_not_tried_again_at_once(cfg, monkeypatch):
    """Every grip fails: the rules planner must move on to other parts instead of repeating the best one."""
    from binpick.supervisor import machine

    s = loose_parts_scene(cfg)
    monkeypatch.setattr(machine.SuctionCup, "sealed", lambda self: False)
    r = run_bin(s, "rules", seed=0, max_attempts=3)
    assert sorted(x.target_part for x in r.records) == [0, 1, 2]


def test_failing_attempts_run_to_the_attempt_limit(cfg, monkeypatch):
    """Spec: a run ends with no_candidate only if nothing can be picked after 3 shakes. Failing picks are picks."""
    from binpick.supervisor import machine

    monkeypatch.setattr(machine.SuctionCup, "sealed", lambda self: False)
    r = run_bin(loose_parts_scene(cfg), "rules", seed=0, max_attempts=8)
    assert r.stop == "attempt_limit" and len(r.records) == 8


def test_a_failed_grip_waits_for_a_shake_unless_another_safe_pick_is_left(cfg, monkeypatch):
    """The only other spots on the same surface are off-center: risky."""
    from binpick.supervisor import machine

    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.11, 0.11, 0.03), 0.4))
    monkeypatch.setattr(machine.SuctionCup, "sealed", lambda self: False)
    r = run_bin(s, "rules", seed=0, max_attempts=2)
    assert r.shakes == 1 and len(r.records) == 2


def test_summary_reports_all_cycle_times(cfg):
    r = run_bin(loose_parts_scene(cfg), "rules", seed=0)
    s = r.summary()
    assert s["cycle_mean_all_s"] == pytest.approx(np.mean([x.cycle_time_s for x in r.records]))
    assert s["sim_time_per_placed_s"] == pytest.approx(r.sim_time_s / r.placed)
    assert s["over_8s"] == sum(x.cycle_time_s > 8.0 for x in r.records if x.outcome == "success")


def test_a_part_lost_in_lift_is_left_alone_until_the_pile_changes(cfg, monkeypatch):
    """Losing the part while lifting means it is stuck: other spots on the same surface would fail too."""
    from binpick.supervisor import machine

    big = PartSpec(0, "box", "plastic", (0.11, 0.11, 0.05), 0.7)   # best target, several spots
    small = PartSpec(1, "box", "plastic", (0.07, 0.07, 0.03), 0.2)
    s = build_scene(cfg, [big, small])
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    for a, dx, h in zip(s.part_qpos_adr, (-0.1, 0.1), (0.05, 0.03)):
        s.data.qpos[a:a + 3] = [c[0] + dx, c[1], lo[2] + h / 2]
    for _ in range(150):
        mujoco.mj_step(s.model, s.data)

    def stuck(self, phase, rec, target, n):
        raise machine._Abort("seal_lost_lift")

    monkeypatch.setattr(machine.Supervisor, "_lift_transport_place", stuck)
    r = run_bin(s, "rules", seed=0, max_attempts=4)
    assert [x.target_part for x in r.records[:2]] == [0, 1]           # moves on at once
    assert r.events[0]["event"] == "shake" and r.shakes >= 1          # then shakes, then may try again
    assert len(r.records) == 4 and r.stop == "attempt_limit"


def test_a_pick_elsewhere_does_not_make_a_stuck_part_worth_retrying(cfg, monkeypatch):
    from binpick.supervisor import machine

    boxes = [PartSpec(0, "box", "plastic", (0.08, 0.08, 0.06), 0.4)] + \
            [PartSpec(k, "box", "plastic", (0.06, 0.06, 0.03), 0.13) for k in (1, 2)]
    s = build_scene(cfg, boxes)
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    for a, p, b in zip(s.part_qpos_adr, ([-0.13, 0], [0.13, 0.07], [0.13, -0.07]), boxes):
        s.data.qpos[a:a + 3] = [c[0] + p[0], c[1] + p[1], lo[2] + b.dims[2] / 2]
    for _ in range(150):
        mujoco.mj_step(s.model, s.data)
    real = machine.Supervisor._lift_transport_place

    def stuck_if_left(self, phase, rec, target, n):
        if target.pos[0] < c[0]:
            raise machine._Abort("seal_lost_lift")
        return real(self, phase, rec, target, n)

    monkeypatch.setattr(machine.Supervisor, "_lift_transport_place", stuck_if_left)
    r = run_bin(s, "rules", seed=0, max_attempts=3)
    assert [x.target_part for x in r.records][0] == 0 and sorted(x.target_part for x in r.records[1:]) == [1, 2]


def only_risky(cfg):
    """One box leaning on a small one: tilted, so never a safe pick."""
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.04), 0.3))
    return s


def test_shakes_before_taking_a_risky_pick(cfg, monkeypatch):
    from binpick.supervisor import run

    real = run.rules_targets
    monkeypatch.setattr(run, "rules_targets", lambda *a, **k: [dataclasses.replace(t, safe=False) for t in real(*a, **k)])
    r = run_bin(only_risky(cfg), "rules", seed=0)
    kinds = [e["event"] for e in r.events]
    assert kinds.count("shake") == cfg.run.max_shakes  # shakes first, then takes it
    assert len(r.records) >= 1 and r.placed == 1 and r.stop == "empty"


def test_a_bin_with_only_failing_risky_picks_still_ends(cfg, monkeypatch):
    from binpick.supervisor import machine, run

    real = run.rules_targets
    monkeypatch.setattr(run, "rules_targets", lambda *a, **k: [dataclasses.replace(t, safe=False) for t in real(*a, **k)])
    monkeypatch.setattr(machine.SuctionCup, "sealed", lambda self: False)
    r = run_bin(only_risky(cfg), "rules", seed=0, max_attempts=4)
    assert r.stop == "attempt_limit" and len(r.records) == 4
    assert r.shakes <= cfg.run.max_shakes + 3  # three before the first one, then one between risky picks


def test_safe_picks_are_taken_without_shaking(cfg):
    assert run_bin(loose_parts_scene(cfg), "rules", seed=0).shakes == 0


def test_the_camera_noise_depends_only_on_the_seed(cfg):
    s = loose_parts_scene(cfg)
    s.sensor_rng = np.random.default_rng(99)  # whatever was used before the run
    a = run_bin(s, "rules", seed=0)
    b = run_bin(loose_parts_scene(cfg), "rules", seed=0)
    assert [x.cycle_time_s for x in a.records] == [x.cycle_time_s for x in b.records]


@pytest.mark.parametrize("cols,stop", [(3, "empty"), (2, "tray_full")])
def test_a_full_tray_and_an_empty_bin_is_called_empty(cfg, cols, stop):
    small = dataclasses.replace(cfg, tray=dataclasses.replace(cfg.tray, rows=1, cols=cols))
    r = run_bin(loose_parts_scene(small), "rules", seed=0)
    assert r.placed == cols and r.stop == stop


def test_shakes_get_harder_if_the_first_does_not_help(cfg, monkeypatch):
    from binpick.supervisor import machine

    hard = []
    real = machine.Supervisor.shake
    monkeypatch.setattr(machine.Supervisor, "shake", lambda self, rng, hard_=False: (hard.append(hard_), real(self, rng, hard_))[1])
    stick = PartSpec(0, "box", "plastic", (0.10, 0.040, 0.040), 0.2)  # no face fits the cup
    run_bin(single_part_scene(cfg, stick), "rules", seed=0)
    assert hard == [False, True, True]


def test_a_hard_shake_keeps_a_full_pile_in_the_bin_and_moves_more(cfg):
    from binpick.sim.settle import make_settled_scene

    moved = {}
    for h in (False, True):
        s, _ = make_settled_scene(cfg, 1000)
        lo, hi = s.bin_inner_bounds()
        p0 = s.data.xpos[s.part_body_ids].copy()
        Supervisor(s, np.random.default_rng(0), seed=0).shake(np.random.default_rng(1), h)
        p = s.data.xpos[s.part_body_ids]
        assert np.all(p[:, :2] > lo[:2]) and np.all(p[:, :2] < hi[:2])
        moved[h] = np.linalg.norm(p - p0, axis=1).mean()
    assert moved[True] > moved[False]


def stick_and_box(cfg):
    s = build_scene(cfg, [PartSpec(0, "box", "plastic", (0.10, 0.040, 0.040), 0.2), PartSpec(1, "box", "plastic", (0.08, 0.08, 0.04), 0.3)])
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    for a, dx in zip(s.part_qpos_adr, (-0.1, 0.1)):
        s.data.qpos[a:a + 3] = [c[0] + dx, c[1], lo[2] + 0.02]
    for _ in range(150):
        mujoco.mj_step(s.model, s.data)
    return s


def test_three_shakes_without_a_pick_before_giving_up_even_after_a_risky_pick(cfg, monkeypatch):
    from binpick.supervisor import run

    real = run.rules_targets
    monkeypatch.setattr(run, "rules_targets", lambda *a, **k: [dataclasses.replace(t, safe=False) for t in real(*a, **k)])
    r = run_bin(stick_and_box(cfg), "rules", seed=0)
    assert r.placed == 1 and r.stop == "no_candidate" and r.shakes == 6


def test_an_unreachable_safe_target_does_not_block_risky_ones(cfg, monkeypatch):
    from binpick.supervisor import run

    real = run.rules_targets

    def far_safe_first(*a, **k):
        ts = real(*a, **k)
        far = [dataclasses.replace(ts[0], pos=np.array([3.0, 0.0, 0.1]), safe=True)] if ts else []
        return far + [dataclasses.replace(t, safe=False) for t in ts]

    monkeypatch.setattr(run, "rules_targets", far_safe_first)
    r = run_bin(stick_and_box(cfg), "rules", seed=0)
    assert r.placed == 1 and r.records[0].outcome == "success"
