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


def test_failures_do_not_reset_the_shake_count(cfg, monkeypatch):
    from binpick.supervisor import machine

    s = loose_parts_scene(cfg)
    monkeypatch.setattr(machine.SuctionCup, "sealed", lambda self: False)
    r = run_bin(s, "rules", seed=0)
    assert r.stop == "no_candidate" and r.shakes == 3 and len(r.records) < 40
