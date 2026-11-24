"""Ground-truth scoring and failure attribution of single attempts."""
import mujoco
import numpy as np

from binpick.planning.oracle import oracle_targets
from binpick.planning.types import GraspTarget
from binpick.sim.parts import PartSpec
from binpick.sim.suction import SuctionCup
from binpick.supervisor import machine
from binpick.supervisor.machine import Supervisor
from helpers import single_part_scene

BOX = PartSpec(0, "box", "plastic", (0.09, 0.07, 0.04), 0.3)


def _scene(cfg, xy=(0.0, 0.0)):
    s = single_part_scene(cfg, BOX, xy_offset=xy)
    for _ in range(150):
        mujoco.mj_step(s.model, s.data)
    return s


def test_wrong_center_estimate_is_misplaced_and_noticed(cfg):
    s = _scene(cfg)
    (t,) = oracle_targets(s)
    bad = GraspTarget(None, t.pos, t.normal, 0.0, t.depth, t.long_axis, center=t.pos + [0.022, 0, 0])
    r = Supervisor(s, np.random.default_rng(0), seed=0).attempt(bad)
    assert r.outcome == "fail" and r.reason == "misplaced" and r.detected


def test_unnoticed_misplacement_is_recorded_as_undetected(cfg, monkeypatch):
    s = _scene(cfg)
    (t,) = oracle_targets(s)
    bad = GraspTarget(None, t.pos, t.normal, 0.0, t.depth, t.long_axis, center=t.pos + [0.022, 0, 0])
    monkeypatch.setattr(machine, "verify_slot", lambda *a: (True, np.zeros(2)))
    r = Supervisor(s, np.random.default_rng(0), seed=0).attempt(bad)
    assert r.reason == "misplaced" and not r.detected


def test_seal_torn_while_landing_off_center_is_misplaced_not_collision(cfg, monkeypatch):
    """The part ended in its slot's area: whatever broke the seal, the result is a misplacement."""
    s = _scene(cfg)
    (t,) = oracle_targets(s)
    bad = GraspTarget(None, t.pos, t.normal, 0.0, t.depth, t.long_axis, center=t.pos + [0.022, 0, 0])
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    real = sup._probe

    def torn(*a, **k):
        out = real(*a, **k)
        if k.get("loaded"):
            sup.cup.off(), sup.cup.update()
            sup.cup.break_cause = "external_contact"
            return False, False
        return out

    monkeypatch.setattr(sup, "_probe", torn)
    r = sup.attempt(bad)
    assert r.sensed == "seal_lost_transport" and r.reason == "misplaced" and r.slot == 0


def test_seal_that_broke_during_the_grip_is_not_called_no_seal(cfg, monkeypatch):
    s = _scene(cfg)
    (t,) = oracle_targets(s)
    monkeypatch.setattr(SuctionCup, "utilization", lambda self: 5.0)
    r = Supervisor(s, np.random.default_rng(0), seed=0).attempt(t)
    assert r.sensed == "no_seal"          # all the robot can tell
    assert r.reason == "seal_lost_grip"   # what happened
    assert r.gripped_part == 0


def test_arm_hitting_the_bin_is_a_collision_even_if_the_part_lands(cfg, monkeypatch):
    s = _scene(cfg)
    (t,) = oracle_targets(s)
    lo, hi = s.bin_inner_bounds()
    wall = GraspTarget(None, np.array([lo[0] - 0.005, (lo[1] + hi[1]) / 2, hi[2] - 0.05]), t.normal, 0.0, 0.04, t.long_axis)
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    assert sup.attempt(wall) is None  # normally the robot sees it coming ...
    monkeypatch.setattr(sup.cell, "path_hits", lambda tr: False)
    r = sup.attempt(wall)             # ... otherwise the cup comes down on the bin wall
    assert r.reason == "collision" and r.arm_hit_n > 50


def test_normal_pick_has_no_arm_hit(cfg):
    s = _scene(cfg)
    r = Supervisor(s, np.random.default_rng(0), seed=0).attempt(oracle_targets(s)[0])
    assert r.outcome == "success" and r.arm_hit_n == 0.0
