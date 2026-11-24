import numpy as np

from binpick.motion.collision import CellCheck
from binpick.motion.ik import DOWN, solve_cup_pose
from binpick.motion.trajectory import JointTrajectory, joint_move
from binpick.planning.oracle import oracle_targets
from binpick.sim.parts import PartSpec
from binpick.sim.scene import build_scene
from binpick.supervisor import machine
from binpick.supervisor.machine import Supervisor
from helpers import single_part_scene, teleport_cup

BOX = PartSpec(0, "box", "plastic", (0.08, 0.08, 0.04), 0.3)


def test_arm_in_the_wall_is_found_and_free_poses_are_not(cfg):
    s = build_scene(cfg, [])
    check = CellCheck(s)
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    wall = solve_cup_pose(s, np.r_[hi[0] - 0.005, c[1], lo[2] + 0.05], DOWN)
    above = solve_cup_pose(s, np.r_[c[:2], hi[2] + 0.1], DOWN)
    inside = solve_cup_pose(s, np.r_[c[:2], lo[2] + 0.05], DOWN)
    table = solve_cup_pose(s, np.r_[0.4, -0.3, -0.02], DOWN)
    assert wall.ok and above.ok and inside.ok and table.ok
    assert check.hits(wall.q) and check.hits(table.q)
    assert not check.hits(above.q) and not check.hits(inside.q)


def test_a_path_through_the_wall_is_found(cfg):
    s = build_scene(cfg, [])
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    a = solve_cup_pose(s, np.r_[c[:2], lo[2] + 0.06], DOWN)
    b = solve_cup_pose(s, np.r_[c[0], hi[1] + 0.15, lo[2] + 0.06], DOWN)
    assert a.ok and b.ok and not CellCheck(s).hits(a.q) and not CellCheck(s).hits(b.q)
    assert CellCheck(s).path_hits(joint_move(a.q, b.q, cfg.motion))


def test_parts_do_not_count(cfg):
    """The robot knows its cell, not where the parts are."""
    s = single_part_scene(cfg, BOX)
    c = s.data.xpos[s.part_body_ids[0]]
    assert not CellCheck(s).hits(solve_cup_pose(s, c, DOWN).q)


def test_checking_leaves_the_simulation_alone(cfg):
    s = single_part_scene(cfg, BOX)
    q, ctrl, t = s.data.qpos.copy(), s.data.ctrl.copy(), s.data.time
    CellCheck(s).hits(np.zeros(7))
    assert np.array_equal(q, s.data.qpos) and np.array_equal(ctrl, s.data.ctrl) and t == s.data.time


def test_a_target_whose_approach_hits_the_cell_is_rejected_before_moving(cfg, monkeypatch):
    s = single_part_scene(cfg, BOX)
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    monkeypatch.setattr(sup.cell, "path_hits", lambda tr: True)
    assert sup.attempt(oracle_targets(s)[0]) is None and sup.records == []
    assert "collision" in sup.events[0]["detail"]


def test_recovery_backs_out_the_way_it_came_if_straight_up_is_impossible(cfg, monkeypatch):
    """Deep in the bin, no line up: a direct joint move to the view pose would sweep the bin."""
    s = single_part_scene(cfg, BOX)
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    monkeypatch.setattr(machine.SuctionCup, "sealed", lambda self: False)  # fails at the bottom of the approach
    real_line, tips = sup._line, []

    def no_line_up(p0, p1, *a, **k):
        if p1[2] > p0[2] + 0.02:
            raise machine.IKError("test")
        return real_line(p0, p1, *a, **k)

    monkeypatch.setattr(sup, "_line", no_line_up)
    sup.observer = lambda sc, ph: tips.append(sc.data.site(sc.cup_site).xpos.copy()) if ph == "retreat" else None
    rec = sup.attempt(oracle_targets(s)[0])
    assert rec.sensed == "no_seal"
    tips = np.array(tips)
    rim = s.bin_inner_bounds()[1][2]
    low = tips[tips[:, 2] < rim]
    assert len(low) and np.ptp(low[:, :2], axis=0).max() < 0.02  # no sideways travel below the rim
    assert tips[-1][2] > rim
