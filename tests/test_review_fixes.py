"""Regression tests for the M2 final-review findings."""
import mujoco
import numpy as np
import pytest

from binpick.motion.ik import cup_rotation, solve_cup_down, yaw_of
from binpick.motion.trajectory import IKError, cartesian_line
from binpick.planning.oracle import oracle_targets
from binpick.sim.parts import PartSpec
from binpick.sim.scene import build_scene
from binpick.sim.settle import make_settled_scene
from binpick.sim.suction import SuctionCup, seal_quality
from binpick.supervisor import machine
from binpick.supervisor.machine import Supervisor
from binpick.supervisor.placement import SlotAllocator
from helpers import part_top, single_part_scene, teleport_cup


# --- 1. lift goes straight up, and wall tear-offs are attributed to collision ---------------
def test_tilted_grasp_lifts_straight_up(cfg):
    s, _ = make_settled_scene(cfg, 1001)
    t = next(t for t in oracle_targets(s) if t.tilt_deg > 8)  # 16 deg face, free to lift
    p = []
    sup = Supervisor(s, np.random.default_rng(0), seed=1001,
                     observer=lambda sc, ph: p.append(sc.data.site(sc.cup_site).xpos.copy()) if ph == "lift" else None)
    sup.attempt(t)
    p = np.array(p)
    assert p[:, 2].max() - p[0, 2] > 0.04          # the lift actually happened (part not torn off at once)
    assert np.abs(p[:, :2] - p[0, :2]).max() < 0.003  # and it went straight up


def test_tear_off_against_wall_is_collision(cfg):
    box = PartSpec(0, "box", "steel", (0.08, 0.08, 0.04), 2.0)
    s = single_part_scene(cfg, box)
    for _ in range(200):
        mujoco.mj_step(s.model, s.data)
    cup = SuctionCup(s, np.random.default_rng(0))
    q = teleport_cup(s, part_top(s) - [0, 0, 0.0005])
    cup.on()
    for _ in range(100):
        mujoco.mj_step(s.model, s.data)
        cup.update()
    assert cup.holding
    lo, hi = s.bin_inner_bounds()
    p0 = part_top(s) + [0, 0, 0.01]
    from binpick.motion.trajectory import execute
    tr = cartesian_line(s, q, p0, np.r_[hi[0] + 0.05, p0[1], p0[2]], np.array([0, 0, -1.0]), 0.0, cfg.motion)
    execute(s, tr, lambda sc: cup.update() or True)
    assert not cup.holding and cup.break_cause == "external_contact"


# --- 2. yaw recovery for tilted cups, and no step at waypoint 0 ---------------------------
@pytest.mark.parametrize("tilt,yaw", [(0.0, 0.4), (np.deg2rad(15), -1.2), (np.deg2rad(20), 2.5)])
def test_yaw_of_inverts_cup_rotation(tilt, yaw):
    axis = np.array([np.sin(tilt), 0.3 * np.sin(tilt), -np.cos(tilt)])
    axis /= np.linalg.norm(axis)
    R = cup_rotation(axis, yaw)
    assert np.allclose(cup_rotation(axis, yaw_of(R)), R, atol=1e-9)


def test_cartesian_line_rejects_jump_at_first_waypoint(cfg):
    s = build_scene(cfg, [])
    lo, hi = s.bin_inner_bounds()
    p = (lo + hi) / 2 + [0, 0, 0.2]
    q = solve_cup_down(s, p, yaw=0.0).q
    with pytest.raises(IKError):
        cartesian_line(s, q, p, p - [0, 0, 0.05], np.array([0, 0, -1.0]), 1.3, cfg.motion)  # 1.3 rad wrist step


# --- 3. seal requires the lip to sit on the target, not on a neighbour -----------------------
def test_neighbour_under_lip_breaks_seal(cfg):
    target = PartSpec(0, "box", "plastic", (0.10, 0.10, 0.03), 0.36)
    plate = PartSpec(1, "box", "plastic", (0.05, 0.10, 0.003), 0.02)
    s = build_scene(cfg, [target, plate])
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    a0, a1 = s.part_qpos_adr
    s.data.qpos[a0:a0 + 3] = [c[0], c[1], lo[2] + 0.015]
    s.data.qpos[a1:a1 + 3] = [c[0] + 0.03, c[1], lo[2] + 0.03 + 0.0035]  # thin plate over half the lip
    mujoco.mj_forward(s.model, s.data)
    teleport_cup(s, [c[0], c[1], lo[2] + 0.03 + 0.005 + 0.0005])  # cup resting on the plate, 5 mm above target
    assert seal_quality(s, 0) == 0.0


# --- 4. slots are used only by releases --------------------------------------------------------
def test_slot_peek_commit():
    al = SlotAllocator(2)
    assert al.peek() == 0 and al.peek() == 0
    al.commit()
    assert al.peek() == 1
    al.commit()
    assert al.peek() is None


def test_failed_transport_does_not_use_a_slot(cfg, monkeypatch):
    s, _ = make_settled_scene(cfg, 1000)
    real = machine.joint_move

    def failing(q0, q1, m, loaded=False):
        if loaded:
            raise IKError("forced transport failure")
        return real(q0, q1, m, loaded)

    monkeypatch.setattr(machine, "joint_move", failing)
    sup = Supervisor(s, np.random.default_rng(0), seed=1000)
    r = sup.attempt(oracle_targets(s)[0])
    assert r.outcome == "fail" and r.reason == "ik_fail" and r.slot is None
    assert sup.slots.peek() == 0


# --- minor 2: display pressure must not consume the sensor noise stream -------------------------
def test_display_pressure_does_not_draw_noise(cfg):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.1, 0.1, 0.03), 0.36))
    cup = SuctionCup(s, np.random.default_rng(0))
    before = cup.rng.bit_generator.state
    cup.pressure_display_kpa()
    assert cup.rng.bit_generator.state == before
