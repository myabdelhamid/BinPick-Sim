import numpy as np
import pytest

from binpick.motion.ik import DOWN
from binpick.motion.trajectory import cartesian_line, execute
from binpick.sim.parts import PartSpec
from binpick.sim.suction import SuctionCup, seal_quality
from helpers import part_top, quat_axis_angle, single_part_scene, teleport_cup

STEEL_BOX = PartSpec(0, "box", "steel", (0.10, 0.10, 0.03), 0.10 * 0.10 * 0.03 * 7850)  # 2.36 kg


def test_flat_top_centered_seals(cfg):
    s = single_part_scene(cfg, STEEL_BOX)
    teleport_cup(s, part_top(s) + [0, 0, 0.001])
    assert seal_quality(s, 0) == pytest.approx(1.0)


def test_cup_far_above_does_not_seal(cfg):
    s = single_part_scene(cfg, STEEL_BOX)
    teleport_cup(s, part_top(s) + [0, 0, 0.02])
    assert seal_quality(s, 0) == 0.0


def test_tilted_30deg_face_vertical_cup_fails(cfg):
    s = single_part_scene(cfg, STEEL_BOX, quat=quat_axis_angle([1, 0, 0], np.deg2rad(30)), z=None)
    teleport_cup(s, part_top(s) + [0, 0, 0.001])
    assert seal_quality(s, 0) == 0.0


def test_thin_cylinder_side_fails(cfg):
    cyl = PartSpec(0, "cylinder", "plastic", (0.040, 0.100), 0.15)
    s = single_part_scene(cfg, cyl, quat=quat_axis_angle([1, 0, 0], np.pi / 2), z=None)
    lo, _ = s.bin_inner_bounds()
    a = s.part_qpos_adr[0]
    s.data.qpos[a + 2] = lo[2] + 0.020
    teleport_cup(s, s.data.qpos[a:a + 3] + [0, 0, 0.020 + 0.001])
    assert seal_quality(s, 0) == 0.0


def test_fat_cylinder_side_seals_partially(cfg):
    cyl = PartSpec(0, "cylinder", "plastic", (0.080, 0.100), 0.6)
    s = single_part_scene(cfg, cyl, quat=quat_axis_angle([1, 0, 0], np.pi / 2), z=None)
    lo, _ = s.bin_inner_bounds()
    a = s.part_qpos_adr[0]
    s.data.qpos[a + 2] = lo[2] + 0.040
    teleport_cup(s, s.data.qpos[a:a + 3] + [0, 0, 0.040 + 0.001])
    # sagitta of a 40 mm radius under a 22 mm lip radius = 6.6 mm -> sealed but below 1
    assert 2 / 3 < seal_quality(s, 0) < 1.0


def test_cup_half_over_edge_fails(cfg):
    box = PartSpec(0, "box", "plastic", (0.06, 0.06, 0.03), 0.13)
    s = single_part_scene(cfg, box)
    teleport_cup(s, part_top(s) + [0.03, 0, 0.001])
    assert seal_quality(s, 0) == 0.0


def _pick_and_lift(s, grip_pos, lift=0.25):  # cup ends 250 mm up: hanging part clears the 210 mm rim
    cup = SuctionCup(s, np.random.default_rng(0))
    q = teleport_cup(s, grip_pos + [0, 0, 0.03])
    tr = cartesian_line(s, q, grip_pos + [0, 0, 0.03], grip_pos - [0, 0, s.cfg.motion.approach_overtravel],
                        DOWN, 0.0, s.cfg.motion)
    execute(s, tr, lambda sc: cup.update() or True)
    cup.on()
    execute(s, tr.__class__(np.array([0.0]), tr.q[-1:]), lambda sc: cup.update() or True,
            hold_s=s.cfg.timing.vacuum_build_s)
    sealed = cup.sealed()
    q = tr.q[-1]
    up = cartesian_line(s, q, grip_pos, grip_pos + [0, 0, lift], DOWN, 0.0, s.cfg.motion)
    execute(s, up, lambda sc: cup.update() or True)
    side = cartesian_line(s, up.q[-1], grip_pos + [0, 0, lift], grip_pos + [0.0, 0.15, lift], DOWN, 0.0, s.cfg.motion)
    execute(s, side, lambda sc: cup.update() or True)
    return cup, sealed


def test_heavy_centered_part_holds_through_lift_and_transport(cfg):
    s = single_part_scene(cfg, STEEL_BOX)
    for _ in range(200):
        import mujoco
        mujoco.mj_step(s.model, s.data)
    z0 = s.data.xpos[s.part_body_ids[0]][2]
    cup, sealed = _pick_and_lift(s, part_top(s))
    assert sealed
    assert cup.sealed() and cup.pressure_kpa() < -55
    assert s.data.xpos[s.part_body_ids[0]][2] > z0 + 0.20


def test_capacity_is_rated_hold_with_safety_factor(cfg):
    from binpick.sim.suction import capacity_n

    s = single_part_scene(cfg, STEEL_BOX)
    sf = capacity_n(s) / (cfg.cup.rated_hold_kg * 9.81)
    assert 2.0 <= sf <= 3.0


def test_grip_does_not_move_part(cfg):
    import mujoco

    s = single_part_scene(cfg, STEEL_BOX)
    for _ in range(200):
        mujoco.mj_step(s.model, s.data)
    teleport_cup(s, part_top(s) - [0, 0, 0.0005])
    p0 = s.data.xpos[s.part_body_ids[0]].copy()
    cup = SuctionCup(s, np.random.default_rng(0))
    cup.on()
    for _ in range(100):
        mujoco.mj_step(s.model, s.data)
        cup.update()
    assert cup.holding
    assert np.linalg.norm(s.data.xpos[s.part_body_ids[0]] - p0) < 1e-3


def test_off_center_heavy_bar_peels_off(cfg):
    bar = PartSpec(0, "box", "steel", (0.110, 0.050, 0.050), 0.110 * 0.05 * 0.05 * 7850)  # 2.16 kg
    s = single_part_scene(cfg, bar)
    import mujoco
    for _ in range(200):
        mujoco.mj_step(s.model, s.data)
    z0 = s.data.xpos[s.part_body_ids[0]][2]
    grip = part_top(s) + [0.110 / 2 - 0.025, 0, 0]  # cup at the end of the bar
    cup, _ = _pick_and_lift(s, grip)
    assert s.data.xpos[s.part_body_ids[0]][2] < z0 + 0.05  # it did not come up with the cup
    assert not cup.sealed()


def test_sensor_reads_ambient_when_off(cfg):
    s = single_part_scene(cfg, STEEL_BOX)
    cup = SuctionCup(s, np.random.default_rng(0))
    cup.update()
    assert abs(cup.pressure_kpa()) < 5 and not cup.sealed()
