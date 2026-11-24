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


def _cylinder_side_q(cfg, dia, length=0.100, lateral=0.0, roll_deg=0.0):
    """Seal quality with the cup on the side of a lying cylinder (axis along world y)."""
    cyl = PartSpec(0, "cylinder", "plastic", (dia, length), 0.15)
    s = single_part_scene(cfg, cyl, quat=quat_axis_angle([1, 0, 0], np.pi / 2), z=None)
    lo, _ = s.bin_inner_bounds()
    a = s.part_qpos_adr[0]
    s.data.qpos[a + 2] = lo[2] + dia / 2
    th = np.deg2rad(roll_deg)
    top = s.data.qpos[a:a + 3] + [lateral, 0, dia / 2 + 0.001]
    teleport_cup(s, top, (np.sin(th), 0, -np.cos(th)))
    return seal_quality(s, 0)


# Piab B40.20 datasheet: minimum curve radius 0.79 in = 20 mm, so a 40 mm cylinder is the limit
def test_cylinder_at_curve_limit_seals(cfg):
    q = _cylinder_side_q(cfg, 0.040)
    assert q > cfg.suction.sealed_threshold_kpa / cfg.suction.vacuum_kpa  # vacuum switch confirms the grip
    assert q < 0.80                                                       # but well below a flat seal


def test_cylinder_below_curve_limit_fails(cfg):
    assert _cylinder_side_q(cfg, 0.036) == 0.0


def test_seal_weakens_as_cylinders_get_thinner(cfg):
    qs = [_cylinder_side_q(cfg, d) for d in (0.080, 0.068, 0.056, 0.044)]
    assert all(x > y for x, y in zip(qs, qs[1:]))


def test_cylinder_shorter_than_cup_fails(cfg):
    assert _cylinder_side_q(cfg, 0.060, length=0.040) == 0.0


def test_cup_beside_the_top_line_of_a_thin_cylinder_fails(cfg):
    assert _cylinder_side_q(cfg, 0.044, lateral=0.015) == 0.0


def test_softer_curve_limit_does_not_loosen_the_tilt_limit(cfg):
    # the datasheet gives no tilt data, so tilt keeps the M2 limits: sealed at 8 deg, gone by 16 deg
    for deg, ok in ((8, True), (16, False)):
        s = single_part_scene(cfg, STEEL_BOX, quat=quat_axis_angle([1, 0, 0], np.deg2rad(deg)), z=None)
        teleport_cup(s, part_top(s) + [0, 0, 0.001])
        q = seal_quality(s, 0)
        assert (q > 2 / 3) if ok else (q == 0.0), (deg, q)


def test_lip_bent_over_a_sharp_edge_fails(cfg):
    # 10 deg tilted box, lip 2 mm past the lower edge: the side face is hit, but a lip cannot fold that sharply
    th = np.deg2rad(10)
    box = PartSpec(0, "box", "plastic", (0.06, 0.08, 0.06), 0.3)
    s = single_part_scene(cfg, box, quat=quat_axis_angle([0, 1, 0], th), z=None)
    R = s.data.xmat[s.part_body_ids[0]].reshape(3, 3)
    c = s.data.xpos[s.part_body_ids[0]] + R @ [0.030 - 0.020, 0, 0.030]
    teleport_cup(s, c + [0, 0, 0.004])
    assert seal_quality(s, 0) == 0.0


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


def _lift_mass(cfg, kg):
    import mujoco

    box = PartSpec(0, "box", "steel", (0.10, 0.10, kg / (0.01 * 7850)), kg)
    s = single_part_scene(cfg, box)
    for _ in range(200):
        mujoco.mj_step(s.model, s.data)
    z0 = s.data.xpos[s.part_body_ids[0]][2]
    cup, sealed = _pick_and_lift(s, part_top(s))
    return cup, sealed, s.data.xpos[s.part_body_ids[0]][2] - z0


def test_heavy_centered_part_holds_through_lift_and_transport(cfg):
    cup, sealed, rise = _lift_mass(cfg, 1.75)
    assert sealed
    assert cup.sealed() and cup.pressure_kpa() < -55
    assert rise > 0.20


def test_pressing_onto_a_part_does_not_break_the_seal(cfg):
    import mujoco

    s = single_part_scene(cfg, STEEL_BOX)
    for _ in range(200):
        mujoco.mj_step(s.model, s.data)
    cup = SuctionCup(s, np.random.default_rng(0))
    top = part_top(s)
    q = teleport_cup(s, top + [0, 0, 0.03])
    tr = cartesian_line(s, q, top + [0, 0, 0.03], top - [0, 0, 0.004], DOWN, 0.0, s.cfg.motion)  # 4 mm too deep
    execute(s, tr, lambda sc: True)
    cup.on()
    execute(s, tr.__class__(np.array([0.0]), tr.q[-1:]), lambda sc: cup.update() or True, hold_s=0.3)
    assert cup.holding, cup.break_cause


def test_capacity_is_the_b40_datasheet_force(cfg):
    from binpick.sim.suction import capacity_n

    s = single_part_scene(cfg, STEEL_BOX)
    assert cfg.suction.vacuum_kpa == 60.0
    assert capacity_n(s) == pytest.approx(39.0, abs=0.1)  # 8.77 lb at 17.72 -inHg (60 kPa)


def test_grip_does_not_move_part(cfg):
    import mujoco

    s = single_part_scene(cfg, STEEL_BOX)
    for _ in range(200):
        mujoco.mj_step(s.model, s.data)
    teleport_cup(s, part_top(s) - [0, 0, 0.0005])
    for _ in range(100):  # the arm settles on the part first, as in a real approach
        mujoco.mj_step(s.model, s.data)
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
