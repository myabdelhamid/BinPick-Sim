import mujoco
import numpy as np
import pytest

from binpick.perception.view import observe_bin
from binpick.planning.rules import rules_targets
from binpick.sim.parts import PartSpec
from binpick.sim.scene import build_scene
from binpick.sim.settle import make_settled_scene
from binpick.sim.suction import seal_quality
from helpers import quat_axis_angle, single_part_scene, teleport_cup


def targets(s):
    mujoco.mj_forward(s.model, s.data)
    return rules_targets(observe_bin(s), s.cfg)


def lying_cylinder(cfg, dia, length=0.10):
    s = single_part_scene(cfg, PartSpec(0, "cylinder", "plastic", (dia, length), 0.2), quat=quat_axis_angle([1, 0, 0], np.pi / 2))
    s.data.qpos[s.part_qpos_adr[0] + 2] = s.bin_inner_bounds()[0][2] + dia / 2
    return s


def two_parts(cfg, a, b, pa, pb, qa=(1, 0, 0, 0), qb=(1, 0, 0, 0)):
    s = build_scene(cfg, [a, b])
    lo, hi = s.bin_inner_bounds()
    base = np.r_[(lo + hi)[:2] / 2, lo[2]]
    for adr, p, q in zip(s.part_qpos_adr, (pa, pb), (qa, qb)):
        s.data.qpos[adr:adr + 3] = base + p
        s.data.qpos[adr + 3:adr + 7] = q
    return s


def test_flat_box_best_target_is_its_center(cfg):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.10, 0.07, 0.04), 0.3), quat=quat_axis_angle([0, 0, 1], 0.4))
    t = targets(s)[0]
    c = s.data.xpos[s.part_body_ids[0]] + [0, 0, 0.02]
    assert np.allclose(t.pos, c, atol=0.003) and np.allclose(t.center, c, atol=0.003)
    assert t.tilt_deg < 1 and t.seal > 0.95 and t.part_index is None
    assert t.depth == pytest.approx(0.04, abs=0.002)  # lying on the floor: the bound is the thickness
    assert abs(t.long_axis @ [np.cos(0.4), np.sin(0.4), 0]) > np.cos(np.deg2rad(3))


def test_steep_face_is_rejected(cfg):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.10, 0.10, 0.04), 0.4), quat=quat_axis_angle([1, 0, 0], np.deg2rad(26)))
    s.data.qpos[s.part_qpos_adr[0] + 2] += 0.03
    assert targets(s) == []


@pytest.mark.parametrize("dia", [0.042, 0.06, 0.08])
def test_cylinder_side_is_not_rejected_for_being_curved(cfg, dia):
    s = lying_cylinder(cfg, dia)
    t = targets(s)[0]
    c = s.data.xpos[s.part_body_ids[0]]
    assert abs(t.pos[0] - c[0]) < 0.003 and t.tilt_deg < 4
    assert 0.7 < t.seal < 0.95
    assert abs(t.long_axis @ [0, 1, 0]) > np.cos(np.deg2rad(5))


def test_flat_before_curved(cfg):
    s = two_parts(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.03), 0.2), PartSpec(1, "cylinder", "plastic", (0.07, 0.10), 0.4),
                  [-0.10, 0, 0.015], [0.10, 0, 0.035], qb=quat_axis_angle([1, 0, 0], np.pi / 2))
    ts = targets(s)
    assert ts[0].pos[0] < ts[-1].pos[0]  # box (lower, flat) first, cylinder (higher, curved) after


def test_higher_part_first(cfg):
    s = two_parts(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.03), 0.2), PartSpec(1, "box", "plastic", (0.08, 0.08, 0.07), 0.5),
                  [-0.10, 0, 0.015], [0.10, 0, 0.035])
    assert targets(s)[0].pos[2] > 0.07


def test_points_near_walls_are_rejected(cfg):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.06, 0.10, 0.04), 0.3))
    lo, _ = s.bin_inner_bounds()
    s.data.qpos[s.part_qpos_adr[0]] = lo[0] + 0.031
    ts = targets(s)
    need = cfg.cup.diameter / 2 + cfg.planner.wall_clearance
    assert all(t.pos[0] - lo[0] >= need - 1e-9 for t in ts)


def test_neighbour_standing_over_the_point_blocks_it(cfg):
    s = two_parts(cfg, PartSpec(0, "box", "plastic", (0.06, 0.06, 0.03), 0.1), PartSpec(1, "box", "plastic", (0.06, 0.06, 0.07), 0.3),
                  [0, 0, 0.015], [0.061, 0, 0.035])
    assert all(t.pos[0] > 0.55 + 0.03 or t.pos[0] < 0.55 + 0.03 - 0.022 for t in targets(s))


def test_face_smaller_than_the_cup_gives_nothing(cfg):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.10, 0.040, 0.040), 0.2))
    assert targets(s) == []


def test_estimated_seal_agrees_with_the_physics(cfg):
    """The camera-based seal estimate must predict the physical seal: no target may be a miss."""
    n = 0
    for seed in (1000, 1001, 1002):
        s, _ = make_settled_scene(cfg, seed)
        for t in targets(s)[:8]:
            q0 = s.data.qpos.copy()
            try:
                teleport_cup(s, t.pos, -t.normal)
            except AssertionError:
                continue
            g = np.zeros(1, np.int32)
            mujoco.mj_ray(s.model, s.data, t.pos + 0.001 * t.normal, -t.normal, None, 1, s.model.geom(s.cup_geom).bodyid[0], g)
            i = s.part_body_ids.index(s.model.geom_bodyid[g[0]])
            assert seal_quality(s, i) == pytest.approx(t.seal, abs=0.12), (seed, t)
            s.data.qpos[:] = q0
            mujoco.mj_forward(s.model, s.data)
            n += 1
    assert n >= 5


def test_free_part_before_a_higher_crowded_one(cfg):
    """A part with taller neighbours against it is likely pinned: prefer one that can be lifted freely."""
    a = PartSpec(0, "box", "plastic", (0.08, 0.08, 0.050), 0.3)
    b = PartSpec(1, "box", "plastic", (0.08, 0.08, 0.040), 0.25)
    walls = [PartSpec(2 + k, "box", "plastic", (0.02, 0.08, 0.07), 0.13) for k in range(2)]
    s = build_scene(cfg, [a, b, *walls])
    lo, hi = s.bin_inner_bounds()
    base = np.r_[(lo + hi)[:2] / 2, lo[2]]
    for adr, p in zip(s.part_qpos_adr, ([-0.1, 0, 0.025], [0.1, 0, 0.020], [-0.1 - 0.0505, 0, 0.035], [-0.1 + 0.0505, 0, 0.035])):
        s.data.qpos[adr:adr + 3] = base + p
    t = targets(s)[0]
    assert t.pos[0] > base[0]  # the lower but free box
