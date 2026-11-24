import numpy as np
import pytest

from binpick.planning.oracle import oracle_targets
from binpick.sim.parts import PartSpec
from binpick.sim.settle import make_settled_scene
from helpers import part_top, quat_axis_angle, single_part_scene


def test_flat_box_target_is_top_center(cfg):
    box = PartSpec(0, "box", "plastic", (0.09, 0.06, 0.03), 0.2)
    s = single_part_scene(cfg, box, quat=quat_axis_angle([0, 0, 1], 0.3))
    (t,) = oracle_targets(s)
    assert np.allclose(t.pos, part_top(s), atol=1e-6)
    assert t.normal @ [0, 0, 1] > 0.9999 and t.tilt_deg < 0.1
    assert t.depth == pytest.approx(0.03)
    assert abs(t.long_axis @ [np.cos(0.3), np.sin(0.3), 0]) > 0.9999  # long side is the 90 mm edge


def test_face_smaller_than_cup_is_skipped(cfg):
    stick = PartSpec(0, "box", "plastic", (0.10, 0.040, 0.040), 0.2)  # every face < 44 mm on one side
    s = single_part_scene(cfg, stick)
    assert oracle_targets(s) == []


def test_upright_cylinder_uses_end(cfg):
    cyl = PartSpec(0, "cylinder", "plastic", (0.06, 0.05), 0.17)
    s = single_part_scene(cfg, cyl)
    (t,) = oracle_targets(s)
    assert t.normal @ [0, 0, 1] > 0.9999 and t.depth == pytest.approx(0.05)


def test_pile_targets_sorted_unoccluded_and_flat_enough(cfg):
    s, _ = make_settled_scene(cfg, 1000)
    ts = oracle_targets(s)
    assert len(ts) >= 1  # a fresh pile exposes 2-4 clear faces (seeds 1000-1005)
    zs = [t.pos[2] for t in ts]
    assert zs == sorted(zs, reverse=True)
    assert all(t.tilt_deg <= 20.0 for t in ts)


def test_thin_lying_cylinder_is_graspable_from_its_side(cfg):
    cyl = PartSpec(0, "cylinder", "plastic", (0.042, 0.100), 0.16)
    s = single_part_scene(cfg, cyl, quat=quat_axis_angle([1, 0, 0], np.pi / 2))
    s.data.qpos[s.part_qpos_adr[0] + 2] = s.bin_inner_bounds()[0][2] + 0.021
    (t,) = oracle_targets(s)
    assert t.normal @ [0, 0, 1] > 0.9999 and t.depth == pytest.approx(0.042)


def test_neighbour_standing_beside_thin_cylinder_blocks_the_cup(cfg):
    from binpick.sim.scene import build_scene

    cyl = PartSpec(0, "cylinder", "plastic", (0.042, 0.100), 0.16)
    wall = PartSpec(1, "box", "plastic", (0.020, 0.10, 0.07), 0.15)  # 70 mm tall, 0.5 mm from the cylinder
    s = build_scene(cfg, [cyl, wall])
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2
    a0, a1 = s.part_qpos_adr
    s.data.qpos[a0:a0 + 3] = [c[0], c[1], lo[2] + 0.021]
    s.data.qpos[a0 + 3:a0 + 7] = quat_axis_angle([1, 0, 0], np.pi / 2)
    s.data.qpos[a1:a1 + 3] = [c[0] + 0.021 + 0.0005 + 0.010, c[1], lo[2] + 0.035]
    assert all(t.part_index != 0 for t in oracle_targets(s))
