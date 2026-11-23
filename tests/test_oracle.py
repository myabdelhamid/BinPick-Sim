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
