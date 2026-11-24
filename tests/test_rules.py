import mujoco
import numpy as np
import pytest

from binpick.perception.view import observe_bin
from binpick.planning.rules import rules_targets
from binpick.sim.parts import PartSpec
from binpick.sim.scene import build_scene
from binpick.sim.settle import make_settled_scene
from binpick.sim.suction import seal_quality
from helpers import quat_axis_angle, quat_mul, single_part_scene, teleport_cup


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


@pytest.mark.parametrize("gap_mm", [0.0, 1.0, 2.0])
def test_never_grips_the_seam_between_two_parts(cfg, gap_mm):
    """Two boxes flush against each other at the same height: whatever the camera can tell apart, the cup must
    not go on the seam, and no target may be centered between them."""
    b = [PartSpec(i, "box", "plastic", (0.08, 0.08, 0.04), 0.3) for i in (0, 1)]
    d = 0.04 + gap_mm * 1e-3 / 2
    s = two_parts(cfg, *b, [-d, 0, 0.02], [d, 0, 0.02])
    for t in targets(s):
        assert abs(t.pos[0] - 0.55) > 0.022 and abs(t.center[0] - 0.55) > 0.03, (t.pos, t.center)


def test_tool_leaning_into_a_wall_is_rejected(cfg):
    """A tilted face low in the bin, facing a wall: the lip fits, but the tool would hit the wall above."""
    import dataclasses

    th = np.deg2rad(14)
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.10, 0.10, 0.04), 0.4), quat=quat_axis_angle([0, 1, 0], -th))
    lo, hi = s.bin_inner_bounds()
    s.data.qpos[s.part_qpos_adr[0]:s.part_qpos_adr[0] + 3] = [lo[0] + 0.062, 0.0, lo[2] + 0.034]
    mujoco.mj_forward(s.model, s.data)
    view = observe_bin(s)

    def at_rim(t):
        return (t.pos + t.normal * (hi[2] - t.pos[2]) / t.normal[2])[0] - lo[0]

    need = cfg.cup.diameter / 2 + cfg.planner.wall_clearance
    loose = dataclasses.replace(cfg, planner=dataclasses.replace(cfg.planner, wall_clearance=-1.0))
    without = rules_targets(view, loose)
    assert without and min(at_rim(t) for t in without) < need  # the check is what rejects them
    assert all(at_rim(t) >= need for t in rules_targets(view, cfg))


def test_obstruction_check(cfg):
    from binpick.planning.rules import _obstructed

    s = two_parts(cfg, PartSpec(0, "box", "plastic", (0.10, 0.10, 0.02), 0.2), PartSpec(1, "box", "plastic", (0.04, 0.10, 0.07), 0.3),
                  [0, 0, 0.010], [0.0705, 0, 0.035])  # tall box face at x = +0.0505
    mujoco.mj_forward(s.model, s.data)
    view, base = observe_bin(s), np.array([0.55, 0.0, s.bin_inner_bounds()[0][2] + 0.02])
    up, lean = np.array([0, 0, 1.0]), np.array([np.sin(np.deg2rad(20)), 0, np.cos(np.deg2rad(20))])
    assert _obstructed(view, base + [0.0505 - 0.023, 0, 0], up, cfg)       # inside cup radius + clearance
    assert not _obstructed(view, base + [0.0505 - 0.030, 0, 0], up, cfg)   # clear
    assert _obstructed(view, base + [0.0505 - 0.036, 0, 0], lean, cfg)     # clear upright, but the tool leans over it
    assert not _obstructed(view, base + [0.0505 - 0.036, 0, 0], up, cfg)


@pytest.mark.parametrize("dia", [0.041, 0.043, 0.046])
@pytest.mark.parametrize("turn", [0.0, 0.6])
def test_thin_cylinder_is_pickable_though_its_steep_sides_are_hard_to_see(cfg, dia, turn):
    s = single_part_scene(cfg, PartSpec(0, "cylinder", "plastic", (dia, 0.09), 0.2),
                          quat=quat_mul(quat_axis_angle([0, 0, 1], turn), quat_axis_angle([1, 0, 0], np.pi / 2)))
    s.data.qpos[s.part_qpos_adr[0] + 2] = s.bin_inner_bounds()[0][2] + dia / 2
    t = targets(s)[0]
    across = np.array([np.cos(turn), np.sin(turn)])
    assert abs((t.pos - s.data.xpos[s.part_body_ids[0]])[:2] @ across) < 0.003
    teleport_cup(s, t.pos, -t.normal)
    assert seal_quality(s, 0) == pytest.approx(t.seal, abs=0.08)


def test_a_lip_past_a_box_edge_is_never_taken_for_a_curve(cfg):
    for w in (0.034, 0.038, 0.042):
        s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.10, w, 0.04), 0.2))
        assert targets(s) == [], w


def test_thin_cylinder_with_a_part_lying_against_its_side_is_rejected(cfg):
    """The lip would land on the neighbour."""
    d = 0.046
    cyl = PartSpec(0, "cylinder", "plastic", (d, 0.09), 0.2)
    box = PartSpec(1, "box", "plastic", (0.06, 0.10, d - 0.006), 0.2)
    s = two_parts(cfg, cyl, box, [0, 0, d / 2], [d / 2 * 0.62 + 0.03, 0, (d - 0.006) / 2], qa=quat_axis_angle([1, 0, 0], np.pi / 2))
    assert all(t.pos[0] > s.data.xpos[s.part_body_ids[1]][0] - 0.03 for t in targets(s))


def test_safe_means_free_centered_and_level(cfg):
    free = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.11, 0.11, 0.04), 0.5))
    t = targets(free)[0]
    assert t.safe and t.crowd == 0
    off_center = rules_targets(observe_bin(free), cfg, avoid=[t.pos])
    assert off_center and not any(x.safe for x in off_center)
    tilted = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.04), 0.3), quat=quat_axis_angle([1, 0, 0], 0.2))
    tilted.data.qpos[tilted.part_qpos_adr[0] + 2] += 0.02
    assert targets(tilted) and not any(t.safe for t in targets(tilted))


def test_a_part_with_another_resting_on_its_edge_is_not_safe(cfg):
    low, top = PartSpec(0, "box", "plastic", (0.10, 0.10, 0.04), 0.4), PartSpec(1, "box", "plastic", (0.06, 0.06, 0.03), 0.13)
    s = two_parts(cfg, low, top, [0, 0, 0.02], [0.05 + 0.02, 0, 0.055])  # 10 mm of overlap on one edge
    c = s.data.xpos[s.part_body_ids[0]]
    assert not any(t.safe for t in targets(s) if t.pos[2] < s.bin_inner_bounds()[0][2] + 0.05), c


def test_safe_targets_come_first(cfg):
    """A free cylinder (curved) before a flat box with something lying on its edge."""
    specs = [PartSpec(0, "box", "plastic", (0.10, 0.10, 0.05), 0.5), PartSpec(1, "box", "plastic", (0.05, 0.05, 0.02), 0.06),
             PartSpec(2, "cylinder", "plastic", (0.06, 0.09), 0.3)]
    s = build_scene(cfg, specs)
    lo, hi = s.bin_inner_bounds()
    base = np.r_[(lo + hi)[:2] / 2, lo[2]]
    for adr, p in zip(s.part_qpos_adr, ([-0.1, 0, 0.025], [-0.1 + 0.06, 0, 0.06], [0.12, 0, 0.03])):
        s.data.qpos[adr:adr + 3] = base + p
    s.data.qpos[s.part_qpos_adr[2] + 3:s.part_qpos_adr[2] + 7] = quat_axis_angle([1, 0, 0], np.pi / 2)
    ts = targets(s)
    k = [t.safe for t in ts]
    assert k == sorted(k, reverse=True) and not all(k)
    assert any(t.pos[0] > base[0] + 0.1 for t in ts if t.safe)


def narrow_box(cfg, w, k):
    rng = np.random.default_rng(k)
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (rng.uniform(0.05, 0.11), w, 0.03), 0.2),
                          quat=quat_axis_angle([0, 0, 1], rng.uniform(0, np.pi)), xy_offset=tuple(rng.uniform(-0.09, 0.09, 2)))
    s.sensor_rng = rng
    return s


@pytest.mark.parametrize("k", range(8))
def test_narrow_face_is_pickable_although_its_edges_are_hard_to_see(cfg, k):
    s = narrow_box(cfg, 0.047, k)
    t = targets(s)[0]
    teleport_cup(s, t.pos, -t.normal)
    assert seal_quality(s, 0) > 0.95


@pytest.mark.parametrize("k", range(8))
def test_face_narrower_than_the_cup_is_not_a_target(cfg, k):
    assert targets(narrow_box(cfg, 0.0425, k)) == []


@pytest.mark.parametrize("gap_mm,push", [(0.0, True), (1.0, True), (6.0, False)])
@pytest.mark.parametrize("lying", [False, True])
def test_lift_direction_points_away_from_a_touching_wall(cfg, gap_mm, push, lying):
    ix = cfg.bin.inner[0]
    s = lying_cylinder(cfg, 0.05) if lying else single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.04), 0.3))
    s.data.qpos[s.part_qpos_adr[0]] = s.bin_inner_bounds()[0][0] + (0.025 if lying else 0.04) + gap_mm * 1e-3
    t = targets(s)[0]
    assert (t.away[0] > 0.9) if push else (t.away[0] < 0.1)
    assert t.away[1] == 0


def test_no_lift_direction_away_from_neighbours(cfg):
    a, b = (PartSpec(k, "box", "plastic", (0.08, 0.08, h), 0.3) for k, h in ((0, 0.05), (1, 0.03)))
    s = two_parts(cfg, a, b, [0, 0, 0.025], [0, 0.0805, 0.015])
    assert all(np.all(t.away == 0) for t in targets(s))


@pytest.mark.parametrize("overlap", [0.006, 0.015])
def test_a_part_under_the_edge_of_another_is_not_safe(cfg, overlap):
    """The strip next to the covering part is often missing from the image."""
    low, top = PartSpec(0, "box", "plastic", (0.109, 0.089, 0.034), 0.4), PartSpec(1, "box", "plastic", (0.06, 0.08, 0.05), 0.3)
    for dx in (1, -1):
        s = two_parts(cfg, low, top, [0, 0, 0.017], [dx * (0.0545 + 0.03 - overlap), 0.01, 0.034 + 0.025])
        z = s.bin_inner_bounds()[0][2] + 0.034
        assert not any(t.safe for t in targets(s) if abs(t.pos[2] - z) < 0.004)


def test_free_sides_before_boxed_in(cfg):
    """Same height, both safe: the one with nothing beside it comes first."""
    specs = [PartSpec(k, "box", "plastic", (0.07, 0.07, 0.05), 0.3) for k in (0, 1)] + \
            [PartSpec(k, "box", "plastic", (0.05, 0.07, 0.03), 0.12) for k in (2, 3)]
    for dx in (1, -1):
        s = build_scene(cfg, specs)
        lo, hi = s.bin_inner_bounds()
        base = np.r_[(lo + hi)[:2] / 2, lo[2]]
        for adr, p in zip(s.part_qpos_adr, ([-dx * 0.1, 0, 0.025], [dx * 0.1, 0, 0.025], [dx * 0.1 + 0.0605, 0, 0.015], [dx * 0.1 - 0.0605, 0, 0.015])):
            s.data.qpos[adr:adr + 3] = base + p
        t = targets(s)[0]
        assert (t.pos[0] - base[0]) * dx < 0


def test_a_part_hidden_under_the_side_of_a_tall_one_is_not_safe(cfg):
    """Seen at a slant, the tall part's side hides a wide strip next to the visible face: unseen is not free."""
    for seed, where in ((1009, [0.0596, 0.0850]), (1002, [0.3576, 0.2366])):
        s, _ = make_settled_scene(cfg, seed)
        s.sensor_rng = np.random.default_rng([seed, 5])
        lo, _ = s.bin_inner_bounds()
        near = [t for t in targets(s) if np.linalg.norm(t.pos[:2] - lo[:2] - where) < 0.02]
        assert not any(t.safe for t in near), seed


@pytest.mark.parametrize("dia,length", [(0.050, 0.0455), (0.050, 0.047), (0.042, 0.046), (0.060, 0.047)])
def test_short_lying_cylinder_is_pickable(cfg, dia, length):
    """Barely longer than the cup: its end edges, like those of a narrow box, must be located from all cells along them."""
    found = 0
    for k in range(6):
        s = single_part_scene(cfg, PartSpec(0, "cylinder", "plastic", (dia, length), 0.2), xy_offset=(0.05, 0.03),
                              quat=quat_mul(quat_axis_angle([0, 0, 1], 0.5 * k), quat_axis_angle([1, 0, 0], np.pi / 2)))
        s.data.qpos[s.part_qpos_adr[0] + 2] = s.bin_inner_bounds()[0][2] + dia / 2
        s.sensor_rng = np.random.default_rng(k)
        ts = targets(s)
        if ts:
            teleport_cup(s, ts[0].pos, -ts[0].normal)
            found += seal_quality(s, 0) > 0.6
    assert found >= 5


def test_cylinder_shorter_than_the_cup_is_not_a_target(cfg):
    for k in range(6):
        s = single_part_scene(cfg, PartSpec(0, "cylinder", "plastic", (0.05, 0.042), 0.2), quat=quat_axis_angle([1, 0, 0], np.pi / 2))
        s.data.qpos[s.part_qpos_adr[0] + 2] = s.bin_inner_bounds()[0][2] + 0.025
        s.sensor_rng = np.random.default_rng(k)
        assert targets(s) == []


@pytest.mark.parametrize("w,safe", [(0.0450, False), (0.0480, True)])
def test_a_face_barely_wider_than_the_cup_is_risky(cfg, w, safe):
    n = []
    for k in range(6):
        ts = targets(narrow_box(cfg, w, 40 + k))
        n += [ts[0].safe] if ts else []
    assert len(n) >= 4 and all(x == safe for x in n)
