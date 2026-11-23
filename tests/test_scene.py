import mujoco
import numpy as np
import pytest

from binpick.sim.scene import build_scene, tray_slot_centers


def test_cup_is_44mm(scene):
    g = scene.model.geom(scene.cup_geom)
    assert g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert 2 * g.size[0] == pytest.approx(0.044, abs=1e-6)


def test_grip_welds_exist_inactive(scene):
    for p in scene.parts:
        eq = scene.model.equality(f"grip_{p.id}")
        assert eq.type == mujoco.mjtEq.mjEQ_WELD
        assert scene.data.eq_active[eq.id] == 0


def test_tray_slots(scene, cfg):
    c = scene.slot_centers
    assert c.shape == (20, 3)
    assert np.allclose(np.diff(c[:5, 0]), 0.120)
    assert np.allclose(c, tray_slot_centers(cfg))


def test_twenty_part_bodies(scene):
    assert len(scene.part_body_ids) == 20
    for pid, spec in zip(scene.part_body_ids, scene.parts):
        assert scene.model.body_mass[pid] == pytest.approx(spec.mass)


def test_zero_parts_supported(cfg):
    s = build_scene(cfg, [])
    assert s.part_body_ids == []


def _step(scene, seconds):
    for _ in range(int(seconds / scene.model.opt.timestep)):
        mujoco.mj_step(scene.model, scene.data)


def test_shaker_moves_and_returns(cfg):
    s = build_scene(cfg, [])
    lo0, _ = s.bin_inner_bounds()
    ax = s.model.actuator(s.shaker_actuators[0]).id
    s.data.ctrl[ax] = cfg.shaker.amplitude
    _step(s, 0.5)
    moved = s.bin_inner_bounds()[0][0] - lo0[0]
    assert moved >= 0.8 * cfg.shaker.amplitude
    s.data.ctrl[ax] = 0.0
    _step(s, 1.0)
    assert np.allclose(s.bin_inner_bounds()[0], lo0, atol=1e-3)
