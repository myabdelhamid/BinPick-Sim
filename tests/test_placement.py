import dataclasses

import mujoco
import numpy as np
import pytest

from binpick.motion.ik import cup_rotation, DOWN
from binpick.sim.parts import PartSpec
from binpick.sim.scene import build_scene
from binpick.supervisor.placement import SlotAllocator, occupied_slots, place_yaw, referee_in_slot, verify_slot

BOX = PartSpec(0, "box", "plastic", (0.09, 0.06, 0.03), 0.2)


def _put_in_slot(cfg, slot, offset=(0.0, 0.0), yaw=0.0):
    s = build_scene(cfg, [BOX])
    c = s.slot_centers[slot]
    a = s.part_qpos_adr[0]
    s.data.qpos[a:a + 3] = [c[0] + offset[0], c[1] + offset[1], c[2] + 0.015]
    s.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(s.model, s.data)
    return s


def test_slot_allocator_never_reuses():
    al = SlotAllocator(3)
    got = []
    for _ in range(3):
        got.append(al.peek())
        al.commit()
    assert got == [0, 1, 2] and al.peek() is None


def test_verify_and_referee_centered(cfg):
    s = _put_in_slot(cfg, 7)
    ok, est = verify_slot(s, 7)
    assert ok and np.linalg.norm(est - s.slot_centers[7][:2]) < 0.003
    assert referee_in_slot(s, 0, 7)


def test_verify_and_referee_reject_offset(cfg):
    s = _put_in_slot(cfg, 7, offset=(0.02, 0.0))
    assert not verify_slot(s, 7)[0]
    assert not referee_in_slot(s, 0, 7)


def test_verify_empty_slot(cfg):
    s = _put_in_slot(cfg, 7)
    assert not verify_slot(s, 3)[0]


@pytest.mark.parametrize("grasp_yaw", [0.0, 0.7, -2.0])
def test_place_yaw_aligns_long_axis_with_tray_x(grasp_yaw):
    long_world = np.array([np.cos(1.1), np.sin(1.1), 0.0])
    R_grasp = cup_rotation(DOWN, grasp_yaw)
    long_cup = R_grasp.T @ long_world
    yaw = place_yaw(long_cup)
    aligned = cup_rotation(DOWN, yaw) @ long_cup
    assert abs(aligned @ [1, 0, 0]) > 0.9999


def test_allocator_skips_slots_seen_occupied():
    al = SlotAllocator(4)
    al.mark_used(1)
    assert al.peek() == 0
    al.commit()
    assert al.peek() == 2
    al.mark_used(3)
    al.commit()
    assert al.peek() is None


def test_noise_does_not_fill_empty_slots(cfg):
    s = build_scene(cfg, [])
    for k in range(30):
        assert occupied_slots(s) == set(), k


def noisier(cfg, sigma):
    return dataclasses.replace(cfg, cameras=dataclasses.replace(cfg.cameras, depth_sigma=sigma))


@pytest.mark.parametrize("sigma", [0.0007, 0.001, 0.002])
def test_stronger_noise_does_not_fill_empty_slots_or_hide_thin_parts(cfg, sigma):
    c = noisier(cfg, sigma)
    empty = build_scene(c, [])
    flat = build_scene(c, [PartSpec(0, "box", "plastic", (0.045, 0.045, 0.02), 0.05)])
    flat.data.qpos[flat.part_qpos_adr[0]:flat.part_qpos_adr[0] + 3] = flat.slot_centers[7] + [0, 0, 0.01]
    mujoco.mj_forward(flat.model, flat.data)
    for k in range(10):
        assert occupied_slots(empty) == set(), k
        assert occupied_slots(flat) == {7}, k


@pytest.mark.parametrize("sigma", [0.0005, 0.001, 0.002])
def test_verify_position_is_steady_under_noise(cfg, sigma):
    s = _put_in_slot(noisier(cfg, sigma), 7, offset=(0.004, -0.003))
    for k in range(10):
        ok, est = verify_slot(s, 7)
        assert ok and np.allclose(est - s.slot_centers[7][:2], [0.004, -0.003], atol=0.0025), (k, est)
