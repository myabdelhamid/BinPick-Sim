import numpy as np
import pytest

from binpick.sim.camera import intrinsics, render
from binpick.sim.parts import sample_parts
from binpick.sim.scene import build_scene


def test_render_shapes(cfg):
    scene = build_scene(cfg, sample_parts(cfg.parts, np.random.default_rng([1000, 0])))
    rgb, d = render(scene, "bin_cam")
    assert rgb.shape == (cfg.cameras.height, cfg.cameras.width, 3) and rgb.dtype == np.uint8
    assert d.shape == rgb.shape[:2] and d.dtype == np.float32
    assert rgb.std() > 5  # not a blank frame


def test_empty_bin_floor_depth(cfg):
    scene = build_scene(cfg, [])
    _, d = render(scene, "bin_cam")
    cam_z = scene.data.cam("bin_cam").xpos[2]
    floor_z = scene.bin_inner_bounds()[0][2]
    h, w = d.shape
    assert d[h // 2, w // 2] == pytest.approx(cam_z - floor_z, abs=0.002)


def test_bin_fills_frame_and_arm_not_visible(cfg):
    scene = build_scene(cfg, [])
    _, d = render(scene, "bin_cam")
    cam_z = scene.data.cam("bin_cam").xpos[2]
    rim_z = scene.bin_inner_bounds()[1][2]
    # nothing in view is closer to the camera than the bin rim (parked arm is out of view)
    assert d.min() >= (cam_z - rim_z) - 0.002


def test_intrinsics(cfg):
    scene = build_scene(cfg, [])
    K = intrinsics(scene, "bin_cam")
    assert K.shape == (3, 3)
    assert K[0, 2] == pytest.approx(cfg.cameras.width / 2) and K[1, 2] == pytest.approx(cfg.cameras.height / 2)
    f = (cfg.cameras.height / 2) / np.tan(np.deg2rad(cfg.cameras.bin_fovy) / 2)
    assert K[0, 0] == pytest.approx(f) and K[1, 1] == pytest.approx(f)
