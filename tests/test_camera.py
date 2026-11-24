import dataclasses

import numpy as np
import pytest
from scipy import ndimage

from binpick.sim.camera import intrinsics, render, sense_depth
from binpick.sim.parts import PartSpec, sample_parts
from binpick.sim.scene import build_scene
from helpers import single_part_scene


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


def _no_noise(cfg):
    return dataclasses.replace(cfg, cameras=dataclasses.replace(cfg.cameras, depth_sigma=0.0, dropout=0.0, edge_dropout=0.0))


def _floor_patch(d):
    h, w = d.shape
    return d[h // 2 - 60:h // 2 + 60, w // 2 - 60:w // 2 + 60]


def test_depth_noise_grows_with_distance_and_is_unbiased(cfg):
    scene = build_scene(cfg, [])
    true = _floor_patch(render(scene, "bin_cam")[1])
    scene.sensor_rng = np.random.default_rng(1)
    err = _floor_patch(sense_depth(scene, "bin_cam")) - true
    want = cfg.cameras.depth_sigma * (np.mean(true) / cfg.cameras.sigma_distance) ** 2
    assert np.nanstd(err) == pytest.approx(want, rel=0.05)
    assert abs(np.nanmean(err)) < 0.05 * want
    assert np.isnan(err).mean() == pytest.approx(cfg.cameras.dropout, rel=0.35)


def test_depth_is_missing_mostly_along_edges(cfg):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.09, 0.06, 0.06), 0.4))
    d = sense_depth(s, "bin_cam")
    true = render(s, "bin_cam")[1]
    jump = ndimage.maximum_filter(true, 3) - ndimage.minimum_filter(true, 3) > cfg.cameras.edge_jump
    assert np.isnan(d[jump]).mean() == pytest.approx(cfg.cameras.edge_dropout, abs=0.1)
    assert np.isnan(d[~jump]).mean() < 2 * cfg.cameras.dropout


def test_noise_follows_the_sensor_stream(cfg):
    scene = build_scene(cfg, [])
    frames = []
    for _ in range(2):
        scene.sensor_rng = np.random.default_rng([7, 5])
        frames.append([sense_depth(scene, "bin_cam") for _ in range(2)])
    assert np.array_equal(frames[0][0], frames[1][0], equal_nan=True) and np.array_equal(frames[0][1], frames[1][1], equal_nan=True)
    assert not np.array_equal(frames[0][0], frames[0][1], equal_nan=True)


def test_zero_noise_is_the_exact_render(cfg):
    scene = build_scene(_no_noise(cfg), [])
    assert np.array_equal(sense_depth(scene, "bin_cam"), render(scene, "bin_cam")[1])
