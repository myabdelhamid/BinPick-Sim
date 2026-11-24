import mujoco
import numpy as np
import pytest

from binpick.perception.view import depth_to_cloud, observe_bin
from binpick.sim.camera import intrinsics, render
from binpick.sim.parts import PartSpec
from binpick.sim.scene import build_scene
from binpick.sim.settle import make_settled_scene
from helpers import quat_axis_angle, single_part_scene


def label_at(view, xy):
    i, j = view.index(np.asarray(xy))
    return view.labels[i, j]


def test_cloud_matches_ray_casts(cfg):
    s, _ = make_settled_scene(cfg, 1000)
    _, d = render(s, "bin_cam")
    cam = s.data.cam("bin_cam")
    P = depth_to_cloud(d, intrinsics(s, "bin_cam"), cam.xpos, cam.xmat.reshape(3, 3))
    assert P.shape == d.shape + (3,)
    rng, g, err = np.random.default_rng(0), np.zeros(1, np.int32), []
    for _ in range(500):
        v, u = rng.integers(0, d.shape[0]), rng.integers(0, d.shape[1])
        ray = P[v, u] - cam.xpos
        n = np.linalg.norm(ray)
        err.append(abs(mujoco.mj_ray(s.model, s.data, cam.xpos, ray / n, None, 1, -1, g) - n))
    assert np.median(err) < 1e-4 and np.percentile(err, 90) < 1e-3


def test_flat_box_is_one_centered_surface(cfg):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.09, 0.06, 0.03), 0.2), xy_offset=(0.05, -0.04))
    v = observe_bin(s)
    c = s.data.xpos[s.part_body_ids[0]]
    lab = label_at(v, c[:2])
    assert lab > 0
    ctr = v.surface_center(lab)
    assert np.allclose(ctr[:2], c[:2], atol=0.002) and ctr[2] == pytest.approx(c[2] + 0.015, abs=0.001)


@pytest.mark.parametrize("gap_mm", [1.0, 1.5, 2.0, 4.0])
@pytest.mark.parametrize("yaw_deg,off", [(0.0, (0.0, 0.0)), (25.0, (0.08, 0.05))])
def test_two_boxes_side_by_side_are_two_surfaces(cfg, gap_mm, yaw_deg, off):
    b = [PartSpec(i, "box", "plastic", (0.08, 0.06, 0.03), 0.17) for i in (0, 1)]
    s = build_scene(cfg, b)
    lo, hi = s.bin_inner_bounds()
    c = (lo + hi) / 2 + [*off, 0]
    u = np.array([np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))])
    d = 0.04 + gap_mm * 1e-3 / 2
    for a, sg in zip(s.part_qpos_adr, (-1, 1)):
        s.data.qpos[a:a + 3] = [*(c[:2] + sg * d * u), lo[2] + 0.015]
        s.data.qpos[a + 3:a + 7] = quat_axis_angle([0, 0, 1], np.deg2rad(yaw_deg))
    mujoco.mj_forward(s.model, s.data)
    v = observe_bin(s)
    l0, l1 = label_at(v, c[:2] - d * u), label_at(v, c[:2] + d * u)
    assert l0 > 0 and l1 > 0 and l0 != l1


def test_lying_cylinder_is_one_surface(cfg):
    s = single_part_scene(cfg, PartSpec(0, "cylinder", "plastic", (0.06, 0.10), 0.3), quat=quat_axis_angle([1, 0, 0], np.pi / 2))
    s.data.qpos[s.part_qpos_adr[0] + 2] = s.bin_inner_bounds()[0][2] + 0.03
    mujoco.mj_forward(s.model, s.data)
    v = observe_bin(s)
    c = s.data.xpos[s.part_body_ids[0]]
    labs = {label_at(v, c[:2] + d) for d in ([0, 0], [0.018, 0], [-0.018, 0], [0, 0.04], [0, -0.04])}
    assert len(labs) == 1 and labs.pop() > 0


def test_box_edge_splits_top_and_side(cfg):
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.08, 0.08, 0.06), 0.4), quat=quat_axis_angle([1, 0, 0], np.deg2rad(35)))
    s.data.qpos[s.part_qpos_adr[0] + 2] += 0.03
    mujoco.mj_forward(s.model, s.data)
    v = observe_bin(s)
    b = s.part_body_ids[0]
    p, R = s.data.xpos[b], s.data.xmat[b].reshape(3, 3)
    top, side = p + R @ [0, 0, 0.03], p + R @ [0, -0.04, 0]  # both faces lean towards the camera
    if (R @ [0, -1, 0])[2] < 0:
        side = p + R @ [0, 0.04, 0]
    assert label_at(v, top[:2]) > 0 and label_at(v, side[:2]) > 0
    assert label_at(v, top[:2]) != label_at(v, side[:2])


def test_empty_bin_and_thin_part(cfg):
    assert observe_bin(build_scene(cfg, [])).is_empty()
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.05, 0.05, 0.02), 0.06))
    assert not observe_bin(s).is_empty()


def test_every_labelled_cell_has_a_height_and_a_normal(cfg):
    for seed in (1000, 1001, 1004):
        s, _ = make_settled_scene(cfg, seed)
        v = observe_bin(s)
        assert np.isfinite(v.z[v.labels > 0]).all() and np.isfinite(v.normals[v.labels > 0]).all(), seed


@pytest.mark.parametrize("off", [(0.0, 0.0), (0.12, 0.08), (-0.13, 0.07)])
def test_flat_face_at_the_floor_has_no_holes(cfg, off):
    """Camera pixels are wider than a cell down there; unfilled cells would break every lip path."""
    s = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.10, 0.10, 0.02), 0.24), xy_offset=off)
    v = observe_bin(s)
    c = s.data.xpos[s.part_body_ids[0]]
    (i0, j0), (i1, j1) = v.index(c[:2] - 0.045), v.index(c[:2] + 0.045)
    assert np.isfinite(v.z[i0:i1, j0:j1]).all()


def test_noise_does_not_fake_parts_or_hide_them(cfg):
    empty = build_scene(cfg, [])
    thin = single_part_scene(cfg, PartSpec(0, "box", "plastic", (0.045, 0.045, 0.02), 0.05), xy_offset=(0.15, 0.10))
    for k in range(30):
        assert observe_bin(empty).is_empty(), k
        assert not observe_bin(thin).is_empty(), k
