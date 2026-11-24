"""Headless RGB + metric depth rendering from the scene's fixed cameras."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy import ndimage

from binpick.sim.scene import Scene


def render(scene: Scene, camera: str, depth: bool = True,
           size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """RGB uint8 (H, W, 3) and depth float32 (H, W) in meters along the optical axis.

    `size` is (width, height); defaults to the configured camera resolution.
    """
    w, h = size or (scene.cfg.cameras.width, scene.cfg.cameras.height)
    mujoco.mj_forward(scene.model, scene.data)
    with mujoco.Renderer(scene.model, height=h, width=w) as r:
        r.update_scene(scene.data, camera=camera)
        rgb = r.render().copy()
        d = None
        if depth:
            r.enable_depth_rendering()
            r.update_scene(scene.data, camera=camera)
            d = r.render().astype(np.float32)
    return rgb, d


def sense_depth(scene: Scene, camera: str) -> np.ndarray:
    """What the depth camera reports: the rendered depth with random error and missing readings (NaN)."""
    c, rng = scene.cfg.cameras, scene.sensor_rng
    d = render(scene, camera)[1]
    jump = ndimage.maximum_filter(d, 3) - ndimage.minimum_filter(d, 3) > c.edge_jump
    missing = rng.random(d.shape) < np.where(jump, c.edge_dropout, c.dropout)
    noisy = d + rng.normal(0.0, 1.0, d.shape) * c.depth_sigma * (d / c.sigma_distance) ** 2
    return np.where(missing, np.nan, noisy).astype(np.float32)


def intrinsics(scene: Scene, camera: str) -> np.ndarray:
    """Pinhole K for the configured resolution (square pixels, principal point at the center)."""
    w, h = scene.cfg.cameras.width, scene.cfg.cameras.height
    fovy = np.deg2rad(scene.model.cam(camera).fovy[0])
    f = (h / 2) / np.tan(fovy / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
