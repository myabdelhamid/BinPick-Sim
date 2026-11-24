"""Depth-image noise reduction, for both cameras."""

from __future__ import annotations

import numpy as np

from binpick.sim.camera import sense_depth
from binpick.sim.scene import Scene


def smooth_depth(scene: Scene, camera: str) -> np.ndarray:
    """One depth image, smoothed with a tolerance that follows the camera's known error."""
    cam, p = scene.cfg.cameras, scene.cfg.perception
    d = sense_depth(scene, camera)
    tol = p.smooth_sigmas * cam.depth_sigma * (d / cam.sigma_distance) ** 2
    return denoise(d, np.maximum(tol, p.smooth_tol_min), p.smooth_reach)


def denoise(depth: np.ndarray, tol: np.ndarray, reach: int) -> np.ndarray:
    """Average each reading with the neighbours (reach pixels around) that differ from it by less than tol (per pixel): random
    error shrinks, height jumps stay sharp. Missing readings stay missing."""
    total, count = np.zeros(depth.shape), np.zeros(depth.shape)
    padded = np.pad(depth.astype(float), reach, constant_values=np.nan)
    h, w = depth.shape
    for di in range(2 * reach + 1):
        for dj in range(2 * reach + 1):
            nb = padded[di:di + h, dj:dj + w]
            near = np.abs(nb - depth) < tol
            total[near] += nb[near]
            count += near
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)
