"""Placement maths, tray-camera verification (robot side) and the ground-truth referee (scoring side)."""

from __future__ import annotations

import numpy as np

from binpick.sim.camera import intrinsics, render
from binpick.sim.scene import Scene

PLACE_TOL = 0.010        # spec: part center within +-10 mm of the slot center
VERIFY_MIN_HEIGHT = 0.003
VERIFY_INSET = 0.008     # ignore pixels near the slot dividers (8 mm tall)
VERIFY_MIN_PIXELS = 50


class SlotAllocator:
    """Hands out tray slots in order; a used slot (filled or misplaced into) is never reused."""

    def __init__(self, n: int):
        self.n, self._next = n, 0

    def next(self) -> int | None:
        if self._next >= self.n:
            return None
        self._next += 1
        return self._next - 1


def place_yaw(long_axis_cup: np.ndarray) -> float:
    """Wrist yaw for a straight-down cup that puts the part's long axis (given in the cup frame)
    along tray x. Cup rotation at yaw is Rz(yaw) Rx(pi), and Rx(pi) maps (x, y, z) -> (x, -y, -z)."""
    a = np.asarray(long_axis_cup, float)
    return float(-np.arctan2(-a[1], a[0]))


def verify_slot(scene: Scene, slot: int) -> tuple[bool, np.ndarray | None]:
    """Robot-side check from the tray camera: is something sitting centered in `slot`?"""
    _, depth = render(scene, "tray_cam")
    K = intrinsics(scene, "tray_cam")
    cam = scene.data.cam("tray_cam").xpos
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    x = cam[0] + (u - K[0, 2]) * depth / K[0, 0]
    y = cam[1] - (v - K[1, 2]) * depth / K[1, 1]
    z = cam[2] - depth
    c = scene.slot_centers[slot]
    half = scene.cfg.tray.slot / 2 - VERIFY_INSET
    mask = (np.abs(x - c[0]) < half) & (np.abs(y - c[1]) < half) & (z > c[2] + VERIFY_MIN_HEIGHT)
    if mask.sum() < VERIFY_MIN_PIXELS:
        return False, None
    est = np.array([x[mask].mean(), y[mask].mean()])
    return bool(np.all(np.abs(est - c[:2]) <= PLACE_TOL)), est


def referee_in_slot(scene: Scene, part_index: int, slot: int) -> bool:
    """Ground-truth scoring: part center within +-10 mm of the slot center (x and y), resting on the tray."""
    p = scene.data.xpos[scene.part_body_ids[part_index]]
    c = scene.slot_centers[slot]
    resting = p[2] < c[2] + max(scene.parts[part_index].dims) + 0.005
    return bool(np.all(np.abs(p[:2] - c[:2]) <= PLACE_TOL) and resting)
