"""Placement maths, tray-camera verification (robot side) and the ground-truth referee (scoring side)."""

from __future__ import annotations

import numpy as np

from binpick.perception.denoise import smooth_depth
from binpick.sim.camera import intrinsics
from binpick.sim.scene import Scene

PLACE_TOL = 0.010        # spec: part center within +-10 mm of the slot center
VERIFY_MIN_HEIGHT = 0.010  # the thinnest part is 20 mm
VERIFY_INSET = 0.008     # ignore pixels near the slot dividers (8 mm tall)
VERIFY_MIN_PIXELS = 50
TOP_PERCENTILE = 95      # the highest readings are noise
TOP_BAND = 0.004         # pixels within 4 mm of the top = the part's top face


class SlotAllocator:
    """Tray slots in order. A slot is used up (commit) only when a part was released into it or is
    seen lying in it; a failure before that leaves it free for the next attempt."""

    def __init__(self, n: int):
        self.n, self._used = n, set()

    def peek(self) -> int | None:
        return next((k for k in range(self.n) if k not in self._used), None)

    def commit(self) -> None:
        self._used.add(self.peek())

    def mark_used(self, slot: int) -> None:
        """Something was seen lying in this slot."""
        self._used.add(slot)


def place_yaw(long_axis_cup: np.ndarray) -> float:
    """Wrist yaw for a straight-down cup that puts the part's long axis (given in the cup frame)
    along tray x. Cup rotation at yaw is Rz(yaw) Rx(pi), and Rx(pi) maps (x, y, z) -> (x, -y, -z)."""
    a = np.asarray(long_axis_cup, float)
    return float(-np.arctan2(-a[1], a[0]))


def _tray_points(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = smooth_depth(scene, "tray_cam")
    K = intrinsics(scene, "tray_cam")
    cam = scene.data.cam("tray_cam").xpos
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    x = cam[0] + (u + 0.5 - K[0, 2]) * depth / K[0, 0]
    y = cam[1] - (v + 0.5 - K[1, 2]) * depth / K[1, 1]
    return x, y, cam[2] - depth


def occupied_slots(scene: Scene) -> set[int]:
    """Slots the tray camera sees something lying in."""
    x, y, z = _tray_points(scene)
    half = scene.cfg.tray.slot / 2 - VERIFY_INSET
    return {k for k, c in enumerate(scene.slot_centers)
            if ((np.abs(x - c[0]) < half) & (np.abs(y - c[1]) < half) & (z > c[2] + VERIFY_MIN_HEIGHT)).sum()
            >= VERIFY_MIN_PIXELS}


def verify_slot(scene: Scene, slot: int) -> tuple[bool, np.ndarray | None]:
    """Robot-side check from the tray camera: is something sitting centered in `slot`?"""
    x, y, z = _tray_points(scene)
    c = scene.slot_centers[slot]
    half = scene.cfg.tray.slot / 2 - VERIFY_INSET
    mask = (np.abs(x - c[0]) < half) & (np.abs(y - c[1]) < half) & (z > c[2] + VERIFY_MIN_HEIGHT)
    if mask.sum() < VERIFY_MIN_PIXELS:
        return False, None
    mask &= z > np.percentile(z[mask], TOP_PERCENTILE) - TOP_BAND  # top face only: side faces seen in perspective bias the centroid
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
