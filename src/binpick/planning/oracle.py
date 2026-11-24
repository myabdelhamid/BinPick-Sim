"""Known-pose grasp selection (M2 stand-in for perception + planning).

Reads part poses from the simulator. Only this module and the referee may do that.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from binpick.sim.scene import Scene
from binpick.sim.suction import wrapped_lip

MAX_TILT_DEG = 20.0
EDGE_MARGIN = 0.002        # face must exceed the cup by this on each side
OCCLUSION_RAY_LEN = 0.3


@dataclass(frozen=True)
class GraspTarget:
    part_index: int
    pos: np.ndarray        # suction point on the face (world)
    normal: np.ndarray     # outward face normal (world); the cup approaches along -normal
    tilt_deg: float        # angle between normal and world +z
    depth: float           # part thickness along the normal (for place height)
    long_axis: np.ndarray  # longest in-face direction (world, unit), for slot alignment


def _faces(scene: Scene, i: int):
    """Yield (center, normal, extent_a, extent_b, depth, long_axis) of graspable flat/curved faces."""
    d, spec = scene.data, scene.parts[i]
    b = scene.part_body_ids[i]
    p, R = d.xpos[b].copy(), d.xmat[b].reshape(3, 3)
    if spec.shape == "box":
        half = np.array(spec.dims) / 2
        for k in range(3):
            o = [j for j in range(3) if j != k]
            long_j = o[0] if half[o[0]] >= half[o[1]] else o[1]
            for sgn in (1, -1):
                n = sgn * R[:, k]
                yield p + n * half[k], n, 2 * half[o[0]], 2 * half[o[1]], 2 * half[k], R[:, long_j]
    else:
        dia, length = spec.dims
        ax = R[:, 2]
        perp = np.cross(ax, [1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.cross(ax, [0, 1.0, 0])
        for sgn in (1, -1):  # flat ends
            yield p + sgn * ax * length / 2, sgn * ax, dia, dia, length, perp / np.linalg.norm(perp)
        up = np.array([0, 0, 1.0]) - ax[2] * ax  # side: the line facing most upward
        if np.linalg.norm(up) > 1e-6 and dia >= 2 * scene.cfg.suction.curve_radius_min:
            n = up / np.linalg.norm(up)
            yield p + n * dia / 2, n, length, length, dia, ax


def _unoccluded(scene: Scene, i: int, pos: np.ndarray, n: np.ndarray) -> bool:
    """The wrapped lip sits on the part, and nothing else stands above the cup-face plane under the cup."""
    m, d = scene.model, scene.data
    r = scene.cfg.cup.diameter / 2
    x = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.column_stack([x, -y, -n])
    if np.isnan(wrapped_lip(scene, i, pos, R, OCCLUSION_RAY_LEN, -1, 8)).any():
        return False
    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        pt = pos + r * (np.cos(th) * x + np.sin(th) * y)
        dist = mujoco.mj_ray(m, d, pt + n * OCCLUSION_RAY_LEN, -n, None, 1, -1, None)
        if 0 <= dist < OCCLUSION_RAY_LEN - 1e-4:
            return False
    return True


def oracle_targets(scene: Scene) -> list[GraspTarget]:
    """Best (most upward, big enough, unoccluded) face per part, highest first."""
    mujoco.mj_forward(scene.model, scene.data)
    cup = scene.cfg.cup.diameter + 2 * EDGE_MARGIN
    lo, hi = scene.bin_inner_bounds()
    out = []
    for i in range(len(scene.parts)):
        pos_i = scene.data.xpos[scene.part_body_ids[i]]
        if not (np.all(pos_i[:2] > lo[:2]) and np.all(pos_i[:2] < hi[:2])):
            continue  # not in the bin
        best = None
        for c, n, ea, eb, depth, long_ax in _faces(scene, i):
            tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
            if tilt > MAX_TILT_DEG or min(ea, eb) < cup:
                continue
            if best is None or tilt < best.tilt_deg:
                best = GraspTarget(i, c, n, tilt, depth, long_ax)
        if best is not None and _unoccluded(scene, i, best.pos, best.normal):
            out.append(best)
    return sorted(out, key=lambda t: -t.pos[2])
