"""Baseline: random visible points on parts. No filters, no scoring. Only the choice of point is random: how the
part is approached and placed is the same as for the rules planner."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from binpick.config import Config
from binpick.perception.view import BinView
from binpick.planning.rules import flat_face, max_depth, surface_rectangle, wall_direction
from binpick.planning.types import GraspTarget


def random_targets(view: BinView, cfg: Config, rng: np.random.Generator, n: int = 50) -> list[GraspTarget]:
    """n random part points in random order; the cup approaches along the local surface normal. The order is
    the only preference: the supervisor takes the first one the arm can reach."""
    ij = np.argwhere(view.part_mask & np.isfinite(view.z))
    if view.is_empty() or len(ij) == 0:
        return []
    # an edge cell has no surface of its own: it belongs to the nearest one
    near = ndimage.distance_transform_edt(view.labels == 0, return_distances=False, return_indices=True)
    faces: dict[int, tuple] = {}
    out = []
    for c in ij[rng.integers(0, len(ij), n)]:
        p = np.r_[view.xy(c), view.z[tuple(c)]]
        own = tuple(near[:, c[0], c[1]])
        lab = int(view.labels[own])
        if lab == 0:
            return []  # nothing but slivers in view
        if lab not in faces:
            flat = flat_face(view, lab)
            faces[lab] = (None if flat is None else flat.n, *surface_rectangle(view, lab)[:2], wall_direction(view, lab, flat is not None, cfg))
        flat_n, center, axis, away = faces[lab]
        normal = view.normals[own] if flat_n is None else flat_n  # on a flat face single cells only add noise
        out.append(GraspTarget(None, p, normal, float(np.degrees(np.arccos(normal[2]))), max_depth(view, p, normal, cfg),
                               axis, center, away=away))
    return out
