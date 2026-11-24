"""Baseline: random visible points on parts. No filters, no scoring."""

from __future__ import annotations

import numpy as np

from binpick.config import Config
from binpick.perception.view import BinView
from binpick.planning.rules import max_depth
from binpick.planning.types import GraspTarget


def random_targets(view: BinView, cfg: Config, rng: np.random.Generator, n: int = 50) -> list[GraspTarget]:
    """n random part points in random order; the cup approaches along the local surface normal. The order is
    the only preference: the supervisor takes the first one the arm can reach."""
    ij = np.argwhere(view.part_mask & np.isfinite(view.normals).all(axis=2))
    if view.is_empty() or len(ij) == 0:
        return []
    out = []
    for c in ij[rng.integers(0, len(ij), n)]:
        p = np.r_[view.xy(c), view.z[tuple(c)]]
        normal = view.normals[tuple(c)]
        a = rng.uniform(0, 2 * np.pi)
        ref = np.array([np.cos(a), np.sin(a), 0.0])
        axis = ref - (ref @ normal) * normal
        out.append(GraspTarget(None, p, normal, float(np.degrees(np.arccos(normal[2]))), max_depth(view, p, normal, cfg),
                               axis / np.linalg.norm(axis)))
    return out
