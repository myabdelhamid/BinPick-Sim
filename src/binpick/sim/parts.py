"""Seeded sampling of part specs (pure NumPy, no MuJoCo)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from binpick.config import PartsConfig

MAX_RESAMPLES = 1000


@dataclass(frozen=True)
class PartSpec:
    id: int
    shape: Literal["box", "cylinder"]
    material: str
    dims: tuple[float, ...]  # box: (sx, sy, sz) full lengths; cylinder: (diameter, length)
    mass: float


def part_volume(spec: PartSpec) -> float:
    if spec.shape == "box":
        sx, sy, sz = spec.dims
        return sx * sy * sz
    d, length = spec.dims
    return math.pi * (d / 2) ** 2 * length


def _draw_dims(cfg: PartsConfig, shape: str, rng: np.random.Generator) -> tuple[float, ...]:
    u = lambda r: float(rng.uniform(*r))  # noqa: E731
    if shape == "box":
        return (u(cfg.box_footprint), u(cfg.box_footprint), u(cfg.box_height))
    return (u(cfg.cyl_diameter), u(cfg.cyl_length))


def sample_parts(cfg: PartsConfig, rng: np.random.Generator) -> list[PartSpec]:
    """Boxes first (ids 0..n_boxes-1), then cylinders.

    Material is drawn once per part; dims are redrawn (same material) until the mass
    falls inside [mass_min, mass_max], so densities stay physical.
    """
    names = list(cfg.materials)
    probs = np.array([cfg.materials[n].prob for n in names])
    shapes = ["box"] * cfg.n_boxes + ["cylinder"] * cfg.n_cylinders
    out: list[PartSpec] = []
    for i, shape in enumerate(shapes):
        material = names[int(rng.choice(len(names), p=probs))]
        density = cfg.materials[material].density
        for _ in range(MAX_RESAMPLES):
            dims = _draw_dims(cfg, shape, rng)
            spec = PartSpec(i, shape, material, dims, 0.0)
            mass = part_volume(spec) * density
            if cfg.mass_min <= mass <= cfg.mass_max:
                out.append(PartSpec(i, shape, material, dims, mass))
                break
        else:
            raise RuntimeError(f"part {i} ({shape}, {material}): no dims within mass range after {MAX_RESAMPLES} tries")
    return out
