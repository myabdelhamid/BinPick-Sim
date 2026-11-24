"""What a planner hands to the supervisor."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GraspTarget:
    part_index: int | None  # only the known-pose oracle fills this in
    pos: np.ndarray         # suction point on the surface (world)
    normal: np.ndarray      # outward surface normal (world); the cup approaches along -normal
    tilt_deg: float         # angle between normal and world +z
    depth: float            # part thickness along the normal; from a camera only an upper bound
    long_axis: np.ndarray   # longest in-face direction (world, unit), for slot alignment
    center: np.ndarray | None = None  # estimated center of the gripped surface (world); None = at pos
    seal: float = 1.0       # estimated seal quality
    score: float = 0.0
    crowd: float = 0.0      # share of the surface's outline with something taller next to it
    safe: bool = True       # free, centered and level as far as the camera can tell
    away: np.ndarray = field(default_factory=lambda: np.zeros(2))  # away from walls the part touches (xy, 0-1 each)
