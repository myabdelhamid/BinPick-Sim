"""Drop parts into the bin one at a time and let them settle (deterministic per seed)."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from binpick.config import Config
from binpick.sim.parts import sample_parts
from binpick.sim.scene import PARK_X0, Scene, build_scene

FINAL_PASSES = 4
MIN_SETTLE_S = 0.2  # always simulate at least this long after a drop


@dataclass(frozen=True)
class PartPose:
    id: int
    pos: np.ndarray
    quat: np.ndarray  # wxyz


class SettleError(RuntimeError):
    def __init__(self, part_id: int, reason: str):
        super().__init__(f"part {part_id}: {reason}")
        self.part_id = part_id


def _set_free(scene: Scene, i: int, pos, quat) -> None:
    adr = scene.part_qpos_adr[i]
    scene.data.qpos[adr:adr + 3] = pos
    scene.data.qpos[adr + 3:adr + 7] = quat
    dof = scene.model.jnt_dofadr[scene.model.body_jntadr[scene.part_body_ids[i]]]
    scene.data.qvel[dof:dof + 6] = 0.0


def _park(scene: Scene, i: int) -> None:
    hz = max(scene.parts[i].dims) / 2 + 0.001
    _set_free(scene, i, [PARK_X0 + 0.2 * scene.parts[i].id, 0.0, hz], [1, 0, 0, 0])


def _part_dofs(scene: Scene, idx: list[int]) -> np.ndarray:
    m = scene.model
    return np.array([m.jnt_dofadr[m.body_jntadr[scene.part_body_ids[i]]] for i in idx], dtype=int)


def _settled(scene: Scene, dofs: np.ndarray) -> bool:
    if len(dofs) == 0:
        return True
    v = scene.data.qvel
    lin = np.array([np.linalg.norm(v[d:d + 3]) for d in dofs])
    ang = np.array([np.linalg.norm(v[d + 3:d + 6]) for d in dofs])
    return bool(lin.max() < scene.cfg.settle.lin_tol and ang.max() < scene.cfg.settle.ang_tol)


def _run_until_settled(scene: Scene, dofs: np.ndarray) -> None:
    m, d, s = scene.model, scene.data, scene.cfg.settle
    n_min, n_max = int(MIN_SETTLE_S / m.opt.timestep), int(s.drop_timeout_s / m.opt.timestep)
    for k in range(n_max):
        mujoco.mj_step(m, d)
        if k >= n_min and k % 10 == 0 and _settled(scene, dofs):
            return


def _inside(scene: Scene, i: int) -> bool:
    lo, hi = scene.bin_inner_bounds()
    p = scene.data.qpos[scene.part_qpos_adr[i]:scene.part_qpos_adr[i] + 3]
    return bool(np.all(p[:2] > lo[:2]) and np.all(p[:2] < hi[:2]) and p[2] > lo[2])


def settle_parts(scene: Scene, rng: np.random.Generator) -> list[PartPose]:
    s = scene.cfg.settle
    scene.park_arm()
    placed: list[int] = []
    for i, part in enumerate(scene.parts):
        for attempt in range(s.max_redrops + 1):
            lo, hi = scene.bin_inner_bounds()
            xy = rng.uniform(lo[:2] + s.inset, hi[:2] - s.inset)
            z = hi[2] + s.drop_height_above_rim + max(part.dims) / 2
            q = rng.normal(size=4)
            _set_free(scene, i, [xy[0], xy[1], z], q / np.linalg.norm(q))
            mujoco.mj_forward(scene.model, scene.data)
            _run_until_settled(scene, _part_dofs(scene, placed + [i]))
            if _inside(scene, i):
                placed.append(i)
                break
            _park(scene, i)
        else:
            raise SettleError(part.id, f"escaped the bin on {s.max_redrops + 1} drops")
    for _ in range(FINAL_PASSES):  # whole pile; stops at the first pass that ends at rest
        _run_until_settled(scene, _part_dofs(scene, placed))
        if _settled(scene, _part_dofs(scene, placed)):
            break
    mujoco.mj_forward(scene.model, scene.data)
    validate_settled(scene)
    return [
        PartPose(p.id, scene.data.qpos[a:a + 3].copy(), scene.data.qpos[a + 3:a + 7].copy())
        for p, a in zip(scene.parts, scene.part_qpos_adr)
    ]


def validate_settled(scene: Scene) -> None:
    """Raise SettleError unless every part is inside the bin and at rest."""
    s = scene.cfg.settle
    for i, part in enumerate(scene.parts):
        if not _inside(scene, i):
            raise SettleError(part.id, "ended outside the bin (knocked out by a later drop)")
        d = _part_dofs(scene, [i])[0]
        v = scene.data.qvel
        if np.linalg.norm(v[d:d + 3]) >= s.lin_tol or np.linalg.norm(v[d + 3:d + 6]) >= s.ang_tol:
            raise SettleError(part.id, "still moving after the final settle pass")


def max_penetration(scene: Scene) -> float:
    """Deepest penetration (m, >= 0) among contacts that involve a part."""
    part_bodies = set(scene.part_body_ids)
    m, d = scene.model, scene.data
    worst = 0.0
    for c in d.contact[: d.ncon]:
        if m.geom_bodyid[c.geom1] in part_bodies or m.geom_bodyid[c.geom2] in part_bodies:
            worst = max(worst, -float(c.dist))
    return worst


def make_settled_scene(cfg: Config, seed: int) -> tuple[Scene, list[PartPose]]:
    parts = sample_parts(cfg.parts, np.random.default_rng([seed, 0]))
    scene = build_scene(cfg, parts)
    return scene, settle_parts(scene, np.random.default_rng([seed, 1]))
