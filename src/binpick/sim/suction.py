"""Suction cup: seal model and load check (ground truth, physics side) and the vacuum sensor the
supervisor reads.

Seal quality q in [0, 1] comes from rays cast from points on the cup lip along the cup axis onto
the part under the cup:
  - any lip ray misses the part (edge, small face, thin curved side)  -> q = 0
  - the lip is not touching (closest hit farther than contact_dist)    -> q = 0
  - otherwise q falls linearly with the lip height spread (tilt, curvature) from gap_ok to gap_max.

Holding: when q > 0 the cup grips the part with a weld constraint (vacuum pulls it onto the lip).
Every step the wrench that weld actually carries is read from the physics and compared with what
the seal can hold, F_avail = q * vacuum * effective area:
    u = max(tension, 0) / F_avail + bending / (F_avail * r_lip * peel_lever_ratio)
        + shear / (shear_friction * F_avail)
u, low-pass filtered over break_filter_s (a seal fails when air leaks in, not on a 2 ms spike),
above 1 breaks the grip: the part falls under physics. Nothing decides the outcome but load vs seal.
(MuJoCo's adhesion actuator was tried first: it only acts through active contacts, so its real
capacity was ~70 % of nominal and fell apart nonlinearly; it could not be calibrated.)
"""

from __future__ import annotations

import mujoco
import numpy as np

from binpick.sim.scene import Scene

RAY_BACKOFF = 0.03  # rays start this far behind the cup face


def _cup_frame(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    site = scene.data.site(scene.cup_site)
    return site.xpos.copy(), site.xmat.reshape(3, 3).copy()


def lip_distances(scene: Scene, part_index: int) -> np.ndarray:
    """Distance from each lip point (in the cup-face plane) to the part surface along the cup axis.
    NaN where the ray misses the part."""
    m, d = scene.model, scene.data
    tip, R = _cup_frame(scene)
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    r = scene.cfg.cup.diameter / 2
    n = scene.cfg.suction.lip_points
    body = scene.part_body_ids[part_index]
    geoms = range(m.body_geomadr[body], m.body_geomadr[body] + m.body_geomnum[body])
    out = np.full(n, np.nan)
    for k, th in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        pnt = tip + r * (np.cos(th) * x + np.sin(th) * y) - RAY_BACKOFF * z
        best = np.inf
        for g in geoms:
            dist = mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g], pnt, z, m.geom_type[g])
            if dist >= 0:
                best = min(best, dist)
        if np.isfinite(best):
            out[k] = best - RAY_BACKOFF
    return out


def seal_quality(scene: Scene, part_index: int) -> float:
    s = scene.cfg.suction
    dist = lip_distances(scene, part_index)
    if np.isnan(dist).any() or dist.min() > s.contact_dist:
        return 0.0
    spread = float(dist.max() - dist.min())
    return float(np.clip(1.0 - (spread - s.gap_ok) / (s.gap_max - s.gap_ok), 0.0, 1.0))


def capacity_n(scene: Scene) -> float:
    """Pull force of a perfect seal: vacuum * effective area."""
    s, r = scene.cfg.suction, scene.cfg.cup.diameter / 2
    return s.vacuum_kpa * 1e3 * np.pi * (s.effective_radius_ratio * r) ** 2


class SuctionCup:
    """Call `update()` once per physics step. The supervisor may only use on/off/pressure_kpa/sealed."""

    def __init__(self, scene: Scene, rng: np.random.Generator):
        self.scene, self.rng = scene, rng
        m = scene.model
        self._cup_body = m.geom(scene.cup_geom).bodyid[0]
        self._part_of_body = {b: i for i, b in enumerate(scene.part_body_ids)}
        self._eq = [m.equality(f"grip_{p.id}").id for p in scene.parts]
        self.active = False
        self.q = 0.0
        self.part_index: int | None = None  # ground truth: part held by the cup (not for the supervisor)
        self.last_utilization = 0.0
        self._broken = False  # seal broke under load: no re-grip until the vacuum is cycled off/on
        self._p = 0.0  # true line pressure, kPa (negative = vacuum)

    def on(self) -> None:
        self.active = True

    def off(self) -> None:
        self.active = False
        self._broken = False  # a new on() may seal again

    @property
    def holding(self) -> bool:
        return self.part_index is not None

    def _part_under_cup(self) -> int | None:
        tip, R = _cup_frame(self.scene)
        z = R[:, 2]
        geomid = np.array([-1], dtype=np.int32)
        # start just inside the cup face (behind it is the tool shaft, a different body)
        dist = mujoco.mj_ray(self.scene.model, self.scene.data, tip - 0.001 * z, z, None, 1,
                             self._cup_body, geomid)
        if dist < 0 or dist > self.scene.cfg.suction.contact_dist or geomid[0] < 0:
            return None
        return self._part_of_body.get(int(self.scene.model.geom_bodyid[geomid[0]]))

    def _attach(self, i: int) -> None:
        m, d = self.scene.model, self.scene.data
        b1, b2 = self._cup_body, self.scene.part_body_ids[i]
        R1 = d.xmat[b1].reshape(3, 3)
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, d.xquat[b1])
        rq = np.zeros(4)
        mujoco.mju_mulQuat(rq, q1inv, d.xquat[b2])
        e = self._eq[i]
        m.eq_data[e, 0:3] = 0.0                              # anchor: part origin (body2 frame)
        m.eq_data[e, 3:6] = R1.T @ (d.xpos[b2] - d.xpos[b1])  # same point in cup frame (body1)
        m.eq_data[e, 6:10] = rq
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1
        self.part_index = i
        self.last_utilization = 0.0

    def _detach(self) -> None:
        if self.part_index is not None:
            self.scene.data.eq_active[self._eq[self.part_index]] = 0
        self.part_index, self.q = None, 0.0

    def _weld_wrench_at_tip(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Net force (world) and moment about the cup tip (world) the cup applies to part i: the weld
        (vacuum) plus the cup-to-part contact forces. Their internal preload (lip pressed onto the
        part) cancels; what remains is the external load the seal has to carry."""
        m, d = self.scene.model, self.scene.data
        n = d.nefc
        mask = (d.efc_type[:n] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) & (d.efc_id[:n] == self._eq[i])
        f = np.where(mask, d.efc_force[:n], 0.0)
        qfrc = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, qfrc, f)  # J^T f, handles sparse and dense Jacobians
        dof = m.jnt_dofadr[m.body_jntadr[self.scene.part_body_ids[i]]]
        gen = qfrc[dof:dof + 6]
        b = self.scene.part_body_ids[i]
        F = gen[:3]
        tau_com = d.xmat[b].reshape(3, 3) @ gen[3:]  # free-joint angular dofs are in the body frame
        tip, _ = _cup_frame(self.scene)
        M = tau_com + np.cross(d.xipos[b] - tip, F)
        f6 = np.zeros(6)
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if {b1, b2} != {self._cup_body, b}:
                continue
            mujoco.mj_contactForce(m, d, k, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]  # contact frame -> world; acts on geom2 along +normal
            if b1 == b:
                fw = -fw
            F = F + fw
            M = M + np.cross(c.pos - tip, fw)
        return F, M

    def utilization(self) -> float:
        if self.part_index is None or self.q <= 0:
            return np.inf
        s, r = self.scene.cfg.suction, self.scene.cfg.cup.diameter / 2
        F, M = self._weld_wrench_at_tip(self.part_index)
        _, R = _cup_frame(self.scene)
        z = R[:, 2]                      # cup axis, pointing from the cup into the part
        tension = float(-F @ z)          # weld pulling the part toward the cup
        shear = float(np.linalg.norm(F + tension * z))
        bending = float(np.linalg.norm(M - (M @ z) * z))
        fa = self.q * capacity_n(self.scene)
        return (max(tension, 0.0) / fa + bending / (fa * r * s.peel_lever_ratio)
                + shear / (s.shear_friction * fa))

    def update(self) -> None:
        m = self.scene.model
        if not self.active:
            self._detach()
        elif self.part_index is None and not self._broken:
            i = self._part_under_cup()
            q = seal_quality(self.scene, i) if i is not None else 0.0
            if q > 0:
                self.q = q
                self._attach(i)
        elif self.part_index is not None:
            a = min(1.0, m.opt.timestep / self.scene.cfg.suction.break_filter_s)
            self.last_utilization += a * (self.utilization() - self.last_utilization)
            if self.last_utilization > 1.0:
                self._detach()
                self._broken = True
        cfg = self.scene.cfg
        tau = max(cfg.timing.vacuum_build_s / 3, m.opt.timestep)
        target = -cfg.suction.vacuum_kpa * self.q
        self._p += (target - self._p) * min(1.0, m.opt.timestep / tau)

    def pressure_kpa(self) -> float:
        return self._p + float(self.rng.normal(0.0, self.scene.cfg.suction.sensor_noise_kpa))

    def sealed(self) -> bool:
        """Grip confirmed (switch-on threshold)."""
        return self.pressure_kpa() < -self.scene.cfg.suction.sealed_threshold_kpa

    def vacuum_present(self) -> bool:
        """Still holding something (switch-off threshold, lower than switch-on: hysteresis)."""
        return self.pressure_kpa() < -self.scene.cfg.suction.lost_threshold_kpa
