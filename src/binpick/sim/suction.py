"""Suction cup: seal model and load check (ground truth, physics side) and the vacuum sensor the
supervisor reads.

The cup is modelled on the Piab B40.20 bellows cup (silicone, item sheet 2026-08-27): 39.0 N pull at
60 kPa, minimum curve radius 20 mm, 13.5 mm vertical travel.

Seal quality q in [0, 1]. The soft lip wraps onto the surface, keeping its length: from the cup center
the surface is followed outwards in lip_points directions until the path is one lip radius long.
  - any path leaves the part (edge, small face, neighbour on top)      -> q = 0
  - any path bends more sharply than the lip can (a sharp edge)        -> q = 0
  - the lip is not touching (closest lip point beyond contact_dist)    -> q = 0
  - lip height spread beyond the bellows travel                        -> q = 0
  - otherwise q = q_tilt * q_curve. q_tilt: best-fit lip plane, 1 up to gap_ok, 0 at gap_max (M2 values;
    the sheet gives no tilt data). q_curve: what is left after removing that plane, falling from 1 (flat)
    to curve_limit_q at the minimum curve radius, 0 beyond it.
curve_limit_q is an assumption: the sheet does not say how much force is left at the limit radius.

Holding: when q > 0 the cup grips the part with a weld constraint (vacuum pulls it onto the lip).
Every step the wrench that weld actually carries is read from the physics and compared with what
the seal can hold, F_avail = q * pull_force_n:
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

RAY_BACKOFF = 0.018  # rays start this far behind the cup face: inside the 20 mm cup, below the tool shaft
WRAP_STEPS = 24     # rays per lip direction when wrapping the lip onto the surface
MAX_LIP_SLOPE = 3.0  # 72 deg: above the 63 deg a lip reaches at the curve limit, below a box edge
CURVE_TOL = 3e-4    # m, ray-march error allowed at the curve limit
EXTERNAL_CONTACT_MIN_N = 5.0  # a break with more contact force than this (and the part's weight) from
                              # anything but the cup is a collision/snag, not a weak seal


def _cup_frame(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    site = scene.data.site(scene.cup_site)
    return site.xpos.copy(), site.xmat.reshape(3, 3).copy()


def wrapped_lip(scene: Scene, part_index: int, pos: np.ndarray, R: np.ndarray, backoff: float,
                exclude_body: int, n_points: int) -> np.ndarray:
    """Height (along the cup axis R[:, 2], from the plane through pos) of each point of a lip wrapped
    onto part `part_index` with the cup centered at pos. NaN where the lip cannot sit on the part.
    Rays are cast against the whole scene except `exclude_body`."""
    m, d = scene.model, scene.data
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    r = scene.cfg.cup.diameter / 2
    body = scene.part_body_ids[part_index]
    geomid = np.array([-1], dtype=np.int32)

    def height(offset: np.ndarray) -> float:
        dist = mujoco.mj_ray(m, d, pos + offset - backoff * z, z, None, 1, exclude_body, geomid)
        if dist < 0 or m.geom_bodyid[geomid[0]] != body:
            return np.nan
        return dist - backoff

    out = np.full(n_points, np.nan)
    h0 = height(np.zeros(3))
    step = r / WRAP_STEPS
    for k, th in enumerate(np.linspace(0, 2 * np.pi, n_points, endpoint=False)):
        u = np.cos(th) * x + np.sin(th) * y
        h, left = h0, r
        for j in range(1, WRAP_STEPS + 1):
            h1 = height(j * step * u)
            if np.isnan(h1) or abs(h1 - h) > MAX_LIP_SLOPE * step:
                break
            seg = np.hypot(step, h1 - h)
            if seg >= left - 1e-9:  # the lip ends on this segment
                out[k] = h + (h1 - h) * left / seg
                break
            h, left = h1, left - seg
    return out


def lip_distances(scene: Scene, part_index: int) -> np.ndarray:
    """Distance from each (wrapped) lip point to the cup-face plane; NaN where it cannot sit on the part."""
    tip, R = _cup_frame(scene)
    cup_body = scene.model.geom(scene.cup_geom).bodyid[0]
    return wrapped_lip(scene, part_index, tip, R, RAY_BACKOFF, cup_body, scene.cfg.suction.lip_points)


def seal_quality(scene: Scene, part_index: int) -> float:
    s, r = scene.cfg.suction, scene.cfg.cup.diameter / 2
    dist = lip_distances(scene, part_index)
    if np.isnan(dist).any() or dist.min() > s.contact_dist or np.ptp(dist) > s.travel:
        return 0.0
    th = np.linspace(0, 2 * np.pi, len(dist), endpoint=False)
    a, b = 2 * np.mean(dist * np.cos(th)), 2 * np.mean(dist * np.sin(th))  # best-fit plane through the lip
    q_tilt = np.clip(1.0 - (2 * np.hypot(a, b) - s.gap_ok) / (s.gap_max - s.gap_ok), 0.0, 1.0)
    drop = np.ptp(dist - a * np.cos(th) - b * np.sin(th))
    limit = s.curve_radius_min * (1 - np.cos(r / s.curve_radius_min))  # lip drop at the minimum curve radius
    if drop > limit + CURVE_TOL:
        return 0.0
    q_curve = 1.0 - (1.0 - s.curve_limit_q) * min(drop / limit, 1.0)
    return float(q_tilt * q_curve)


def capacity_n(scene: Scene) -> float:
    """Pull force of a perfect seal (datasheet value at the line vacuum)."""
    return scene.cfg.suction.pull_force_n


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
        self.break_cause = ""  # ground truth, for failure attribution only: "overload" | "external_contact"
        self._broken = False  # seal broke under load: no re-grip until the vacuum is cycled off/on
        self._p = 0.0  # true line pressure, kPa (negative = vacuum)

    def on(self) -> None:
        self.active = True
        self.break_cause = ""

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
        return (tension / fa + bending / (fa * r * s.peel_lever_ratio)
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
                ext = self._external_contact_n(self.part_index)
                weight = self.scene.model.body_mass[self.scene.part_body_ids[self.part_index]] * 9.81
                self.break_cause = "external_contact" if ext > max(EXTERNAL_CONTACT_MIN_N, weight) else "overload"
                self._detach()
                self._broken = True
        cfg = self.scene.cfg
        tau = max(cfg.timing.vacuum_build_s / 3, m.opt.timestep)
        target = -cfg.suction.vacuum_kpa * self.q
        self._p += (target - self._p) * min(1.0, m.opt.timestep / tau)

    def _external_contact_n(self, i: int) -> float:
        m, d = self.scene.model, self.scene.data
        b, total, f6 = self.scene.part_body_ids[i], 0.0, np.zeros(6)
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b in (b1, b2) and self._cup_body not in (b1, b2):
                mujoco.mj_contactForce(m, d, k, f6)
                total += abs(f6[0])
        return total

    def pressure_display_kpa(self) -> float:
        """Noise-free line pressure for videos/plots (does not consume the sensor noise stream)."""
        return self._p

    def pressure_kpa(self) -> float:
        return self._p + float(self.rng.normal(0.0, self.scene.cfg.suction.sensor_noise_kpa))

    def sealed(self) -> bool:
        """Grip confirmed (switch-on threshold)."""
        return self.pressure_kpa() < -self.scene.cfg.suction.sealed_threshold_kpa

    def vacuum_present(self) -> bool:
        """Still holding something (switch-off threshold, lower than switch-on: hysteresis)."""
        return self.pressure_kpa() < -self.scene.cfg.suction.lost_threshold_kpa
