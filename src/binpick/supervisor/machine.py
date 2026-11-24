"""Pick-and-place state machine:
capture -> plan -> approach -> grip -> lift -> transport -> place -> release -> retreat -> verify.

The supervisor senses only what a real cell has: the arm's own joint positions, the vacuum sensor and
the cameras. Ground truth is used only by the referee (scoring) and the lost-part bookkeeping.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import mujoco
import numpy as np

from binpick.motion.collision import CellCheck
from binpick.motion.ik import DOWN, cup_rotation, solve_cup_pose, yaw_of
from binpick.motion.trajectory import IKError, JointTrajectory, cartesian_line, cartesian_path, execute, joint_move, reverse
from binpick.planning.types import GraspTarget
from binpick.sim.scene import G, PARK_X0, PREFIX, Scene
from binpick.sim.suction import SuctionCup
from binpick.supervisor.placement import SlotAllocator, place_yaw, occupied_slots, referee_in_slot, verify_slot
from binpick.supervisor.records import AttemptRecord

HOVER_ABOVE_RIM = 0.10   # approach lines start at least this far above the bin rim
PROBE_BLANK_S = 0.12     # contact probing ignores the start of the move (the arm is still speeding up)
PROBE_WINDOW_S = 0.02    # lag is compared with its mean 20-40 ms earlier
SHAKE_RAMP_S = 0.3
ARM_HIT_MIN_N = 50.0     # scoring: arm or tool hitting the cell harder than this is a collision
SEAL_POLL_STEPS = 10     # read the vacuum sensor every 10 physics steps (20 ms)

Observer = Callable[[Scene, str], None]


class _Abort(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class Supervisor:
    def __init__(self, scene: Scene, rng: np.random.Generator, seed: int, observer: Observer | None = None):
        self.scene, self.seed, self.observer = scene, seed, observer
        self.cup = SuctionCup(scene, rng)
        self.cell = CellCheck(scene)
        cfg = scene.cfg
        self.slots = SlotAllocator(cfg.tray.rows * cfg.tray.cols)
        self.records: list[AttemptRecord] = []
        self.events: list[dict] = []
        self.lost: set[int] = set()
        self._phase = "idle"
        self.last_target: GraspTarget | None = None
        self.last_success_sensed = False  # the robot's own verdict on the last attempt (not the referee's)
        self.set_down_felt = False  # contact was felt on the last set-down
        self.felt_mass_kg = 0.0  # weight of the part, felt at the end of the lift
        self._watch_seal = False
        self._low_polls = 0
        self._k = 0
        r = solve_cup_pose(scene, np.asarray(cfg.layout.view_pos), DOWN, 0.0)
        assert r.ok, "layout.view_pos is unreachable"
        self._view_q = r.q
        m = scene.model
        self._tool_kg = float(sum(m.body_mass[b] for b in range(m.nbody) if m.body(b).name in (PREFIX + "suction_tool", PREFIX + "cup_body")))
        names = [m.body(b).name for b in m.geom_bodyid]
        self._is_arm_geom = np.array([n.startswith("panda/") for n in names])
        self._is_part_geom = np.array([n.startswith("part_") for n in names])
        self._cup_geom_id = m.geom(scene.cup_geom).id
        self._held_part: int | None = None
        self._arm_hit_n = 0.0
        self._clear = True  # arm out of both cameras' view (parked at start)
        self._tip = scene.model.site(scene.cup_site).id
        self._dofs = [int(scene.model.joint(j).dofadr[0]) for j in scene.arm_joint_names]

    # ---- low level -------------------------------------------------------------------------
    def _q(self) -> np.ndarray:
        return np.array([self.scene.data.joint(j).qpos[0] for j in self.scene.arm_joint_names])

    def _on_step(self, scene: Scene) -> bool:
        self.cup.update()
        self._log_truth()
        if self.observer is not None:
            self.observer(scene, self._phase)
        self._k += 1
        if self._watch_seal and self._k % SEAL_POLL_STEPS == 0:
            self._low_polls = 0 if self.cup.vacuum_present() else self._low_polls + 1
            if self._low_polls >= self.scene.cfg.suction.lost_debounce_polls:
                return False
        return True

    def _run(self, traj: JointTrajectory, hold_s: float | None = None, sideways_gain: float = 0.0) -> bool:
        return execute(self.scene, traj, self._on_step, hold_s=hold_s, sideways_gain=sideways_gain)

    def _pull(self) -> float:
        """Downward pull on the cup (N) that the arm's joint torques carry. Only meaningful at rest."""
        m, d, J = self.scene.model, self.scene.data, np.zeros((6, self.scene.model.nv))
        mujoco.mj_jacSite(m, d, J[:3], J[3:], self._tip)
        tau = d.actuator_force[[m.actuator(a).id for a in self.scene.arm_actuator_names]]
        return float((np.linalg.pinv(J[:, self._dofs].T) @ tau)[2]) - self._tool_kg * G

    def _wait(self, seconds: float) -> None:
        q = self._q()
        self._run(JointTrajectory(np.array([0.0]), q[None, :]), hold_s=seconds)

    def _line(self, p0, p1, axis, yaw, loaded: bool = True) -> JointTrajectory:
        return cartesian_line(self.scene, self._q(), p0, p1, axis, yaw, self.scene.cfg.motion, loaded=loaded)

    def _ik_min_motion(self, pos: np.ndarray, axis: np.ndarray, yaw: float | None = None):
        """IK seeded from the current joints with joint1 pre-rotated toward `pos`. With yaw=None the
        (round) cup keeps its current heading relative to the arm, so the wrist does not spin."""
        q = self._q()
        base = np.asarray(self.scene.cfg.layout.robot_base_pos)
        p_cur, R = self._cup_pose()
        d_theta = np.arctan2(pos[1] - base[1], pos[0] - base[0]) - np.arctan2(p_cur[1] - base[1], p_cur[0] - base[0])
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
        q0 = q.copy()
        q0[0] += d_theta
        if yaw is None:
            yaw = float(np.arctan2(R[1, 0], R[0, 0])) + d_theta
        r = solve_cup_pose(self.scene, np.asarray(pos, float), np.asarray(axis, float), yaw, q0=q0)
        if not r.ok:  # the current posture is a bad start: try the standard one
            r = solve_cup_pose(self.scene, np.asarray(pos, float), np.asarray(axis, float), yaw)
        return r

    def _cup_pose(self) -> tuple[np.ndarray, np.ndarray]:
        site = self.scene.data.site(self.scene.cup_site)
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    def _rim(self) -> float:
        return float(self.scene.bin_inner_bounds()[1][2])

    # ---- one attempt -----------------------------------------------------------------------
    def attempt(self, targets: GraspTarget | Callable[[], list[GraspTarget]]) -> AttemptRecord | None:
        """Run one pick-and-place. `targets` is one target, or a planner call made once the cameras are clear
        that returns targets in order of preference; the first the arm can reach is used. Returns None if none
        is (planning rejections are not attempts; they are logged in `events`)."""
        scene, cfg, d = self.scene, self.scene.cfg, self.scene.data
        t0 = d.time
        times: dict[str, float] = {}

        def phase(name: str, fn: Callable[[], None]) -> None:
            self._phase = name
            ts = d.time
            try:
                fn()
            finally:
                times[name] = times.get(name, 0.0) + d.time - ts

        def capture():
            if not self._clear:
                self._to_view()
            self._wait(cfg.timing.capture_s)

        phase("capture", capture)

        # ---- plan: everything up to the grip must be feasible before the arm moves
        plan: dict = {}

        def do_plan():
            candidates = [targets] if isinstance(targets, GraspTarget) else targets()
            self._wait(cfg.timing.plan_s)
            for t in candidates:
                try:
                    plan.update(self._plan_approach(t), target=t)
                    return
                except IKError as e:
                    self.events.append({"seed": self.seed, "time": d.time, "event": "plan_rejected", "detail": str(e)})

        phase("plan", do_plan)
        if not plan:
            return None
        target: GraspTarget = plan["target"]
        n = np.asarray(target.normal, float)

        rec = AttemptRecord(self.seed, len(self.records) + 1, self._truth_part(target), tilt_deg=target.tilt_deg,
                            est_seal=target.seal)
        self.records.append(rec)
        self.last_target, self._clear = target, False
        self._way_in = plan["approach"]
        self._held_part, self._arm_hit_n = None, 0.0
        try:
            phase("approach", lambda: (self._run(joint_move(self._q(), plan["hover_q"], cfg.motion)),
                                       self._run(plan["approach"]),
                                       self._probe(plan["near"], target.pos - n * cfg.motion.touch_past, -n, plan["yaw"],
                                                   cfg.motion.touch_speed, loaded=False)))
            self._grip(phase, rec)
            self._lift_transport_place(phase, rec, target, n)
            phase("release", lambda: (self.cup.off(), self._wait(cfg.timing.release_s)))
            rec.slot, self._pending_slot = self._pending_slot, None
            self.slots.commit()
            p, _ = self._cup_pose()
            phase("retreat", lambda: (self._run(self._line(p, p + [0, 0, cfg.motion.retreat_dist], DOWN,
                                                             self._place_yaw, loaded=False)),
                                      self._to_view()))

            def verify():
                self._wait(cfg.timing.verify_s)
                rec.verify_ok = verify_slot(scene, rec.slot)[0] and not self.cup.vacuum_present()
                self._note_occupied()

            phase("verify", verify)
        except _Abort as a:
            rec.reason = rec.sensed = a.reason
            self._recover(phase)
            self._slot_after_failure(rec)
        except IKError:
            rec.reason = rec.sensed = "ik_fail"
            self._recover(phase)
            self._slot_after_failure(rec)

        self.last_success_sensed = rec.verify_ok
        self._score(rec)
        rec.cycle_time_s = d.time - t0
        rec.phase_times = times
        self._phase = "idle"
        return rec

    def _plan_approach(self, t: GraspTarget) -> dict:
        cfg, n = self.scene.cfg, np.asarray(t.normal, float)
        near = t.pos + n * cfg.motion.touch_start
        hover_dist = max(cfg.motion.pregrasp_dist, (self._rim() + HOVER_ABOVE_RIM - t.pos[2]) / max(n[2], 0.2))
        hover = t.pos + n * hover_dist
        r = self._ik_min_motion(hover, -n)
        if not r.ok:
            raise IKError("hover unreachable")
        approach = cartesian_line(self.scene, r.q, hover, near, -n, r.yaw, cfg.motion, loaded=False)
        if self.cell.path_hits(joint_move(self._q(), r.q, cfg.motion)) or self.cell.path_hits(approach):
            raise IKError("collision with the cell on the way in")
        return {"hover_q": r.q, "approach": approach, "near": near, "yaw": r.yaw}

    def _truth_part(self, t: GraspTarget) -> int | None:
        """Ground truth, for the log only: which part lies under the chosen point."""
        geom = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(self.scene.model, self.scene.data, t.pos + 0.003 * t.normal, -np.asarray(t.normal, float), None, 1,
                      -1, geom)
        body = int(self.scene.model.geom_bodyid[geom[0]]) if geom[0] >= 0 else -1
        return self.scene.part_body_ids.index(body) if body in self.scene.part_body_ids else None

    def _grip(self, phase, rec: AttemptRecord) -> None:
        cfg = self.scene.cfg

        def grip():
            self.cup.on()
            self._wait(cfg.timing.vacuum_build_s)
            if not self.cup.sealed():
                raise _Abort("no_seal")

        phase("grip", grip)

    def _lift_transport_place(self, phase, rec: AttemptRecord, target: GraspTarget, n: np.ndarray) -> None:
        cfg, scene = self.scene.cfg, self.scene
        p_grip, R_grip = self._cup_pose()
        long_cup = R_grip.T @ target.long_axis
        center = target.pos if target.center is None else target.center
        offset_cup = R_grip.T @ (center - target.pos)  # from the cup to the part's estimated center
        clear_z = self._rim() + target.depth + cfg.motion.clear_above_rim

        def lift():
            self._watch_seal, self._low_polls = True, 0
            # Straight up, whatever the face tilt: lifting along a tilted normal drags parts that lean on
            # a wall (or a neighbour) into it. Cup orientation is kept.
            p1 = np.r_[p_grip[:2], max(clear_z, p_grip[2] + cfg.motion.lift_dist)]
            # The first millimetres also go a little sideways, away from what touches the part.
            side = p_grip + np.r_[np.asarray(target.away) * cfg.motion.lift_away, cfg.motion.lift_away_rise]
            m = dataclasses.replace(cfg.motion, max_cart_acc=cfg.motion.lift_acc)
            path = cartesian_path(scene, self._q(), [p_grip, side, np.r_[side[:2], p1[2]]], -n, self._grasp_yaw(), m)
            if not self._run(path, sideways_gain=cfg.motion.lift_sideways_gain):
                raise _Abort("seal_lost_lift")
            self.felt_mass_kg = self._pull() / G

        phase("lift", lift)
        slot = self.slots.peek()
        if slot is None:
            raise _Abort("tray_full")
        self._pending_slot = slot  # used up only on release (or if a failed part ends up lying in it)
        slot_c = scene.slot_centers[slot]
        self._place_yaw = place_yaw(long_cup)
        top = slot_c[2] + target.depth + cfg.motion.place_clearance  # cup height that surely clears the tray
        pre_z = max(clear_z, top + cfg.motion.pregrasp_dist)
        xy = {}

        def transport():
            # the part's center, not the cup, goes to the slot center (the wrist may turn by pi, so solve twice)
            at = lambda yaw: slot_c[:2] - (cup_rotation(DOWN, yaw) @ np.r_[offset_cup[:2], 0.0])[:2]  # noqa: E731
            r = self._ik_min_motion(np.r_[at(self._place_yaw), pre_z], DOWN, self._place_yaw)
            if r.ok:
                r = self._ik_min_motion(np.r_[at(r.yaw), pre_z], DOWN, r.yaw)
            if not r.ok:
                raise IKError("pre-place unreachable")
            self._place_yaw, xy["cup"] = r.yaw, at(r.yaw)
            heavy = self.felt_mass_kg > cfg.motion.heavy_kg
            m = dataclasses.replace(cfg.motion, max_joint_acc_loaded=cfg.motion.max_joint_acc_heavy) if heavy else cfg.motion
            if not self._run(joint_move(self._q(), r.q, m, loaded=True)):
                raise _Abort("seal_lost_transport")

        phase("transport", transport)

        def place():
            c = xy["cup"]
            if not self._run(self._line(np.r_[c, pre_z], np.r_[c, top], DOWN, self._place_yaw)):
                raise _Abort("seal_lost_transport")
            self._set_down(c, top, slot_c[2])
            self._watch_seal = False

        phase("place", place)

    def _probe(self, p0: np.ndarray, p1: np.ndarray, axis: np.ndarray, yaw: float, speed: float,
               loaded: bool) -> tuple[bool, bool]:
        """Move slowly from p0 toward p1 until contact is felt: the arm's own joint readings start to lag behind the
        commanded joints along `axis`. Returns (completed or touched, touched)."""
        scene = self.scene
        m = dataclasses.replace(scene.cfg.motion, max_cart_vel=speed)
        tr = cartesian_line(scene, self._q(), p0, p1, axis, yaw, m, loaded=loaded)
        t0, J, lags, touched = scene.data.time, np.zeros((3, scene.model.nv)), [], []
        axis = np.asarray(axis, float)
        k, blank = int(PROBE_WINDOW_S / scene.model.opt.timestep), PROBE_BLANK_S / scene.model.opt.timestep

        def step(sc: Scene) -> bool:
            if not self._on_step(sc):
                return False
            mujoco.mj_jacSite(sc.model, sc.data, J, None, self._tip)
            lags.append(axis @ J[:, self._dofs] @ (tr.sample(sc.data.time - t0) - self._q()))
            if len(lags) > blank and lags[-1] - np.mean(lags[-2 * k:-k]) > m.probe_lag:
                touched.append(True)
                return False
            return True

        done = execute(scene, tr, step)
        if touched:
            self._wait(scene.cfg.motion.settle_after_move_s)  # come to rest; the next hold then starts unloaded
        return done or bool(touched), bool(touched)

    def _set_down(self, xy: np.ndarray, z_top: float, tray_z: float) -> None:
        """Lower the part until it touches the tray (the camera cannot see its thickness)."""
        m, parts = self.scene.cfg.motion, self.scene.cfg.parts
        z_end = tray_z + min(parts.box_height[0], parts.cyl_diameter[0]) - m.probe_past
        ok, self.set_down_felt = self._probe(np.r_[xy, z_top], np.r_[xy, z_end], DOWN, self._place_yaw, m.probe_speed, loaded=True)
        if not ok:
            raise _Abort("seal_lost_transport")

    def _slot_after_failure(self, rec: AttemptRecord) -> None:
        """A failed part seen lying in the pending slot (tray camera) uses that slot up."""
        slot, self._pending_slot = self._pending_slot, None
        if slot is not None:
            if verify_slot(self.scene, slot)[1] is not None:
                rec.slot = slot
            self._note_occupied()

    def _note_occupied(self) -> None:
        """Part of the tray check after a placement or a drop: never place into a slot something lies in."""
        for k in occupied_slots(self.scene):
            self.slots.mark_used(k)

    _pending_slot: int | None = None

    def _to_view(self) -> None:
        """Move to the pose that is in neither camera's view: always the same joints, so posture cannot drift."""
        self._run(joint_move(self._q(), self._view_q, self.scene.cfg.motion))
        self._clear = True

    def _grasp_yaw(self) -> float:
        return yaw_of(self._cup_pose()[1])

    _place_yaw = 0.0

    def _recover(self, phase) -> None:
        """After a failure: vacuum off, back the cup straight up above the rim."""
        self._watch_seal = False
        cfg = self.scene.cfg

        def retreat():
            self.cup.off()
            p, R = self._cup_pose()
            up = np.r_[p[:2], max(p[2] + cfg.motion.retreat_dist, self._rim() + HOVER_ABOVE_RIM)]
            try:
                self._run(self._line(p, up, R[:, 2], yaw_of(R), loaded=False))
            except IKError:
                if p[2] < self._rim() + HOVER_ABOVE_RIM:  # inside the bin: back out along the approach
                    self._run(joint_move(self._q(), self._way_in.q[-1], cfg.motion))
                    self._run(reverse(self._way_in))
            self._to_view()

        phase("retreat", retreat)

    # ---- shaking ---------------------------------------------------------------------------
    def shake(self, rng: np.random.Generator) -> None:
        """Shake the bin with the arm out of the way, then give the parts time to settle."""
        scene, sh = self.scene, self.scene.cfg.shaker
        if not self._clear:
            self._to_view()
        self.events.append({"seed": self.seed, "time": scene.data.time, "event": "shake"})
        self._phase = "shake"
        acts = [scene.model.actuator(a).id for a in scene.shaker_actuators]
        phase, t0 = rng.uniform(0, 2 * np.pi, 2), scene.data.time
        q = self._q()
        hold = JointTrajectory(np.array([0.0]), q[None, :])

        def step(sc: Scene) -> bool:
            t = sc.data.time - t0
            # fade in and out so the bin starts and ends at home
            w = np.clip(min(t, sh.duration_s - t) / SHAKE_RAMP_S, 0.0, 1.0)
            sc.data.ctrl[acts] = w * sh.amplitude * np.sin(2 * np.pi * sh.freq_hz * t + phase)
            return self._on_step(sc)

        execute(scene, hold, step, hold_s=sh.duration_s + sh.settle_s)
        scene.data.ctrl[acts] = 0.0
        self._phase = "idle"

    # ---- scoring (ground truth) ------------------------------------------------------------
    def _log_truth(self) -> None:
        """What really happens, for the attempt record only (never read by the robot)."""
        m, d = self.scene.model, self.scene.data
        if self.cup.holding:
            self._held_part = self.cup.part_index
        g = d.contact.geom[:d.ncon]
        arm, part = self._is_arm_geom[g], self._is_part_geom[g]
        cup = g == self._cup_geom_id
        # arm or tool against bin, tray or floor; or anything but the cup lip against a part
        hits = arm.any(axis=1) & ~arm.all(axis=1) & ~(cup.any(axis=1) & part.any(axis=1))
        f = np.zeros(6)
        for k in np.flatnonzero(hits):
            mujoco.mj_contactForce(m, d, int(k), f)
            self._arm_hit_n = max(self._arm_hit_n, float(f[0]))

    def _score(self, rec: AttemptRecord) -> None:
        scene, d = self.scene, self.scene.data
        mujoco.mj_forward(scene.model, d)
        rec.gripped_part = part = self._held_part
        rec.arm_hit_n = self._arm_hit_n if self._arm_hit_n > ARM_HIT_MIN_N else 0.0
        rec.referee_ok = part is not None and rec.slot is not None and referee_in_slot(scene, part, rec.slot)
        rec.detected = rec.verify_ok == rec.referee_ok
        if rec.arm_hit_n:  # spec: a collision during motion is a failed attempt, wherever the part ends up
            rec.outcome, rec.reason = "fail", "collision"
        elif rec.referee_ok:
            rec.outcome, rec.reason = "success", ""
        else:
            rec.outcome = "fail"
            if not rec.reason or rec.slot is not None:
                rec.reason = "misplaced"  # it lies in its slot area, off center
            elif rec.reason == "no_seal" and part is not None:
                rec.reason = "seal_lost_grip"  # sealed, then broke before the vacuum was confirmed
            elif rec.reason.startswith("seal_lost") and self.cup.break_cause == "external_contact":
                rec.reason = "collision"  # root cause: torn off by contact with the bin or another part
        lo, hi = scene.bin_inner_bounds()
        for i, b in enumerate(scene.part_body_ids):
            if i in self.lost:
                continue
            p = d.xpos[b]
            in_bin = np.all(p[:2] > lo[:2]) and np.all(p[:2] < hi[:2])
            in_tray = any(np.all(np.abs(p[:2] - c[:2]) < scene.cfg.tray.slot / 2) for c in scene.slot_centers)
            if not in_bin and not in_tray:
                self._park_lost(i)
                rec.lost_parts.append(scene.parts[i].id)

    def _park_lost(self, i: int) -> None:
        d, a = self.scene.data, self.scene.part_qpos_adr[i]
        d.qpos[a:a + 3] = [PARK_X0 + 0.2 * self.scene.parts[i].id, 1.0, max(self.scene.parts[i].dims)]
        d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.lost.add(i)
        mujoco.mj_forward(self.scene.model, d)
