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

from binpick.motion.ik import DOWN, cup_rotation, solve_cup_pose, yaw_of
from binpick.motion.trajectory import IKError, JointTrajectory, cartesian_line, execute, joint_move
from binpick.planning.types import GraspTarget
from binpick.sim.scene import PARK_X0, Scene
from binpick.sim.suction import SuctionCup
from binpick.supervisor.placement import SlotAllocator, place_yaw, referee_in_slot, verify_slot
from binpick.supervisor.records import AttemptRecord

HOVER_ABOVE_RIM = 0.10   # approach lines start at least this far above the bin rim
PROBE_BLANK_S = 0.12     # contact probing ignores the start of the move (the arm is still speeding up)
PROBE_WINDOW_S = 0.02    # lag is compared with its mean 20-40 ms earlier
SHAKE_RAMP_S = 0.3
SEAL_POLL_STEPS = 10     # read the vacuum sensor every 10 physics steps (20 ms)

Observer = Callable[[Scene, str], None]


class _Abort(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class Supervisor:
    def __init__(self, scene: Scene, rng: np.random.Generator, seed: int, observer: Observer | None = None):
        self.scene, self.seed, self.observer = scene, seed, observer
        self.cup = SuctionCup(scene, rng)
        cfg = scene.cfg
        self.slots = SlotAllocator(cfg.tray.rows * cfg.tray.cols)
        self.records: list[AttemptRecord] = []
        self.events: list[dict] = []
        self.lost: set[int] = set()
        self._phase = "idle"
        self.n_candidates = 0  # targets offered by the planner in the last attempt
        self._watch_seal = False
        self._low_polls = 0
        self._k = 0
        self._clear = True  # arm out of both cameras' view (parked at start)
        self._tip = scene.model.site(scene.cup_site).id
        self._dofs = [int(scene.model.joint(j).dofadr[0]) for j in scene.arm_joint_names]

    # ---- low level -------------------------------------------------------------------------
    def _q(self) -> np.ndarray:
        return np.array([self.scene.data.joint(j).qpos[0] for j in self.scene.arm_joint_names])

    def _on_step(self, scene: Scene) -> bool:
        self.cup.update()
        if self.observer is not None:
            self.observer(scene, self._phase)
        self._k += 1
        if self._watch_seal and self._k % SEAL_POLL_STEPS == 0:
            self._low_polls = 0 if self.cup.vacuum_present() else self._low_polls + 1
            if self._low_polls >= self.scene.cfg.suction.lost_debounce_polls:
                return False
        return True

    def _run(self, traj: JointTrajectory, hold_s: float | None = None) -> bool:
        return execute(self.scene, traj, self._on_step, hold_s=hold_s)

    def _run_steps(self, traj: JointTrajectory, on_step: Callable[[Scene], bool]) -> bool:
        return execute(self.scene, traj, on_step)

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
        return solve_cup_pose(self.scene, np.asarray(pos, float), np.asarray(axis, float), yaw, q0=q0)

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
            self.n_candidates = len(candidates)
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
        self._clear = False
        try:
            phase("approach", lambda: (self._run(joint_move(self._q(), plan["hover_q"], cfg.motion)),
                                       self._run(plan["approach"])))
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

            phase("verify", verify)
        except _Abort as a:
            rec.reason = rec.sensed = a.reason
            self._recover(phase)
            self._slot_after_failure(rec)
        except IKError:
            rec.reason = rec.sensed = "ik_fail"
            self._recover(phase)
            self._slot_after_failure(rec)

        self._score(rec)
        rec.cycle_time_s = d.time - t0
        rec.phase_times = times
        self._phase = "idle"
        return rec

    def _plan_approach(self, t: GraspTarget) -> dict:
        cfg, n = self.scene.cfg, np.asarray(t.normal, float)
        grip = t.pos - n * cfg.motion.approach_overtravel
        hover_dist = max(cfg.motion.pregrasp_dist, (self._rim() + HOVER_ABOVE_RIM - t.pos[2]) / max(n[2], 0.2))
        hover = t.pos + n * hover_dist
        r = self._ik_min_motion(hover, -n)
        if not r.ok:
            raise IKError("hover unreachable")
        return {"hover_q": r.q, "approach": cartesian_line(self.scene, r.q, hover, grip, -n, r.yaw, cfg.motion, loaded=False)}

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
            rec.gripped_part = self.cup.part_index  # ground truth, for the record only
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
            ok = self._run(self._line(p_grip, p1, -n, self._grasp_yaw()))
            if not ok:
                raise _Abort("seal_lost_lift")

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
            if not self._run(joint_move(self._q(), r.q, cfg.motion, loaded=True)):
                raise _Abort("seal_lost_transport")

        phase("transport", transport)

        def place():
            c = xy["cup"]
            if not self._run(self._line(np.r_[c, pre_z], np.r_[c, top], DOWN, self._place_yaw)):
                raise _Abort("seal_lost_transport")
            self._set_down(c, top, slot_c[2])
            self._watch_seal = False

        phase("place", place)

    def _set_down(self, xy: np.ndarray, z_top: float, tray_z: float) -> None:
        """Lower the part until it touches the tray. The camera cannot see its thickness, so contact is felt: the
        arm's own joint readings start to lag behind the commanded joints."""
        scene, m = self.scene, self.scene.cfg.motion
        z_end = tray_z + min(scene.cfg.parts.box_height[0], scene.cfg.parts.cyl_diameter[0]) - m.probe_past
        slow = dataclasses.replace(m, max_cart_vel=m.probe_speed)
        tr = cartesian_line(scene, self._q(), np.r_[xy, z_top], np.r_[xy, z_end], DOWN, self._place_yaw, slow)
        t0, J, lags, touched = scene.data.time, np.zeros((3, scene.model.nv)), [], []

        def step(sc: Scene) -> bool:
            if not self._on_step(sc):
                return False
            mujoco.mj_jacSite(sc.model, sc.data, J, None, self._tip)
            lags.append(-J[2, self._dofs] @ (tr.sample(sc.data.time - t0) - self._q()))  # tip above its command, m
            k = int(PROBE_WINDOW_S / sc.model.opt.timestep)
            if len(lags) > PROBE_BLANK_S / sc.model.opt.timestep and lags[-1] - np.mean(lags[-2 * k:-k]) > m.probe_lag:
                touched.append(True)
                return False
            return True

        if not self._run_steps(tr, step) and not touched:
            raise _Abort("seal_lost_transport")

    def _slot_after_failure(self, rec: AttemptRecord) -> None:
        """A failed part seen lying in the pending slot (tray camera) uses that slot up."""
        slot, self._pending_slot = self._pending_slot, None
        if slot is not None and verify_slot(self.scene, slot)[1] is not None:
            rec.slot = slot
            self.slots.commit()

    _pending_slot: int | None = None

    def _to_view(self) -> None:
        """Move to the pose that is in neither camera's view."""
        pos = np.asarray(self.scene.cfg.layout.view_pos)
        r = self._ik_min_motion(pos, DOWN)
        if not r.ok:  # from an awkward pose: any solution will do
            r = solve_cup_pose(self.scene, pos, DOWN, 0.0)
        self._run(joint_move(self._q(), r.q, self.scene.cfg.motion))
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
                self._wait(0.2)
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
    def _score(self, rec: AttemptRecord) -> None:
        scene, d = self.scene, self.scene.data
        mujoco.mj_forward(scene.model, d)
        part = rec.gripped_part
        rec.referee_ok = part is not None and rec.slot is not None and referee_in_slot(scene, part, rec.slot)
        if rec.referee_ok:
            rec.outcome, rec.reason = "success", ""
            rec.detected = rec.verify_ok
        else:
            rec.outcome = "fail"
            if not rec.reason:
                rec.reason = "misplaced"
            elif rec.reason.startswith("seal_lost") and self.cup.break_cause == "external_contact":
                rec.reason = "collision"  # root cause: torn off by contact with the bin or another part
                rec.detected = not rec.verify_ok
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
