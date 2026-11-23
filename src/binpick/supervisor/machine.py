"""Pick-and-place state machine:
capture -> plan -> approach -> grip -> lift -> transport -> place -> release -> retreat -> verify.

The supervisor senses only what a real cell has: the arm's own joint positions, the vacuum sensor and
the cameras. Ground truth is used only by the referee (scoring) and the lost-part bookkeeping.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from binpick.motion.ik import DOWN, solve_cup_pose, yaw_of
from binpick.motion.trajectory import IKError, JointTrajectory, cartesian_line, execute, joint_move
from binpick.planning.oracle import GraspTarget
from binpick.sim.scene import PARK_X0, Scene
from binpick.sim.suction import SuctionCup
from binpick.supervisor.placement import SlotAllocator, place_yaw, referee_in_slot, verify_slot
from binpick.supervisor.records import AttemptRecord

HOVER_ABOVE_RIM = 0.10   # approach lines start at least this far above the bin rim
READY_ABOVE_RIM = 0.25   # ready pose: cup down over the bin center; out of the tray camera's view
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
        self._watch_seal = False
        self._low_polls = 0
        self._k = 0
        lo, hi = scene.bin_inner_bounds()
        self.ready_pos = np.r_[(lo[:2] + hi[:2]) / 2, hi[2] + READY_ABOVE_RIM]

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
    def attempt(self, target: GraspTarget) -> AttemptRecord | None:
        """Run one pick-and-place. Returns None if planning rejects the target before any motion
        (not an attempt, logged in `events`)."""
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

        phase("capture", lambda: self._wait(cfg.timing.capture_s))

        # ---- plan: everything up to the grip must be feasible before the arm moves
        n = np.asarray(target.normal, float)
        axis = -n
        grip = target.pos - n * cfg.motion.approach_overtravel
        hover_dist = max(cfg.motion.pregrasp_dist, (self._rim() + HOVER_ABOVE_RIM - target.pos[2]) / max(n[2], 0.2))
        hover = target.pos + n * hover_dist
        plan = {}

        def do_plan():
            self._wait(cfg.timing.plan_s)
            r = self._ik_min_motion(hover, axis)
            if not r.ok:
                raise IKError("hover unreachable")
            plan["hover_q"], plan["yaw"] = r.q, r.yaw
            plan["approach"] = cartesian_line(scene, r.q, hover, grip, axis, r.yaw, cfg.motion, loaded=False)

        try:
            phase("plan", do_plan)
        except IKError as e:
            self.events.append({"seed": self.seed, "time": d.time, "event": "plan_rejected",
                                "target_part": target.part_index, "detail": str(e)})
            return None

        rec = AttemptRecord(self.seed, len(self.records) + 1, target.part_index)
        self.records.append(rec)
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
                                      self._to_ready()))

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
        release_p = slot_c + [0, 0, cfg.motion.place_clearance + target.depth]
        pre_place = np.r_[slot_c[:2], max(clear_z, release_p[2] + cfg.motion.pregrasp_dist)]

        def transport():
            r = self._ik_min_motion(pre_place, DOWN, self._place_yaw)
            if not r.ok:
                raise IKError("pre-place unreachable")
            self._place_yaw = r.yaw
            if not self._run(joint_move(self._q(), r.q, cfg.motion, loaded=True)):
                raise _Abort("seal_lost_transport")

        phase("transport", transport)

        def place():
            if not self._run(self._line(pre_place, release_p, DOWN, self._place_yaw)):
                raise _Abort("seal_lost_transport")
            self._watch_seal = False

        phase("place", place)

    def _slot_after_failure(self, rec: AttemptRecord) -> None:
        """A failed part seen lying in the pending slot (tray camera) uses that slot up."""
        slot, self._pending_slot = self._pending_slot, None
        if slot is not None and verify_slot(self.scene, slot)[1] is not None:
            rec.slot = slot
            self.slots.commit()

    _pending_slot: int | None = None

    def _to_ready(self) -> None:
        r = self._ik_min_motion(self.ready_pos, DOWN)
        if not r.ok:
            raise IKError("ready pose unreachable")
        self._run(joint_move(self._q(), r.q, self.scene.cfg.motion))

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

        phase("retreat", retreat)

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
