"""One bin, start to finish: attempt after attempt, shaking when nothing can be picked."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from binpick.perception.view import observe_bin
from binpick.planning.random_pick import random_targets
from binpick.planning.rules import rules_targets
from binpick.sim.scene import Scene
from binpick.supervisor.machine import Supervisor
from binpick.supervisor.placement import referee_in_slot
from binpick.supervisor.records import AttemptRecord

PLANNERS = ("rules", "random")


@dataclass
class BinResult:
    seed: int
    planner: str
    stop: str                 # empty | tray_full | no_candidate | attempt_limit
    emptied: bool             # ground truth: every part is in its slot
    placed: int               # ground truth, at the end of the run
    left_in_bin: int
    lost: int
    misplaced: int
    shakes: int
    sim_time_s: float
    records: list[AttemptRecord] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        ok = [r for r in self.records if r.outcome == "success"]
        reasons: dict[str, int] = {}
        for r in self.records:
            if r.outcome != "success":
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        ct = [r.cycle_time_s for r in ok]
        return {"seed": self.seed, "planner": self.planner, "stop": self.stop, "emptied": self.emptied,
                "placed": self.placed, "left_in_bin": self.left_in_bin, "lost": self.lost, "misplaced": self.misplaced,
                "shakes": self.shakes, "attempts": len(self.records), "successes": len(ok), "fail_reasons": reasons,
                "cycle_mean_s": float(np.mean(ct)) if ct else None, "cycle_max_s": float(np.max(ct)) if ct else None,
                "sim_time_s": self.sim_time_s}


def run_bin(scene: Scene, planner: str, seed: int, observer: Callable[[Scene, str, Supervisor], None] | None = None,
            max_attempts: int | None = None) -> BinResult:
    if planner not in PLANNERS:
        raise ValueError(f"planner: expected one of {PLANNERS}, got {planner!r}")
    cfg = scene.cfg
    limit = cfg.run.max_attempts if max_attempts is None else max_attempts
    sup = Supervisor(scene, np.random.default_rng([seed, 2]), seed=seed)
    if observer is not None:
        sup.observer = lambda s, phase: observer(s, phase, sup)
    pick_rng, shake_rng = np.random.default_rng([seed, 3]), np.random.default_rng([seed, 4])
    seen = {"empty": False}

    def propose():
        view = observe_bin(scene)
        seen["empty"] = view.is_empty()
        if seen["empty"]:
            return []
        return rules_targets(view, cfg) if planner == "rules" else random_targets(view, cfg, pick_rng)

    t0, shakes, in_a_row = scene.data.time, 0, 0
    while True:
        if len(sup.records) >= limit:
            stop = "attempt_limit"
            break
        if sup.slots.peek() is None:
            stop = "tray_full"
            break
        if sup.attempt(propose) is not None:
            in_a_row = 0
            continue
        if seen["empty"]:
            stop = "empty"
            break
        if in_a_row == cfg.run.max_shakes:
            stop = "no_candidate"
            break
        sup.shake(shake_rng)
        shakes, in_a_row = shakes + 1, in_a_row + 1

    # ---- scoring (ground truth)
    mujoco.mj_forward(scene.model, scene.data)
    slot_of = {r.gripped_part: r.slot for r in sup.records if r.outcome == "success"}
    in_slot = {i for i, slot in slot_of.items() if referee_in_slot(scene, i, slot)}
    lo, hi = scene.bin_inner_bounds()
    n = len(scene.parts)
    pos = [scene.data.xpos[b] for b in scene.part_body_ids]
    in_bin = {i for i in range(n) if i not in sup.lost and np.all(pos[i][:2] > lo[:2]) and np.all(pos[i][:2] < hi[:2])}
    return BinResult(seed=seed, planner=planner, stop=stop, emptied=len(in_slot) == n, placed=len(in_slot),
                     left_in_bin=len(in_bin), lost=len(sup.lost),
                     misplaced=n - len(in_slot) - len(in_bin) - len(sup.lost), shakes=shakes,
                     sim_time_s=float(scene.data.time - t0), records=sup.records, events=sup.events)
