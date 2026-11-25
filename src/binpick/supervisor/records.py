"""Attempt records and CSV logging."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

PHASES = ("capture", "plan", "approach", "grip", "lift", "transport", "place", "release", "retreat", "verify")
REASONS = ("no_seal", "seal_lost_grip", "seal_lost_lift", "seal_lost_transport", "misplaced", "ik_fail", "collision", "tray_full",
           "no_candidate")


@dataclass
class AttemptRecord:
    seed: int
    attempt: int
    target_part: int | None
    gripped_part: int | None = None      # ground truth: what the cup actually held (may differ from target)
    slot: int | None = None
    released: bool = False               # the robot itself set the part down and let go of it
    outcome: str = "fail"                # "success" only if released and the referee finds the part in its slot
    reason: str = ""                     # root cause (one of REASONS) when outcome == "fail"
    sensed: str = ""                     # what the supervisor's own sensors concluded (may differ from reason)
    detected: bool = True                # supervisor's own sensors agreed with the outcome
    verify_ok: bool = False              # supervisor's tray-camera verdict
    referee_ok: bool = False             # ground-truth verdict (scoring)
    lost_parts: list[int] = field(default_factory=list)
    arm_hit_n: float = 0.0               # hardest contact of arm/tool with the cell (ground truth), N; 0 = none
    cycle_time_s: float = 0.0
    tilt_deg: float = 0.0
    est_seal: float = 1.0                # planner's seal estimate
    safe: bool = True                    # planner: free, centered and level
    crowd: float = 0.0                   # planner: share of the outline with something taller beside it
    felt_kg: float = 0.0                 # weight the arm felt at the end of the lift
    phase_times: dict[str, float] = field(default_factory=dict)


def write_csv(records: list[AttemptRecord], path: str | Path) -> None:
    cols = ["seed", "attempt", "target_part", "gripped_part", "slot", "outcome", "reason", "sensed", "detected",
            "verify_ok", "referee_ok", "lost_parts", "cycle_time_s", "tilt_deg", "est_seal", "arm_hit_n", "safe", "crowd", "felt_kg", "released"] + [f"t_{p}" for p in PHASES]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([r.seed, r.attempt, "" if r.target_part is None else r.target_part, "" if r.gripped_part is None else r.gripped_part,
                        "" if r.slot is None else r.slot, r.outcome, r.reason, r.sensed, int(r.detected),
                        int(r.verify_ok),
                        int(r.referee_ok), ";".join(map(str, r.lost_parts)), f"{r.cycle_time_s:.3f}",
                        f"{r.tilt_deg:.1f}", f"{r.est_seal:.3f}", f"{r.arm_hit_n:.0f}", int(r.safe), f"{r.crowd:.3f}", f"{r.felt_kg:.2f}", int(r.released)]
                       + [f"{r.phase_times.get(p, 0.0):.3f}" for p in PHASES])
