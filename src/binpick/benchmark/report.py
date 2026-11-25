"""Turn a benchmark folder into report.md and summary.json. Reads only the files in that folder."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

TARGETS = {"success": 0.95, "emptying": 0.95, "cycle": 8.0}
RESAMPLES = 10_000
Z = 1.96
REASONS = {
    "no_seal": "no seal formed", "seal_lost_grip": "seal lost before the grip was confirmed", "seal_lost_lift": "lost in lift (load)",
    "seal_lost_transport": "lost in transport", "collision": "approach stopped on unexpected contact, or part torn off by something", "misplaced": "not within ±10 mm of its slot",
    "ik_fail": "arm could not follow the path", "tray_full": "no free slot"}

DISCLOSURES = """\
These notes describe the default settings.

- **Part weight limit 1.75 kg.** Chosen by the owner after seeing heavier parts drop, and justified afterwards: Schmalz safety factors 1.5 and 2.0 give 2.1 and 1.6 kg for a straight lift at 2.5 m/s² (so 1.75 kg is above the stricter one), and sideways transport lowers both. It redraws about 4% of parts compared with 2.5 kg. There were no manufacturer part weights available.
- **The cup** is modelled on the Piab B40.20 datasheet (39 N at 60 kPa, 20 mm minimum curve radius). The sheet gives no tilt data and no force at the curve limit; those two are assumptions. Carrying limits with a centered grip, measured 2026-11-24 before the lift was improved: flat 100 mm box 1.75 kg, 60 mm tall box 1.75 kg, 60 mm cylinder by its end 1.5 kg, cylinder sides 1.25 kg. Re-measured on the current robot: 60 mm end 1.75 kg, cylinder sides 1.5 kg (60 and 80 mm fail at 1.75 kg).
- **Frozen test.** Parts, cup and contact physics have not changed since commit 6bba2f3, except three fixes so that bins whose parts creep or jitter still start (b5b16fb, 7eca2ad, 8d0f978); bins that started before start the same.
- **Cycle time is partly assumed.** Photo 0.3 s, vacuum build-up 0.15 s, release 0.25 s and tray check 0.2 s are assumed. All planning is charged 0.5 s; on the reference Mac it takes 0.41 s on average and up to 0.6 s. (Milestone 5 charged 0.2 s; that code needed about 1.6 s with motion planning.) Together 1.4 s of every cycle are fixed.
- **Small parts.** The cup's 44 mm lip does not fit on a cylinder side, or a box face, narrower than about 43–44 mm. Parts were frozen before this was known and were not changed; such parts count as not placed.
- **Camera noise:** 0.5 mm per pixel at 0.65 m (growing with distance²), 1% missing pixels, half the pixels missing at edges. Independent per pixel, so milder than correlated noise of the same size. No calibration error.
- **What the robot senses:** two depth cameras, a noisy vacuum sensor, and its own joint angles and torques (noise-free, arm gravity perfectly compensated). It knows the range of part sizes. Positions and weights of parts are never given to it.
- **Carrying:** the arm feels the part's weight at the end of the lift and sets its acceleration from that and the planner's seal estimate. The random baseline has no seal estimate and counts every seal as perfect.
- **Joint limits:** speeds and accelerations stay within the Panda datasheet. Milestone 5's moves exceeded a joint's speed limit by up to 15%, so its 8.09 s was a little too short.
- **Order of picks:** safe picks (free, centered, level) come before the owner's "flat before curved"; with no safe pick the bin is shaken first, up to three times. Shakes are not part of any pick's cycle time; they are in the sim time per placed part.
- **Friction:** parts rub on bin and tray with the simulator default 1.0; part on part 0.5–0.6.
- **Not attempts, for both planners:** targets the arm cannot reach, or whose approach would hit the cell. They are in events.jsonl.
- **Seeds.** Development used seeds 1000 and up. Seeds 1–25 were first run on 2026-11-25 for milestone 5 (rules 95.4%, 22 of 25 bins emptied, 8.09 s); any later run of them is a rerun after fixes made with those results in hand, and no longer a blind test."""


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _why_left(p: dict, stop: dict) -> str:
    if p["state"] == "not run":
        return "bin did not run"
    if p["state"] == "lost":
        return "fell out of the cell"
    if p["state"] == "misplaced":
        return "lies in the tray, not centered in a slot of its own"
    tried = "tried" if int(p["targeted"]) else f"never tried ({stop[p['planner'], p['seed']]})"
    return {"never": "no side the 44 mm lip fits on, ", "end": "fits the lip only standing on its end, "}.get(p["grippable"], "") + tried


def _wilson(k: int, n: int) -> tuple[float, float]:
    c, h = (k + Z**2 / 2) / (n + Z**2), Z * np.sqrt(k * (n - k) / n + Z**2 / 4) / (n + Z**2)
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


def _success_range(per_bin: list[tuple[int, int]]) -> tuple[float, float] | None:
    """Approximate 95% range of the success rate if other bins of the same kind had been drawn (bins resampled, not
    attempts: attempts in one bin are related)."""
    k = np.array([b for b in per_bin if b[1] > 0])
    if len(k) < 2:
        return None
    pick = np.random.default_rng(0).integers(0, len(k), (RESAMPLES, len(k)))
    rates = k[pick, 0].sum(axis=1) / k[pick, 1].sum(axis=1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def _planner(name: str, meta: dict, attempts: list[dict], bins: list[dict], parts: list[dict]) -> dict:
    A = [a for a in attempts if a["planner"] == name]
    B = [b for b in bins if b["planner"] == name]
    P = [p for p in parts if p["planner"] == name]
    if sorted(int(b["seed"]) for b in B) != sorted(meta["seeds"]):
        raise ValueError(f"{name}: bins.csv does not hold exactly the seeds of meta.json")
    for b in B:  # the files must tell one story
        mine = [a for a in A if a["seed"] == b["seed"]]
        ok = sum(a["outcome"] == "success" for a in mine)
        placed = sum(p["state"] == "placed" for p in P if p["seed"] == b["seed"])
        if (len(mine), ok, placed) != (int(b["attempts"]), int(b["successes"]), int(b["placed"])):
            raise ValueError(f"{name} seed {b['seed']}: attempts.csv, bins.csv and parts.csv disagree")
    good = [a for a in A if a["outcome"] == "success"]
    cyc, cyc_all = [float(a["cycle_time_s"]) for a in good], [float(a["cycle_time_s"]) for a in A]
    total = lambda key: sum(int(b[key]) for b in B)  # noqa: E731
    placed, sim = total("placed"), sum(float(b["sim_time_s"]) for b in B)
    state = {(p["seed"], p["part"]): p["state"] for p in P}
    stop = {(b["planner"], b["seed"]): b["stop"] for b in B}
    by_bin = [(int(b["successes"]), int(b["attempts"])) for b in B]
    return {
        "attempts": len(A), "successes": len(good), "success_rate": len(good) / len(A) if A else None, "success_range": _success_range(by_bin),
        "bins": len(B), "emptied": total("emptied"), "emptying_rate": total("emptied") / len(B), "parts": len(P), "placed": placed,
        "emptying_range": _wilson(total("emptied"), len(B)),
        "knocked_out": sum(state[a["seed"], str(a["gripped_part"])] != "placed" for a in good),
        "dropped_into_slot": sum(a["outcome"] != "success" and a["referee_ok"] == "1" and float(a["arm_hit_n"]) == 0 for a in A), "left": total("left"), "lost": total("lost"), "misplaced": total("misplaced"),
        "cycle_mean_s": float(np.mean(cyc)) if cyc else None, "cycle_median_s": float(np.median(cyc)) if cyc else None,
        "cycle_max_s": max(cyc, default=None), "over_8s": sum(c > TARGETS["cycle"] for c in cyc),
        "cycle_all_mean_s": float(np.mean(cyc_all)) if cyc_all else None, "cycle_all_max_s": max(cyc_all, default=None),
        "sim_time_per_placed_s": sim / placed if placed else None, "shakes": total("shakes"), "hard_shakes": total("hard_shakes"),
        "failures": dict(Counter(a["reason"] for a in A if a["outcome"] != "success").most_common()),
        "stops": dict(Counter(b["stop"] for b in B).most_common()), "errors": [int(b["seed"]) for b in B if b["error"]],
        "risky_attempts": sum(a["safe"] == "0" for a in A), "verdict_wrong": sum(a["detected"] == "0" for a in A),
        "arm_hits": sum(float(a["arm_hit_n"]) > 0 for a in A),
        "left_over": dict(Counter(_why_left(p, stop) for p in P if p["state"] != "placed").most_common()),
        "ungrippable_bins": len({p["seed"] for p in P if p["grippable"] == "never"}),
    }


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _s(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f} s"


def _verdict(met: bool | None) -> str:
    return "–" if met is None else "met" if met else "**missed**"


def make_report(folder: str | Path) -> tuple[str, dict]:
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    attempts, bins, parts = (_read(folder / f"{n}.csv") for n in ("attempts", "bins", "parts"))
    S = {p: _planner(p, meta, attempts, bins, parts) for p in meta["planners"]}
    names, main = meta["planners"], S[meta["planners"][0]]
    row = lambda label, f: f"| {label} | " + " | ".join(f(S[p]) for p in names) + " |"  # noqa: E731
    head = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    seeds = meta["seeds"]
    span = f"{seeds[0]}–{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) and len(seeds) > 1 else ", ".join(map(str, seeds))
    L = [f"# Benchmark report: seeds {span}", ""]
    if meta["max_attempts"]:
        L += [f"**Not a full benchmark:** every bin was stopped after {meta['max_attempts']} attempts.", ""]
    if not meta["default_config"]:
        L += ["**Not the default settings:** see config.yaml in this folder. The notes at the end describe the default settings.", ""]
    if meta["dirty"]:
        L += ["**Run from uncommitted code:** these numbers cannot be reproduced from the commit below.", ""]
    L += [f"{main['bins']} bins × {main['parts'] // main['bins']} parts per planner. Both planners get the same bins.", "",
          f"## Targets ({names[0]} planner)", "", "| Metric | Target | Result | |", "|---|---|---|---|",
          f"| Pick success per attempt | ≥ 95% | {_pct(main['success_rate'])} | {_verdict(None if main['success_rate'] is None else main['success_rate'] >= TARGETS['success'])} |",
          f"| Mean cycle time per successful pick | ≤ 8 s | {_s(main['cycle_mean_s'])} | {_verdict(None if main['cycle_mean_s'] is None else main['cycle_mean_s'] <= TARGETS['cycle'])} |",
          f"| Bin-emptying rate | ≥ 95% | {_pct(main['emptying_rate'])} | {_verdict(main['emptying_rate'] >= TARGETS['emptying'])} |", "",
          "## Side by side", "", *head,
          row("Successful picks ÷ attempts", lambda r: f"{r['successes']} / {r['attempts']}"),
          row("**Success per attempt**", lambda r: f"**{_pct(r['success_rate'])}**"),
          row("… approximate 95% range (resampling bins)", lambda r: "–" if r["success_range"] is None else f"{_pct(r['success_range'][0])} to {_pct(r['success_range'][1])}"),
          row("**Bins emptied**", lambda r: f"**{r['emptied']} / {r['bins']} ({_pct(r['emptying_rate'])})**"),
          row("… 95% range", lambda r: f"{_pct(r['emptying_range'][0])} to {_pct(r['emptying_range'][1])}"),
          row("Parts placed", lambda r: f"{r['placed']} / {r['parts']} ({_pct(r['placed'] / r['parts'])})"),
          row("Parts left in bin / lost / misplaced", lambda r: f"{r['left']} / {r['lost']} / {r['misplaced']}"),
          row("Successful picks later knocked out of their slot", lambda r: str(r["knocked_out"])),
          row("Failed attempts whose part fell into its slot anyway", lambda r: str(r["dropped_into_slot"])),
          row("Cycle time, successful picks: mean", lambda r: _s(r["cycle_mean_s"])),
          row("… median", lambda r: _s(r["cycle_median_s"])),
          row("… slowest", lambda r: _s(r["cycle_max_s"])),
          row("… over 8 s", lambda r: f"{r['over_8s']} / {r['successes']}"),
          row("Cycle time, all attempts: mean / slowest", lambda r: f"{_s(r['cycle_all_mean_s'])} / {_s(r['cycle_all_max_s'])}"),
          row("Sim time per placed part, with shakes and failures", lambda r: _s(r["sim_time_per_placed_s"])),
          row("Shakes (of those hard)", lambda r: f"{r['shakes']} ({r['hard_shakes']})"),
          row("Robot's own verdict wrong", lambda r: f"{r['verdict_wrong']} / {r['attempts']} attempts"),
          row("Arm or tool hit something (> 50 N)", lambda r: f"{r['arm_hits']} attempts"),
          row("Bins that failed to run", lambda r: str(len(r["errors"]))), "",
          "Cycle time runs from the start of the bin photo to the end of the tray check and nothing overlaps (the owner's strict definition); "
          "taking photos while the arm moves would shorten it. Part of it is assumed, see the notes below. An attempt is a success only if "
          "the robot set the part down, let go, and the part rests within ±10 mm of its slot's center.", "",
          "## Failed attempts", "", "| Cause | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    causes = [c for c, _ in Counter({c: sum(S[p]["failures"].get(c, 0) for p in names) for p in names for c in S[p]["failures"]}).most_common()]
    L += [row(f"{REASONS.get(c, c)} (`{c}`)", lambda r, c=c: str(r["failures"].get(c, 0))) for c in causes]
    L += ["", "## Parts not placed, and why", "", *head]
    whys = [c for c, _ in Counter({c: sum(S[p]["left_over"].get(c, 0) for p in names) for p in names for c in S[p]["left_over"]}).most_common()]
    L += [row(c, lambda r, c=c: str(r["left_over"].get(c, 0))) for c in whys]
    stop = {(b["planner"], b["seed"]): b["stop"] for b in bins}
    L += ["", f"{main['ungrippable_bins']} of {main['bins']} bins hold a part with no side the cup's lip fits on (geometry only).", "",
          f"Parts the {names[0]} planner did not place:", "", "| Seed | Part | Shape, mm | kg | End state | Why |", "|---|---|---|---|---|---|"]
    L += [f"| {p['seed']} | {p['part']} | {p['shape']} {p['dims_mm']} | {float(p['mass_kg']):.2f} | {p['state']} | {_why_left(p, stop)} |"
          for p in parts if p["planner"] == names[0] and p["state"] != "placed"]
    L += ["", "## Every bin", "", "| Planner | Seed | Placed | Attempts | Shakes | Ended | Left / lost / misplaced |", "|---|---|---|---|---|---|---|"]
    L += [f"| {b['planner']} | {b['seed']} | {b['placed']}/{b['parts']} | {b['attempts']} | {b['shakes']} | "
          f"{'ERROR' if b['error'] else b['stop']} | {b['left']} / {b['lost']} / {b['misplaced']} |" for b in bins]
    errors = [b for b in bins if b["error"]]
    if errors:
        L += ["", "### Bins that failed to run", "", "They count as not emptied and all their parts as not placed. Attempts made before the failure are not in the success rate, ranges, "
              "cycle times or failure table."]
        L += [f"- {b['planner']} seed {b['seed']}: ERROR {b['error']}" for b in errors]
    L += ["", "## What to know before using these numbers", "", DISCLOSURES, "", "## Reproduce", "", "```bash", meta["command"] or "uv run binpick-bench", "```", "",
          f"- Code: commit `{meta['commit']}`{' with uncommitted changes' if meta['dirty'] else ''}. Config SHA-256 `{meta['config_sha256'][:16]}`.",
          f"- Machine: {meta['platform']}; " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()) + ".",
          f"- Started {meta['started']}, {meta['wall_s'] / 60:.0f} minutes with {meta['workers']} workers. Results do not depend on the number of workers; "
          "another machine may differ slightly.", ""]
    return "\n".join(L), S


def write_report(folder: str | Path) -> dict:
    text, summary = make_report(folder)
    (Path(folder) / "report.md").write_text(text)
    (Path(folder) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
