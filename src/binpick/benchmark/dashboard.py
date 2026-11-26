"""One self-contained HTML page that shows benchmark runs. Every number comes from the run folders (report.py does the
counting); the page only draws it."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from binpick.benchmark.report import TARGETS, make_report

PAGE = Path(__file__).with_name("dashboard.html")


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_data(folder: str | Path, label: str, note: str) -> dict:
    """What the page needs from one benchmark folder."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    _, summary = make_report(folder)
    attempts, bins, parts = (_read(folder / f"{n}.csv") for n in ("attempts", "bins", "parts"))
    seeds, lead = meta["seeds"], "rules" if "rules" in meta["planners"] else meta["planners"][0]
    span = f"{seeds[0]}–{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) and len(seeds) > 1 else ", ".join(map(str, seeds))
    keep = ("attempts", "successes", "success_rate", "success_range", "bins", "emptied", "emptying_rate", "emptying_range", "parts",
            "placed", "cycle_mean_s", "cycle_median_s", "cycle_max_s", "over_8s", "sim_time_per_placed_s", "shakes", "failures",
            "verdict_wrong", "arm_hits", "errors")
    return {
        "label": label, "note": note, "seeds": span, "commit": meta["commit"][:7], "dirty": meta["dirty"],
        "default_config": meta.get("default_config", True), "max_attempts": meta["max_attempts"],
        "planners": {p: {k: summary[p][k] for k in keep} for p in meta["planners"]},
        "bins": [{"planner": b["planner"], "seed": int(b["seed"]), "placed": int(b["placed"]), "parts": int(b["parts"]),
                  "attempts": int(b["attempts"]), "shakes": int(b["shakes"]), "emptied": b["emptied"] == "1",
                  "stop": "error" if b["error"] else b["stop"]} for b in bins],
        "cycles": {p: [round(float(a["cycle_time_s"]), 3) for a in attempts if a["planner"] == p and a["outcome"] == "success"]
                   for p in meta["planners"]},
        "left": [{"planner": p["planner"], "seed": int(p["seed"]), "part": int(p["part"]), "shape": p["shape"], "dims": p["dims_mm"],
                  "kg": float(p["mass_kg"]), "state": p["state"], "grippable": p["grippable"], "tried": int(p["targeted"]) > 0}
                 for p in parts if p["state"] != "placed" and p["planner"] == lead],
    }


def write_dashboard(runs: list[tuple[str, str, str]], out: str | Path) -> Path:
    """runs: (folder, label, note). The first run is shown first."""
    data = {"targets": TARGETS, "runs": [run_data(*r) for r in runs]}
    text = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")  # nothing in the data can end the script
    page = PAGE.read_text().replace("/*DATA*/null", text).replace("{{TITLE}}", html.escape("BinPick-Sim results"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
