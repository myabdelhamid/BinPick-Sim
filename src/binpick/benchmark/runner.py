"""Run many bins, one process each, and gather what they logged into CSV files.

Every bin is a single-threaded process seeded only by its own seed, so results do not depend on the number of workers.
A bin that crashes or hangs is reported by name and counted as not emptied; it is never dropped.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np

from binpick.config import DEFAULT_PATH, load_config
from binpick.sim.parts import PartSpec, sample_parts

RESERVED = range(1, 26)      # run only by the official benchmark, from committed code
BIN_TIMEOUT_S = 3600.0       # a bin takes a few minutes
KNOWN = {"attempts.csv", "bins.csv", "parts.csv", "events.jsonl", "meta.json", "config.yaml", "report.md", "summary.json", "bins"}
BIN_COLUMNS = ["planner", "seed", "stop", "emptied", "placed", "left", "lost", "misplaced", "shakes", "hard_shakes", "attempts", "successes",
               "parts", "sim_time_s", "wall_s", "error"]
PART_COLUMNS = ["planner", "seed", "part", "shape", "material", "dims_mm", "mass_kg", "state", "targeted", "grippable"]


def parse_seeds(text: str) -> list[int]:
    """"1-25" or "3,7-9"."""
    out: list[int] = []
    for item in text.split(","):
        a, _, b = item.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def git_state() -> tuple[str, bool]:
    """(commit, uncommitted changes?) of the code being run."""
    root = Path(__file__).resolve().parents[3]
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)  # noqa: E731
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return "unknown", True
    return head.stdout.strip(), bool(git("status", "--porcelain").stdout.strip())


def grippable(p: PartSpec, cfg) -> str:
    """Can the frozen cup seal on this part at all? "any": on a side it can lie on; "end": a cylinder only standing on its
    end; "never". (Geometry only: a centered lip must fit.)"""
    lip = cfg.cup.diameter
    if p.shape == "box":
        return "any" if sorted(p.dims)[1] >= lip else "never"
    dia, length = p.dims
    if length >= lip and dia >= 2 * cfg.suction.curve_radius_min:
        return "any"
    return "end" if dia >= lip else "never"


def bin_command(seed: int, planner: str, out: str, config: str, max_attempts: int | None) -> list[str]:
    return [sys.executable, "-m", "binpick.benchmark.runner", str(seed), planner, out, config, str(max_attempts or 0)]


def _one_bin(seed: int, planner: str, out: str, config: str, max_attempts: int | None) -> None:
    """Runs inside the bin's own process."""
    from binpick.sim.settle import make_settled_scene
    from binpick.supervisor.records import write_csv
    from binpick.supervisor.run import run_bin

    t0 = time.perf_counter()
    cfg = load_config(config)
    scene, _ = make_settled_scene(cfg, seed)
    res = run_bin(scene, planner, seed, max_attempts=max_attempts)
    base = Path(out) / "bins" / f"{planner}_{seed}"
    write_csv(res.records, f"{base}.csv")
    Path(f"{base}.json").write_text(json.dumps({**res.summary(), "part_state": res.part_state, "events": res.events,
                                                "wall_s": time.perf_counter() - t0}, default=float))


def _run_process(job: tuple) -> str:
    """Error text, or "" if the bin finished."""
    seed, planner, out, *_ = job
    env = {**os.environ, **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    try:
        p = subprocess.run(bin_command(*job), capture_output=True, text=True, env=env, timeout=BIN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"timed out after {BIN_TIMEOUT_S:.0f} s"
    if p.returncode != 0:
        return f"exit code {p.returncode}: {' '.join(p.stderr.split())[-300:]}"
    if not (Path(out) / "bins" / f"{planner}_{seed}.json").exists():
        return "finished without a result"
    return ""


def _write(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def run_benchmark(seeds: list[int], planners: list[str], out: str | Path, workers: int, config: str | Path | None = None,
                  max_attempts: int | None = None, overwrite: bool = False, command: str = "") -> list[str]:
    """Run every (planner, seed), write the CSVs, return the errors."""
    out = Path(out).resolve()
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} already exists; pass --overwrite to replace it")
        if not {p.name for p in out.iterdir()} <= KNOWN:
            raise SystemExit(f"{out} is not a benchmark folder; nothing deleted")
        shutil.rmtree(out)
    (out / "bins").mkdir(parents=True)
    shutil.copy(config or DEFAULT_PATH, out / "config.yaml")
    cfg = load_config(out / "config.yaml")
    commit, dirty = git_state()
    started, t0 = datetime.datetime.now().astimezone().isoformat(timespec="seconds"), time.perf_counter()
    jobs = [(s, p, str(out), str(out / "config.yaml"), max_attempts) for p in planners for s in seeds]

    def work(job: tuple) -> str:
        err = _run_process(job)
        print(f"{job[1]} seed {job[0]}: {'ERROR ' + err if err else 'done'}", flush=True)
        return err

    with ThreadPoolExecutor(workers) as ex:
        errs = list(ex.map(work, jobs))

    attempts, bins, parts = [], [], []
    with open(out / "events.jsonl", "w") as ev:
        for (seed, planner, *_), err in zip(jobs, errs):
            specs = sample_parts(cfg.parts, np.random.default_rng([seed, 0]))
            state, targeted = ["not run"] * len(specs), [0] * len(specs)
            row = dict.fromkeys(BIN_COLUMNS, 0) | {"planner": planner, "seed": seed, "stop": "error", "left": len(specs), "parts": len(specs),
                                                   "sim_time_s": 0.0, "wall_s": 0.0, "error": err}
            if not err:
                base = out / "bins" / f"{planner}_{seed}"
                r = json.loads(Path(f"{base}.json").read_text())
                state = r["part_state"]
                with open(f"{base}.csv", newline="") as f:
                    mine = [{"planner": planner, **a} for a in csv.DictReader(f)]
                attempts += mine
                for a in mine:
                    if a["target_part"] != "":
                        targeted[int(a["target_part"])] += 1
                for e in r["events"]:
                    ev.write(json.dumps({"planner": planner, **e}) + "\n")
                row |= {k: r[k] for k in ("stop", "placed", "lost", "misplaced", "shakes", "hard_shakes", "attempts", "successes")}
                row |= {"left": r["left_in_bin"], "emptied": int(r["emptied"]), "sim_time_s": round(r["sim_time_s"], 3), "wall_s": round(r["wall_s"], 1)}
            bins.append(row)
            parts += [{"planner": planner, "seed": seed, "part": p.id, "shape": p.shape, "material": p.material,
                       "dims_mm": "x".join(f"{1000 * x:.1f}" for x in p.dims), "mass_kg": round(p.mass, 3), "state": s, "targeted": t,
                       "grippable": grippable(p, cfg)} for p, s, t in zip(specs, state, targeted)]
    _write(out / "attempts.csv", list(attempts[0]) if attempts else ["planner", "seed", "attempt"], attempts)
    _write(out / "bins.csv", BIN_COLUMNS, bins)
    _write(out / "parts.csv", PART_COLUMNS, parts)
    (out / "meta.json").write_text(json.dumps({
        "seeds": seeds, "planners": planners, "commit": commit, "dirty": dirty, "max_attempts": max_attempts,
        "config_sha256": hashlib.sha256((out / "config.yaml").read_bytes()).hexdigest(), "workers": workers, "command": command,
        "started": started, "wall_s": round(time.perf_counter() - t0, 1), "platform": f"{platform.platform()} ({platform.machine()})",
        "versions": {"python": platform.python_version(), **{k: version(k) for k in ("mujoco", "mink", "numpy", "scipy", "open3d")}}}, indent=2))
    return [f"{p} seed {s}: {e}" for (s, p, *_), e in zip(jobs, errs) if e]


if __name__ == "__main__":
    _one_bin(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) or None)
