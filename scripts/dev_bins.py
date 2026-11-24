"""Run whole dev bins (seeds >= 1000) in parallel and print every result.
Usage: uv run python scripts/dev_bins.py FIRST_SEED END_SEED PLANNER [OUT_DIR] [WORKERS]"""
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def one(a):
    seed, planner, out = a
    from binpick.config import load_config
    from binpick.sim.settle import make_settled_scene
    from binpick.supervisor.records import write_csv
    from binpick.supervisor.run import run_bin

    scene, _ = make_settled_scene(load_config(), seed)
    res = run_bin(scene, planner, seed)
    d = Path(out) / f"{planner}_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    write_csv(res.records, d / "attempts.csv")
    (d / "events.json").write_text(json.dumps(res.events, default=float))
    s = res.summary()
    (d / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def in_own_process(k, planner, out):
    """One bin per process: a hard crash (or a kill) loses that bin only, and its exit code is reported."""
    summary = Path(out) / f"{planner}_{k}" / "summary.json"
    summary.unlink(missing_ok=True)
    p = subprocess.run([sys.executable, __file__, "--one", str(k), planner, out], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"exit code {p.returncode}: {p.stderr.strip()[-300:]}")
    return json.loads(summary.read_text())


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one((int(sys.argv[2]), sys.argv[3], sys.argv[4]))
        sys.exit(0)
    a, b, planner = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    assert a >= 1000, "seeds 1-25 are reserved for the benchmark"
    out = sys.argv[4] if len(sys.argv) > 4 else "outputs/dev"
    rows, errors = [], []
    with ThreadPoolExecutor(int(sys.argv[5]) if len(sys.argv) > 5 else 6) as ex:
        jobs = {ex.submit(in_own_process, k, planner, out): k for k in range(a, b)}
        for job in as_completed(jobs):  # one failing bin must not lose the others; it is reported, never dropped
            if job.exception() is not None:
                errors.append(jobs[job])
                print(jobs[job], "ERROR", repr(job.exception()), flush=True)
                continue
            s = job.result()
            rows.append(s)
            print(s["seed"], f"placed {s['placed']}/20 attempts {s['attempts']} shakes {s['shakes']} stop {s['stop']}",
                  f"left {s['left_in_bin']} lost {s['lost']} misplaced {s['misplaced']}", s["fail_reasons"], flush=True)
    n, ok = sum(s["attempts"] for s in rows), sum(s["successes"] for s in rows)
    mean = lambda key, w: sum(s[key] * s[w] for s in rows if s[w] and s[key] is not None) / max(sum(s[w] for s in rows), 1)  # noqa: E731
    reasons = Counter()
    for s in rows:
        reasons.update(s["fail_reasons"])
    placed = sum(s["placed"] for s in rows)
    print(f"{planner}: success {ok}/{n} = {100 * ok / max(n, 1):.1f}%  emptied {sum(s['emptied'] for s in rows)}/{b - a}"
          f"  placed {placed}/{20 * (b - a)}  errors {errors}  failures {dict(reasons)}")
    print(f"cycle: successful picks mean {mean('cycle_mean_s', 'successes'):.2f} s max {max([s['cycle_max_s'] or 0 for s in rows], default=0):.2f} s,"
          f" over 8 s {sum(s['over_8s'] for s in rows)}/{ok}; all attempts mean {mean('cycle_mean_all_s', 'attempts'):.2f} s;"
          f" sim time per placed part incl. shakes {sum(s['sim_time_s'] for s in rows) / max(placed, 1):.1f} s")
