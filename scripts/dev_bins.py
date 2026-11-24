"""Run whole dev bins (seeds >= 1000) in parallel and print every result.
Usage: uv run python scripts/dev_bins.py FIRST_SEED END_SEED PLANNER [OUT_DIR] [WORKERS]"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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


if __name__ == "__main__":
    a, b, planner = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    assert a >= 1000, "seeds 1-25 are reserved for the benchmark"
    out = sys.argv[4] if len(sys.argv) > 4 else "outputs/dev"
    rows = []
    with ProcessPoolExecutor(int(sys.argv[5]) if len(sys.argv) > 5 else 6) as ex:
        for s in ex.map(one, [(k, planner, out) for k in range(a, b)]):
            rows.append(s)
            print(s["seed"], f"placed {s['placed']}/20 attempts {s['attempts']} shakes {s['shakes']} stop {s['stop']}",
                  f"left {s['left_in_bin']} lost {s['lost']} misplaced {s['misplaced']}", s["fail_reasons"], flush=True)
    n, ok = sum(s["attempts"] for s in rows), sum(s["successes"] for s in rows)
    ct = [s["cycle_mean_s"] * s["successes"] for s in rows if s["successes"]]
    reasons = Counter()
    for s in rows:
        reasons.update(s["fail_reasons"])
    print(f"{planner}: success {ok}/{n} = {100 * ok / max(n, 1):.1f}%  emptied {sum(s['emptied'] for s in rows)}/{len(rows)}"
          f"  placed {sum(s['placed'] for s in rows)}/{20 * len(rows)}  mean cycle {sum(ct) / max(ok, 1):.2f} s"
          f"  max {max((s['cycle_max_s'] or 0) for s in rows):.2f} s  failures {dict(reasons)}")
