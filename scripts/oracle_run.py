"""M2 evidence run: known-pose oracle, top target, N picks per seed.
Usage: uv run python scripts/oracle_run.py FIRST_SEED END_SEED PICKS   (e.g. 1000 1010 3)"""
from collections import Counter
import numpy as np, sys
from binpick.config import load_config
from binpick.sim.settle import make_settled_scene
from binpick.planning.oracle import oracle_targets
from binpick.supervisor.machine import Supervisor
cfg = load_config(); res = []
seeds = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1000, 1001, 1002, 1003)
picks = int(sys.argv[3]) if len(sys.argv) > 3 else 3
for seed in seeds:
    s, _ = make_settled_scene(cfg, seed)
    sup = Supervisor(s, np.random.default_rng(0), seed=seed)
    for _ in range(picks):
        ts = oracle_targets(s)
        if not ts: break
        r = sup.attempt(ts[0])
        if r:
            res.append(r)
            if r.outcome != "success": print(seed, r.attempt, r.reason, "(sensed:", r.sensed + ")", s.parts[r.target_part].shape, np.round(s.parts[r.target_part].dims, 3), round(s.parts[r.target_part].mass, 2))
ct = [r.cycle_time_s for r in res if r.outcome == "success"]
print("attempts", len(res), "success", len(ct), "mean", round(np.mean(ct), 2), "max", round(max(ct), 2))
print("phase means", {k: round(np.mean([r.phase_times.get(k, 0) for r in res if r.outcome == "success"]), 2) for k in res[0].phase_times})
print("failure reasons", dict(Counter(r.reason for r in res if r.outcome != "success")))
