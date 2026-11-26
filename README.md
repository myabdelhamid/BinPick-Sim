# BinPick-Sim

A simulated robot cell: a Franka Panda with a 44 mm suction cup empties a cluttered bin of 20 mixed parts (boxes and cylinders, up
to 1.75 kg) into a 5 × 4 tray. It works only from what a real cell senses (a noisy depth camera, a vacuum sensor and its own joints),
and a one-command benchmark measures it against a random baseline on the same bins.

**Results** (25 bins × 20 parts per run): [docs/RESULTS.md](docs/RESULTS.md) · dashboard: [docs/dashboard.html](docs/dashboard.html) ·
2-minute video: [docs/demo.mp4](docs/demo.mp4)

| Rules planner | Success per attempt (target 95%) | Bins emptied (target 95%) | Mean cycle (target 8 s) | Random baseline |
|---|---|---|---|---|
| **Unused seeds 1240–1264**, blind | 94.5% | 20 / 25 | 7.38 s | 3.4% |
| Seeds 1–25, final code (rerun after fixes) | 95.6% | 24 / 25 | 7.40 s | 3.2% |
| Seeds 1–25, first run, blind | 95.4% | 22 / 25 | 8.09 s | 3.6% |

The unused seeds are the fairest estimate. In the final runs, every bin not emptied has one cylinder too short for the 44 mm cup to
grip on its side; in the first run two heavy parts also fell out of the cell.
It is a simulation: what is assumed is listed in [RESULTS.md](docs/RESULTS.md#what-is-real-and-what-is-assumed).

## Setup

Needs [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned (Open3D has no 3.13 wheels for Apple Silicon).

```bash
uv sync
```

## Run the benchmark

```bash
uv run binpick-bench
```

Runs the rules planner and the random baseline on the same 25 bins (seeds 1–25, 500 parts per planner), headless and in parallel, in about
30 minutes on an M-series Mac (keep the lid open: closed, the Mac overheats and stalls). Other seeds: `--seeds 1240-1264 --out
outputs/unused`. Writes `outputs/benchmark/`:

| File | What it is |
|---|---|
| `report.md` | the results, both planners side by side, failure causes and what to know before using them |
| `attempts.csv` | every attempt: outcome, reason, cycle time, time per phase, felt weight |
| `bins.csv`, `parts.csv` | one row per bin, one row per part (placed, left, lost, misplaced; why) |
| `events.jsonl` | shakes and rejected targets |
| `summary.json`, `meta.json`, `config.yaml` | the numbers for machines; code version, machine and exact settings |

The same seed gives the same result whatever the number of workers. A bin that fails to run is named in the report and counts as not
emptied. Seeds 1–25 run only from committed code.

```bash
uv run binpick-dashboard                         # redraw docs/dashboard.html from the delivered runs
uv run python scripts/demo_video.py              # redraw docs/demo.mp4 (about 5 minutes)
```

## Look at single bins

```bash
uv run binpick-run --seed 1240 --planner rules --video     # empty one bin, film it: outputs/run_seed1240_rules/
uv run binpick-scene --seed 1000                           # images of a settled bin: outputs/scene_seed1000/
uv run mjpython -m binpick.view --seed 1000                # interactive viewer (macOS)
```

Use seeds 1000 and up for anything but the benchmark.

## How it works

1. **Scene** (`sim/`): 15 boxes and 5 cylinders with seeded random sizes and materials are dropped into the bin and settle.
   The cup holds a part only if its lip seals (edges, tilt and curvature break the seal) and lets go when the load (weight,
   acceleration, peel, being pulled against a neighbour) exceeds what the seal holds: 39 N, from the Piab B40 datasheet.
2. **Perception** (`perception/`): the depth image (with noise and missing pixels) becomes a point cloud and a 1 mm height map,
   split into smooth surfaces.
3. **Planning** (`planning/rules.py`): on every surface, where would the cup seal? Rejects tilt over 20°, edges, walls and
   obstacles; picks free, centered, level spots first; shakes the bin when none is left. `planning/random_pick.py` is the baseline.
4. **Motion** (`motion/`): inverse kinematics on the arm's own model, straight-line and joint moves within the Panda's joint
   limits, a collision check against the cell, a guarded approach that stops on unexpected contact.
5. **Supervisor** (`supervisor/`): capture → plan → approach → grip → lift → transport → place → verify, or recover and retry.
   It feels the part's weight and carries heavy parts slower, checks the tray with a second camera, and logs every attempt.
6. **Benchmark** (`benchmark/`): one process per bin, CSVs, report, dashboard.

All settings are in [`configs/default.yaml`](configs/default.yaml). The design and every decision are in
[`docs/superpowers/specs/`](docs/superpowers/specs/) and the per-milestone plans; evidence for each milestone is in
[`docs/checkpoints/`](docs/checkpoints/).

## Tests

```bash
uv run pytest -q
```

About 330 tests, about 10 minutes. CI runs them headless on Ubuntu (`MUJOCO_GL=egl`), including a two-bin benchmark.

## Credits

The Franka Panda model is vendored from MuJoCo Menagerie (Apache-2.0, see [`assets/franka_emika_panda/VENDORED.md`](assets/franka_emika_panda/VENDORED.md)).
