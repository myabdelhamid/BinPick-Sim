# BinPick-Sim

Simulated robot cell: a Franka Panda with a 44 mm suction cup empties a cluttered bin of 20 mixed parts into a 5×4 tray grid, measured by a reproducible benchmark.

Status: **milestone 5** (benchmark). The arm picks what its noisy depth camera sees, places parts in tray slots, retries and shakes the bin; one command measures it.

## Setup

Needs [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned (Open3D has no 3.13 wheels for Apple Silicon).

```bash
uv sync
```

## Build a scene and save images

```bash
uv run binpick-scene --seed 1000
```

Writes `outputs/scene_seed1000/`:

| File | What it is |
|---|---|
| `rgb.png` | top-down bin camera, 640×480 |
| `depth.npy` | metric depth (float32, meters) from the same camera |
| `depth_mm.png` | the same depth as 16-bit millimeters |
| `depth_preview.png` | colorized depth for viewing |
| `tray_rgb.png` | top-down tray camera (used for placement checks later) |
| `overview.png` | 3/4 view of the whole cell |
| `parts.json` | every part: shape, material, size, mass, settled pose, plus bin bounds, slot centers and camera intrinsics |

Seeds 1–25 are reserved for the benchmark. Use seeds ≥ 1000 for development.

## Pick-and-place demo (known part poses)

```bash
uv run binpick-demo --seed 1000 --picks 3 --video
```

Writes `outputs/demo_seed1000/attempts.csv` (one row per attempt: outcome, failure reason, cycle time, time per phase), `events.json` (planning rejections), and `demo.mp4`.

How suction works: the cup seals only if all 16 lip points land on the part and the surface height under the lip varies by less than the bellows can absorb (edges, tilt and curvature break the seal). A sealed cup holds the part; every physics step the grip's real load (tension, peel moment, shear) is compared with what the seal can hold (39 N, the Piab B40 datasheet pull at 60 kPa). Overload breaks the seal and the part falls. The robot only sees a noisy vacuum sensor and the cameras.

## Benchmark

```bash
uv run binpick-bench
```

Runs the rules planner and the random baseline on the same 25 bins (seeds 1–25, 500 parts each), headless and in parallel (about 30 minutes on an M-series Mac; keep the lid open). Writes `outputs/benchmark/`:

| File | What it is |
|---|---|
| `report.md` | the results, side by side, with failure causes and what to know before using them |
| `attempts.csv` | every attempt: outcome, reason, cycle time, time per phase |
| `bins.csv`, `parts.csv` | one row per bin, one row per part |
| `events.jsonl` | shakes and rejected targets |
| `summary.json`, `meta.json`, `config.yaml` | the numbers for machines; code version, machine and settings |

The same seed gives the same result whatever the number of workers. A bin that fails to run is named in the report and counts as not emptied.

## Empty one bin and film it

```bash
uv run binpick-run --seed 1220 --planner rules --video
```

## Look at it interactively (macOS)

```bash
uv run mjpython -m binpick.view --seed 1000
```

## Tests

```bash
uv run pytest -q
```

CI runs the same tests headless on Ubuntu (`MUJOCO_GL=egl`).

## Layout

- `configs/default.yaml`: every setting (part counts and ranges, materials, cup, bin, tray, cameras, shaker, settling, layout)
- `src/binpick/sim/`: parts sampler, scene builder (MjSpec), drop-and-settle, cameras
- `src/binpick/motion/ik.py`: cup-pointing-down IK (mink)
- `assets/franka_emika_panda/`: vendored MuJoCo Menagerie Panda (Apache-2.0, see `VENDORED.md`)
- `docs/superpowers/specs/`: design; `docs/checkpoints/`: milestone evidence
