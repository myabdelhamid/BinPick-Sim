# BinPick-Sim

Simulated robot cell: a Franka Panda with a 44 mm suction cup empties a cluttered bin of 20 mixed parts into a 5×4 tray grid, measured by a reproducible benchmark.

Status: **milestone 1** (scene). The cell builds and settles a seeded bin and saves camera images. Picking comes in M2–M3.

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
