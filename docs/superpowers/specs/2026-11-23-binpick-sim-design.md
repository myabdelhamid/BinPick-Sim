# BinPick-Sim — Design

Date: 2026-11-23
Status: draft for approval, owner answers from the builder check included (covers the whole project; milestone 1 in build-ready detail)

## 1. Intent

A simulated robot cell that empties a cluttered bin of mixed parts with a suction gripper, places each part in a tray slot, and proves its performance with a reproducible benchmark. It is a proof of concept for a robotics founder pitching a manufacturer. **Honest, measured results matter more than a polished demo**: every number in the final report must come from one reproducible command, with failures counted, not hidden.

Roles: the builder (Claude) writes all code, scenes, tests, benchmark and report. The owner reviews at each milestone checkpoint and decides anything that changes scope or targets.

## 2. Goals, non-goals, anti-goals

Goals (v1):
- Franka Panda (MuJoCo Menagerie) + custom 44 mm suction cup. (Originally a MuJoCo adhesion actuator; replaced in M2 by a seal-gated weld with a load check, see §7. Owner approved 2026-11-24. Since then the cup is modelled on the Piab B40.20 datasheet, §7.)
- Bin of 20 randomly dropped parts per seed: 15 boxes, 5 cylinders.
- Output tray with a 5 × 4 grid (20 slots).
- One fixed top-down RGB-D camera over the bin with sensor noise, used for picking. Plus one fixed top-down camera over the tray, used only for the verify step (owner-decided addition; see §7).
- Rule-based pick planner and a random baseline planner, swappable.
- Benchmark: seeds 1–25 × 20 parts = 500 parts per planner, headless, parallel, CSV per attempt, one command.

Non-goals: two-finger grasping, complex/transparent parts, learned grasp models, real hardware, photorealism.

Anti-goals (fail even if it "works"):
- Suction that just attaches the part regardless of grip quality.
- Cherry-picking seeds or reporting only good runs.
- A success definition looser than "part ends inside its slot ±10 mm".
- Results that can't be reproduced from a clean checkout with one command.

## 3. Targets and metric definitions

| Metric | Target | Stretch |
|---|---|---|
| Pick success per attempt | ≥ 95 % | ≥ 98 % |
| Cycle time per pick (sim time) | ≤ 8 s | ≤ 5 s |
| Bin-emptying rate | ≥ 95 % | — |

- **Attempt (owner-decided):** an attempt starts when the arm begins approaching a chosen suction point. Each retry is a new attempt. "No candidate" results, shakes and planning-time rejections (for example IK infeasible before any motion) are logged as events but are **not** attempts. Collisions, seal losses and misplacements during motion are failed attempts. The same rule applies to both planners.
- **Success rate** = successful picks ÷ all attempts.
- **Successful pick** = the part's final resting position is inside its assigned slot, within ±10 mm of the slot center in x and y, after release and settling. This is scored by the benchmark referee from sim ground truth.
- **Lost and misplaced parts (owner-decided, strict):**
  - A part dropped back into the bin stays there and can be picked again.
  - A part that ends up outside the bin and not in a slot (floor, tray rim) is a failed attempt. It is removed from the scene and recorded as *lost*.
  - A part placed outside its slot's ±10 mm is a failed attempt (`misplaced`). It stays where it is, its slot counts as used, and it is not re-picked in v1.
- **Bin-emptying rate (owner-decided, strict)** = fraction of the 25 bins where **all 20 parts end inside their slots**. Reported alongside it: per-part placement rate, and lost/misplaced/left-in-bin counts.
- **Cycle time** = sim time from start of capture to end of verify for that attempt.
- A bin run ends when it's empty, when no safe pick remains after 3 shakes, or after 40 attempts.
- **Reproducibility (builder's call):** each bin is an independent single-threaded process seeded only by its seed, so results are identical regardless of worker count. Reported numbers come from one reference machine (the owner's Mac), and its platform and versions are written into the report. Results on other platforms may differ slightly. CI runs a 2-seed smoke benchmark and checks that it completes and produces a well-formed CSV, not exact numbers.
- Missed targets are reported as the real number plus the main causes.

## 4. Environment and constraints

- macOS Apple Silicon, no NVIDIA GPU. Sim runs on CPU.
- **Python 3.12**, managed by `uv`. Python 3.13 is installed, but Open3D has no 3.13 wheels for macOS arm64, so the project pins 3.12.
- Dependencies: `mujoco` (≥ 3.2, provides `mjpython` and `MjSpec`), `mink`, `numpy`, `open3d`, `pyyaml`, `imageio`/`pillow`, `pytest`.
- The interactive viewer needs `mjpython`. Everything else, including all tests and benchmarks, runs headless with `python`.
- Menagerie is not on pip. The Panda model (`franka_emika_panda/panda_nohand.xml` + meshes + Apache-2.0 LICENSE) is vendored under `assets/`.
- CI: GitHub Actions on Ubuntu with `MUJOCO_GL=egl` (fall back to `osmesa`) for headless rendering.
- Units: SI (m, kg, s). All randomness comes from `numpy.random.default_rng(seed)`.

## 5. Approaches considered (scene construction)

1. **Programmatic scene via `MjSpec` (chosen).** Build the world in Python for each seed: load the vendored Panda, attach the cup to its flange, add the bin, tray, camera and 20 sampled parts, then compile. Clean composition, and part masses and inertias always match their sizes. Easy to unit-test.
2. Templated XML strings per seed. Works, but attaching the arm by string manipulation is brittle and harder to test.
3. One fixed XML with sizes edited at runtime. Rejected: mass, inertia and contacts no longer match the edited geometry.

The owner approved approach 1.

## 6. Part and cell specification (owner-decided numbers)

Parts (all ranges uniform, sampled per part):
- **Boxes (15):** footprint 45–110 mm on each side, height 20–70 mm.
- **Cylinders (5):** diameter 40–80 mm, length 40–110 mm.
- **Material** per part: 60 % plastic, 30 % aluminium, 10 % steel.
- **Mass**: every part weighs between about 50 g and 2.5 kg.

Densities (builder's call): plastic 1200 kg/m³, aluminium 2700 kg/m³, steel 7850 kg/m³. Mass = density × volume. **Enforcing the 2.5 kg cap (builder's call):** if a sampled part is over 2.5 kg, its dimensions are resampled with the same material until it fits. This keeps densities physical. Steel parts therefore skew small, which is realistic. Parts below 50 g are resampled the same way.

Suction cup:
- 44 mm outer diameter. Rigid cylinder collision geom, about 20 mm long, on a short tool body at the Panda flange (`attachment_site`). Tool length is the builder's call.
- Rated hold: about 3 kg. With a good seal, every part (≤ 2.5 kg) can be lifted and transported. **Only a bad grip makes a part fall.**

Bin: 400 × 300 × 200 mm inside, 10 mm walls, on a table (height is the builder's call).

**Shaking (builder's call):** the bin sits on a shaker base with two horizontal slide joints (x, y) driven by position actuators, as industrial vibratory tables do. A shake is a short seeded oscillation (amplitude and frequency in the config) followed by settling, with the robot parked clear. It is in the M1 scene so the layout and reachability tests already include it. The robot never stirs parts.

Tray: 5 × 4 grid of 120 × 120 mm slots (600 × 480 mm overall). Low dividers or painted slot lines (builder's call; dividers must not stop a correctly placed part from settling inside ±10 mm).

Layout: robot base, bin and tray positions are the builder's call. Hard constraint, checked by a test: with the cup pointing straight down, IK reaches (a) every tray slot center at place height and (b) the bin floor at each inner corner, inset by cup radius + wall-clearance margin. If the layout can't meet this, the builder may put the robot on a pedestal or move the tray. Shrinking the slots or the grid needs the owner's decision.

## 7. Suction realism principle (fixed now, detailed in M2/M4)

MuJoCo's adhesion actuator sticks to anything it touches, so on its own it is unrealistic. It also turned out not to be calibratable: it only pulls through active contacts, so its real hold was ~70% of nominal and collapsed nonlinearly (measured in M2). The M2 design is:
- When the seal model gives q > 0, a weld constraint holds the part to the cup (the vacuum).
- Every step the cup-to-part interface load (weld + lip contact forces) is measured and compared with what the seal can hold, q × 39.0 N (Piab B40.20 datasheet pull at 60 kPa; was an ideal 74 N until 2026-11-24): utilization = tension/F + bending/(F·r·0.5) + shear/(0.5·F), filtered over 20 ms. Pressing (negative tension) counts as extra clamping. Above 1 the grip breaks and cannot re-seal until the vacuum is cycled.
- A **seal model** decides how much force is available. It is computed every control step from the *ground-truth* cup-to-part geometry in sim, never from the planner's estimate.
- Seal quality `q ∈ [0, 1]` drops with: angle between the cup axis and the surface normal, curvature or non-planarity of the surface under the cup lip, and the fraction of the cup lip that is off the face (edges, small faces).
- Available hold force = `q × 39.0 N`. 39 N is 4.0 kg static, so only 1.3× the 3 kg rating. **Open owner decision:** a perfect centered grip holds 2.0 kg at current speeds but loses 2.25 and 2.5 kg in transport, so "every part is liftable" does not hold above about 2 kg.
- Curves follow the B40's 20 mm minimum curve radius. The lip wraps onto the surface; a 40 mm cylinder is the thinnest that seals. **Assumption:** `q` falls linearly to 0.75 at that radius (the sheet gives no force there). Tilt limits are unchanged from M2 because the sheet gives no tilt data. The modelled lip stays 44 mm (B40: 43 mm lip, 46 mm body).
- The part detaches when the required force exceeds what is available. Required force = weight + inertial load during lift and transport + peel from an off-center grip (moment arm from cup center to the part's center of mass).
- Failure reasons are logged from the model: `no_seal`, `seal_lost_lift`, `seal_lost_transport`, `misplaced`, `ik_fail`, `collision`, `no_candidate`.

**What the supervisor may see (owner-decided: real sensors only).** The supervisor never reads part poses from the simulator. It gets:
- a simulated **vacuum-pressure sensor** in the cup, derived from seal quality, with noise and a threshold, as a real cell would have;
- the **bin camera**, to re-capture after each attempt;
- the **tray camera**, to check placement in the verify step.

Sim ground truth is used only by the seal physics and by the benchmark referee that scores success.

This is what makes good grips succeed and bad grips fail, as the owner requires. Exact formulas and thresholds are designed in M2 and calibrated in M4 against simple physical sanity cases (flat top center → hold; 30° tilt → fail; cylinder side below 40 mm → fail (was 40 mm, changed with the B40 cup)). Calibration is never done against benchmark results.

## 8. Architecture

One Python package `binpick`, one module per stage. Plain inputs and outputs (dataclasses and NumPy arrays), so modules can be swapped.

```
binpick/
  config.py        # load/validate configs/default.yaml → frozen dataclasses
  sim/             # 1. Simulation
    parts.py       #    sample part specs from seed (pure, no MuJoCo)
    scene.py       #    MjSpec world: panda + cup, bin, tray, camera, parts
    settle.py      #    drop parts one at a time, settle, validate
    camera.py      #    render RGB + metric depth (+ noise, M4)
    suction.py     #    seal model, vacuum grip with load check, vacuum sensor (M2/M4)
  perception/      # 2. depth → point cloud (M3)
  planning/        # 3. rules planner, random baseline (M3)
  motion/          # 4. mink IK + trajectories (M2)
  supervisor/      # 5. state machine, failures, metrics (M2–M5)
  benchmark/       # parallel runner, CSV, report (M5)
  cli.py           # entry points
configs/default.yaml   # ALL settings: counts, ranges, densities, seed, noise, tray grid, timings
assets/franka_emika_panda/  # vendored Menagerie model + LICENSE
tests/
.github/workflows/ci.yml
```

Supervisor state machine (M2+): capture → plan → approach → grip → lift → transport → place → verify. On failure, go to retry, or shake (up to 3) when no candidate passes.

Pick rules (M3), as specified by the owner: candidate points on visible surfaces. Seal check: plane fit under a 44 mm disk, reject if tilt > 20° or plane residual is too large. Curved parts are never rejected just for being curved. Reject points near walls or out of reach. Score by height, seal quality, wall clearance and distance to the part center. Flat before curved. Shake if nothing passes. The random baseline picks a random visible point with no filters.

Placement rules (M2): next free slot. Rotate the wrist so the part's long side lines up with the slot. Lower to just above the tray surface, then release.

## 9. Milestone 1 — build-ready detail

Deliverables:
1. Project skeleton: `pyproject.toml` (uv, Python 3.12), package layout above, `configs/default.yaml`, vendored Panda, `.gitignore`, minimal README, CI workflow running pytest headless.
2. `sim/parts.py`: deterministic part sampler implementing §6.
3. `sim/scene.py`: MjSpec world with Panda (no hand) + 44 mm cup + adhesion actuator (ctrl 0..1; seal model not yet active), bin on its x/y shaker base (actuators present, held at zero), 5 × 4 tray, top-down camera over the bin, top-down camera over the tray, an overview camera, lights, floor. A test checks that a commanded shake moves the bin and that it returns to its home position.
4. `sim/settle.py`: parts dropped one at a time from above the rim at random x/y inside the bin (inset margin) with uniformly random orientation. After each drop, step until all parts are still (linear speed < 1 mm/s, angular < 0.01 rad/s) or a timeout. At the end, every part's center must be inside the bin. A part that escapes is re-dropped from the same seeded RNG stream, so the result stays deterministic. The robot is parked out of the camera view while settling.
5. `sim/camera.py`: headless render of RGB (uint8) and metric depth (float32 m) from the top-down camera. Default 640 × 480; fovy chosen so the bin fills most of the frame. Intrinsics are exposed for M3.
6. CLI `binpick-scene --seed N` writes `outputs/scene_seed{N}/`: `rgb.png`, `depth.npy` (meters), `depth_mm.png` (16-bit), `depth_preview.png` (colorized), `tray_rgb.png` (empty tray from the tray camera), `overview.png` (perspective view of the whole cell), `parts.json` (per part: id, shape, material, dims, mass, settled pose).
7. Viewer: `mjpython -m binpick.view --seed N` opens the settled scene interactively.

M1 tests (pytest, headless):
- Config loads; counts are 15 boxes and 5 cylinders.
- Sampler: all dimensions within range, every mass in [0.05, 2.5] kg, materials drawn from the three, same seed gives identical specs, different seeds give different specs.
- Cup geom diameter is 44 mm. The adhesion actuator exists.
- Tray has 20 slots on a 120 mm pitch.
- Reachability IK test from §6 passes (uses mink).
- Settling: all 20 parts end inside the bin, at rest, with no penetration deeper than a small tolerance (for example 2 mm). Same seed gives the same poses on the same machine (tolerance 1e-6).
- Camera: correct image shapes. On an empty-bin scene, measured depth of the bin floor matches the geometry within 2 mm.

M1 checkpoint gives the owner: summary, one command, test output, the RGB/depth/overview images for a seed, and a "how to explain this" note.

## 10. Later milestones (from the owner's brief, unchanged)

2. Motion + state machine: pick from a known pose, place in a slot. Seal model v1.
3. Perception + planning: point cloud, seal check, suction-point selection. First full bin emptied.
4. Reliability: retries, failure detection, collision checks, bin shaking, depth noise. Seal model calibration.
5. Benchmark: both planners on seeds 1–25, headless, parallel, CSV of every attempt, report (success per attempt, bin-emptying rate, average/slowest cycle, failure breakdown). Tests + CI.
6. Fixes for the top failure causes, then a full rerun.
7. Delivery: README, side-by-side results report, dashboard, 2-minute demo video.

**Seed hygiene (builder's call, owner may override):** during development and tuning (M2–M4, M6) use seeds ≥ 1000. Seeds 1–25 are touched only by benchmark runs, so the reported numbers aren't tuned on the bins they're measured on.

## 11. Left to the builder

Exact cell layout and pedestal height, tool length, contact and solver parameters, friction per material, camera mounting height and fovy, settle timeouts, file formats beyond those listed, internal module APIs, and seal-model formulas (within the §7 principle).
