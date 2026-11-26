# BinPick-Sim: results

*Final code: commit `b22dd8b`, measured 2026-11-25 on an Apple M-series Mac. Every number here comes from the benchmark folders in
[`docs/checkpoints/`](checkpoints/) and can be reproduced with one command. A visual version is in [`dashboard.html`](dashboard.html).*

## In one paragraph

A simulated Franka Panda with a 44 mm suction cup empties a cluttered bin of 20 mixed parts (boxes and cylinders, up to 1.75 kg)
into a 5 × 4 tray, using only what a real cell would sense: a noisy top-down depth camera, a vacuum sensor, and its own joints.
On **25 bins nobody had looked at before**, it placed **495 of 500 parts**; **94.5%** of its pick attempts succeeded, with a mean cycle of
**7.4 s**. Picking points at random on the same bins succeeded **3.4%** of the time. It met the cycle-time target, came close on
success per attempt (target 95%) and emptied **20 of 25 bins** (target 95%). Each bin it could not empty had one small cylinder left
that the 44 mm cup cannot grip lying on its side.

## Targets

| | Target | Unused seeds 1240–1264 (blind) | Seeds 1–25, final code (rerun) | Seeds 1–25, first run (blind) |
|---|---|---|---|---|
| Pick success per attempt | ≥ 95% | **94.5%** (495 / 524) – missed | **95.6%** (499 / 522) – met | 95.4% (497 / 521) – met |
| Bins emptied (all 20 parts in their slots) | ≥ 95% | **80%** (20 / 25) – missed | **96%** (24 / 25) – met | 88% (22 / 25) – missed |
| Mean cycle time, successful picks | ≤ 8 s | **7.38 s** – met | **7.40 s** – met | 8.09 s – missed |
| Parts placed | – | 495 / 500 | 499 / 500 | 497 / 500 |
| Random baseline, success per attempt | – | 3.4% (34 / 1000) | 3.2% (32 / 1000) | 3.6% (36 / 1000) |

**Which number to quote.** Seeds 1–25 are the benchmark the brief asked for. They were first run blind (right-hand column). Their
failures were then studied and fixed, so the rerun on the final code is no longer a blind test. The seeds 1240–1264 were run once,
on the final code, with nobody having looked at them: **that column is the fairest estimate of how the final robot does on new
bins.**

**How sure.** With 25 bins, the success rate on other bins like these would likely lie between 90% and 97%, and the share of bins
emptied between 61% and 91% (blind run; approximate 95% ranges: success resampled over bins, bins emptied a Wilson interval). The first
number is fairly well determined; the second is not, because 25 bins is few.

## Both planners, side by side (unused seeds 1240–1264)

| | Rules planner | Random baseline |
|---|---|---|
| Success per attempt | **94.5%** | 3.4% |
| Successful picks / attempts | 495 / 524 | 34 / 1000 |
| Bins emptied | 20 / 25 | 0 / 25 |
| Parts placed | 495 / 500 | 35 / 500 |
| Mean / median / slowest cycle, successful picks | 7.38 / 7.23 / 11.58 s | 8.02 / 7.97 / 8.85 s |
| Successful picks over 8 s | 75 / 495 | 17 / 34 |
| Sim time per placed part, including shakes and failures | 9.3 s | 101.6 s |
| Robot's own verdict different from the referee | 0 / 524 | 0 / 1000 |
| Arm or tool hit something hard (> 50 N) | 1 attempt | 3 attempts |

The random baseline gets the same camera, motion, sensing, recovery and placing; only the choice of the point to grip is random.
Each run stops after 40 attempts, which is why random leaves most parts in the bin.

## Where it fails

**Failed attempts** (rules planner, 29 of 524 on the unused seeds; 23 of 522 on seeds 1–25):

| Cause | Unused seeds | Seeds 1–25 |
|---|---|---|
| Part torn off by its neighbours or a wall while lifting | 23 | 10 |
| Arm touched something on the approach (134 N) | 1 | 0 |
| Seal gave way under the load while lifting | 4 | 11 |
| No seal formed, or lost before it was confirmed | 1 | 2 |
| Dropped while carrying, arm could not follow its path | 0 | 0 |

Two kinds of failure remain: parts torn off because they are held by a neighbour or a wall, and grips that give way under the load
while lifting. **One part accounts for 11 of the 29 failures on the unused seeds:** in seed 1264 a 0.27 kg plastic cylinder was
torn off 11 times. The planner rated it a safe pick every time, and after every shake it tried it again. It was finally placed on the
twelfth try and the bin was emptied. Without that bin the rate would be 475 / 493 = 96.3%. This is a weakness of the planner: it does
not stop trying a part that keeps tearing off.

**Bins not emptied.** All six parts left over in the two final runs are short cylinders (40–43 mm long). The cup's 44 mm lip cannot
seal on the side of a cylinder shorter than about 43 mm. Two of the six (also 42–43 mm across) fit the cup on no side at all. The
other four fit only standing on an end; the runs ended with them lying down, after three shakes in a row brought no pick. In the 50
final bins, 11 held cylinders that fit only standing on an end (12 such parts, 8 of them still placed) and 2 held one that fits
nowhere. So
this is mostly a limit of the cup and the part mix, and partly of the robot, which has no way of standing a part up.

**Cycle time** (7.4 s on average): approach 1.4 s, lift 1.05 s, transport 1.4 s, set-down 1.0 s, retreat 1.1 s, plus fixed costs
of 1.4 s for the photo (0.3), planning (0.5), vacuum build-up (0.15), release (0.25) and the tray check (0.2). Nothing overlaps; a
real cell would take the next photo and plan while the arm is over the tray, which could save up to 0.8 s (the photo and planning
times; not measured). Including shakes and failed
attempts, a part leaves the bin every 9.3 s.

## How it got here

| Stage | What changed | Success | Bins emptied | Cycle |
|---|---|---|---|---|
| Camera picking (M3), 10 fresh bins | Pick from depth images only; no camera noise yet | 72.8% | 0 / 10 | 8.1 s |
| Reliability (M4), 20 fresh bins | Camera noise; free parts first; straighter lift, away from walls; harder shakes | 92.6–97.3% | 16–18 / 20 | 8.1 s |
| Benchmark (M5), seeds 1–25 | Strict scoring: the robot must let go itself | 95.4% | 22 / 25 | 8.09 s |
| Fixes (M6), unused seeds | Carrying speed from the felt weight; no path failures; faster planning; joint limits enforced | 94.5% | 20 / 25 | 7.38 s |

M5's 8.09 s was too short in two ways: some moves went up to 15% over a joint's speed limit, and planning was charged 0.2 s though
that code needed about 1.0 s (1.6 s with all motion planning). The final code keeps every move within the Panda datasheet and charges
planning at what it needs.

## What is real and what is assumed

Read these before using the numbers outside the simulation.

- **Simulation, not hardware.** MuJoCo physics with a Franka Panda model and contact settings chosen once and then frozen.
  Part-to-bin friction is the simulator default (1.0), higher than the parts' own 0.5–0.6.
- **The cup** is modelled on the Piab B40.20 datasheet: 39 N at 60 kPa, 20 mm minimum curve radius. How much it holds when tilted,
  and at the curve limit, are assumptions (the sheet does not say).
- **Part weights** are capped at 1.75 kg. The cap was chosen by the project owner after seeing heavier parts fall, and justified
  afterwards: Schmalz's safety factors 1.5 and 2.0 give 2.1 and 1.6 kg for a straight lift at 2.5 m/s², so **1.75 kg is above the
  stricter one**, and sideways transport lowers both. Compared with a 2.5 kg cap it redraws about 4% of parts.
  **No manufacturer part weights were available**; real ones may change the results.
- **What the cup carries, by shape** (centered grip, measured on the robot before the lift was improved; current figures after the
  arrow): flat 100 mm box 1.75 kg; 60 mm tall box 1.75 kg; 60 mm cylinder by its end 1.5 → 1.75 kg; cylinder sides 1.25 → 1.5 kg
  (60 and 80 mm cylinders fail at 1.75 kg on their side). So the heaviest cylinders cannot be carried lying down.
- **Camera noise** is 0.5 mm per pixel at 0.65 m, with missing pixels at edges. It is independent from pixel to pixel, which is
  milder than a real camera's noise of the same size.
- **What the robot senses:** the depth cameras, a noisy vacuum sensor, and its own joint angles and torques (assumed noise-free,
  arm gravity perfectly compensated). It knows the range of part sizes in the catalogue, but never where the parts are or what they
  weigh; it feels the weight after lifting.
- **Cycle time** includes 1.4 s of assumed fixed times; planning is charged at what this code needs on the Mac (0.41 s on average,
  up to 0.6 s, charged 0.5 s).
- **Pick order** puts safe picks (free, centered, level) before the owner's rule "flat before curved".
- **Random baseline** gets the same everything; it has no seal estimate, so it carries every part as if the seal were perfect.

## Reproduce

```bash
uv sync
uv run binpick-bench                                             # seeds 1-25, both planners, about 30 minutes
uv run binpick-bench --seeds 1240-1264 --out outputs/unused     # the blind set
uv run binpick-dashboard                                         # redraw docs/dashboard.html
```

Each bin is a separate single-threaded process seeded only by its seed, so results do not depend on the number of workers. Another
machine or library version may differ slightly. The run folders hold every attempt (`attempts.csv`), every part (`parts.csv`), every
bin (`bins.csv`), the exact settings (`config.yaml`) and the code version (`meta.json`).

## What we would do next

1. **A second, smaller cup** (or a cup change) for the parts the 44 mm cup cannot grip; this is what limits the bin-emptying rate
   in the final runs.
2. **Stop retrying a part that keeps tearing off** (seed 1264): count failures per part and leave it until the others are out.
3. **Real part data from the client**: weights, materials and shapes, then rerun.
4. **Take the next photo and plan while placing**, up to 0.8 s per pick (estimate).
5. **Pinned parts:** feel the load while lifting and back off before the grip tears; or nudge a neighbour aside.
6. **Hardware check** of the two assumptions that matter most: the cup's hold when tilted, and the camera's real noise.
