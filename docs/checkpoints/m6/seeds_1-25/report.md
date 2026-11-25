# Benchmark report: seeds 1–25

25 bins × 20 parts per planner. Both planners get the same bins.

## Targets (rules planner)

| Metric | Target | Result | |
|---|---|---|---|
| Pick success per attempt | ≥ 95% | 95.6% | met |
| Mean cycle time per successful pick | ≤ 8 s | 7.40 s | met |
| Bin-emptying rate | ≥ 95% | 96.0% | met |

## Side by side

| | rules | random |
|---|---|---|
| Successful picks ÷ attempts | 499 / 522 | 32 / 1000 |
| **Success per attempt** | **95.6%** | **3.2%** |
| … approximate 95% range (resampling bins) | 93.8% to 97.3% | 2.2% to 4.2% |
| **Bins emptied** | **24 / 25 (96.0%)** | **0 / 25 (0.0%)** |
| … 95% range | 80.5% to 99.3% | 0.0% to 13.3% |
| Parts placed | 499 / 500 (99.8%) | 32 / 500 (6.4%) |
| Parts left in bin / lost / misplaced | 1 / 0 / 0 | 458 / 0 / 10 |
| Successful picks later knocked out of their slot | 0 | 0 |
| Failed attempts whose part fell into its slot anyway | 0 | 0 |
| Cycle time, successful picks: mean | 7.40 s | 8.02 s |
| … median | 7.27 s | 7.97 s |
| … slowest | 10.34 s | 9.65 s |
| … over 8 s | 78 / 499 | 15 / 32 |
| Cycle time, all attempts: mean / slowest | 7.23 s / 10.34 s | 3.62 s / 12.41 s |
| Sim time per placed part, with shakes and failures | 9.02 s | 113.02 s |
| Shakes (of those hard) | 191 (36) | 0 (0) |
| Robot's own verdict wrong | 0 / 522 attempts | 0 / 1000 attempts |
| Arm or tool hit something (> 50 N) | 0 attempts | 9 attempts |
| Bins that failed to run | 0 | 0 |

Cycle time runs from the start of the bin photo to the end of the tray check and nothing overlaps (the owner's strict definition); taking photos while the arm moves would shorten it. Part of it is assumed, see the notes below. An attempt is a success only if the robot set the part down, let go, and the part rests within ±10 mm of its slot's center.

## Failed attempts

| Cause | rules | random |
|---|---|---|
| no seal formed (`no_seal`) | 2 | 525 |
| approach stopped on unexpected contact, or part torn off by something (`collision`) | 10 | 403 |
| lost in lift (load) (`seal_lost_lift`) | 11 | 13 |
| seal lost before the grip was confirmed (`seal_lost_grip`) | 0 | 17 |
| not within ±10 mm of its slot (`misplaced`) | 0 | 10 |

## Parts not placed, and why

| | rules | random |
|---|---|---|
| tried | 0 | 312 |
| never tried (attempt_limit) | 0 | 143 |
| lies in the tray, not centered in a slot of its own | 0 | 10 |
| fits the lip only standing on its end, tried | 0 | 2 |
| no side the 44 mm lip fits on, never tried (no_candidate) | 1 | 0 |
| no side the 44 mm lip fits on, never tried (attempt_limit) | 0 | 1 |

1 of 25 bins hold a part with no side the cup's lip fits on (geometry only).

Parts the rules planner did not place:

| Seed | Part | Shape, mm | kg | End state | Why |
|---|---|---|---|---|---|
| 20 | 18 | cylinder 42.8x41.2 | 0.07 | left | no side the 44 mm lip fits on, never tried (no_candidate) |

## Every bin

| Planner | Seed | Placed | Attempts | Shakes | Ended | Left / lost / misplaced |
|---|---|---|---|---|---|---|
| rules | 1 | 20/20 | 21 | 3 | empty | 0 / 0 / 0 |
| rules | 2 | 20/20 | 21 | 5 | empty | 0 / 0 / 0 |
| rules | 3 | 20/20 | 20 | 7 | empty | 0 / 0 / 0 |
| rules | 4 | 20/20 | 20 | 8 | empty | 0 / 0 / 0 |
| rules | 5 | 20/20 | 24 | 9 | empty | 0 / 0 / 0 |
| rules | 6 | 20/20 | 21 | 9 | empty | 0 / 0 / 0 |
| rules | 7 | 20/20 | 20 | 8 | empty | 0 / 0 / 0 |
| rules | 8 | 20/20 | 20 | 4 | empty | 0 / 0 / 0 |
| rules | 9 | 20/20 | 22 | 7 | empty | 0 / 0 / 0 |
| rules | 10 | 20/20 | 21 | 4 | empty | 0 / 0 / 0 |
| rules | 11 | 20/20 | 20 | 9 | empty | 0 / 0 / 0 |
| rules | 12 | 20/20 | 20 | 6 | empty | 0 / 0 / 0 |
| rules | 13 | 20/20 | 21 | 5 | empty | 0 / 0 / 0 |
| rules | 14 | 20/20 | 21 | 6 | empty | 0 / 0 / 0 |
| rules | 15 | 20/20 | 21 | 7 | empty | 0 / 0 / 0 |
| rules | 16 | 20/20 | 23 | 12 | empty | 0 / 0 / 0 |
| rules | 17 | 20/20 | 20 | 10 | empty | 0 / 0 / 0 |
| rules | 18 | 20/20 | 20 | 9 | empty | 0 / 0 / 0 |
| rules | 19 | 20/20 | 20 | 4 | empty | 0 / 0 / 0 |
| rules | 20 | 19/20 | 19 | 9 | no_candidate | 1 / 0 / 0 |
| rules | 21 | 20/20 | 21 | 15 | empty | 0 / 0 / 0 |
| rules | 22 | 20/20 | 21 | 10 | empty | 0 / 0 / 0 |
| rules | 23 | 20/20 | 22 | 9 | empty | 0 / 0 / 0 |
| rules | 24 | 20/20 | 22 | 6 | empty | 0 / 0 / 0 |
| rules | 25 | 20/20 | 21 | 10 | empty | 0 / 0 / 0 |
| random | 1 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 2 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 3 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 4 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 5 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 6 | 4/20 | 40 | 0 | attempt_limit | 16 / 0 / 0 |
| random | 7 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 8 | 0/20 | 40 | 0 | attempt_limit | 20 / 0 / 0 |
| random | 9 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 10 | 0/20 | 40 | 0 | attempt_limit | 19 / 0 / 1 |
| random | 11 | 3/20 | 40 | 0 | attempt_limit | 17 / 0 / 0 |
| random | 12 | 1/20 | 40 | 0 | attempt_limit | 18 / 0 / 1 |
| random | 13 | 0/20 | 40 | 0 | attempt_limit | 20 / 0 / 0 |
| random | 14 | 1/20 | 40 | 0 | attempt_limit | 18 / 0 / 1 |
| random | 15 | 0/20 | 40 | 0 | attempt_limit | 19 / 0 / 1 |
| random | 16 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 17 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 18 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 19 | 0/20 | 40 | 0 | attempt_limit | 20 / 0 / 0 |
| random | 20 | 2/20 | 40 | 0 | attempt_limit | 16 / 0 / 2 |
| random | 21 | 0/20 | 40 | 0 | attempt_limit | 18 / 0 / 2 |
| random | 22 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 23 | 1/20 | 40 | 0 | attempt_limit | 17 / 0 / 2 |
| random | 24 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 25 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |

## What to know before using these numbers

These notes describe the default settings.

- **Part weight limit 1.75 kg.** Chosen by the owner after seeing heavier parts drop, and justified afterwards: Schmalz safety factors 1.5 and 2.0 give 2.1 and 1.6 kg for a straight lift at 2.5 m/s² (so 1.75 kg is above the stricter one), and sideways transport lowers both. It redraws about 4% of parts compared with 2.5 kg. There were no manufacturer part weights available.
- **The cup** is modelled on the Piab B40.20 datasheet (39 N at 60 kPa, 20 mm minimum curve radius). The sheet gives no tilt data and no force at the curve limit; those two are assumptions. Carrying limits with a centered grip, measured 2026-11-24 before the lift was improved: flat 100 mm box 1.75 kg, 60 mm tall box 1.75 kg, 60 mm cylinder by its end 1.5 kg, cylinder sides 1.25 kg. Re-measured on the current robot: 60 mm end 1.75 kg, cylinder sides 1.5 kg (60 and 80 mm fail at 1.75 kg).
- **Frozen test.** Parts, cup and contact physics have not changed since commit 6bba2f3, except three fixes so that bins whose parts creep or jitter still start (b5b16fb, 7eca2ad, 8d0f978); bins that started before start the same.
- **Cycle time is partly assumed.** Photo 0.3 s, vacuum build-up 0.15 s, release 0.25 s and tray check 0.2 s are assumed. All planning is charged 0.5 s; on the reference Mac it takes 0.41 s on average and up to 0.6 s. (Milestone 5 charged 0.2 s; that code needed about 1.6 s with motion planning.) Together 1.4 s of every cycle are fixed.
- **Small parts.** The cup's 44 mm lip does not fit on a cylinder side, or a box face, narrower than about 43–44 mm. Parts were frozen before this was known and were not changed; such parts count as not placed.
- **Camera noise:** 0.5 mm per pixel at 0.65 m (growing with distance²), 1% missing pixels, half the pixels missing at edges. Independent per pixel, so milder than correlated noise of the same size. No calibration error.
- **What the robot senses:** two depth cameras, a noisy vacuum sensor, and its own joint angles and torques (noise-free, arm gravity perfectly compensated). It knows the range of part sizes. Positions and weights of parts are never given to it.
- **Carrying:** the arm feels the part's weight at the end of the lift and sets its acceleration from that and the planner's seal estimate. The random baseline has no seal estimate and counts every seal as perfect.
- **Joint limits:** speeds and accelerations stay within the Panda datasheet. Milestone 5's moves exceeded a joint's speed limit by up to 15%, so its 8.09 s was a little too short.
- **Order of picks:** safe picks (free, centered, level) come before the owner's "flat before curved"; with no safe pick the bin is shaken first, up to three times. Shakes are not part of any pick's cycle time; they are in the sim time per placed part.
- **Friction:** parts rub on bin and tray with the simulator default 1.0; part on part 0.5–0.6.
- **Not attempts, for both planners:** targets the arm cannot reach, or whose approach would hit the cell. They are in events.jsonl.
- **Seeds.** Development used seeds 1000 and up. Seeds 1–25 were first run on 2026-11-25 for milestone 5 (rules 95.4%, 22 of 25 bins emptied, 8.09 s); any later run of them is a rerun after fixes made with those results in hand, and no longer a blind test.

## Reproduce

```bash
uv run binpick-bench --out outputs/benchmark_m6
```

- Code: commit `b22dd8b0a9dc20955a0776832f6c9ae531a4fb05`. Config SHA-256 `10546640fa729905`.
- Machine: macOS-26.5.1-arm64-arm-64bit (arm64); python 3.12.15, mujoco 4.0.1, mink 1.4.0, numpy 2.5.5, scipy 1.18.1, open3d 0.20.0.
- Started 2026-11-25T10:47:36+01:00, 20 minutes with 6 workers. Results do not depend on the number of workers; another machine may differ slightly.
