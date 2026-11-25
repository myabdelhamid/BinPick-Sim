# Benchmark report: seeds 1–25

25 bins × 20 parts per planner. Both planners get the same bins.

## Targets (rules planner)

| Metric | Target | Result | |
|---|---|---|---|
| Pick success per attempt | ≥ 95% | 95.4% | met |
| Mean cycle time per successful pick | ≤ 8 s | 8.09 s | **missed** |
| Bin-emptying rate | ≥ 95% | 88.0% | **missed** |

## Side by side

| | rules | random |
|---|---|---|
| Successful picks ÷ attempts | 497 / 521 | 36 / 1000 |
| **Success per attempt** | **95.4%** | **3.6%** |
| … approximate 95% range (resampling bins) | 93.0% to 97.5% | 2.4% to 4.8% |
| **Bins emptied** | **22 / 25 (88.0%)** | **0 / 25 (0.0%)** |
| … 95% range | 70.0% to 95.8% | 0.0% to 13.3% |
| Parts placed | 497 / 500 (99.4%) | 36 / 500 (7.2%) |
| Parts left in bin / lost / misplaced | 1 / 2 / 0 | 455 / 0 / 9 |
| Successful picks later knocked out of their slot | 0 | 0 |
| Failed attempts whose part fell into its slot anyway | 0 | 0 |
| Cycle time, successful picks: mean | 8.09 s | 8.84 s |
| … median | 8.02 s | 8.83 s |
| … slowest | 11.02 s | 9.95 s |
| … over 8 s | 258 / 497 | 35 / 36 |
| Cycle time, all attempts: mean / slowest | 7.90 s / 11.02 s | 3.68 s / 11.46 s |
| Sim time per placed part, with shakes and failures | 9.64 s | 102.30 s |
| Shakes (of those hard) | 192 (39) | 0 (0) |
| Robot's own verdict wrong | 0 / 521 attempts | 0 / 1000 attempts |
| Arm or tool hit something (> 50 N) | 0 attempts | 3 attempts |
| Bins that failed to run | 0 | 0 |

Cycle time runs from the start of the bin photo to the end of the tray check and nothing overlaps (the owner's strict definition); taking photos while the arm moves would shorten it. Part of it is assumed, see the notes below. An attempt is a success only if the robot set the part down, let go, and the part rests within ±10 mm of its slot's center.

## Failed attempts

| Cause | rules | random |
|---|---|---|
| no seal formed (`no_seal`) | 0 | 556 |
| approach stopped on unexpected contact, or part torn off by something (`collision`) | 10 | 371 |
| lost in lift (load) (`seal_lost_lift`) | 6 | 17 |
| seal lost before the grip was confirmed (`seal_lost_grip`) | 1 | 10 |
| not within ±10 mm of its slot (`misplaced`) | 0 | 9 |
| arm could not follow the path (`ik_fail`) | 5 | 0 |
| lost in transport (`seal_lost_transport`) | 2 | 1 |

## Parts not placed, and why

| | rules | random |
|---|---|---|
| tried | 0 | 288 |
| never tried (attempt_limit) | 0 | 164 |
| lies in the tray, not centered in a slot of its own | 0 | 9 |
| fell out of the cell | 2 | 0 |
| no side the 44 mm lip fits on, never tried (no_candidate) | 1 | 0 |
| fits the lip only standing on its end, tried | 0 | 1 |
| no side the 44 mm lip fits on, never tried (attempt_limit) | 0 | 1 |
| fits the lip only standing on its end, never tried (attempt_limit) | 0 | 1 |

1 of 25 bins hold a part with no side the cup's lip fits on (geometry only).

Parts the rules planner did not place:

| Seed | Part | Shape, mm | kg | End state | Why |
|---|---|---|---|---|---|
| 13 | 19 | cylinder 61.4x72.5 | 1.69 | lost | fell out of the cell |
| 20 | 18 | cylinder 42.8x41.2 | 0.07 | left | no side the 44 mm lip fits on, never tried (no_candidate) |
| 23 | 7 | box 94.7x100.1x67.0 | 1.72 | lost | fell out of the cell |

## Every bin

| Planner | Seed | Placed | Attempts | Shakes | Ended | Left / lost / misplaced |
|---|---|---|---|---|---|---|
| rules | 1 | 20/20 | 21 | 4 | empty | 0 / 0 / 0 |
| rules | 2 | 20/20 | 20 | 7 | empty | 0 / 0 / 0 |
| rules | 3 | 20/20 | 21 | 8 | empty | 0 / 0 / 0 |
| rules | 4 | 20/20 | 25 | 10 | empty | 0 / 0 / 0 |
| rules | 5 | 20/20 | 21 | 8 | empty | 0 / 0 / 0 |
| rules | 6 | 20/20 | 20 | 8 | empty | 0 / 0 / 0 |
| rules | 7 | 20/20 | 20 | 5 | empty | 0 / 0 / 0 |
| rules | 8 | 20/20 | 21 | 7 | empty | 0 / 0 / 0 |
| rules | 9 | 20/20 | 22 | 5 | empty | 0 / 0 / 0 |
| rules | 10 | 20/20 | 20 | 5 | empty | 0 / 0 / 0 |
| rules | 11 | 20/20 | 20 | 4 | empty | 0 / 0 / 0 |
| rules | 12 | 20/20 | 22 | 10 | empty | 0 / 0 / 0 |
| rules | 13 | 19/20 | 23 | 6 | empty | 0 / 1 / 0 |
| rules | 14 | 20/20 | 21 | 7 | empty | 0 / 0 / 0 |
| rules | 15 | 20/20 | 20 | 10 | empty | 0 / 0 / 0 |
| rules | 16 | 20/20 | 21 | 11 | empty | 0 / 0 / 0 |
| rules | 17 | 20/20 | 20 | 9 | empty | 0 / 0 / 0 |
| rules | 18 | 20/20 | 21 | 7 | empty | 0 / 0 / 0 |
| rules | 19 | 20/20 | 21 | 6 | empty | 0 / 0 / 0 |
| rules | 20 | 19/20 | 19 | 9 | no_candidate | 1 / 0 / 0 |
| rules | 21 | 20/20 | 22 | 11 | empty | 0 / 0 / 0 |
| rules | 22 | 20/20 | 20 | 5 | empty | 0 / 0 / 0 |
| rules | 23 | 19/20 | 20 | 12 | empty | 0 / 1 / 0 |
| rules | 24 | 20/20 | 20 | 9 | empty | 0 / 0 / 0 |
| rules | 25 | 20/20 | 20 | 9 | empty | 0 / 0 / 0 |
| random | 1 | 0/20 | 40 | 0 | attempt_limit | 20 / 0 / 0 |
| random | 2 | 0/20 | 40 | 0 | attempt_limit | 19 / 0 / 1 |
| random | 3 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 4 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 5 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 6 | 3/20 | 40 | 0 | attempt_limit | 16 / 0 / 1 |
| random | 7 | 0/20 | 40 | 0 | attempt_limit | 19 / 0 / 1 |
| random | 8 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 9 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 10 | 2/20 | 40 | 0 | attempt_limit | 17 / 0 / 1 |
| random | 11 | 0/20 | 40 | 0 | attempt_limit | 20 / 0 / 0 |
| random | 12 | 2/20 | 40 | 0 | attempt_limit | 17 / 0 / 1 |
| random | 13 | 0/20 | 40 | 0 | attempt_limit | 20 / 0 / 0 |
| random | 14 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 15 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 16 | 0/20 | 40 | 0 | attempt_limit | 19 / 0 / 1 |
| random | 17 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 18 | 3/20 | 40 | 0 | attempt_limit | 17 / 0 / 0 |
| random | 19 | 4/20 | 40 | 0 | attempt_limit | 16 / 0 / 0 |
| random | 20 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 21 | 1/20 | 40 | 0 | attempt_limit | 17 / 0 / 2 |
| random | 22 | 1/20 | 40 | 0 | attempt_limit | 18 / 0 / 1 |
| random | 23 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 24 | 4/20 | 40 | 0 | attempt_limit | 16 / 0 / 0 |
| random | 25 | 0/20 | 40 | 0 | attempt_limit | 20 / 0 / 0 |

## What to know before using these numbers

These notes describe the default settings.

- **Part weight limit 1.75 kg.** Chosen by the owner after seeing heavier parts drop, and justified afterwards: Schmalz safety factors 1.5 and 2.0 give 2.1 and 1.6 kg for a straight lift at 2.5 m/s² (so 1.75 kg is above the stricter one), and sideways transport lowers both. It redraws about 4% of parts compared with 2.5 kg. There were no manufacturer part weights available.
- **The cup** is modelled on the Piab B40.20 datasheet (39 N at 60 kPa, 20 mm minimum curve radius). The sheet gives no tilt data and no force at the curve limit; those two are assumptions. Carrying limits with a centered grip, measured 2026-11-24 before the lift was improved: flat 100 mm box 1.75 kg, 60 mm tall box 1.75 kg, 60 mm cylinder by its end 1.5 kg, cylinder sides 1.25 kg. Re-measured on the current robot: 60 mm end 1.75 kg, cylinder sides 1.5 kg (60 and 80 mm fail at 1.75 kg).
- **Frozen test.** Parts, cup and contact physics have not changed since commit 6bba2f3, except three fixes so that bins whose parts creep or jitter still start (b5b16fb, 7eca2ad, 8d0f978); bins that started before start the same.
- **Cycle time is partly assumed.** Photo 0.3 s, planning 0.2 s, vacuum build-up 0.15 s, release 0.25 s and tray check 0.2 s are fixed settings, 1.1 s of every cycle. This unoptimised code really needs about 1.0 s to plan on the reference Mac, not 0.2 s; charging that would add about 0.8 s to every cycle.
- **Small parts.** The cup's 44 mm lip does not fit on a cylinder side, or a box face, narrower than about 43–44 mm. Parts were frozen before this was known and were not changed; such parts count as not placed.
- **Camera noise:** 0.5 mm per pixel at 0.65 m (growing with distance²), 1% missing pixels, half the pixels missing at edges. Independent per pixel, so milder than correlated noise of the same size. No calibration error.
- **What the robot senses:** two depth cameras, a noisy vacuum sensor, and its own joint angles and torques (noise-free, arm gravity perfectly compensated). It knows the range of part sizes. Positions and weights of parts are never given to it.
- **Order of picks:** safe picks (free, centered, level) come before the owner's "flat before curved"; with no safe pick the bin is shaken first, up to three times. Shakes are not part of any pick's cycle time; they are in the sim time per placed part.
- **Friction:** parts rub on bin and tray with the simulator default 1.0; part on part 0.5–0.6.
- **Not attempts, for both planners:** targets the arm cannot reach, or whose approach would hit the cell. They are in events.jsonl.
- **Seeds.** Development used seeds 1000 and up. Seeds 1–25 were first run on 2026-11-25 for milestone 5; any later run of them is a rerun after fixes and no longer a blind test.

## Reproduce

```bash
uv run binpick-bench
```

- Code: commit `242ed2e71cd1cc734e31610a1cc58c5a95502010`. Config SHA-256 `0176a75dfafd35fb`.
- Machine: macOS-26.5.1-arm64-arm-64bit (arm64); python 3.12.15, mujoco 4.0.1, mink 1.4.0, numpy 2.5.5, scipy 1.18.1, open3d 0.20.0.
- Started 2026-11-25T03:25:34+01:00, 29 minutes with 6 workers. Results do not depend on the number of workers; another machine may differ slightly.
