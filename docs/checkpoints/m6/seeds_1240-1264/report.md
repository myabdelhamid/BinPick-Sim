# Benchmark report: seeds 1240–1264

25 bins × 20 parts per planner. Both planners get the same bins.

## Targets (rules planner)

| Metric | Target | Result | |
|---|---|---|---|
| Pick success per attempt | ≥ 95% | 94.5% | **missed** |
| Mean cycle time per successful pick | ≤ 8 s | 7.38 s | met |
| Bin-emptying rate | ≥ 95% | 80.0% | **missed** |

## Side by side

| | rules | random |
|---|---|---|
| Successful picks ÷ attempts | 495 / 524 | 34 / 1000 |
| **Success per attempt** | **94.5%** | **3.4%** |
| … approximate 95% range (resampling bins) | 90.4% to 97.3% | 2.4% to 4.5% |
| **Bins emptied** | **20 / 25 (80.0%)** | **0 / 25 (0.0%)** |
| … 95% range | 60.9% to 91.1% | 0.0% to 13.3% |
| Parts placed | 495 / 500 (99.0%) | 35 / 500 (7.0%) |
| Parts left in bin / lost / misplaced | 5 / 0 / 0 | 461 / 1 / 3 |
| Successful picks later knocked out of their slot | 0 | 0 |
| Failed attempts whose part fell into its slot anyway | 0 | 1 |
| Cycle time, successful picks: mean | 7.38 s | 8.02 s |
| … median | 7.23 s | 7.97 s |
| … slowest | 11.58 s | 8.85 s |
| … over 8 s | 75 / 495 | 17 / 34 |
| Cycle time, all attempts: mean / slowest | 7.19 s / 11.58 s | 3.56 s / 11.64 s |
| Sim time per placed part, with shakes and failures | 9.34 s | 101.62 s |
| Shakes (of those hard) | 224 (43) | 0 (0) |
| Robot's own verdict wrong | 0 / 524 attempts | 0 / 1000 attempts |
| Arm or tool hit something (> 50 N) | 1 attempts | 3 attempts |
| Bins that failed to run | 0 | 0 |

Cycle time runs from the start of the bin photo to the end of the tray check and nothing overlaps (the owner's strict definition); taking photos while the arm moves would shorten it. Part of it is assumed, see the notes below. An attempt is a success only if the robot set the part down, let go, and the part rests within ±10 mm of its slot's center.

## Failed attempts

| Cause | rules | random |
|---|---|---|
| no seal formed (`no_seal`) | 0 | 536 |
| approach stopped on unexpected contact, or part torn off by something (`collision`) | 24 | 409 |
| lost in lift (load) (`seal_lost_lift`) | 4 | 9 |
| seal lost before the grip was confirmed (`seal_lost_grip`) | 1 | 7 |
| not within ±10 mm of its slot (`misplaced`) | 0 | 3 |
| lost in transport (`seal_lost_transport`) | 0 | 1 |
| arm could not follow the path (`ik_fail`) | 0 | 1 |

## Parts not placed, and why

| | rules | random |
|---|---|---|
| tried | 0 | 310 |
| never tried (attempt_limit) | 0 | 141 |
| fits the lip only standing on its end, tried | 0 | 6 |
| fits the lip only standing on its end, never tried (no_candidate) | 4 | 0 |
| lies in the tray, not centered in a slot of its own | 0 | 3 |
| fits the lip only standing on its end, never tried (attempt_limit) | 0 | 3 |
| no side the 44 mm lip fits on, never tried (no_candidate) | 1 | 0 |
| fell out of the cell | 0 | 1 |
| no side the 44 mm lip fits on, tried | 0 | 1 |

1 of 25 bins hold a part with no side the cup's lip fits on (geometry only).

Parts the rules planner did not place:

| Seed | Part | Shape, mm | kg | End state | Why |
|---|---|---|---|---|---|
| 1240 | 18 | cylinder 47.8x43.1 | 0.09 | left | fits the lip only standing on its end, never tried (no_candidate) |
| 1253 | 18 | cylinder 42.2x40.2 | 0.07 | left | no side the 44 mm lip fits on, never tried (no_candidate) |
| 1256 | 19 | cylinder 47.3x43.1 | 0.20 | left | fits the lip only standing on its end, never tried (no_candidate) |
| 1260 | 17 | cylinder 50.2x40.7 | 0.10 | left | fits the lip only standing on its end, never tried (no_candidate) |
| 1263 | 17 | cylinder 55.1x40.8 | 0.12 | left | fits the lip only standing on its end, never tried (no_candidate) |

## Every bin

| Planner | Seed | Placed | Attempts | Shakes | Ended | Left / lost / misplaced |
|---|---|---|---|---|---|---|
| rules | 1240 | 19/20 | 20 | 12 | no_candidate | 1 / 0 / 0 |
| rules | 1241 | 20/20 | 22 | 9 | empty | 0 / 0 / 0 |
| rules | 1242 | 20/20 | 22 | 9 | empty | 0 / 0 / 0 |
| rules | 1243 | 20/20 | 21 | 13 | empty | 0 / 0 / 0 |
| rules | 1244 | 20/20 | 20 | 6 | empty | 0 / 0 / 0 |
| rules | 1245 | 20/20 | 20 | 3 | empty | 0 / 0 / 0 |
| rules | 1246 | 20/20 | 20 | 4 | empty | 0 / 0 / 0 |
| rules | 1247 | 20/20 | 22 | 6 | empty | 0 / 0 / 0 |
| rules | 1248 | 20/20 | 20 | 10 | empty | 0 / 0 / 0 |
| rules | 1249 | 20/20 | 21 | 10 | empty | 0 / 0 / 0 |
| rules | 1250 | 20/20 | 21 | 11 | empty | 0 / 0 / 0 |
| rules | 1251 | 20/20 | 20 | 6 | empty | 0 / 0 / 0 |
| rules | 1252 | 20/20 | 21 | 6 | empty | 0 / 0 / 0 |
| rules | 1253 | 19/20 | 19 | 14 | no_candidate | 1 / 0 / 0 |
| rules | 1254 | 20/20 | 21 | 9 | empty | 0 / 0 / 0 |
| rules | 1255 | 20/20 | 21 | 14 | empty | 0 / 0 / 0 |
| rules | 1256 | 19/20 | 21 | 11 | no_candidate | 1 / 0 / 0 |
| rules | 1257 | 20/20 | 21 | 5 | empty | 0 / 0 / 0 |
| rules | 1258 | 20/20 | 20 | 11 | empty | 0 / 0 / 0 |
| rules | 1259 | 20/20 | 20 | 6 | empty | 0 / 0 / 0 |
| rules | 1260 | 19/20 | 19 | 9 | no_candidate | 1 / 0 / 0 |
| rules | 1261 | 20/20 | 21 | 7 | empty | 0 / 0 / 0 |
| rules | 1262 | 20/20 | 20 | 7 | empty | 0 / 0 / 0 |
| rules | 1263 | 19/20 | 20 | 9 | no_candidate | 1 / 0 / 0 |
| rules | 1264 | 20/20 | 31 | 17 | empty | 0 / 0 / 0 |
| random | 1240 | 0/20 | 40 | 0 | attempt_limit | 20 / 0 / 0 |
| random | 1241 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1242 | 1/20 | 40 | 0 | attempt_limit | 18 / 0 / 1 |
| random | 1243 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1244 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1245 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1246 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 1247 | 1/20 | 40 | 0 | attempt_limit | 18 / 1 / 0 |
| random | 1248 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1249 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1250 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1251 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 1252 | 2/20 | 40 | 0 | attempt_limit | 17 / 0 / 1 |
| random | 1253 | 0/20 | 40 | 0 | attempt_limit | 20 / 0 / 0 |
| random | 1254 | 5/20 | 40 | 0 | attempt_limit | 15 / 0 / 0 |
| random | 1255 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1256 | 3/20 | 40 | 0 | attempt_limit | 17 / 0 / 0 |
| random | 1257 | 2/20 | 40 | 0 | attempt_limit | 18 / 0 / 0 |
| random | 1258 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1259 | 4/20 | 40 | 0 | attempt_limit | 16 / 0 / 0 |
| random | 1260 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1261 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1262 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |
| random | 1263 | 0/20 | 40 | 0 | attempt_limit | 19 / 0 / 1 |
| random | 1264 | 1/20 | 40 | 0 | attempt_limit | 19 / 0 / 0 |

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
uv run binpick-bench --seeds 1240-1264 --out outputs/benchmark_m6_unused
```

- Code: commit `b22dd8b0a9dc20955a0776832f6c9ae531a4fb05`. Config SHA-256 `10546640fa729905`.
- Machine: macOS-26.5.1-arm64-arm-64bit (arm64); python 3.12.15, mujoco 4.0.1, mink 1.4.0, numpy 2.5.5, scipy 1.18.1, open3d 0.20.0.
- Started 2026-11-25T10:28:02+01:00, 20 minutes with 6 workers. Results do not depend on the number of workers; another machine may differ slightly.
