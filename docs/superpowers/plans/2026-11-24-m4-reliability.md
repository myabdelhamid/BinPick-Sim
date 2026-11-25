# Milestone 4 (Reliability) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (owner chose native execution). Steps use checkbox (`- [ ]`) syntax.

**Goal:** The cell keeps working with a noisy camera and things going wrong: it picks parts that are free first, notices failures itself, avoids hitting the cell, recovers without sweeping the bin, and shakes when no safe pick is left.

**Architecture:** Noise lives in `sim/camera.py` (`sense_depth`); perception denoises. The rules planner marks each target safe or risky; `supervisor/run.py` shakes before taking risky ones. The supervisor gains a watched approach, a load-watched lift and a recovery that cannot sweep the bin.

**Tech Stack:** as M3. **Spec:** `docs/superpowers/specs/2026-11-23-binpick-sim-design.md` (§3, §7 test freeze, §8)

## Diagnosis (2026-11-24, M3 code, dev seeds 1000–1009 and 1010–1021, ground truth for analysis only)
- Most failures break the seal within 2 mm of lift. Peel (bending) is the largest share of the load: something leaning on one side of the part, a wall rubbing, or a cup away from the part's center.
- Success by what the camera could already see (321 attempts, 68% overall): nothing taller along the outline 88%, some (1–3%) 47%, more 25%; cup within 1.5 mm of the visible center 77%, over 4 mm 28%; tilt under 1° 74%, over 10° 45%. All three good ("safe", half of all attempts): **93%**; the rest 44%.
- 26 of 29 parts left on fresh seeds were never tried: the planner's 2 mm edge margin and its flat limit for thin cylinders.
- Part-to-bin friction in the simulator is 1.0 (MuJoCo default, the larger of the two), though the part materials say 0.5–0.6. Frozen; reported to the owner, not changed.

## Global Constraints
- Test freeze (§7): no changes to parts, cup or contact physics.
- The robot reads only cameras (noisy), vacuum sensor, its own joints. Ground truth only in seal physics and referee.
- Dev seeds ≥ 1000; seeds 1–25 never. Fresh seeds 1200–1209 are run once, after development.
- Every new setting in `configs/default.yaml`. Failures are counted, never hidden.

## Review Focus
1. A bin where nothing is ever "safe": the run must still end, within the shake and attempt limits.
2. Noise making an empty bin look non-empty (or a part invisible) or an empty slot look occupied.
3. Recovery after a failure while the cup is deep in the bin with a part still attached or wedged.
4. A pick that lifts two parts.
5. Random baseline: must not get the rules planner's safety checks, but must get the same motion, sensing and recovery.

### Task 1: Depth noise (done first; commit "feat(camera)")
`sense_depth(scene, camera)`: Gaussian error 0.5 mm at 0.65 m growing with distance², 1% missing pixels, half the pixels next to a >5 mm jump missing; both cameras; own stream `[seed, 5]` in `run_bin`. Between industrial structured light (60–80 µm) and low-cost stereo (RMS up to 0.4% ≈ 2.6 mm). Perception: edge-preserving average, pinhole fill, thin-cylinder flanks continued from a fitted cylinder. Tests: noise size, bias, edges, stream, zero noise; thin cylinders 41–46 mm pickable and seal estimate right; box edge never taken for a curve; neighbour on a flank rejected.

### Task 2: Noise end to end
`run_bin` seeds the stream; verify and occupied-slot checks hold under noise. Tests: 30 noisy looks at an empty bin are all empty, at one 20 mm part never empty; 30 noisy looks at an empty tray see no occupied slot; a run is repeatable.

### Task 3: Edge margin
`edge_margin` 0.002 → value chosen on dev bins from {0, 0.5, 1, 2} mm. Test: a 46 mm box face is pickable.

### Task 4: Safe picks first
`GraspTarget.safe: bool`, `crowd: float`. Outline share with something taller within `crowd_reach`; cup within `safe_offset` of the surface center; tilt under `safe_tilt_deg`. Order: safe before risky, then flat before curved, then score. `run_bin`: no safe target → shake (max 3 in a row), then take the best risky one; nothing at all after 3 shakes → `no_candidate`. Tests: ordering; a bin with only risky targets shakes first and then picks; run ends.

### Task 5: Load-watched lift
The arm's own torques (command vs reading) give the load on the cup. Start the lift slowly, speed up when it runs free; yield sideways/tilt to the load within limits; give up (`seal_lost_lift` path, but with the vacuum still on: lower, release) when stuck. Kept only if it beats Task 4 alone on dev bins.

### Task 6: Collision checks and recovery
Watched hover-to-near line (stop on contact); planned lines keep arm links clear of the bin walls; `_recover` goes straight up or stops, never a joint move from inside the bin. Tests for each.

### Task 7: Failure detection
Two parts lifted (camera after lift is not available: tray camera sees two) or part dropped in transport: robot's verdict matches the referee in ≥ 95% of attempts. Test on recorded runs; report disagreement.

### Task 8: Dev bins, fresh bins, noise sweep (0, 0.5, 1, 2 mm), fresh reviewer, fix pass, checkpoint.

## Outcome (2026-11-25)
Done: Tasks 1–4, 6, the lift of Task 5 (stiffer and away from walls; yielding to the load was tried three ways and was worse), the measurements and the review fix pass. Added: narrow faces and short cylinders, guarded approach, harder shakes, "face barely wider than the cup is risky".
Not built: giving up a stuck lift before the seal breaks (Task 5), detecting two lifted parts and a standing test of the robot's verdict against the referee (Task 7; measured instead: 0 disagreements in 1,266 rules attempts on the three 20-bin runs), checks of transport and retreat paths (Task 6; no arm hits seen).
