# Milestone 3 (Perception + Planning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (owner chose native execution). Steps use checkbox (`- [ ]`) syntax.

**Goal:** The cell picks from what its camera sees, with no part poses from the simulator: depth image → point cloud → suction points → seal check → score → pick, until the first full bin is emptied into the tray. A random baseline planner runs through the same loop.

**Architecture:** `perception/view.py` turns one depth image into a point cloud (Open3D) and a 1 mm top-down height map with surface labels. `planning/rules.py` and `planning/random_pick.py` return ranked `GraspTarget`s from that view. The supervisor takes a propose-function, feels for contact on place (part height is not visible from above), and `supervisor/run.py` loops one bin with shakes.

**Tech Stack:** as M2, plus Open3D and scipy.ndimage.

**Spec:** `docs/superpowers/specs/2026-11-23-binpick-sim-design.md` (§3 metrics, §7 test freeze, §8 pick rules)

## Findings that shaped the plan (measured 2026-11-24, seed 1000)
- Depth error vs ray casts with pixel centers at +0.5: p90 0.005 mm. The ARB_clip_control warning is harmless. `verify_slot` ignores the half pixel (about 0.5 mm): fixed in Task 1.
- Arm pose with the cup at (0.39, 0.22, 0.45) appears in neither camera. It replaces the ready pose (which blocked the bin camera) and is used for capture and verify.

## Global Constraints
- Test freeze (§7): no changes to parts, cup or contact physics.
- Supervisor, perception and rules/random planners read only: camera images, vacuum sensor, the arm's own joint positions and commands. `planning/oracle.py`, the seal physics and the referee keep ground truth.
- Dev seeds ≥ 1000 only. Seeds 1–25 are never run here.
- Every new setting goes in `configs/default.yaml` (`perception`, `planner`, `run`).
- Depth noise is M4, but nothing here may depend on noise-free depth: smoothing and margins are sized in mm, not micrometres.

## Review Focus
1. Two parts side by side at the same height: one surface label covering both puts the estimated center on the seam.
2. A part partly hidden under another: the visible patch's centroid is not the part's center, so it is placed off-center.
3. Part height unknown: it must not be dropped from height or pressed hard into the tray.
4. The arm in the bin camera's view at capture time after a failure.
5. A bin the camera sees as empty while a thin part (20 mm) remains, and the reverse.

---
### Task 1: Point cloud and height map
**Files:** create `src/binpick/perception/view.py`; modify `supervisor/placement.py` (half pixel), `config.py`, `configs/default.yaml`; test `tests/test_view.py`
**Produces:**
- `depth_to_cloud(depth, K, cam_pos, cam_R) -> (H, W, 3)` world points (Open3D, pixel centers at +0.5)
- `BinView(z, origin, cell, labels, edges, normals, lo, hi)`; `observe_bin(scene) -> BinView`; `BinView.is_empty() -> bool`
Tests: cloud points lie within 1 mm of ray-cast truth (median); a flat box's top gives one label whose centroid is within 2 mm of its center; two boxes 4 mm apart give two labels; a lying 60 mm cylinder gives one label; empty bin → `is_empty()`, one 20 mm thin box → not empty.

### Task 2: Rules planner
**Files:** create `src/binpick/planning/rules.py`, `planning/types.py` (GraspTarget moves here; `part_index: int | None`, new `center`, `seal`, `score`); test `tests/test_rules.py`
**Produces:** `rules_targets(view, cfg) -> list[GraspTarget]` best first.
Seal check on the height map: wrapped lip (16 directions), plane fit to the points under the cup, reject tilt > 20°, lip off the surface / over an edge / unknown, lip drop beyond the B40 curve limit; cup and tool must clear the walls and anything taller within 24 mm. Score: height, seal, wall clearance, distance to surface center; flat tier before curved tier.
Tests: flat box → top center ±3 mm; 25° box → no target; 40 mm lying cylinder → target with normal up (not rejected for being curved); flat box ranked before a taller cylinder; box against the wall with < 22 mm clearance at its edge → no target near the wall; box under a taller neighbour's overhang → neighbour first; long axis within 3°.

### Task 3: Random baseline
**Files:** create `planning/random_pick.py`; test `tests/test_random_pick.py`
`random_targets(view, cfg, rng, n=50)`: uniformly random visible part points, approach along the local surface normal, no filters or scoring. Tests: all points on parts, deterministic per rng, includes tilted/edge points.

### Task 4: Supervisor on perception
**Files:** modify `supervisor/machine.py`, `motion/trajectory.py`, `supervisor/records.py`; test `tests/test_supervisor.py`, `tests/test_contact.py`
- `attempt(target_or_propose)`: capture at the view pose, plan once, take the first feasible target (rejected ones logged; their time is charged to this attempt).
- Contact-sensed place: descend, watching tip lag from joint commands vs readings (no new sensor); stop on contact; release. Start height from the largest depth the part can have (distance to the bin floor along the normal, ≤ 110 mm).
- Place offset: cup offset from the surface center is subtracted so the part's center lands on the slot center.
- After a failure, return to the view pose. Record `target_part` by ray (log only).
Tests: view pose is in neither camera; unknown-height box is placed with < 15 N contact and < 10 mm drop; grasp 12 mm off-center still lands within ±10 mm.

### Task 5: Bin loop and command
**Files:** create `supervisor/run.py`, modify `cli.py`, `pyproject.toml` (`binpick-run --seed --planner --video`); test `tests/test_run.py`
`run_bin(scene, planner, seed) -> BinResult`: ends when the camera sees an empty bin, the tray is full, no candidate after 3 shakes in a row, or 40 attempts. Shake: arm at view pose, oscillate shaker, settle. CSV, events, summary.

### Task 6: Dev bins, final review, checkpoint
Run seeds 1000–1009, both planners, all reported. Video of the first fully emptied bin. Fresh reviewer, fix pass, checkpoint.
