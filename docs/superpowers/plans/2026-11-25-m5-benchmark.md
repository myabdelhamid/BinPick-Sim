# Milestone 5 (Benchmark) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (owner chose native execution). Steps use checkbox (`- [ ]`) syntax.

**Goal:** One command runs both planners on seeds 1–25 (500 parts each), headless and in parallel, logs every attempt, and writes a report that can be reproduced.

**Architecture:** `benchmark/runner.py` starts one process per bin and gathers its files into CSVs. `benchmark/report.py` turns the CSVs into `report.md` and `summary.json`; it reads nothing else. `cli.bench_main` is the command.

**Spec:** §3 (metrics), §7 (test freeze, what the report must state), §10.5. **Owner, 2026-11-25:** run now with the 1.75 kg cap (real weights may come later: then rerun and report both); strict cycle time, nothing overlaps.

## Global Constraints
- Test freeze (§7). No robot changes in M5 except bugs that stop a bin from running, each reported.
- Seeds 1–25 are run once, by the final command, after the review. Development and review use seeds ≥ 1000.
- Every failure counts. A bin that cannot run is reported by name, counts as not emptied, and is in the denominator (25).
- Results do not depend on the number of workers. Reserved seeds run only from a clean git tree.

## Review Focus
1. A bin process that crashes or hangs: it must not vanish from the totals.
2. Numbers in the report that do not match the CSV (denominators, parts not attempts, knocked-out parts).
3. Same seeds, different worker count: identical CSVs.
4. Running it twice into the same folder: no mixing of old and new results.
5. A planner with no successes (no cycle times): the report must not crash or divide by zero.

### Task 1: Log what the report needs
`AttemptRecord.safe`, `crowd`, `felt_kg` (CSV columns); shake events carry `hard`; `BinResult.part_state: list[str]` ("placed" | "left" | "lost" | "misplaced"), `hard_shakes`. Tests in test_run/test_supervisor.

### Task 2: Runner
`run_benchmark(seeds, planners, out, workers, config, max_attempts=None) -> list[str] errors`; one process per bin with single-threaded maths; writes `attempts.csv` (with planner), `bins.csv`, `parts.csv`, `events.jsonl`, `meta.json` (commit, dirty, versions, machine, config and its hash, command, wall time), `config.yaml`. Refuses an existing folder without `--overwrite`, and reserved seeds from a dirty tree without `--allow-dirty`. Tests: crash of one bin reported and counted; workers 1 and 3 give identical CSVs.

### Task 3: Report
`make_report(folder) -> (markdown, summary dict)`: targets table, both planners side by side (success per attempt with a 95% range from resampling bins, bin-emptying rate, parts placed, cycle mean / median / slowest, share over 8 s, sim time per placed part including shakes and failures), failure breakdown, per-bin table, parts not placed and why (no side the cup can seal on, only standing on an end, tried and failed, never tried), robot's verdict vs referee, and the disclosures of §7. Tests on hand-made CSVs with known answers, including zero successes and a crashed bin.

### Task 4: Command, smoke test, README
`uv run binpick-bench` (defaults: seeds 1–25, both planners). Two-seed smoke test (seeds 1000–1001, 2 attempts each) in pytest, so also in CI.

### Task 5: Fresh review on dev seeds, fix pass, then the official run and the checkpoint.
