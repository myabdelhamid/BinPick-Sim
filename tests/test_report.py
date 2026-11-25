import csv
import json

import pytest

from binpick.benchmark.report import make_report
from binpick.benchmark.runner import BIN_COLUMNS, PART_COLUMNS, grippable
from binpick.sim.parts import PartSpec

ATTEMPT = ["planner", "seed", "attempt", "target_part", "gripped_part", "outcome", "reason", "detected", "cycle_time_s", "safe", "arm_hit_n"]


def write(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def bin_row(planner, seed, **kw):
    return {"planner": planner, "seed": seed, "stop": "empty", "emptied": 1, "placed": 2, "left": 0, "lost": 0, "misplaced": 0, "shakes": 1,
            "hard_shakes": 0, "attempts": 2, "successes": 2, "parts": 2, "sim_time_s": 30.0, "wall_s": 1.0, "error": "", **kw}


def part_row(planner, seed, pid, **kw):
    return {"planner": planner, "seed": seed, "part": pid, "shape": "box", "material": "plastic", "dims_mm": "80x60x40", "mass_kg": 0.2,
            "state": "placed", "targeted": 1, "grippable": "any", **kw}


def attempt(planner, seed, k, outcome="success", reason="", cycle=7.0, **kw):
    return {"planner": planner, "seed": seed, "attempt": k, "target_part": 0, "gripped_part": 0, "outcome": outcome, "reason": reason,
            "detected": 1, "cycle_time_s": cycle, "safe": 1, "arm_hit_n": 0, **kw}


@pytest.fixture()
def folder(tmp_path):
    """rules: seed 1 empties (2/2), seed 2 places 1 of 2 in 3 attempts (one of them knocked out later), seed 3 crashed. random: nothing works."""
    write(tmp_path / "attempts.csv", ATTEMPT, [
        attempt("rules", 1, 1, cycle=6.0), attempt("rules", 1, 2, cycle=9.0),
        attempt("rules", 2, 1, cycle=7.5), attempt("rules", 2, 2, "success", cycle=8.5), attempt("rules", 2, 3, "fail", "collision", 5.0, safe=0, detected=0),
        attempt("random", 1, 1, "fail", "no_seal", 3.0), attempt("random", 2, 1, "fail", "no_seal", 3.0), attempt("random", 3, 1, "fail", "collision", 3.0)])
    write(tmp_path / "bins.csv", BIN_COLUMNS, [
        bin_row("rules", 1), bin_row("rules", 2, emptied=0, placed=1, left=1, stop="no_candidate", attempts=3, shakes=4, hard_shakes=2, sim_time_s=45.0),
        bin_row("rules", 3, stop="error", emptied=0, placed=0, left=2, attempts=0, successes=0, shakes=0, sim_time_s=0.0, error="exit code 1: boom"),
        *[bin_row("random", k, stop="attempt_limit", emptied=0, placed=0, left=2, attempts=1, successes=0, sim_time_s=9.0) for k in (1, 2, 3)]])
    write(tmp_path / "parts.csv", PART_COLUMNS, [
        *[part_row("rules", 1, p) for p in (0, 1)], part_row("rules", 2, 0),
        part_row("rules", 2, 1, shape="cylinder", dims_mm="42x41", state="left", targeted=0, grippable="never"),
        *[part_row("rules", 3, p, state="not run", targeted=0) for p in (0, 1)],
        *[part_row("random", s, p, state="left", targeted=int(p == 0)) for s in (1, 2, 3) for p in (0, 1)]])
    (tmp_path / "meta.json").write_text(json.dumps({"seeds": [1, 2, 3], "planners": ["rules", "random"], "commit": "abc", "dirty": False,
                                                    "max_attempts": None, "config_sha256": "x", "platform": "p", "versions": {}, "workers": 2,
                                                    "started": "t", "wall_s": 1.0, "command": "c"}))
    return tmp_path


def test_numbers_come_straight_from_the_files(folder):
    _, s = make_report(folder)
    r = s["rules"]
    assert r["attempts"] == 5 and r["successes"] == 4 and r["success_rate"] == pytest.approx(0.8)
    assert r["bins"] == 3 and r["emptied"] == 1 and r["emptying_rate"] == pytest.approx(1 / 3)  # the crashed bin counts
    assert r["placed"] == 3 and r["parts"] == 6 and r["knocked_out"] == 1
    assert r["cycle_mean_s"] == pytest.approx((6 + 9 + 7.5 + 8.5) / 4) and r["cycle_max_s"] == 9.0 and r["over_8s"] == 2
    assert r["cycle_all_mean_s"] == pytest.approx((6 + 9 + 7.5 + 8.5 + 5) / 5) and r["cycle_all_max_s"] == 9.0
    assert r["sim_time_per_placed_s"] == pytest.approx(75 / 3)
    assert r["failures"] == {"collision": 1} and r["errors"] == [3] and r["risky_attempts"] == 1
    assert r["verdict_wrong"] == 1 and r["arm_hits"] == 0
    assert r["left_over"] == {"no side the cup can seal on": 1, "bin did not run": 2}


def test_a_planner_without_any_success_is_reported(folder):
    text, s = make_report(folder)
    r = s["random"]
    assert r["successes"] == 0 and r["cycle_mean_s"] is None and r["sim_time_per_placed_s"] is None
    assert r["failures"] == {"no_seal": 2, "collision": 1}
    assert "nan" not in text.lower() and "none" not in text.lower()


def test_report_says_what_the_owner_requires(folder):
    text, _ = make_report(folder)
    for must in ("1.75 kg", "no manufacturer part weights", "ERROR", "boom", "Zero overlap", "abc", "missed", "seed 3"):
        assert must in text, must
    assert "| Pick success per attempt | ≥ 95% |" in text


def test_range_is_repeatable_and_contains_the_rate(folder):
    a, b = make_report(folder)[1]["rules"], make_report(folder)[1]["rules"]
    assert a["success_range"] == b["success_range"]
    lo, hi = a["success_range"]
    assert 0 <= lo <= a["success_rate"] <= hi <= 1


def test_mismatched_files_are_refused(folder):
    rows = list(csv.DictReader(open(folder / "attempts.csv")))
    write(folder / "attempts.csv", ATTEMPT, rows[1:])  # one attempt missing
    with pytest.raises(ValueError, match="rules seed 1"):
        make_report(folder)


@pytest.mark.parametrize("shape,dims,want", [
    ("box", (0.08, 0.06, 0.02), "any"), ("box", (0.10, 0.0435, 0.030), "never"), ("box", (0.10, 0.0435, 0.06), "any"),
    ("cylinder", (0.05, 0.08), "any"), ("cylinder", (0.042, 0.041), "never"), ("cylinder", (0.06, 0.042), "end"), ("cylinder", (0.042, 0.06), "any")])
def test_grippable(cfg, shape, dims, want):
    assert grippable(PartSpec(0, shape, "plastic", dims, 0.2), cfg) == want
