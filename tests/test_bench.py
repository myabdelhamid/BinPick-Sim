import csv
import json
import sys

import pytest

from binpick import cli
from binpick.benchmark import runner


def test_smoke_benchmark_two_seeds(tmp_path):
    """What CI runs: both planners on two dev bins, a few attempts each."""
    out = tmp_path / "bench"
    assert cli.bench_main(["--seeds", "1000-1001", "--max-attempts", "2", "--workers", "2", "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out / "attempts.csv")))
    assert {(r["planner"], r["seed"]) for r in rows} == {(p, s) for p in ("rules", "random") for s in ("1000", "1001")}
    assert all(r["outcome"] in ("success", "fail") and float(r["cycle_time_s"]) > 0 for r in rows)
    bins = list(csv.DictReader(open(out / "bins.csv")))
    assert len(bins) == 4 and all(b["error"] == "" for b in bins)
    assert len(list(csv.DictReader(open(out / "parts.csv")))) == 4 * 20
    meta = json.loads((out / "meta.json").read_text())
    assert meta["seeds"] == [1000, 1001] and len(meta["commit"]) == 40 and meta["max_attempts"] == 2
    text = (out / "report.md").read_text()
    assert "stopped after 2 attempts" in text  # not a full benchmark, and says so
    assert (out / "summary.json").exists() and (out / "config.yaml").exists()
    with pytest.raises(SystemExit, match="already exists"):
        cli.bench_main(["--seeds", "1000", "--out", str(out)])


def test_results_do_not_depend_on_the_number_of_workers(tmp_path):
    texts = []
    for w in (1, 3):
        out = tmp_path / f"w{w}"
        runner.run_benchmark([1000, 1001], ["random"], out, workers=w, max_attempts=2)
        texts.append((out / "attempts.csv").read_text() + (out / "parts.csv").read_text())
    assert texts[0] == texts[1]


def test_a_crashing_bin_is_reported_and_counted(tmp_path, monkeypatch):
    real = runner.bin_command
    monkeypatch.setattr(runner, "bin_command", lambda seed, *a: [sys.executable, "-c", "import os; os._exit(9)"] if seed == 1001 else real(seed, *a))
    out = tmp_path / "bench"
    assert cli.bench_main(["--seeds", "1000-1001", "--planners", "random", "--max-attempts", "1", "--out", str(out)]) == 1
    bins = {b["seed"]: b for b in csv.DictReader(open(out / "bins.csv"))}
    assert bins["1001"]["stop"] == "error" and "exit code" in bins["1001"]["error"] and bins["1001"]["emptied"] == "0"
    assert len(list(csv.DictReader(open(out / "parts.csv")))) == 40
    s = json.loads((out / "summary.json").read_text())["random"]
    assert s["bins"] == 2 and s["errors"] == [1001] and s["parts"] == 40


def test_a_hanging_bin_is_stopped_and_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "bin_command", lambda *a: [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(runner, "BIN_TIMEOUT_S", 2.0)
    out = tmp_path / "bench"
    errors = runner.run_benchmark([1001], ["random"], out, workers=1, max_attempts=1)
    assert len(errors) == 1 and "timed out" in errors[0]


def test_overwrite_leaves_nothing_of_the_old_run(tmp_path):
    out = tmp_path / "bench"
    out.mkdir()
    (out / "bins").mkdir()
    (out / "attempts.csv").write_text("old")
    (out / "bins" / "rules_7.json").write_text("{}")
    runner.run_benchmark([1000], ["random"], out, workers=1, max_attempts=1, overwrite=True)
    assert {r["seed"] for r in csv.DictReader(open(out / "attempts.csv"))} == {"1000"}


def test_overwrite_refuses_a_folder_that_is_not_a_benchmark(tmp_path):
    (tmp_path / "thesis.docx").write_text("precious")
    with pytest.raises(SystemExit, match="not a benchmark folder"):
        cli.bench_main(["--seeds", "1000", "--out", str(tmp_path), "--overwrite"])
    assert (tmp_path / "thesis.docx").exists()


def test_reserved_seeds_need_a_clean_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "git_state", lambda: ("0" * 40, True))
    with pytest.raises(SystemExit, match="uncommitted"):
        cli.bench_main(["--seeds", "1-2", "--out", str(tmp_path / "b")])


@pytest.mark.parametrize("text,want", [("1-3", [1, 2, 3]), ("5", [5]), ("1,4-5", [1, 4, 5])])
def test_seed_lists(text, want):
    assert runner.parse_seeds(text) == want
