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
        runner.run_benchmark([1000, 1001], ["rules", "random"], out, workers=w, max_attempts=2)
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
    assert [x["stop"] for x in csv.DictReader(open(out / "bins.csv"))] == ["error"]


def test_overwrite_leaves_nothing_of_the_old_run(tmp_path):
    out = tmp_path / "bench"
    out.mkdir()
    (out / "bins").mkdir()
    (out / runner.MARKER).write_text("")
    (out / "attempts.csv").write_text("old")
    (out / "bins" / "rules_7.json").write_text("{}")
    runner.run_benchmark([1000], ["random"], out, workers=1, max_attempts=1, overwrite=True)
    assert {r["seed"] for r in csv.DictReader(open(out / "attempts.csv"))} == {"1000"}
    assert not (out / "bins" / "rules_7.json").exists()


def test_overwrite_refuses_a_folder_that_is_not_a_benchmark(tmp_path):
    (tmp_path / "thesis.docx").write_text("precious")
    with pytest.raises(SystemExit, match="not a benchmark folder"):
        cli.bench_main(["--seeds", "1000", "--out", str(tmp_path), "--overwrite"])
    assert (tmp_path / "thesis.docx").exists()


def test_reserved_seeds_need_a_clean_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "git_state", lambda: ("0" * 40, True))
    monkeypatch.setattr(runner, "_run_process", lambda job: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="uncommitted"):
        runner.run_benchmark([1, 2], ["random"], tmp_path / "b", workers=1)
    assert not (tmp_path / "b").exists()


@pytest.mark.parametrize("kw,match", [
    ({"seeds": [1000, 1000]}, "twice"), ({"seeds": []}, "no seeds"), ({"planners": ["rules", "rules"]}, "twice"), ({"planners": ["x"]}, "unknown planner"),
    ({"workers": 0}, "workers"), ({"config": "missing.yaml"}, "missing.yaml")])
def test_bad_input_is_refused_before_anything_is_deleted(tmp_path, kw, match):
    out = tmp_path / "bench"
    (out / "bins").mkdir(parents=True)
    (out / runner.MARKER).write_text("")
    (out / "attempts.csv").write_text("old")
    args = {"seeds": [1000], "planners": ["random"], "out": out, "workers": 1, "overwrite": True} | kw
    with pytest.raises(SystemExit, match=match):
        runner.run_benchmark(**args)
    assert (out / "attempts.csv").read_text() == "old"


def test_overwrite_can_reuse_the_folders_own_settings(tmp_path):
    out = tmp_path / "bench"
    runner.run_benchmark([1000], ["random"], out, workers=1, max_attempts=1)
    before = (out / "attempts.csv").read_text()
    runner.run_benchmark([1000], ["random"], out, workers=1, max_attempts=1, config=out / "config.yaml", overwrite=True)
    assert (out / "attempts.csv").read_text() == before
    assert json.loads((out / "meta.json").read_text())["default_config"]


@pytest.mark.parametrize("files", [{"config.yaml": "other app"}, {"bins/precious.docx": "x", "meta.json": "{}"}])
def test_overwrite_refuses_lookalikes(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(text)
    with pytest.raises(SystemExit, match="not a benchmark folder"):
        runner.run_benchmark([1000], ["random"], tmp_path, workers=1, overwrite=True)
    assert all((tmp_path / name).read_text() == text for name, text in files.items())


def test_a_corrupt_result_costs_one_bin_only(tmp_path, monkeypatch):
    out = tmp_path / "bench"
    real = runner._run_process

    def corrupt(job):
        err = real(job)
        if job[0] == 1001:
            (out / "bins" / "random_1001.json").write_text("{half")
        return err

    monkeypatch.setattr(runner, "_run_process", corrupt)
    errors = runner.run_benchmark([1000, 1001], ["random"], out, workers=1, max_attempts=1)
    assert len(errors) == 1 and "1001" in errors[0]
    assert {x["seed"]: x["stop"] for x in csv.DictReader(open(out / "bins.csv"))}["1001"] == "error"


def test_code_changed_during_the_run_is_recorded(tmp_path, monkeypatch):
    states = iter([("a" * 40, False), ("a" * 40, True)])
    monkeypatch.setattr(runner, "git_state", lambda: next(states))
    runner.run_benchmark([1000], ["random"], tmp_path / "b", workers=1, max_attempts=1)
    assert json.loads((tmp_path / "b" / "meta.json").read_text())["dirty"]


@pytest.mark.parametrize("text,want", [("1-3", [1, 2, 3]), ("5", [5]), ("1,4-5", [1, 4, 5])])
def test_seed_lists(text, want):
    assert runner.parse_seeds(text) == want
