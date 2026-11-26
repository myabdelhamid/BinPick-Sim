import json
import re

from binpick import cli
from binpick.benchmark.report import make_report


def page_data(path):
    return json.loads(re.search(r"const DATA = (\{.*?\});\n", path.read_text()).group(1))


def test_dashboard_shows_the_report_numbers(tmp_path):
    out = tmp_path / "d.html"
    assert cli.dashboard_main(["--out", str(out)]) == 0
    data = page_data(out)
    assert [r["label"] for r in data["runs"]] == [r[1] for r in cli.DEFAULT_RUNS]
    for run, (folder, *_) in zip(data["runs"], cli.DEFAULT_RUNS):
        want = make_report(folder)[1]
        for p, s in run["planners"].items():
            assert (s["successes"], s["attempts"], s["emptied"], s["placed"]) == \
                (want[p]["successes"], want[p]["attempts"], want[p]["emptied"], want[p]["placed"])
            assert len(run["cycles"][p]) == want[p]["successes"]
        assert len([b for b in run["bins"] if b["planner"] == "rules"]) == 25
        assert len(run["left"]) == want["rules"]["parts"] - want["rules"]["placed"]


def test_dashboard_is_self_contained_and_escapes_its_data(tmp_path):
    out = tmp_path / "d.html"
    cli.dashboard_main(["--out", str(out)])
    text = out.read_text()
    assert "http://" not in text.replace("http://www.w3.org/2000/svg", "") and "https://" not in text
    assert "/*DATA*/" not in text and "</script>" in text
    assert text.count("</script>") == 1  # the data cannot close the script early


def test_a_hostile_label_cannot_break_out_of_the_script(tmp_path):
    from binpick.benchmark.dashboard import write_dashboard

    folder = cli.DEFAULT_RUNS[0][0]
    out = write_dashboard([(folder, "<!--<script>alert(1)</script>", "</script><b>")], tmp_path / "d.html")
    text = out.read_text()
    assert text.count("<script>") == 1 and text.count("</script>") == 1 and "<!--" not in text
    assert page_data(out)["runs"][0]["label"] == "<!--<script>alert(1)</script>"


def test_the_delivered_documents_quote_the_benchmark_numbers():
    """README and RESULTS are written by hand: their headline numbers must still match the run folders."""
    from pathlib import Path

    root = Path(cli.__file__).resolve().parents[2]
    readme, results = (root / "README.md").read_text(), (root / "docs" / "RESULTS.md").read_text()
    for folder, *_ in cli.DEFAULT_RUNS:
        s = make_report(folder)[1]
        r, q = s["rules"], s["random"]
        for text in (readme, results):
            for want in (f"{100 * r['success_rate']:.1f}%", f"{r['emptied']} / {r['bins']}", f"{r['cycle_mean_s']:.2f} s",
                         f"{100 * q['success_rate']:.1f}%"):
                assert want in text, (folder, want)
        assert f"{r['successes']} / {r['attempts']}" in results, folder
