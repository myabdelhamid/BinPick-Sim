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
