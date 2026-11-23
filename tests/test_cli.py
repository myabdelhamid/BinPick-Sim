import json

from binpick.cli import scene_main

FILES = ["rgb.png", "depth.npy", "depth_mm.png", "depth_preview.png", "tray_rgb.png", "overview.png", "parts.json"]


def test_cli_writes_outputs_and_is_repeatable(tmp_path):
    texts = []
    for _ in range(2):
        assert scene_main(["--seed", "1000", "--out", str(tmp_path)]) == 0
        texts.append((tmp_path / "scene_seed1000" / "parts.json").read_text())
    out = tmp_path / "scene_seed1000"
    for f in FILES:
        assert (out / f).exists(), f
    assert texts[0] == texts[1]
    j = json.loads(texts[0])
    assert j["seed"] == 1000 and len(j["parts"]) == 20
    assert {"id", "shape", "material", "dims", "mass", "pos", "quat"} <= j["parts"][0].keys()


def test_demo_writes_csv_and_video(tmp_path):
    import csv

    from binpick.cli import demo_main

    assert demo_main(["--seed", "1000", "--picks", "1", "--video", "--out", str(tmp_path)]) == 0
    out = tmp_path / "demo_seed1000"
    rows = list(csv.DictReader(open(out / "attempts.csv")))
    assert len(rows) == 1 and rows[0]["outcome"] in ("success", "fail")
    assert (out / "demo.mp4").stat().st_size > 10_000
