"""Two-minute demo video: what the robot sees, the rules planner emptying a bin it was never tuned on, the random
baseline on the same bin, and the benchmark results. Usage:
    uv run python scripts/demo_video.py [OUT.mp4]
Numbers on the cards come from the filmed run, which is checked against the benchmark's record of the same bin, and from the
benchmark folders in docs/checkpoints."""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"  # as the benchmark runs its bins, so the filmed run is the benchmark's run

import csv
import sys
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from binpick.benchmark.report import make_report
from binpick.cli import depth_preview
from binpick.config import load_config
from binpick.perception.view import depth_to_cloud, observe_bin
from binpick.planning.rules import rules_targets
from binpick.sim.camera import intrinsics, render, sense_depth
from binpick.sim.settle import make_settled_scene
from binpick.supervisor.run import run_bin

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 30
SEED = 1240                 # first of the unused seeds: picked by its number, not by how it went
RULES_SPEED, RANDOM_SPEED, RANDOM_SIM_S = 4.0, 4.0, 64.0
BG, INK, INK2, BLUE, ORANGE, GOOD, BAD = (252, 252, 251), (11, 11, 11), (82, 81, 78), (42, 120, 214), (235, 104, 52), (0, 99, 0), (180, 47, 47)
RUNS = [("docs/checkpoints/m6/seeds_1240-1264", "Unused seeds 1240–1264 (blind)"), ("docs/checkpoints/m6/seeds_1-25", "Seeds 1–25, final code (rerun)"),
        ("docs/checkpoints/m5", "Seeds 1–25, first run (blind)")]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path, index in (("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0), ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0)):
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Video:
    def __init__(self, path: Path):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
        self.frames = 0

    def add(self, img: Image.Image, seconds: float = 1 / FPS) -> None:
        a = np.asarray(img.convert("RGB"))
        for _ in range(max(1, round(seconds * FPS))):
            self.w.append_data(a)
            self.frames += 1

    def close(self) -> None:
        self.w.close()


def card(title: str, lines: list[str], foot: str = "") -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 100
    for part in textwrap.wrap(title, 40):
        d.text((90, y), part, font=font(48, True), fill=INK)
        y += 60
    y += 40
    for line in lines:
        for k, part in enumerate(textwrap.wrap(line, 62) or [""]):
            d.text((90 if k == 0 else 118, y), ("• " if k == 0 and line else "  ") + part if line else "", font=font(30), fill=INK2)
            y += 44
        y += 10
    if foot:
        d.text((90, H - 80), foot, font=font(22), fill=INK2)
    return img


def caption(img: Image.Image, left: str, right: str = "") -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle((0, H - 64, W, H), fill=(252, 252, 251))
    d.text((24, H - 48), left, font=font(26, True), fill=INK)
    if right:
        d.text((W - 24 - d.textlength(right, font=font(26)), H - 48), right, font=font(26), fill=INK2)
    return img


def what_it_sees(cfg) -> Image.Image:
    scene, _ = make_settled_scene(cfg, SEED)
    scene.sensor_rng = np.random.default_rng([SEED, 5])
    rgb = render(scene, "bin_cam")[0]
    depth = sense_depth(scene, "bin_cam")
    view = observe_bin(scene)
    targets = rules_targets(view, cfg)
    # the planner's top-down map: each surface its own shade, chosen pick points on top (green safe, orange risky)
    rng = np.random.default_rng(0)
    shades = (rng.uniform(120, 230, (view.labels.max() + 1, 3))).astype(np.uint8)
    shades[0] = (235, 235, 232)
    # drawn as the camera sees it: every pixel takes the surface of the map cell it falls in
    cam = scene.data.cam("bin_cam")
    K, pos, R = intrinsics(scene, "bin_cam"), cam.xpos.copy(), cam.xmat.reshape(3, 3).copy()
    P = depth_to_cloud(np.nan_to_num(depth, nan=0.0), K, pos, R)
    ij = np.round(np.nan_to_num((P[..., :2] - view.origin) / view.cell, nan=-1.0)).astype(int)
    inside = np.all((ij >= 0) & (ij < np.array(view.labels.shape)), axis=-1) & np.isfinite(depth)
    lab = np.full(depth.shape + (3,), 235, np.uint8)
    lab[~np.isfinite(depth)] = 40  # no reading
    lab[inside] = shades[view.labels[ij[inside][:, 0], ij[inside][:, 1]]]
    top = Image.fromarray(lab)
    d = ImageDraw.Draw(top)
    for t in targets:
        c = R.T @ (t.pos - pos)                     # camera frame: x right, y up, looking along -z
        u, v = K[0, 0] * c[0] / -c[2] + K[0, 2], K[1, 1] * -c[1] / -c[2] + K[1, 2]
        d.ellipse((u - 9, v - 9, u + 9, v + 9), fill=GOOD if t.safe else ORANGE, outline=(255, 255, 255), width=3)
    filled = np.nan_to_num(depth, nan=np.nanmax(depth))
    panels = [("Colour image (for people)", Image.fromarray(rgb)), ("Depth, with noise", Image.fromarray(depth_preview(filled))),
              ("Surfaces and pick points", top)]
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 30), "What the robot sees", font=font(40, True), fill=INK)
    d.text((40, 84), "It uses only the depth image: smooth surfaces, then where a 44 mm cup can seal (green: safe pick, orange: risky).",
           font=font(22), fill=INK2)
    for k, (name, p) in enumerate(panels):
        p = p.copy()
        p.thumbnail((392, 420))
        x = 40 + k * 408
        img.paste(p, (x, 150))
        d.text((x, 150 + p.height + 12), name, font=font(24, True), fill=INK)
    return img


def film_run(video: Video, cfg, planner: str, speed: float, until_s: float | None, label: str) -> dict:
    scene, _ = make_settled_scene(cfg, SEED)
    renderer = mujoco.Renderer(scene.model, height=H, width=W)
    state = {"next": scene.data.time, "t0": scene.data.time}

    class Stop(Exception):
        pass

    def observe(sc, phase, sup):
        t = sc.data.time - state["t0"]
        if until_s is not None and t > until_s:
            raise Stop
        if sc.data.time + 1e-9 < state["next"]:
            return
        state["next"] += speed / FPS
        recs = sup.records
        ok = sum(r.outcome == "success" for r in recs)
        last = ""
        if recs and recs[-1].outcome in ("success", "fail") and phase in ("capture", "plan", "shake", "idle"):
            r = recs[-1]
            last = "last: placed" if r.outcome == "success" else f"last: failed ({r.reason.replace('_', ' ')})"
        renderer.update_scene(sc.data, camera="overview_cam")
        img = Image.fromarray(renderer.render())
        right = f"attempt {len(recs)}   ·   {ok} placed   ·   {phase}   {last}"
        caption(img, f"{label}   ·   {speed:.0f}× speed", right)
        video.add(img)

    result = {}
    try:
        res = run_bin(scene, planner, SEED, observer=observe)
        result = res.summary()
    except Stop:
        pass
    renderer.close()
    return result


def results_card() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 40), "Benchmark: 25 bins × 20 parts, both planners on the same bins", font=font(36, True), fill=INK)
    head = ["", "Success per attempt", "Bins emptied", "Mean cycle", "Random baseline"]
    xs = [60, 450, 700, 880, 1050]
    for x, h in zip(xs, head):
        d.text((x, 130), h, font=font(22, True), fill=INK2)
    y = 180
    for folder, name in RUNS:
        s = make_report(ROOT / folder)[1]
        r, q = s["rules"], s["random"]
        cells = [name, f"{100 * r['success_rate']:.1f}%", f"{r['emptied']} / {r['bins']}", f"{r['cycle_mean_s']:.2f} s", f"{100 * q['success_rate']:.1f}%"]
        for k, (x, c) in enumerate(zip(xs, cells)):
            color = INK
            if k == 1:
                color = GOOD if r["success_rate"] >= 0.95 else BAD
            if k == 2:
                color = GOOD if r["emptying_rate"] >= 0.95 else BAD
            if k == 3:
                color = GOOD if r["cycle_mean_s"] <= 8 else BAD
            d.text((x, y), c, font=font(30 if k else 26, k > 0), fill=color)
            if color != INK:
                d.text((x, y + 38), "met" if color == GOOD else "missed", font=font(18), fill=color)
        y += 90
    d.text((60, y + 10), "Targets: 95% success per attempt, 95% of bins emptied, 8 s mean cycle.", font=font(24), fill=INK2)
    d.text((60, y + 50), "The unused seeds are the fairest test: nobody had looked at them before the final run.", font=font(24), fill=INK2)
    return img


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "demo.mp4"
    cfg = load_config()
    v = Video(out)
    v.add(card("BinPick-Sim", ["A simulated Franka Panda with a 44 mm suction cup empties a bin of 20 mixed parts into a 5 × 4 tray.",
                                "It only uses what a real cell senses: a noisy depth camera, a vacuum sensor, its own joints.",
                                "Every attempt counts. A pick succeeds only if the part rests within 10 mm of its slot's center."],
               "Simulation (MuJoCo). Seed 1240: the first of 25 seeds nobody had looked at before the final benchmark."), 7)
    v.add(what_it_sees(cfg), 10)
    rules = film_run(v, cfg, "rules", RULES_SPEED, None, f"Rules planner · seed {SEED}")
    bench = ROOT / RUNS[0][0]
    row = next(b for b in csv.DictReader(open(bench / "bins.csv")) if b["planner"] == "rules" and int(b["seed"]) == SEED)
    assert (rules["placed"], rules["attempts"], rules["shakes"]) == (int(row["placed"]), int(row["attempts"]), int(row["shakes"])), \
        "the filmed run differs from the benchmark's run of this bin"
    why = {"never": "fits the 44 mm cup on no side", "end": "fits the 44 mm cup only standing on an end"}
    left = [p for p in csv.DictReader(open(bench / "parts.csv")) if p["planner"] == "rules" and int(p["seed"]) == SEED and p["state"] != "placed"]
    v.add(card(f"Rules planner: {rules['placed']} of 20 placed, {rules['successes']} of {rules['attempts']} attempts succeeded",
               [f"Mean cycle {rules['cycle_mean_s']:.1f} s per successful pick, {rules['shakes']} shakes.",
                *[f"Left over: a {p['shape']} of {p['dims_mm']} mm, which {why[p['grippable']]}." if p["grippable"] in why
                  else f"Left over: a {p['shape']} of {p['dims_mm']} mm ({p['state']})." for p in left],
                "Next: the random baseline on the same bin. Same camera, arm and placing; only the pick point is random."]), 7)
    film_run(v, cfg, "random", RANDOM_SPEED, RANDOM_SIM_S, f"Random · seed {SEED} · first {RANDOM_SIM_S:.0f} s shown")
    v.add(results_card(), 14)
    v.add(card("What limits it", ["Parts wedged between neighbours or against a wall: lifting tears the grip off.",
                                  "Short cylinders: the 44 mm cup cannot seal on their side. In the final runs, every bin not emptied had one.",
                                  "One part torn off 11 times in one bin: the planner does not stop retrying a part that keeps failing.",
                                  "It is a simulation: the cup's hold when tilted, the camera noise and the part weights are assumptions."],
               "Results, dashboard and every logged attempt: docs/RESULTS.md, docs/dashboard.html, docs/checkpoints/"), 9)
    v.add(card("Reproduce", ["uv run binpick-bench   (seeds 1–25, both planners, about 30 minutes)",
                             "uv run binpick-bench --seeds 1240-1264 --out outputs/unused",
                             "uv run binpick-dashboard"]), 5)
    v.close()
    print(f"wrote {out}: {v.frames / FPS:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
