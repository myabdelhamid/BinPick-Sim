"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from binpick.config import load_config
from binpick.sim.camera import intrinsics, render
from binpick.sim.settle import make_settled_scene

# Colorized depth: dark blue (near) -> yellow (far), 5 anchors interpolated to 256 entries.
_ANCHORS = np.array([[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]], float)
_LUT = np.stack([np.interp(np.linspace(0, 4, 256), np.arange(5), _ANCHORS[:, c]) for c in range(3)], 1).astype(np.uint8)


def depth_preview(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(depth, [1, 99])
    idx = np.clip((depth - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return _LUT[idx]


def scene_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="binpick-scene", description="Build and settle a seeded bin, save images.")
    ap.add_argument("--seed", type=int, default=None, help="scene seed (default: config seed)")
    ap.add_argument("--config", default=None, help="path to YAML config (default: configs/default.yaml)")
    ap.add_argument("--out", default="outputs", help="output root directory")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    scene, poses = make_settled_scene(cfg, seed)
    out = Path(args.out) / f"scene_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    rgb, depth = render(scene, "bin_cam")
    Image.fromarray(rgb).save(out / "rgb.png")
    np.save(out / "depth.npy", depth)
    Image.fromarray(np.clip(depth * 1000, 0, 65535).astype(np.uint16)).save(out / "depth_mm.png")
    Image.fromarray(depth_preview(depth)).save(out / "depth_preview.png")
    Image.fromarray(render(scene, "tray_cam", depth=False)[0]).save(out / "tray_rgb.png")
    Image.fromarray(render(scene, "overview_cam", depth=False, size=(1280, 960))[0]).save(out / "overview.png")

    lo, hi = scene.bin_inner_bounds()
    meta = {
        "seed": seed,
        "bin_inner_bounds": [lo.tolist(), hi.tolist()],
        "slot_centers": scene.slot_centers.tolist(),
        "intrinsics": {c: intrinsics(scene, c).tolist() for c in ("bin_cam", "tray_cam")},
        "parts": [
            {"id": p.id, "shape": p.shape, "material": p.material, "dims": list(p.dims), "mass": p.mass,
             "pos": pose.pos.tolist(), "quat": pose.quat.tolist()}
            for p, pose in zip(scene.parts, poses)
        ],
    }
    (out / "parts.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")
    return 0


class _VideoRecorder:
    """Overview-camera frames at a fixed sim-time rate, with a status line burned in."""

    def __init__(self, scene, path: Path, fps: int = 30, size: tuple[int, int] = (960, 720)):
        import imageio.v2 as imageio
        import mujoco

        self.scene, self.fps, self.next_t = scene, fps, scene.data.time
        self.renderer = mujoco.Renderer(scene.model, height=size[1], width=size[0])
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
        self.status = ""

    def __call__(self, scene, phase: str, pressure_kpa: float) -> None:
        if scene.data.time + 1e-9 < self.next_t:
            return
        self.next_t += 1.0 / self.fps
        from PIL import Image, ImageDraw

        self.renderer.update_scene(scene.data, camera="overview_cam")
        img = Image.fromarray(self.renderer.render())
        ImageDraw.Draw(img).text(
            (12, 10), f"t={scene.data.time:6.2f}s  {self.status}  phase={phase:<9}  vacuum={pressure_kpa:6.1f} kPa",
            fill=(20, 20, 20))
        self.writer.append_data(np.asarray(img))

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


def demo_main(argv: list[str] | None = None) -> int:
    """M2 demo: pick parts from their known poses, place them in the tray, log and optionally film it."""
    from binpick.planning.oracle import oracle_targets
    from binpick.supervisor.machine import Supervisor
    from binpick.supervisor.records import write_csv

    ap = argparse.ArgumentParser(prog="binpick-demo", description="Known-pose pick-and-place demo.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--picks", type=int, default=3, help="number of attempts")
    ap.add_argument("--video", action="store_true", help="write demo.mp4 from the overview camera")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    scene, _ = make_settled_scene(cfg, seed)
    out = Path(args.out) / f"demo_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    rec = _VideoRecorder(scene, out / "demo.mp4") if args.video else None
    sup = Supervisor(scene, np.random.default_rng([seed, 2]), seed=seed,
                     observer=(lambda s, ph: rec(s, ph, sup.cup.pressure_display_kpa())) if rec else None)
    try:
        while len(sup.records) < args.picks:
            targets = oracle_targets(scene)
            if not targets:
                sup.events.append({"seed": seed, "time": scene.data.time, "event": "no_candidate"})
                break
            for t in targets:
                if rec:
                    rec.status = f"attempt {len(sup.records) + 1}: part {scene.parts[t.part_index].id}"
                if sup.attempt(t) is not None:
                    break
            else:
                sup.events.append({"seed": seed, "time": scene.data.time, "event": "no_reachable_candidate"})
                break
    finally:
        if rec:
            rec.close()

    write_csv(sup.records, out / "attempts.csv")
    (out / "events.json").write_text(json.dumps(sup.events, indent=2, default=float))
    for r in sup.records:
        print(f"attempt {r.attempt}: part {scene.parts[r.target_part].id} -> {r.outcome}"
              f"{'' if r.outcome == 'success' else ' (' + r.reason + ')'}  slot={r.slot}  cycle={r.cycle_time_s:.2f}s")
    ok = sum(r.outcome == "success" for r in sup.records)
    print(f"{ok}/{len(sup.records)} successful; wrote {out}")
    return 0


def run_main(argv: list[str] | None = None) -> int:
    """Run one bin from the camera image until it is empty (or the run ends), log and optionally film it."""
    from binpick.supervisor.records import write_csv
    from binpick.supervisor.run import PLANNERS, run_bin

    ap = argparse.ArgumentParser(prog="binpick-run", description="Empty one bin using the camera.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--planner", choices=PLANNERS, default="rules")
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--video", action="store_true", help="write run.mp4 from the overview camera")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    scene, _ = make_settled_scene(cfg, seed)
    out = Path(args.out) / f"run_seed{seed}_{args.planner}"
    out.mkdir(parents=True, exist_ok=True)
    rec = _VideoRecorder(scene, out / "run.mp4") if args.video else None

    def film(s, ph, sup):
        ok = sum(r.outcome == "success" for r in sup.records)
        rec.status = f"{args.planner}: attempt {len(sup.records)}, {ok} placed"
        rec(s, ph, sup.cup.pressure_display_kpa())

    try:
        res = run_bin(scene, args.planner, seed, max_attempts=args.max_attempts, observer=film if rec else None)
    finally:
        if rec:
            rec.close()
    write_csv(res.records, out / "attempts.csv")
    (out / "events.json").write_text(json.dumps(res.events, indent=2, default=float))
    (out / "summary.json").write_text(json.dumps(res.summary(), indent=2))
    for r in res.records:
        print(f"attempt {r.attempt}: part {r.target_part} -> {r.outcome}"
              f"{'' if r.outcome == 'success' else ' (' + r.reason + ')'}  slot={r.slot}  cycle={r.cycle_time_s:.2f}s")
    print(f"{res.planner}, seed {seed}: {res.placed}/{len(scene.parts)} placed in {len(res.records)} attempts, "
          f"{res.shakes} shakes, stopped: {res.stop}; wrote {out}")
    return 0


def bench_main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write the report."""
    from binpick.benchmark import runner
    from binpick.benchmark.report import write_report
    from binpick.supervisor.run import PLANNERS

    ap = argparse.ArgumentParser(prog="binpick-bench", description="Run every planner on every seed and write a report.")
    ap.add_argument("--seeds", default="1-25", help='for example "1-25" or "1000,1005-1010"')
    ap.add_argument("--planners", default=",".join(PLANNERS))
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--max-attempts", type=int, default=None, help="stop bins early (smoke test only; the report says so)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/benchmark")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing benchmark folder")
    ap.add_argument("--allow-dirty", action="store_true", help="run the reserved seeds from uncommitted code (the report says so)")
    args = ap.parse_args(argv)

    seeds, planners = runner.parse_seeds(args.seeds), args.planners.split(",")
    command = " ".join(["uv run binpick-bench", *(sys.argv[1:] if argv is None else argv)])
    errors = runner.run_benchmark(seeds, planners, args.out, args.workers, args.config, args.max_attempts, args.overwrite, command,
                                  args.allow_dirty)
    for name, r in write_report(args.out).items():
        rate = "-" if r["success_rate"] is None else f"{100 * r['success_rate']:.1f}%"
        print(f"{name}: success {r['successes']}/{r['attempts']} = {rate}, bins emptied {r['emptied']}/{r['bins']}, parts placed {r['placed']}/{r['parts']}")
    for e in errors:
        print("ERROR", e)
    print(f"report: {Path(args.out) / 'report.md'}")
    return 1 if errors else 0


DEFAULT_RUNS = [
    ("docs/checkpoints/m6/seeds_1240-1264", "Final, unused seeds", "Final code on 25 seeds nobody had used: the blind test."),
    ("docs/checkpoints/m6/seeds_1-25", "Final, seeds 1–25", "Final code on the benchmark seeds, after fixes made with milestone 5's results in hand: not blind."),
    ("docs/checkpoints/m5", "First run, seeds 1–25", "Milestone 5: the first run of seeds 1-25, blind."),
]


def dashboard_main(argv: list[str] | None = None) -> int:
    """Draw benchmark runs as one HTML page."""
    from binpick.benchmark.dashboard import write_dashboard

    ap = argparse.ArgumentParser(prog="binpick-dashboard", description="One HTML page showing benchmark runs.")
    ap.add_argument("--run", nargs=3, action="append", metavar=("FOLDER", "LABEL", "NOTE"),
                    help="a benchmark folder, its short label and a one-line note (repeat; default: the delivered runs)")
    ap.add_argument("--out", default="docs/dashboard.html")
    args = ap.parse_args(argv)
    print(f"wrote {write_dashboard([tuple(r) for r in (args.run or DEFAULT_RUNS)], args.out)}")
    return 0
