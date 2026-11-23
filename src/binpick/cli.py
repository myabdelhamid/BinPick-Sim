"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
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
