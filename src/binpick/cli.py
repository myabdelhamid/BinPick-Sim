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
