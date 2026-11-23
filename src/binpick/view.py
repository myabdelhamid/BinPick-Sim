"""Interactive viewer for a settled scene. On macOS run with: uv run mjpython -m binpick.view --seed N"""

from __future__ import annotations

import argparse

import mujoco.viewer

from binpick.config import load_config
from binpick.sim.settle import make_settled_scene


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Open a settled BinPick-Sim scene in the MuJoCo viewer.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    scene, _ = make_settled_scene(cfg, cfg.seed if args.seed is None else args.seed)
    mujoco.viewer.launch(scene.model, scene.data)


if __name__ == "__main__":
    main()
