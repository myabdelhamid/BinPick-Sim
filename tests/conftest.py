from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def write_patched(tmp_path: Path, patch: dict) -> Path:
    data = yaml.safe_load(DEFAULT_YAML.read_text())
    p = tmp_path / "patched.yaml"
    p.write_text(yaml.safe_dump(_deep_merge(data, patch)))
    return p


@pytest.fixture(scope="session")
def cfg():
    from binpick.config import load_config

    return load_config()


@pytest.fixture(scope="module")
def scene(cfg):
    import numpy as np

    from binpick.sim.parts import sample_parts
    from binpick.sim.scene import build_scene

    return build_scene(cfg, sample_parts(cfg.parts, np.random.default_rng([1000, 0])))
