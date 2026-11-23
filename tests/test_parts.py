import numpy as np
import pytest

from binpick.sim.parts import part_volume, sample_parts


def test_counts_and_ranges(cfg):
    ps = sample_parts(cfg.parts, np.random.default_rng([1, 0]))
    assert [p.shape for p in ps].count("box") == 15
    assert [p.shape for p in ps].count("cylinder") == 5
    assert [p.id for p in ps] == list(range(20))
    for p in ps:
        assert 0.05 <= p.mass <= 2.5
        assert p.mass == pytest.approx(part_volume(p) * cfg.parts.materials[p.material].density)
        if p.shape == "box":
            assert all(0.045 <= d <= 0.110 for d in p.dims[:2]) and 0.020 <= p.dims[2] <= 0.070
        else:
            assert 0.040 <= p.dims[0] <= 0.080 and 0.040 <= p.dims[1] <= 0.110


@pytest.mark.parametrize("seed", [0, 1, 2**31 - 1])
def test_deterministic(cfg, seed):
    a = sample_parts(cfg.parts, np.random.default_rng([seed, 0]))
    b = sample_parts(cfg.parts, np.random.default_rng([seed, 0]))
    assert a == b


def test_seeds_differ(cfg):
    a = sample_parts(cfg.parts, np.random.default_rng([1, 0]))
    b = sample_parts(cfg.parts, np.random.default_rng([2, 0]))
    assert a != b


def test_material_mix_over_many_seeds(cfg):
    mats = [p.material for s in range(200) for p in sample_parts(cfg.parts, np.random.default_rng([s, 0]))]
    assert mats.count("plastic") / len(mats) == pytest.approx(0.6, abs=0.03)
    assert mats.count("steel") / len(mats) == pytest.approx(0.1, abs=0.02)
