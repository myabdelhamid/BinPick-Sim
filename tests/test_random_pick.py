import numpy as np

from binpick.perception.view import observe_bin
from binpick.planning.random_pick import random_targets
from binpick.sim.settle import make_settled_scene


def test_random_targets_are_visible_unfiltered_and_repeatable(cfg):
    s, _ = make_settled_scene(cfg, 1000)
    v = observe_bin(s)
    a = random_targets(v, cfg, np.random.default_rng(3), n=200)
    b = random_targets(v, cfg, np.random.default_rng(3), n=200)
    assert len(a) == 200 and all(np.array_equal(x.pos, y.pos) for x, y in zip(a, b))
    assert all(v.part_mask[v.index(t.pos[:2])] for t in a)
    assert any(t.tilt_deg > 25 for t in a)  # nothing is filtered out
    assert all(0 < t.depth <= 0.110 for t in a)
    ys = np.array([t.pos[:2] for t in a])
    assert np.ptp(ys, axis=0).min() > 0.15  # spread over the bin


def test_empty_bin_gives_no_targets(cfg):
    from binpick.sim.scene import build_scene

    assert random_targets(observe_bin(build_scene(cfg, [])), cfg, np.random.default_rng(0)) == []
