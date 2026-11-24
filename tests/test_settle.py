import dataclasses

import numpy as np
import pytest

from binpick.sim.settle import SettleError, make_settled_scene, max_penetration


@pytest.fixture(scope="module")
def settled(cfg):
    return make_settled_scene(cfg, seed=1000)


def test_all_parts_settle_inside_bin(settled):
    scene, poses = settled
    lo, hi = scene.bin_inner_bounds()
    assert len(poses) == 20
    for p in poses:
        assert np.all(p.pos[:2] > lo[:2]) and np.all(p.pos[:2] < hi[:2]) and p.pos[2] > lo[2]


def test_scene_at_rest(settled):
    scene, _ = settled
    assert np.max(np.abs(scene.data.qvel)) < 0.01


def test_no_deep_penetration(settled):
    scene, _ = settled
    assert max_penetration(scene) < 0.002


def test_same_seed_same_poses(cfg, settled):
    _, a = settled
    _, b = make_settled_scene(cfg, 1000)
    for pa, pb in zip(a, b):
        assert np.allclose(pa.pos, pb.pos, atol=1e-6) and np.allclose(pa.quat, pb.quat, atol=1e-6)


def test_escape_raises(cfg):
    bad = dataclasses.replace(cfg, settle=dataclasses.replace(cfg.settle, inset=-0.3, max_redrops=2))
    with pytest.raises(SettleError) as e:
        make_settled_scene(bad, 1002)
    assert isinstance(e.value.part_id, int)


def test_validate_rejects_part_knocked_out(cfg):
    from binpick.sim.settle import validate_settled

    scene, _ = make_settled_scene(cfg, 1003)
    validate_settled(scene)  # a good pile passes
    a = scene.part_qpos_adr[3]
    scene.data.qpos[a:a + 3] = [0.0, -0.8, 0.05]  # on the floor, outside the bin
    with pytest.raises(SettleError) as e:
        validate_settled(scene)
    assert e.value.part_id == 3


def test_validate_rejects_moving_pile(cfg):
    import mujoco

    from binpick.sim.settle import validate_settled

    scene, _ = make_settled_scene(cfg, 1004)
    m = scene.model
    dof = m.jnt_dofadr[m.body_jntadr[scene.part_body_ids[5]]]
    scene.data.qvel[dof] = 0.5
    with pytest.raises(SettleError) as e:
        validate_settled(scene)
    assert e.value.part_id == 5


@pytest.mark.parametrize("seed", [1030, 1031])
def test_slow_piles_get_the_time_they_need(cfg, seed):
    make_settled_scene(cfg, seed)  # needed up to 2.3 s more than one final pass; raised SettleError before
