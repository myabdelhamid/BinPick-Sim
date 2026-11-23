import numpy as np
import pytest

from binpick.planning.oracle import GraspTarget, oracle_targets
from binpick.sim.parts import PartSpec
from binpick.sim.settle import make_settled_scene
from binpick.supervisor.machine import Supervisor
from binpick.supervisor.records import write_csv
from helpers import quat_axis_angle, single_part_scene


@pytest.fixture(scope="module")
def two_picks(cfg):
    scene, _ = make_settled_scene(cfg, 1000)
    trace = []
    sup = Supervisor(scene, np.random.default_rng(0), seed=1000,
                     observer=lambda s, phase: trace.append((phase, s.data.time, s.data.xpos[s.part_body_ids].copy())))
    recs = []
    for _ in range(2):
        t = oracle_targets(scene)[0]
        recs.append((t, sup.attempt(t)))
    return scene, sup, recs, trace


def test_first_pick_succeeds(two_picks):
    _, _, recs, _ = two_picks
    t, r = recs[0]
    assert r.outcome == "success", r
    assert r.reason == "" and r.referee_ok and r.verify_ok
    assert r.gripped_part == t.part_index and r.slot == 0
    assert r.cycle_time_s < 10.0  # M2 sanity bound; the 8 s target is measured in the M5 benchmark


def test_second_pick_uses_next_slot(two_picks):
    _, _, recs, _ = two_picks
    assert recs[1][1].slot == 1 and recs[1][1].attempt == 2


def test_part_clears_rim_during_transport(two_picks):
    scene, _, recs, trace = two_picks
    rim = scene.bin_inner_bounds()[1][2]
    i = recs[0][1].gripped_part
    hs = scene.parts[i].dims
    lo, hi = scene.bin_inner_bounds()
    r = max(hs)
    over_bin = [pos[i][2] for phase, _, pos in trace if phase in ("lift", "transport")
                and np.all(pos[i][:2] > lo[:2] - r) and np.all(pos[i][:2] < hi[:2] + r)]
    over_bin = over_bin[len(over_bin) // 2:]  # second half: the part is out of the pile and moving away
    assert over_bin and min(over_bin) - r / 2 > rim - 0.005  # conservative: half the largest dim below center


def test_phase_times_add_up(two_picks):
    _, _, recs, _ = two_picks
    r = recs[0][1]
    assert sum(r.phase_times.values()) == pytest.approx(r.cycle_time_s, abs=0.01)
    assert {"capture", "plan", "approach", "grip", "lift", "transport", "place", "verify"} <= r.phase_times.keys()


def test_bad_grasp_point_gives_no_seal(cfg):
    cyl = PartSpec(0, "cylinder", "plastic", (0.040, 0.100), 0.15)  # 40 mm side: cannot seal
    s = single_part_scene(cfg, cyl, quat=quat_axis_angle([1, 0, 0], np.pi / 2))
    lo, _ = s.bin_inner_bounds()
    a = s.part_qpos_adr[0]
    s.data.qpos[a + 2] = lo[2] + 0.020
    import mujoco
    mujoco.mj_forward(s.model, s.data)
    top = s.data.qpos[a:a + 3] + [0, 0, 0.020]
    t = GraspTarget(0, top, np.array([0, 0, 1.0]), 0.0, 0.04, np.array([0, 1.0, 0]))
    r = Supervisor(s, np.random.default_rng(0), seed=0).attempt(t)
    assert r.outcome == "fail" and r.reason == "no_seal" and r.slot is None


def test_unreachable_target_is_not_an_attempt(cfg):
    box = PartSpec(0, "box", "plastic", (0.09, 0.06, 0.03), 0.2)
    s = single_part_scene(cfg, box)
    t = GraspTarget(0, np.array([2.0, 2.0, 0.1]), np.array([0, 0, 1.0]), 0.0, 0.03, np.array([1.0, 0, 0]))
    sup = Supervisor(s, np.random.default_rng(0), seed=0)
    assert sup.attempt(t) is None
    assert sup.events[-1]["event"] == "plan_rejected"


def test_csv(two_picks, tmp_path):
    _, sup, _, _ = two_picks
    p = tmp_path / "a.csv"
    write_csv(sup.records, p)
    lines = p.read_text().splitlines()
    assert len(lines) == 3 and lines[0].startswith("seed,attempt,")
