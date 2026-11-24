"""Does a planned arm motion hit the cell? The robot knows its own geometry and the cell's (bin, tray, floor), not
where the parts are, so it checks against an empty copy of the cell."""

from __future__ import annotations

import mujoco
import numpy as np

from binpick.motion.trajectory import JointTrajectory
from binpick.sim.scene import PREFIX, Scene

PATH_STEP_S = 0.04  # a path is checked this often (at most about 30 mm of cup travel)


class CellCheck:
    def __init__(self, scene: Scene):
        self._cell = scene.empty_cell()
        m = self._cell.model
        names = [m.body(b).name for b in m.geom_bodyid]
        self._arm = np.array([n.startswith(PREFIX) and n != PREFIX + "link0" for n in names])
        self._base = np.array([n == PREFIX + "link0" for n in names])
        self._adr = [int(m.joint(j).qposadr[0]) for j in self._cell.arm_joint_names]

    def hits(self, q: np.ndarray) -> bool:
        m, d = self._cell.model, self._cell.data
        d.qpos[self._adr] = q
        mujoco.mj_fwdPosition(m, d)
        g = d.contact.geom[:d.ncon]
        return bool(np.any(self._arm[g].any(axis=1) & ~self._arm[g].all(axis=1) & ~self._base[g].any(axis=1)))

    def path_hits(self, tr: JointTrajectory) -> bool:
        n = max(2, int(np.ceil(tr.duration / PATH_STEP_S)) + 1)
        return any(self.hits(tr.sample(t)) for t in np.linspace(0, tr.duration, n))
