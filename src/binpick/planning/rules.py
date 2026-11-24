"""Rule-based suction planner. Input: the height map seen by the bin camera. No simulator state.

For every candidate point the cup is laid on the seen surface the way the physical lip would wrap
(16 directions, see sim/suction.py); a plane is fitted to the points under it. Rejected:
  - tilt of that plane above max_tilt_deg;
  - lip over an edge, a jump, or ground the camera could not see;
  - deviation from the plane beyond the cup's curve limit (curved surfaces inside it are kept);
  - cup or tool too close to a wall, or anything taller standing in their way.
The rest are scored; flat surfaces come before curved ones.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from binpick.config import Config
from binpick.perception.view import BinView
from binpick.planning.types import GraspTarget

LIP_STEPS = 16
CREASE_SPAN = 3   # lip steps (about 4 mm) over which a surface direction is measured


def lip_radius(cfg: Config) -> float:
    """Cup radius plus a margin for camera and arm position error: the lip must also fit a little further out."""
    return cfg.cup.diameter / 2 + cfg.planner.edge_margin


def _frames(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit tangents per normal (N, 3)."""
    x = np.cross(np.tile([0.0, 1.0, 0.0], (len(n), 1)), n)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, np.cross(n, x)


def _surface_z(view: BinView, xy: np.ndarray) -> np.ndarray:
    ij = (xy - view.origin) / view.cell
    return ndimage.map_coordinates(view.z, [ij[..., 0].ravel(), ij[..., 1].ravel()], order=1, cval=np.nan).reshape(xy.shape[:-1])


def _lay_cup(view: BinView, pos: np.ndarray, n: np.ndarray, cfg: Config):
    """Wrap the lip around each (pos, n). Returns lip heights below the cup plane (N, K), NaN where the lip
    cannot sit, and the plane normal fitted to all points under the cup."""
    r, K = lip_radius(cfg), cfg.suction.lip_points
    step = r / LIP_STEPS
    x, y = _frames(n)
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    u = np.cos(th)[None, :, None] * x[:, None] + np.sin(th)[None, :, None] * y[:, None]  # (N, K, 3)
    d = step * np.arange(1, LIP_STEPS + 1)
    flat = pos[:, None, None] + d[None, None, :, None] * u[:, :, None]                    # (N, K, S, 3)
    z = _surface_z(view, flat[..., :2])
    # height below the cup plane along the cup axis of the surface point above/below each sample
    h = -(z - flat[..., 2]) * n[:, None, None, 2]
    prev = np.concatenate([np.zeros_like(h[..., :1]), h[..., :-1]], axis=2)
    dh = h - prev
    seg = np.hypot(step, dh)
    bad = ~(np.abs(dh) <= cfg.perception.max_slope * step)
    arc = np.cumsum(seg, axis=2)
    reached = arc >= r - 1e-9  # False from the first unseen (NaN) sample on
    end = np.where(reached.any(axis=2), np.argmax(reached, axis=2), LIP_STEPS - 1)         # first segment past the lip
    take = lambda a: np.take_along_axis(a, end[..., None], axis=2)[..., 0]  # noqa: E731
    left = r - (take(arc) - take(seg))
    lip = take(prev) + take(dh) * left / take(seg)
    used = np.arange(LIP_STEPS)[None, None] <= end[..., None]
    lip = np.where(~reached.any(axis=2) | (bad & used).any(axis=2) | _creased(h, used, step, np.deg2rad(cfg.perception.crease_deg)), np.nan, lip)
    # plane fit to the points under the cup (in tangent coordinates)
    px, py = d[None, :] * np.cos(th)[:, None], d[None, :] * np.sin(th)[:, None]
    w = (used & np.isfinite(h)).astype(float)
    hz = np.where(np.isfinite(h), h, 0.0)
    A = np.stack([np.broadcast_to(px, h.shape), np.broadcast_to(py, h.shape), np.ones_like(h)], axis=-1)
    AtA = np.einsum("nks,nksi,nksj->nij", w, A, A) + 1e-12 * np.eye(3)
    sol = np.linalg.solve(AtA, np.einsum("nks,nksi,nks->ni", w, A, hz)[..., None])[..., 0]
    fit = n + sol[:, :1] * x + sol[:, 1:2] * y
    return lip, fit / np.linalg.norm(fit, axis=1, keepdims=True)


def _creased(h: np.ndarray, used: np.ndarray, step: float, limit: float) -> np.ndarray:
    """True per (cup, direction) if the surface profile along that diameter turns sharply: a lip can follow a
    curve but not a fold (box edge, seam between two leaning parts)."""
    k, (_, K, S) = CREASE_SPAN, h.shape
    opp = (np.arange(K) + K // 2) % K
    prof = np.concatenate([h[:, opp, ::-1], np.zeros_like(h[..., :1]), h], axis=2)      # full diameter, 2S+1 samples
    seen = np.concatenate([used[:, opp, ::-1], np.ones_like(used[..., :1]), used], axis=2)
    ang = np.arctan2(prof[..., k:] - prof[..., :-k], k * step)
    turn = np.abs(ang[..., k:] - ang[..., :-k])
    return ((turn > limit) & seen[..., 2 * k:] & seen[..., :-2 * k]).any(axis=2)


def _lip_drop(lip: np.ndarray) -> np.ndarray:
    """Lip height spread left after removing the best-fit plane (as in the physical seal)."""
    th = np.linspace(0, 2 * np.pi, lip.shape[1], endpoint=False)
    a, b = 2 * np.mean(lip * np.cos(th), axis=1), 2 * np.mean(lip * np.sin(th), axis=1)
    return np.ptp(lip - a[:, None] * np.cos(th) - b[:, None] * np.sin(th), axis=1)


def _wall_distance(view: BinView, pos: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Smallest distance from the cup axis to a wall, at the surface and where the tool crosses the rim."""
    rim = pos + n * ((view.hi[2] - pos[:, 2]) / n[:, 2])[:, None]
    d = [np.minimum(p[:, :2] - view.lo[:2], view.hi[:2] - p[:, :2]).min(axis=1) for p in (pos, rim)]
    return np.minimum(*d)


def _obstructed(view: BinView, p: np.ndarray, n: np.ndarray, cfg: Config) -> bool:
    """Something seen stands inside the cup/tool cylinder above the cup plane."""
    pl, half = cfg.planner, 0.07
    (i0, j0), (i1, j1) = view.index(p[:2] - half), view.index(p[:2] + half)
    ii, jj = np.mgrid[i0:i1 + 1, j0:j1 + 1]
    q = np.dstack([view.origin[0] + ii * view.cell, view.origin[1] + jj * view.cell, view.z[i0:i1 + 1, j0:j1 + 1]]) - p
    h = q @ n
    lat = np.linalg.norm(q - h[..., None] * n, axis=2)
    return bool(np.any((h > pl.obstacle_height) & (lat < cfg.cup.diameter / 2 + pl.obstacle_clearance)))


def _rectangle(view: BinView, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Center and long axis (world) of the smallest rectangle around the visible surface, seen along its mean
    normal. Unlike the centroid, it stays right when a corner of the face is hidden."""
    pts, n = view.surface_points(label), view.normals[view.labels == label].mean(axis=0)
    n /= np.linalg.norm(n)
    p = pts.mean(axis=0)
    x, y = (f[0] for f in _frames(n[None]))
    uv = np.column_stack([(pts - p) @ x, (pts - p) @ y])
    try:
        hull = uv[ConvexHull(uv).vertices]
    except QhullError:
        return p, x
    best = None
    for e in np.roll(hull, -1, axis=0) - hull:
        a = e / np.linalg.norm(e)
        R = np.array([a, [-a[1], a[0]]])
        c = hull @ R.T
        size = np.ptp(c, axis=0)
        if best is None or size.prod() < best[0]:
            best = (size.prod(), (c.max(axis=0) + c.min(axis=0)) / 2 @ R, R[int(np.argmax(size))])
    _, mid, ax = best
    return p + mid[0] * x + mid[1] * y, ax[0] * x + ax[1] * y


def max_depth(view: BinView, p: np.ndarray, n: np.ndarray, cfg: Config) -> float:
    """Largest thickness the part can have along n: it cannot reach below the floor, nor exceed the largest part."""
    longest = max(cfg.parts.box_footprint[1], cfg.parts.cyl_length[1])
    return float(min(longest, (p[2] - view.lo[2]) / max(n[2], 0.2)))


def rules_targets(view: BinView, cfg: Config) -> list[GraspTarget]:
    pl, s, r = cfg.planner, cfg.suction, cfg.cup.diameter / 2
    k = max(1, int(round(pl.candidate_step / view.cell)))
    grid = np.zeros(view.z.shape, bool)
    grid[::k, ::k] = True
    ij = np.argwhere(grid & (view.labels > 0))
    centers = [view.index(view.surface_center(lab)[:2]) for lab in np.unique(view.labels) if lab > 0]
    centers = [c for c in centers if view.labels[c] > 0]
    ij = np.vstack([ij, np.array(centers, dtype=int).reshape(-1, 2)])
    if len(ij) == 0:
        return []
    lab = view.labels[tuple(ij.T)]
    pos = np.column_stack([view.xy(ij), view.z[tuple(ij.T)]])
    n = view.normals[tuple(ij.T)]
    ok = n[:, 2] > np.cos(np.deg2rad(60))
    pos, n, lab = pos[ok], n[ok], lab[ok]
    for _ in range(2):  # align the cup with the plane under it
        if len(pos) == 0:
            return []
        _, n = _lay_cup(view, pos, n, cfg)
        ok = np.isfinite(n).all(axis=1) & (n[:, 2] > np.cos(np.deg2rad(60)))
        pos, n, lab = pos[ok], n[ok], lab[ok]
    lip, _ = _lay_cup(view, pos, n, cfg)
    tilt = np.degrees(np.arccos(np.clip(n[:, 2], -1, 1)))
    drop = _lip_drop(lip)
    limit = s.curve_radius_min * (1 - np.cos(lip_radius(cfg) / s.curve_radius_min))
    wall = _wall_distance(view, pos, n)
    ok = (tilt <= pl.max_tilt_deg) & (drop <= limit) & (np.ptp(lip, axis=1) <= s.travel) & (wall >= r + pl.wall_clearance)

    out: list[tuple[bool, GraspTarget]] = []
    for c in np.unique(lab[ok]):
        sel = np.where(ok & (lab == c))[0]
        center, axis = _rectangle(view, c)
        seal = 1.0 - (1.0 - s.curve_limit_q) * drop[sel] / limit
        off = np.linalg.norm((pos[sel] - center)[:, :2], axis=1)
        score = (pl.w_height * (pos[sel, 2] - view.lo[2]) / (view.hi[2] - view.lo[2]) + pl.w_seal * seal
                 + pl.w_wall * np.minimum(wall[sel] / pl.wall_saturation, 1.0)
                 - pl.w_center * off / r)
        kept = 0
        for m in np.argsort(-score):
            g = sel[m]
            if _obstructed(view, pos[g], n[g], cfg):
                continue
            out.append((bool(drop[g] > pl.flat_drop), GraspTarget(
                None, pos[g], n[g], float(tilt[g]), max_depth(view, pos[g], n[g], cfg), axis, center,
                float(seal[m]), float(score[m]))))
            kept += 1
            if kept == pl.per_surface:
                break
    out.sort(key=lambda t: (t[0], -t[1].score))
    return [t for _, t in out]
