"""Bin camera depth image -> point cloud (Open3D) -> top-down height map with surface labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy import ndimage

from binpick.sim.camera import intrinsics, render
from binpick.sim.scene import Scene

NORMAL_SPAN = 2  # cells on each side used for slopes and for comparing neighbouring normals


def depth_to_cloud(depth: np.ndarray, K: np.ndarray, cam_pos: np.ndarray, cam_R: np.ndarray) -> np.ndarray:
    """(H, W, 3) world points. Depth values belong to pixel centers (u + 0.5, v + 0.5)."""
    h, w = depth.shape
    k = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2] - 0.5, K[1, 2] - 0.5)
    T = np.eye(4)  # Open3D camera (x right, y down, z forward) in world; MuJoCo camera is (x right, y up, z back)
    T[:3, :3] = cam_R @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = cam_pos
    pc = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)), k, np.linalg.inv(T),
        depth_scale=1.0, depth_trunc=100.0, project_valid_depth_only=False)
    return np.asarray(pc.points).reshape(h, w, 3).copy()


@dataclass(frozen=True)
class BinView:
    """Height map of the bin interior. Index [i, j] is x = origin[0] + i*cell, y = origin[1] + j*cell."""
    z: np.ndarray        # (nx, ny) surface height, NaN where the camera saw nothing
    labels: np.ndarray   # (nx, ny) smooth-surface id, 0 = floor, wall, edge or unknown
    normals: np.ndarray  # (nx, ny, 3) outward unit normals (NaN at holes)
    origin: np.ndarray   # (2,) xy of cell [0, 0]
    cell: float
    lo: np.ndarray       # bin inner bounds
    hi: np.ndarray
    part_mask: np.ndarray  # (nx, ny) cells that are part surface
    min_part_cells: int

    def index(self, xy: np.ndarray) -> tuple[int, int]:
        ij = np.round((np.asarray(xy) - self.origin) / self.cell).astype(int)
        return tuple(np.clip(ij, 0, np.array(self.z.shape) - 1))

    def xy(self, ij: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(ij) * self.cell

    def surface_points(self, label: int) -> np.ndarray:
        ij = np.argwhere(self.labels == label)
        return np.column_stack([self.xy(ij), self.z[tuple(ij.T)]])

    def surface_center(self, label: int) -> np.ndarray:
        """Centroid of the visible surface patch (world)."""
        return self.surface_points(label).mean(axis=0)

    def is_empty(self) -> bool:
        return int(self.part_mask.sum()) < self.min_part_cells


def _height_map(P: np.ndarray, lo: np.ndarray, hi: np.ndarray, cell: float) -> np.ndarray:
    n = np.round((hi[:2] - lo[:2]) / cell).astype(int) + 1
    p = P.reshape(-1, 3)
    ij = np.round((p[:, :2] - lo[:2]) / cell).astype(int)
    keep = np.all((ij >= 0) & (ij < n), axis=1)
    z = np.full(n, -np.inf)
    np.maximum.at(z, tuple(ij[keep].T), p[keep, 2])
    z[~np.isfinite(z)] = np.nan
    # pixels are up to ~1.05 mm apart at the floor: fill single-cell gaps from their neighbours
    pad = np.pad(z, 1, mode="edge")
    near = np.stack([np.roll(pad, (i, j), (0, 1))[1:-1, 1:-1] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    fill = np.isnan(z) & (np.isfinite(near).sum(axis=0) >= 4)
    z[fill] = np.nanmedian(near[:, fill], axis=0)
    return z


def _normals(z: np.ndarray, cell: float) -> np.ndarray:
    k = NORMAL_SPAN
    gx, gy = np.full_like(z, np.nan), np.full_like(z, np.nan)
    gx[k:-k] = (z[2 * k:] - z[:-2 * k]) / (2 * k * cell)
    gy[:, k:-k] = (z[:, 2 * k:] - z[:, :-2 * k]) / (2 * k * cell)
    n = np.dstack([-gx, -gy, np.ones_like(z)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def _edges(z: np.ndarray, normals: np.ndarray, cell: float, max_slope: float, crease_cos: float) -> np.ndarray:
    """Cells on a height jump or on a sharp crease (where the surface direction changes quickly)."""
    e = np.zeros(z.shape, bool)
    k = NORMAL_SPAN
    for ax in (0, 1):
        jump = np.abs(np.diff(z, axis=ax)) > max_slope * cell
        a = [slice(None)] * 2
        b = [slice(None)] * 2
        a[ax], b[ax] = slice(None, -1), slice(1, None)
        e[tuple(a)] |= jump
        e[tuple(b)] |= jump
        c, lo_, hi_ = [slice(None)] * 2, [slice(None)] * 2, [slice(None)] * 2
        c[ax], lo_[ax], hi_[ax] = slice(k, -k), slice(None, -2 * k), slice(2 * k, None)
        dot = np.sum(normals[tuple(lo_)] * normals[tuple(hi_)], axis=2)
        e[tuple(c)] |= ~(dot >= crease_cos)  # NaN counts as an edge
    return e | np.isnan(z) | np.isnan(normals).any(axis=2)


def observe_bin(scene: Scene) -> BinView:
    cfg = scene.cfg.perception
    _, depth = render(scene, "bin_cam")
    cam = scene.data.cam("bin_cam")
    P = depth_to_cloud(depth, intrinsics(scene, "bin_cam"), cam.xpos, cam.xmat.reshape(3, 3))
    lo, hi = scene.bin_inner_bounds()
    z = _height_map(P, lo, hi, cfg.cell)
    normals = _normals(z, cfg.cell)
    part = z > lo[2] + cfg.min_part_height
    m = int(np.ceil(cfg.wall_margin / cfg.cell))  # wall faces seen at a slant land on the border cells
    part[:m] = part[-m:] = False
    part[:, :m] = part[:, -m:] = False
    edges = _edges(z, normals, cfg.cell, cfg.max_slope, np.cos(np.deg2rad(cfg.crease_deg)))
    labels, n = ndimage.label(part & ~edges)
    size = ndimage.sum_labels(np.ones_like(z), labels, np.arange(n + 1))
    labels[size[labels] < cfg.min_surface_mm2 * 1e-6 / cfg.cell**2] = 0
    return BinView(z=z, labels=labels, normals=normals, origin=lo[:2].copy(), cell=cfg.cell, lo=lo, hi=hi, part_mask=part,
                   min_part_cells=int(cfg.empty_mm2 * 1e-6 / cfg.cell**2))
