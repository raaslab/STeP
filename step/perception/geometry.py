"""Point-cloud geometry for perceived objects: cleaning, support-plane
fitting, and 2.5D heightmap box decomposition.

Objects become a set of vertical boxes, one per run of equal-height cells
in a top-surface heightmap, matching how LIBERO's own assets are modelled
(a bowl is ~40 boxes, hollow inside). A convex hull would fill a bowl's
cavity and make rim grasps and placing into containers look infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PerceptionConfig


def clean_points(points: np.ndarray, cfg: PerceptionConfig) -> np.ndarray:
    """Voxel downsample, drop statistical outliers, keep the largest cluster."""
    import open3d as o3d

    if len(points) < cfg.cluster_min_points:
        return np.zeros((0, 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    pcd = pcd.voxel_down_sample(cfg.voxel_size_m)
    if len(pcd.points) > cfg.outlier_nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(cfg.outlier_nb_neighbors, cfg.outlier_std_ratio)
    if len(pcd.points) < cfg.cluster_min_points:
        return np.asarray(pcd.points)
    labels = np.asarray(pcd.cluster_dbscan(eps=cfg.cluster_eps_m, min_points=cfg.cluster_min_points))
    pts = np.asarray(pcd.points)
    if labels.max() < 0:
        return pts
    largest = np.bincount(labels[labels >= 0]).argmax()
    return pts[labels == largest]


def fit_support_plane(points: np.ndarray, cfg: PerceptionConfig, max_planes: int = 4) -> float | None:
    """Height of the largest horizontal plane (table, or floor in floor-arena
    scenes). Gravity is known from the robot's mounting, so only planes whose
    normal is within ~10 degrees of +Z qualify."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    pcd = pcd.voxel_down_sample(0.01)
    best: tuple[int, float] | None = None
    for _ in range(max_planes):
        if len(pcd.points) < 100:
            break
        (a, b, c, d), inliers = pcd.segment_plane(
            cfg.table_ransac_threshold_m, 3, cfg.table_ransac_iterations
        )
        if abs(c) > 0.985:
            z = float(-d / c)
            if best is None or len(inliers) > best[0]:
                best = (len(inliers), z)
        pcd = pcd.select_by_index(inliers, invert=True)
    return None if best is None else best[1]


@dataclass
class BoxGeom:
    center_local: np.ndarray  # (3,) in the body frame
    half_size: np.ndarray  # (3,)


@dataclass
class ObjectShape:
    origin_world: np.ndarray  # bottom-centre of the footprint, matching LIBERO's body origins
    yaw: float  # body frame rotation about world Z
    boxes: list[BoxGeom]
    top_z: float
    bottom_z: float


def _footprint_yaw(xy: np.ndarray) -> float:
    """Long-axis yaw from 2D PCA, or 0 for round-ish footprints so a bowl's
    grid isn't spun by noise."""
    centered = xy - xy.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(centered.T))
    if evals[0] <= 1e-12 or evals[1] / evals[0] < 1.5**2:
        return 0.0
    major = evecs[:, 1]
    return float(np.arctan2(major[1], major[0]))


def heightmap_shape(
    points: np.ndarray,
    support_z: float,
    cfg: PerceptionConfig,
    cell_m: float = 0.01,
    height_quantum_m: float = 0.005,
) -> ObjectShape | None:
    """Box decomposition of an object's cleaned world points.

    The bottom is the lowest observed point, snapped to the support plane when
    close to it; side views nearly always see an object down to whatever it
    rests on, which is the cabinet top or a plate as often as the table.
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import Delaunay

    pts = np.asarray(points, dtype=np.float64)
    pts = pts[pts[:, 2] > support_z + cfg.table_clearance_m]
    if len(pts) < cfg.cluster_min_points:
        return None

    # 5th percentile: a few pixels of whatever the object sits on bleed into
    # the bottom of its mask, which would drag a stacked object down to the table.
    bottom_z = float(np.percentile(pts[:, 2], 5))
    # Snap to the support when close to it, or when only the top surface was
    # seen (a view from above shows no sides to measure the bottom from).
    if bottom_z - support_z < 0.015 or float(pts[:, 2].max()) - bottom_z < 0.01:
        bottom_z = support_z

    yaw = _footprint_yaw(pts[:, :2])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    center_xy = 0.5 * (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0))
    local_xy = (pts[:, :2] - center_xy) @ R

    lo = local_xy.min(axis=0) - cell_m
    idx = np.floor((local_xy - lo) / cell_m).astype(int)
    shape = idx.max(axis=0) + 2
    height = np.full(shape, -np.inf)
    np.maximum.at(height, (idx[:, 0], idx[:, 1]), pts[:, 2])
    observed = np.isfinite(height)

    # Footprint = convex fill of the observed cells; unobserved cells inside it
    # (the far side of a box, the interior just behind a bowl's near rim) take
    # the height of the nearest observed cell.
    gx, gy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    obs_cells = np.argwhere(observed)
    if len(obs_cells) >= 3 and np.linalg.matrix_rank(obs_cells - obs_cells.mean(axis=0)) == 2:
        inside = Delaunay(obs_cells).find_simplex(np.stack([gx, gy], axis=-1).reshape(-1, 2)) >= 0
        footprint = inside.reshape(shape) | observed
    else:
        footprint = observed
    _, (ix, iy) = distance_transform_edt(~observed, return_indices=True)
    filled = height[ix, iy]
    top = np.where(footprint, filled, -np.inf)
    q = np.zeros(shape, dtype=int)
    q[footprint] = np.round((top[footprint] - bottom_z) / height_quantum_m).astype(int)

    boxes: list[BoxGeom] = []
    # Run-length along y per x row, then merge identical runs across rows.
    open_runs: dict[tuple[int, int, int], int] = {}
    finished: list[tuple[int, int, int, int, int]] = []
    for x in range(shape[0]):
        runs: list[tuple[int, int, int]] = []
        y = 0
        while y < shape[1]:
            if q[x, y] <= 0:
                y += 1
                continue
            y0, h = y, q[x, y]
            while y < shape[1] and q[x, y] == h:
                y += 1
            runs.append((y0, y, h))
        next_open: dict[tuple[int, int, int], int] = {}
        for run in runs:
            next_open[run] = open_runs.pop(run, x)
        for (y0, y1, h), x0 in open_runs.items():
            finished.append((x0, x, y0, y1, h))
        open_runs = next_open
    for (y0, y1, h), x0 in open_runs.items():
        finished.append((x0, shape[0], y0, y1, h))

    origin_xy = center_xy
    pad = cfg.hull_padding_m
    for x0, x1, y0, y1, h in finished:
        cx = lo[0] + 0.5 * (x0 + x1) * cell_m
        cy = lo[1] + 0.5 * (y0 + y1) * cell_m
        hz = 0.5 * h * height_quantum_m
        boxes.append(
            BoxGeom(
                center_local=np.array([cx, cy, hz]),
                half_size=np.array([0.5 * (x1 - x0) * cell_m + pad, 0.5 * (y1 - y0) * cell_m + pad, hz + pad / 2]),
            )
        )
    if not boxes:
        return None
    return ObjectShape(
        origin_world=np.array([origin_xy[0], origin_xy[1], bottom_z]),
        yaw=yaw,
        boxes=boxes,
        top_z=float(bottom_z + q.max() * height_quantum_m),
        bottom_z=bottom_z,
    )


def fit_plane_normal(points: np.ndarray, threshold_m: float) -> tuple[np.ndarray, float] | None:
    """RANSAC plane (unit normal n, offset d with n.x + d = 0)."""
    import open3d as o3d

    if len(points) < 30:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    (a, b, c, d), inliers = pcd.segment_plane(threshold_m, 3, 500)
    if len(inliers) < 20:
        return None
    n = np.array([a, b, c])
    norm = np.linalg.norm(n)
    return n / norm, float(d / norm)
