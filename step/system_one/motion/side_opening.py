"""Containers entered from the side (e.g. a microwave): an interior region
site with the container's own roof above it. A top-down drop can't reach
inside, so pick, transport and place switch to a horizontal insertion."""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import numpy as np

from .scene_query import SceneQuery


@dataclass
class SideOpening:
    container_body: str
    region_site: str
    region_center: np.ndarray
    region_half: np.ndarray
    floor_z: float
    ceiling_z: float
    # Unit horizontal vector pointing into the container through the opening.
    insert_dir: np.ndarray
    # Distance from region_center back out to the opening plane, along -insert_dir.
    front_offset_m: float
    # Midpoint between the side walls' inner faces -- a handle-held hand spans
    # nearly the whole opening width, so it has to go in centered.
    opening_center_xy: np.ndarray | None = None


def _instance_prefix(name: str) -> str | None:
    match = re.match(r"^(.*?_\d+)(?:_|$)", name)
    return match.group(1) if match else None


def _inside(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> bool:
    return bool(np.all(np.abs(point - center) <= half))


def find_side_opening(scene: SceneQuery, goal_name: str | None) -> SideOpening | None:
    if not goal_name:
        return None
    prefix = _instance_prefix(goal_name)
    if prefix is None:
        return None
    main_body = f"{prefix}_main"
    try:
        scene.body_id(main_body)
    except Exception:
        return None
    # The container's own fixed walls only -- a hinged door is a child body
    # and doesn't define the opening.
    walls = [scene.geom_world_aabb(g) for g in scene.body_collision_geom_ids(main_body, include_children=False)]
    if not walls:
        return None

    model = scene.model
    for sid in range(model.nsite):
        site = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if not site or not site.startswith(prefix + "_") or not site.endswith("_region") or "init" in site:
            continue
        center = np.asarray(scene.site_pose(site)[0], dtype=float)
        xy_cover = [
            (c, h) for c, h in walls if abs(center[0] - c[0]) <= h[0] and abs(center[1] - c[1]) <= h[1]
        ]
        roofs = [c[2] - h[2] for c, h in xy_cover if c[2] - h[2] > center[2]]
        floors = [c[2] + h[2] for c, h in xy_cover if c[2] + h[2] < center[2]]
        if not roofs or not floors:
            continue

        open_dirs = []
        for d in (np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, -1.0, 0])):
            samples = [center + t * d for t in np.arange(0.0, 0.6, 0.005)]
            if not any(_inside(p, c, h) for p in samples for c, h in walls):
                open_dirs.append(d)
        if len(open_dirs) != 1:
            continue
        out_dir = open_dirs[0]
        # The opening plane: where the roof/floor slabs end along out_dir.
        front = max(float(np.dot(c, out_dir) + np.dot(h, np.abs(out_dir))) for c, h in xy_cover)
        site_half = np.abs(scene.data.site_xmat[sid].reshape(3, 3)) @ model.site_size[sid]
        lateral = np.array([-out_dir[1], out_dir[0], 0.0])
        faces = []
        for sign in (1.0, -1.0):
            hit = next(
                (t for t in np.arange(0.0, 0.6, 0.002) if any(_inside(center + sign * t * lateral, c, h) for c, h in walls)),
                None,
            )
            faces.append(None if hit is None else sign * hit)
        opening_center_xy = None
        if None not in faces:
            opening_center_xy = (center + 0.5 * (faces[0] + faces[1]) * lateral)[:2]
        return SideOpening(
            container_body=main_body,
            region_site=site,
            region_center=center,
            region_half=site_half,
            floor_z=float(max(floors)),
            ceiling_z=float(min(roofs)),
            insert_dir=-out_dir,
            front_offset_m=front - float(np.dot(center, out_dir)),
            opening_center_xy=opening_center_xy,
        )
    return None


def support_top_z_below(scene: SceneQuery, xy: np.ndarray, below_z: float, exclude_bodies: set[str]) -> float | None:
    """Height of the highest collision surface under xy (e.g. the table), ignoring exclude_bodies."""
    excluded = {scene.body_id(b) for b in exclude_bodies}
    best = None
    for gid in range(scene.model.ngeom):
        if scene.model.geom_contype[gid] == 0 and scene.model.geom_conaffinity[gid] == 0:
            continue
        if int(scene.model.geom_bodyid[gid]) in excluded or scene.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c, h = scene.geom_world_aabb(gid)
        top = float(c[2] + h[2])
        if abs(xy[0] - c[0]) <= h[0] and abs(xy[1] - c[1]) <= h[1] and top < below_z and (best is None or top > best):
            best = top
    return best


def staging_point(scene: SceneQuery, side: SideOpening, object_name: str, clearance_m: float = 0.12) -> np.ndarray | None:
    """A table spot straight out from the opening, where the object is set
    down so it can be re-grasped by its handle with the hand pointing in."""
    xy = side.region_center[:2] - side.insert_dir[:2] * (side.front_offset_m + clearance_m)
    top = support_top_z_below(scene, xy, side.floor_z, {object_name})
    return None if top is None else np.array([xy[0], xy[1], top])
