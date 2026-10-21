"""Attached-object lifecycle: once a grasp is verified, the object's
collision geometry rides along with the gripper for the rest of
planning/execution instead of sitting in the static world at its pre-grasp
resting pose.

Mirrors Cap-X's curobo_api.attach_objects_to_robot pattern in spirit
(record a grasp-relative transform, stop treating the object as a
free-standing obstacle), adapted to CollisionChecker/SceneQuery.
collision.compose_world_pose re-derives the object's hypothetical world
pose from the gripper's FK on every check, so no separate "remove from
world" bookkeeping is needed.
"""

from __future__ import annotations

import numpy as np

from .scene_query import BodyExtent, SceneQuery, quat_to_rotmat_wxyz
from .types import AttachedObjectState


def attached_sphere_infeasible_for_goal(attached: AttachedObjectState, goal_radius_m: float | None) -> bool:
    """Whether the attached object's sphere approximation is provably too
    large to ever fit within a goal region of radius goal_radius_m,
    regardless of approach.

    The sphere fit bins the object along its longest axis, which can't
    shrink for a flat, plate-shaped object resting on its edge (e.g. a book
    standing upright) -- a thin book toward a narrow caddy slot can be
    geometrically impossible for the sphere to clear even though the real
    object fits fine, fighting the MPC's clearance cost for the whole leg.

    Conservative: if every fitted sphere already exceeds the goal radius,
    it's better to stop treating the carried object as an obstacle for the
    rest of this leg than keep contorting the plan away from an unwinnable conflict.
    """
    if goal_radius_m is None or not attached.collision_spheres_object_frame:
        return False
    smallest_sphere_radius = min(radius for _offset, radius in attached.collision_spheres_object_frame)
    return smallest_sphere_radius > float(goal_radius_m)


def fit_spheres_to_extent(extent: BodyExtent, n_spheres: int = 3) -> list[tuple[np.ndarray, float]]:
    """Approximate a body's AABB as n_spheres spheres spread along its
    longest axis, each sized to circumscribe the cross-section rectangle
    perpendicular to that axis -- can only over-cover the true geometry,
    never under-cover it.

    A single sphere is conservative for anything elongated; more spheres
    tighten the fit along the long axis. Offsets are in the body's own
    local frame (including extent.center_offset).
    """
    he = np.asarray(extent.half_extents, dtype=np.float64)
    axis = int(np.argmax(he))
    other = [i for i in range(3) if i != axis]
    radius = float(np.hypot(he[other[0]], he[other[1]]))
    n = max(1, int(n_spheres))
    span = max(float(he[axis]) - radius, 0.0)
    offsets_along_axis = np.linspace(-span, span, n) if n > 1 else np.array([0.0])

    spheres: list[tuple[np.ndarray, float]] = []
    for a in offsets_along_axis:
        local = np.array(extent.center_offset, dtype=np.float64)
        local[axis] += a
        spheres.append((local, radius))
    return spheres


def fit_spheres_from_geoms(scene: SceneQuery, body_name: str, n_spheres: int = 3) -> list[tuple[np.ndarray, float]]:
    """Fit n_spheres collision spheres from a body's real collision
    geoms' extreme surface points, grouped into bins along the body's
    longest local axis; each sphere encloses every sample point in its bin.

    Tighter than fit_spheres_to_extent's merged-AABB-diagonal formula
    whenever the true shape is round or irregular: LIBERO objects are often
    convex-decomposed into wedges circling a round profile, whose unioned
    AABBs describe a shape closer to a square than the circle they tile,
    overestimating enough to make placements above a basket read as
    colliding with the rim.

    An elongated cross-section (e.g. a moka pot's handle) is binned along
    its second axis too; one sphere per long-axis slice made the pot ~2x its
    real width, so two pots could never fit side by side on a burner.

    Fit on dense face samples, then any sample left outside every sphere
    grows its nearest sphere, so the sample set is always fully covered.
    Falls back to fit_spheres_to_extent if the body has no discrete collision geoms.
    """
    points_per_edge = 16
    points = scene.body_surface_points_dense_local(body_name, points_per_edge=points_per_edge)
    if len(points) == 0:
        extent = scene.body_extent_local(body_name)
        return fit_spheres_to_extent(extent, n_spheres) if extent is not None else [(np.zeros(3), 0.05)]

    span = points.max(axis=0) - points.min(axis=0)
    long_ax, mid_ax, short_ax = (int(a) for a in np.argsort(span)[::-1])
    ratio = float(span[mid_ax]) / max(float(span[short_ax]), 1e-6)
    n_long = max(1, int(n_spheres))
    n_mid = min(3, int(np.ceil(1.5 * ratio))) if ratio > 1.3 else 1

    def _bin_index(values: np.ndarray, n: int) -> np.ndarray:
        lo, hi = float(values.min()), float(values.max())
        if n == 1 or hi - lo < 1e-9:
            return np.zeros(len(values), dtype=int)
        return np.minimum(((values - lo) / (hi - lo) * n).astype(int), n - 1)

    long_bin = _bin_index(points[:, long_ax], n_long)
    mid_bin = _bin_index(points[:, mid_ax], n_mid)
    centers: list[np.ndarray] = []
    radii: list[float] = []
    for i in range(n_long):
        for j in range(n_mid):
            member_pts = points[(long_bin == i) & (mid_bin == j)]
            if len(member_pts) == 0:
                continue
            center = member_pts.mean(axis=0)
            centers.append(center)
            radii.append(float(np.max(np.linalg.norm(member_pts - center, axis=1))))
    if not centers:
        center = points.mean(axis=0)
        return [(center, float(np.max(np.linalg.norm(points - center, axis=1))))]

    center_arr = np.asarray(centers)
    radius_arr = np.asarray(radii)
    dists = np.linalg.norm(points[:, None, :] - center_arr[None, :, :], axis=2)
    uncovered = np.all(dists > radius_arr[None, :], axis=1)
    for k in np.nonzero(uncovered)[0]:
        nearest = int(np.argmin(dists[k] - radius_arr))
        radius_arr[nearest] = max(radius_arr[nearest], float(dists[k, nearest]))
    # Pad by half the face-sample spacing so geometry between samples stays covered.
    radius_arr += 0.5 * float(np.max(span)) / (points_per_edge - 1)
    return [(c, float(r)) for c, r in zip(center_arr, radius_arr)]


def refresh_grasp_transform(attached: AttachedObjectState, scene: SceneQuery, hand_site: str) -> None:
    """Re-read the hand-to-object transform from the live sim. A heavy or
    off-center object (e.g. a moka pot held near its lid) pivots in the
    fingers after attach, leaving the spheres at the grasp-time pose while the
    real object swings into neighbors. Only call while it's actually held."""
    hand_pos, hand_quat = scene.site_pose(hand_site)
    obj_pos, obj_quat = scene.body_pose(attached.object_name)
    T_hand = np.eye(4)
    T_hand[:3, :3] = quat_to_rotmat_wxyz(hand_quat)
    T_hand[:3, 3] = hand_pos
    T_obj = np.eye(4)
    T_obj[:3, :3] = quat_to_rotmat_wxyz(obj_quat)
    T_obj[:3, 3] = obj_pos
    attached.grasp_transform_hand_from_object = np.linalg.inv(T_hand) @ T_obj


class AttachedObjectManager:
    def __init__(self) -> None:
        self._state: AttachedObjectState | None = None

    @property
    def is_attached(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> AttachedObjectState | None:
        return self._state

    def attach(
        self,
        scene: SceneQuery,
        object_name: str,
        hand_site: str,
        step: int = 0,
        n_spheres: int = 3,
    ) -> AttachedObjectState:
        """Record the current object-to-hand transform and mark the object
        attached. Call this right after grasp verification succeeds (e.g. once
        _is_grasped() first returns True), while the object is still at
        its true grasped pose relative to the gripper."""
        hand_pos, hand_quat = scene.site_pose(hand_site)
        obj_pos, obj_quat = scene.body_pose(object_name)

        R_hand = quat_to_rotmat_wxyz(hand_quat)
        T_hand = np.eye(4)
        T_hand[:3, :3] = R_hand
        T_hand[:3, 3] = hand_pos

        R_obj = quat_to_rotmat_wxyz(obj_quat)
        T_obj = np.eye(4)
        T_obj[:3, :3] = R_obj
        T_obj[:3, 3] = obj_pos

        T_hand_from_obj = np.linalg.inv(T_hand) @ T_obj

        extent = scene.body_extent_local(object_name)
        radius = float(np.linalg.norm(extent.half_extents)) if extent is not None else 0.05
        spheres = fit_spheres_from_geoms(scene, object_name, n_spheres)

        self._state = AttachedObjectState(
            object_name=object_name,
            grasp_transform_hand_from_object=T_hand_from_obj,
            collision_radius_m=radius,
            attached_at_step=step,
            collision_spheres_object_frame=spheres,
        )
        return self._state

    def detach(self) -> None:
        self._state = None
