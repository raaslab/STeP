"""Attached-object lifecycle: once a grasp is verified, the object's collision
geometry rides along with the gripper for the rest of planning/execution
(lift, transport, place) instead of sitting in the static world at its
pre-grasp resting pose.

Mirrors Cap-X's ``curobo_api.attach_objects_to_robot``/``plan_with_grasped_
object`` pattern in spirit (record a grasp-relative transform, stop treating
the object as a free-standing world obstacle) but adapted to
``CollisionChecker``/``SceneQuery`` instead of cuRobo's sphere-fit
``WorldConfig``. See ``collision.py``'s ``compose_world_pose``, which
re-derives the object's hypothetical world pose from the gripper's current FK
pose plus the stored transform on every collision check -- so there's no
separate "remove from world" bookkeeping needed: once an
``AttachedObjectState`` is passed into ``CollisionChecker.check(...,
attached=...)``, the object follows the gripper for that check and its
original resting-pose geoms are simply not queried as a free-standing
obstacle.
"""

from __future__ import annotations

import numpy as np

from .scene_query import BodyExtent, SceneQuery, quat_to_rotmat_wxyz
from .types import AttachedObjectState


def attached_sphere_infeasible_for_goal(attached: AttachedObjectState, goal_radius_m: float | None) -> bool:
    """Whether the attached object's own sphere approximation is provably
    too large to ever fit within a goal region of radius ``goal_radius_m``,
    regardless of how the arm approaches it.

    The sphere fit (see ``fit_spheres_from_geoms``) bins the object along its
    longest local axis -- correct for an elongated object but structurally
    unable to shrink for a flat, plate-shaped one resting on its edge (e.g. a
    book standing upright), since every cross-section perpendicular to its
    height stays close to its full face. A thin book carried toward a narrow
    caddy slot can then be geometrically impossible for the *sphere* to
    clear even though the real object would fit fine, and the MPC's
    clearance cost fights that unwinnable conflict for the whole transport.

    Conservative in the safe direction: if every one of the object's fitted
    spheres already exceeds the goal's radius, no sphere among them could
    ever satisfy a clearance test there, so it's better to stop treating the
    (over-approximated) carried object as an obstacle for the rest of this
    leg and let the arm reach the goal, than to keep contorting the plan
    away from a conflict the real object may not even have.
    """
    if goal_radius_m is None or not attached.collision_spheres_object_frame:
        return False
    smallest_sphere_radius = min(radius for _offset, radius in attached.collision_spheres_object_frame)
    return smallest_sphere_radius > float(goal_radius_m)


def fit_spheres_to_extent(extent: BodyExtent, n_spheres: int = 3) -> list[tuple[np.ndarray, float]]:
    """Approximate a body's AABB as ``n_spheres`` spheres spread along its
    longest axis, each sized to circumscribe the cross-section rectangle
    perpendicular to that axis (radius = hypotenuse of the other two half-
    extents) -- can only over-cover the true geometry, never under-cover it,
    matching the "must not underestimate the object" rule used elsewhere in
    this pipeline.

    A single sphere (n_spheres=1) is fine for roughly cube-shaped/round
    objects but very conservative for anything elongated; more spheres
    tighten the approximation along the long axis. Returned offsets are in
    the body's own local frame (relative to its origin, already including
    ``extent.center_offset``).
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
    """Fit ``n_spheres`` collision spheres directly from a body's real
    collision geoms' extreme surface points (box corners / round-primitive
    rings, see ``SceneQuery.body_surface_points_local``), grouped into bins
    along the body's longest local axis; each bin sphere is sized to
    enclose every sample point that falls in it.

    Tighter than ``fit_spheres_to_extent``'s "treat the whole merged AABB as
    a worst-case rectangle, use its diagonal" formula whenever the true
    shape is round or irregular. LIBERO objects are commonly
    convex-decomposed into many thin wedge-shaped pieces circling a round
    profile, and the union of their individually-rotated AABBs describes a
    shape closer to a square than the circle those wedges actually tile --
    taking that square's diagonal as the fitted radius overestimates enough
    to make placement candidates above a modestly-sized basket read as
    colliding with the rim. Sampling real surface points and fitting
    directly to those avoids compounding that error across neighboring
    pieces.

    Still guaranteed to never under-cover the real geometry, since every
    sampled point is a true point on (or just outside, for the
    round-primitive ring) the geom's surface. Falls back to
    ``fit_spheres_to_extent`` if the body has no discrete collision geoms.
    """
    points = scene.body_surface_points_local(body_name)
    if len(points) == 0:
        extent = scene.body_extent_local(body_name)
        return fit_spheres_to_extent(extent, n_spheres) if extent is not None else [(np.zeros(3), 0.05)]

    span = points.max(axis=0) - points.min(axis=0)
    axis = int(np.argmax(span))
    lo, hi = float(points[:, axis].min()), float(points[:, axis].max())
    n = max(1, int(n_spheres))
    edges = np.linspace(lo, hi, n + 1)
    edges[-1] += 1e-9  # inclusive of the max-position point on the last bin

    spheres: list[tuple[np.ndarray, float]] = []
    for i in range(n):
        mask = (points[:, axis] >= edges[i]) & (points[:, axis] < edges[i + 1])
        if not np.any(mask):
            continue
        member_pts = points[mask]
        center = member_pts.mean(axis=0)
        radius = float(np.max(np.linalg.norm(member_pts - center, axis=1)))
        spheres.append((center, radius))
    if not spheres:
        center = points.mean(axis=0)
        radius = float(np.max(np.linalg.norm(points - center, axis=1)))
        spheres.append((center, radius))
    return spheres


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
        ``_is_grasped()`` first returns True), while the object is still at
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
