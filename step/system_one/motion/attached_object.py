"""Attached-object lifecycle: once a grasp is verified, the object's collision
geometry rides along with the gripper for the rest of planning/execution
(lift, transport, place) instead of sitting in the static world at its
pre-grasp resting pose.

Mirrors Cap-X's ``curobo_api.attach_objects_to_robot``/``plan_with_grasped_
object`` pattern in spirit (record a grasp-relative transform, stop treating
the object as a free-standing world obstacle, use its geometry attached to
the end effector for collision during transport) but adapted to
``CollisionChecker``/``SceneQuery`` instead of cuRobo's ``WorldConfig`` sphere-
fit representation -- see ``collision.py``'s ``compose_world_pose``, which is
what actually re-derives the object's hypothetical world pose from the
gripper's current FK pose plus the stored transform on every collision check,
so there's no separate "remove from world" bookkeeping needed: as long as an
``AttachedObjectState`` is passed into ``CollisionChecker.check(...,
attached=...)``, the object is repositioned to follow the gripper for that
check and its *original* resting-pose geoms are simply not queried as a
free-standing obstacle (the caller stops passing ``target_object_body`` for
free-space obstacle treatment once attached).
"""

from __future__ import annotations

import mujoco
import numpy as np

from .scene_query import BodyExtent, SceneQuery, quat_to_rotmat_wxyz
from .types import AttachedObjectState


def fit_spheres_to_extent(extent: BodyExtent, n_spheres: int = 3) -> list[tuple[np.ndarray, float]]:
    """Approximate a body's AABB as ``n_spheres`` spheres spread along its
    longest axis, each sized to circumscribe the cross-section rectangle
    perpendicular to that axis (radius = hypotenuse of the other two half-
    extents) -- this can only over-cover the true geometry, never
    under-cover it, matching the "must not underestimate the object" rule
    used everywhere else in this pipeline (grasp width estimation, etc.).

    A single sphere (n_spheres=1) is fine for roughly cube-shaped/round
    objects, but is very conservative for anything elongated (e.g. treats a
    30cm-long object as a 30cm-diameter ball); more spheres tighten the
    approximation along the long axis. Returned offsets are in the body's own
    local frame (i.e. relative to its origin, already including
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


def _geom_sample_points_local(scene: SceneQuery, gid: int, body_mat: np.ndarray, body_pos: np.ndarray) -> np.ndarray:
    """A handful of extreme surface points of one geom, expressed in the
    body's local frame -- box corners for box/mesh geoms, a ring of points
    for round primitives. Used by ``fit_spheres_from_geoms`` to fit spheres
    directly from real geometry instead of each geom's own worst-case
    bounding-sphere radius, which double-counts extent along axes several
    neighboring geoms already share (e.g. a dozen wedge-shaped pieces that
    each individually span an object's full height -- binning by height
    then can't separate them, and each wedge's own bounding sphere already
    balloons to cover that shared full-height reach)."""
    gpos = scene.data.geom_xpos[gid]
    gmat = scene.data.geom_xmat[gid].reshape(3, 3)
    size = scene.model.geom_size[gid]
    gtype = scene.model.geom_type[gid]
    rel_mat = body_mat.T @ gmat
    rel_pos = body_mat.T @ (gpos - body_pos)

    if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        return np.zeros((0, 3))
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(size[0])
        local_pts = np.array(
            [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r], [0, 0, 0]]
        )
    elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, half_len = float(size[0]), float(size[1])
        n_ring = 16
        angles = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
        # Slightly inflate the sampled ring radius so a finite number of
        # angular samples still safely bounds the true (continuous) round
        # surface at any angle in between.
        r_safe = r / np.cos(np.pi / n_ring)
        local_pts = np.array(
            [[r_safe * np.cos(a), r_safe * np.sin(a), z] for z in (-half_len, half_len) for a in angles]
            + [[0, 0, 0]]
        )
    else:  # box, mesh: 8 corners of the local half-extent box
        hx, hy, hz = scene._geom_local_half_extent(gtype, size)
        local_pts = np.array(
            [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
    return (rel_mat @ local_pts.T).T + rel_pos


def fit_spheres_from_geoms(scene: SceneQuery, body_name: str, n_spheres: int = 3) -> list[tuple[np.ndarray, float]]:
    """Fit ``n_spheres`` collision spheres directly from a body's real
    collision geoms' extreme surface points (box corners / round-primitive
    rings, see ``_geom_sample_points_local``), grouped into bins along the
    body's longest local axis; each bin sphere is sized to enclose every
    sample point that falls in it.

    Tighter than ``fit_spheres_to_extent``'s single "treat the whole merged
    AABB as a worst-case rectangle at every slice, use its diagonal" formula
    whenever the true shape is round or irregular. LIBERO objects are
    commonly convex-decomposed into a dozen-plus thin wedge-shaped pieces
    circling a round profile; the union of their individually-rotated AABBs
    describes a shape close to a square, not the circle those wedges
    actually tile, so taking that square's diagonal as the fitted radius
    (what ``fit_spheres_to_extent`` does) came out ~50% too large for a real
    soup can -- ~6cm for a can whose real cross-section is ~4cm -- on its
    own enough to make every placement candidate above a modestly-sized
    basket read as colliding with the rim. Sampling real corner/surface
    points instead and fitting directly to *those* (rather than to each
    geom's own already-inflated bounding sphere) avoids compounding that
    error across neighboring pieces.

    Still guaranteed to never under-cover the real geometry: every sampled
    point is a true point on (or, for the round-primitive ring, very close
    to and outside) the geom's surface, so a sphere containing all of them
    contains the geom. Falls back to ``fit_spheres_to_extent`` if the body
    has no discrete collision geoms to inspect. Offsets returned in the
    body's local frame.
    """
    gids = scene.body_collision_geom_ids(body_name)
    if not gids:
        extent = scene.body_extent_local(body_name)
        return fit_spheres_to_extent(extent, n_spheres) if extent is not None else [(np.zeros(3), 0.05)]

    bid = scene.body_id(body_name)
    body_pos = scene.data.xpos[bid]
    body_mat = scene.data.xmat[bid].reshape(3, 3)

    point_sets = [_geom_sample_points_local(scene, g, body_mat, body_pos) for g in gids]
    points = np.concatenate([p for p in point_sets if len(p)], axis=0)

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
