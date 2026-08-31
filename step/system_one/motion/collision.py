"""3D collision checking on a scratch MuJoCo model/data (never the live sim).

Three complementary checks:

1. ``SceneQuery.compute_contacts()`` (``mj_collision``) -- authoritative for
   *actual penetration* of the robot's own real geometry, and, crucially,
   already respects the model's own self-collision exclusions (adjacent-link
   ``<exclude>``/``contype``/``conaffinity`` pairs baked into robosuite's
   Panda MJCF), so it's the right tool for self-collision and robot-vs-
   environment contact.
2. ``mujoco.mj_geomDistance`` -- an explicit pairwise signed-distance query,
   independent of any geom's compiled contact margin, used for the robot's
   configurable-margin clearance vs. the environment. This never mutates
   ``model.geom_margin`` (which would leak across every ``SceneQuery`` sharing
   that MjModel), so it's the only side-effect-free way to get a non-zero
   margin.
3. **Sphere-approximated attached-object collision** -- once an object is
   grasped, checking its *exact* geometry (LIBERO objects are often
   convex-decomposed into 20+ small geoms, see ``attached_object.py``) against
   every relevant environment geom on every planning query is both expensive
   and unnecessary. Mirroring Cap-X's ``curobo_api.attach_objects_to_robot``
   (which sphere-fits the grasped object's mesh onto a robot link via
   ``SphereFitType``), the attached object is instead approximated as a small
   set of spheres (``AttachedObjectState.collision_spheres_object_frame``,
   computed once at grasp time by ``attached_object.fit_spheres_to_extent``)
   and checked with a simple analytic sphere-vs-AABB distance against every
   other body in the scene. This is what actually makes it feasible to plan a
   collision-free placement trajectory that reasons about the carried
   object's geometry (e.g. avoiding a container's rim) instead of just moving
   the gripper as a point.
"""

from __future__ import annotations

import math
from enum import Enum

import mujoco
import numpy as np

from .config import CollisionConfig
from .scene_query import SceneQuery, quat_to_rotmat_wxyz
from .types import AttachedObjectState, CollisionResult, Contact


class CollisionPhase(str, Enum):
    FREE_SPACE = "free_space"  # every real obstacle (incl. a carried/attached object's true
    # target) must be avoided; no intentional contact is permitted yet.
    TERMINAL_APPROACH = "terminal_approach"  # pick: gripper<->target-object contact permitted
    # (final grasp descent -- the object being grasped is expected to touch the gripper).
    PLACE_APPROACH = "place_approach"  # place: attached-object<->goal-container contact
    # permitted (final insertion descent -- the carried object is expected to touch the
    # container it's being placed into, e.g. resting on a basket's floor).


def compose_world_pose(
    hand_pos: np.ndarray, hand_quat_wxyz: np.ndarray, transform_hand_from_object: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World pose of an object given the hand's world pose and a fixed 4x4
    hand-from-object transform (stored at grasp time)."""
    R_hand = quat_to_rotmat_wxyz(hand_quat_wxyz)
    T_hand = np.eye(4)
    T_hand[:3, :3] = R_hand
    T_hand[:3, 3] = hand_pos
    T_world_obj = T_hand @ transform_hand_from_object
    pos = T_world_obj[:3, 3]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T_world_obj[:3, :3].reshape(9))
    return pos, quat


def required_hand_pose_for_object_pose(
    object_pos: np.ndarray, object_quat_wxyz: np.ndarray, transform_hand_from_object: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``compose_world_pose``: given the *desired* world pose of a
    rigidly-held object and the fixed hand-from-object grasp transform, solve
    for the hand pose that achieves it. Used by ``PlaceSkill`` to turn a
    candidate object placement into an IK target for the arm."""
    R_obj = quat_to_rotmat_wxyz(object_quat_wxyz)
    T_obj = np.eye(4)
    T_obj[:3, :3] = R_obj
    T_obj[:3, 3] = object_pos
    T_hand = T_obj @ np.linalg.inv(transform_hand_from_object)
    pos = T_hand[:3, 3]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T_hand[:3, :3].reshape(9))
    return pos, quat


def sphere_aabb_distance(
    sphere_center: np.ndarray, sphere_radius: float, aabb_center: np.ndarray, aabb_half_extents: np.ndarray
) -> float:
    """Signed distance from a sphere to an axis-aligned box (negative =
    penetrating). Standard closest-point formula: clamp the sphere center
    into the box to get the closest surface point; if the center is already
    inside, fall back to the (negative) distance to the nearest face."""
    delta = sphere_center - aabb_center
    outside = np.maximum(np.abs(delta) - aabb_half_extents, 0.0)
    if np.any(outside > 0):
        return float(np.linalg.norm(outside)) - sphere_radius
    # Center is inside the box: signed penetration = distance to the nearest face.
    inside_depth = float(np.min(aabb_half_extents - np.abs(delta)))
    return -inside_depth - sphere_radius


class CollisionChecker:
    def __init__(
        self,
        scene: SceneQuery,
        joint_names: list[str],
        ee_site: str,
        robot_body_names: list[str],
        gripper_body_names: list[str],
        config: CollisionConfig | None = None,
    ) -> None:
        self.base_scene = scene
        self.joint_names = joint_names
        self.ee_site = ee_site
        self.config = config or CollisionConfig()
        self.robot_geom_ids: set[int] = set()
        for b in robot_body_names:
            self.robot_geom_ids.update(scene.body_geom_ids(b, include_children=False))
        self.gripper_geom_ids: set[int] = set()
        for b in gripper_body_names:
            self.gripper_geom_ids.update(scene.body_geom_ids(b, include_children=False))
        self.robot_geom_ids |= self.gripper_geom_ids
        self.robot_body_ids: set[int] = {scene.body_id(b) for b in set(robot_body_names) | set(gripper_body_names)}
        # Every other body with real geometry -- candidate obstacles for the
        # attached-object sphere check. Cached once: the *set* of bodies in
        # the scene doesn't change across planning queries, only their poses
        # (which are re-read fresh on every call). Body 0 is always MuJoCo's
        # "world" body -- every other body is its descendant, so
        # body_extent_world("world") would aggregate the *entire scene* into
        # one giant AABB via body_geom_ids' include_children walk. World-
        # owned geoms (e.g. a bare ground plane) are excluded rather than
        # handled specially: a plane has no finite extent to bound anyway,
        # and every real obstacle worth avoiding here (table, containers,
        # other objects) is its own proper body.
        self._obstacle_body_names: list[str] = [
            mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in range(1, scene.model.nbody)
            if b not in self.robot_body_ids and mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b)
        ]

    def _permitted(
        self,
        g1: int,
        g2: int,
        phase: CollisionPhase,
        target_object_geom_ids: set[int],
    ) -> bool:
        """Whether (g1, g2) is an *intentional* contact, not a violation (real
        robot geoms only -- see ``_attached_sphere_violations`` for the
        sphere-approximated attached-object side).

        Matches by actual geom-id membership in the gripper/target geom sets
        -- not by name substrings against the object's *body* name, which
        doesn't work: LIBERO object geoms are named like
        ``alphabet_soup_1_g7``, which is not a substring match against the
        body name ``alphabet_soup_1_main`` (a real bug caught by
        ``tests/test_pick_skill_or.py::test_pick_single_object_succeeds``,
        which failed with every grasp candidate rejected as
        "in collision" for touching the very object it was grasping).
        """
        gripper_touch = g1 in self.gripper_geom_ids or g2 in self.gripper_geom_ids
        if not gripper_touch:
            return False
        other = g2 if g1 in self.gripper_geom_ids else g1
        if phase == CollisionPhase.TERMINAL_APPROACH and other in target_object_geom_ids:
            return True
        return False

    def _attached_sphere_violations(
        self,
        scratch: SceneQuery,
        attached: AttachedObjectState,
        phase: CollisionPhase,
        goal_container_body: str | None,
        margin: float,
    ) -> tuple[list[Contact], float]:
        """Sphere-approximated collision of the carried object against every
        other body in the scene (see module docstring). Returns
        (violations, min_clearance).

        Checked **per-geom**, not against each body's merged bounding box: a
        body assembled from several geoms (e.g. a basket's floor + walls) has
        empty space *between* them -- its hollow interior -- that a single
        merged AABB would flatten into solid, since the box spans from the
        floor geom's bottom to the wall geoms' top across the whole
        footprint. That made every hover pose above an open container read
        as "in collision" even directly over the empty opening, not touching
        any actual wall/floor geom (the root cause of placement failing to
        find *any* feasible hover pose above a basket)."""
        hand_pos, hand_quat = scratch.site_pose(self.ee_site)
        obj_pos, obj_quat = compose_world_pose(hand_pos, hand_quat, attached.grasp_transform_hand_from_object)
        R_obj = quat_to_rotmat_wxyz(obj_quat)
        world_spheres = [
            (obj_pos + R_obj @ local_offset, radius)
            for local_offset, radius in attached.collision_spheres_object_frame
        ]

        permitted_body = goal_container_body if phase == CollisionPhase.PLACE_APPROACH else None

        violations: list[Contact] = []
        min_clearance = math.inf
        for body_name in self._obstacle_body_names:
            if body_name == attached.object_name or body_name == permitted_body:
                continue
            for gid in scratch.body_collision_geom_ids(body_name):
                aabb_center, half_extents = scratch.geom_world_aabb(gid)
                for sphere_center, radius in world_spheres:
                    dist = sphere_aabb_distance(sphere_center, radius, aabb_center, half_extents)
                    min_clearance = min(min_clearance, dist)
                    if dist < margin:
                        violations.append(
                            Contact(f"attached:{attached.object_name}", scratch.geom_name(gid), dist)
                        )
        return violations, min_clearance

    def check(
        self,
        q: np.ndarray,
        phase: CollisionPhase = CollisionPhase.FREE_SPACE,
        target_object_body: str | None = None,
        attached: AttachedObjectState | None = None,
        goal_container_body: str | None = None,
        check_margin: bool = False,
    ) -> CollisionResult:
        """Validity check at one configuration.

        ``check_margin=False`` (the default, used by the OMPL state-validity
        checker where speed matters -- this runs on the order of 1e3-1e4 times
        per plan) only enforces *actual* penetration for the robot's real
        geometry. The attached-object sphere check always applies its
        configurable margin regardless of ``check_margin`` -- it's cheap
        enough (a handful of sphere-vs-AABB distance calls) not to need the
        same fast/slow split as the exact-geometry robot check.
        """
        scratch = self.base_scene.clone()
        scratch.set_qpos(self.joint_names, q)

        target_object_geom_ids: set[int] = set()
        if target_object_body:
            target_object_geom_ids = set(scratch.body_geom_ids(target_object_body))
        # The attached object's *real* geoms are still sitting wherever they
        # physically are in the cloned live scene -- not moved to follow this
        # hypothetical candidate `q` (that's what the sphere approximation is
        # for, using the correct FK-derived pose for `q`). So they must be
        # excluded from the exact-geometry check below, or a candidate `q`
        # far from the object's *current* pose could spuriously read as
        # colliding (or a `q` that happens to land near the stale position
        # could spuriously read as fine) against a position that has nothing
        # to do with where the object would actually be at that `q`.
        attached_real_geom_ids: set[int] = (
            set(scratch.body_geom_ids(attached.object_name)) if attached is not None else set()
        )

        # 1. Hard penetration check via mj_collision (respects model self-exclusions).
        # Restricted to contacts involving the robot/gripper -- objects resting on the
        # floor/table/each other are pre-existing scene state, not something the arm's
        # motion is responsible for avoiding (the attached object is handled separately,
        # by sphere approximation, below).
        violations: list[Contact] = []
        min_clearance = math.inf
        for g1_name, g2_name, dist in scratch.compute_contacts():
            g1, g2 = scratch.geom_id(g1_name), scratch.geom_id(g2_name)
            if g1 not in self.robot_geom_ids and g2 not in self.robot_geom_ids:
                continue
            if g1 in attached_real_geom_ids or g2 in attached_real_geom_ids:
                continue
            min_clearance = min(min_clearance, dist)
            if dist >= 0:
                continue
            if self._permitted(g1, g2, phase, target_object_geom_ids):
                continue
            violations.append(Contact(g1_name, g2_name, dist))

        if check_margin:
            margin = self.config.env_margin_m
            exclude_ids = self.robot_geom_ids | attached_real_geom_ids
            env_ids = [g for g in range(scratch.model.ngeom) if g not in exclude_ids]
            for g1 in self.robot_geom_ids:
                for g2 in env_ids:
                    if self._permitted(g1, g2, phase, target_object_geom_ids):
                        continue
                    d = mujoco.mj_geomDistance(scratch.model, scratch.data, g1, g2, margin, None)
                    min_clearance = min(min_clearance, d)
                    if d < margin:
                        violations.append(Contact(scratch.geom_name(g1), scratch.geom_name(g2), d))

        # 2. Sphere-approximated attached-object collision.
        if attached is not None:
            attached_violations, attached_clearance = self._attached_sphere_violations(
                scratch, attached, phase, goal_container_body, self.config.attached_margin_m
            )
            violations.extend(attached_violations)
            min_clearance = min(min_clearance, attached_clearance)

        return CollisionResult(
            in_collision=len(violations) > 0,
            contacts=violations,
            min_clearance=min_clearance,
        )

    def check_edge(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        resolution: float,
        phase: CollisionPhase = CollisionPhase.FREE_SPACE,
        target_object_body: str | None = None,
        attached: AttachedObjectState | None = None,
        goal_container_body: str | None = None,
        check_margin: bool = False,
    ) -> CollisionResult:
        """Validate an entire path segment at a configurable joint-space
        resolution, not just its OMPL vertices."""
        q_start = np.asarray(q_start, dtype=np.float64)
        q_end = np.asarray(q_end, dtype=np.float64)
        dist = float(np.linalg.norm(q_end - q_start))
        n_steps = max(1, int(math.ceil(dist / max(resolution, 1e-6))))
        for i in range(n_steps + 1):
            alpha = i / n_steps
            q = q_start + alpha * (q_end - q_start)
            result = self.check(
                q,
                phase=phase,
                target_object_body=target_object_body,
                attached=attached,
                goal_container_body=goal_container_body,
                check_margin=check_margin,
            )
            if result.in_collision:
                return result
        return CollisionResult(in_collision=False, contacts=[], min_clearance=math.inf)
