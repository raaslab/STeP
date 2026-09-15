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
        # A grasp only ever needs the fingerPADS on the object -- this is the
        # same definition robosuite's own _check_grasp uses (left_fingerpad/
        # right_fingerpad geom groups). Everything else on the gripper (the
        # palm/hand body, the non-pad finger segments) getting "permitted" to
        # touch the target during TERMINAL_APPROACH -- what an earlier
        # version of _permitted() did, treating every gripper_body_names geom
        # as equally acceptable -- masks a real physical impossibility: if
        # the intended grasp height requires the palm to descend below the
        # object's top surface (e.g. the object is taller than the fingers
        # are long), the palm hitting the object's top was never flagged as
        # a collision, so a "validated" descent could still be one the real
        # gripper physically can't complete. Derived from the existing
        # `*_pad_collision` geom-naming convention rather than a new
        # constructor parameter, so this applies without touching any of
        # this class's call sites.
        self.fingerpad_geom_ids: set[int] = {
            g for g in self.gripper_geom_ids if scene.geom_name(g).endswith("_pad_collision")
        }
        # Broader than fingerpad_geom_ids: every geom belonging to a finger
        # (both the pad and the rigid segment above it), but never the palm.
        # Used only for the small compliance tolerance in
        # ``_finger_overlap_within_tolerance`` below -- real fingers (and
        # MuJoCo's own soft contact) have a little give; a rigid wrist/palm
        # assembly doesn't, so the palm never gets any tolerance.
        self.finger_geom_ids: set[int] = {g for g in self.gripper_geom_ids if "finger" in scene.geom_name(g)}
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

        Only fingerpad<->target contact is ever permitted, not just any
        gripper-body geom -- see ``fingerpad_geom_ids``'s definition above
        for why the palm/hand and the non-pad finger segments must stay
        strictly excluded.

        Matches by actual geom-id membership in the gripper/target geom sets
        -- not by name substrings against the object's *body* name, which
        doesn't work: LIBERO object geoms are named like
        ``alphabet_soup_1_g7``, which is not a substring match against the
        body name ``alphabet_soup_1_main`` (a real bug caught by
        ``tests/test_pick_skill_or.py::test_pick_single_object_succeeds``,
        which failed with every grasp candidate rejected as
        "in collision" for touching the very object it was grasping).
        """
        fingerpad_touch = g1 in self.fingerpad_geom_ids or g2 in self.fingerpad_geom_ids
        if not fingerpad_touch:
            return False
        other = g2 if g1 in self.fingerpad_geom_ids else g1
        if phase == CollisionPhase.TERMINAL_APPROACH and other in target_object_geom_ids:
            return True
        return False

    def _finger_overlap_within_tolerance(
        self,
        g1: int,
        g2: int,
        phase: CollisionPhase,
        target_object_geom_ids: set[int],
        dist: float,
    ) -> bool:
        """A small, configurable amount of interpenetration between a FINGER
        geom (pad or the rigid segment above it -- never the palm, see
        ``finger_geom_ids``) and the grasp target is tolerated during
        TERMINAL_APPROACH, mirroring real gripper compliance: a rigid
        open-loop check requires the fully-open gripper to already be clear
        of the object before ever closing, but a real (or MuJoCo's own
        soft-contact) gripper can close through a few mm of an object
        slightly wider than its rated opening. See
        ``CollisionConfig.finger_overlap_tolerance_m``'s docstring for the
        concrete case this fixes.
        """
        if phase != CollisionPhase.TERMINAL_APPROACH or self.config.finger_overlap_tolerance_m <= 0.0:
            return False
        finger_touch = g1 in self.finger_geom_ids or g2 in self.finger_geom_ids
        if not finger_touch:
            return False
        other = g2 if g1 in self.finger_geom_ids else g1
        if other not in target_object_geom_ids:
            return False
        return dist >= -self.config.finger_overlap_tolerance_m

    def _attached_world_spheres(
        self, scratch: SceneQuery, attached: AttachedObjectState
    ) -> list[tuple[np.ndarray, float]]:
        """The carried object's fitted spheres placed in the world at whatever
        configuration ``scratch`` currently holds -- i.e. carried rigidly by the
        hand, not left at the object's stale live pose."""
        hand_pos, hand_quat = scratch.site_pose(self.ee_site)
        obj_pos, obj_quat = compose_world_pose(
            hand_pos, hand_quat, attached.grasp_transform_hand_from_object
        )
        R_obj = quat_to_rotmat_wxyz(obj_quat)
        return [
            (obj_pos + R_obj @ local_offset, radius)
            for local_offset, radius in attached.collision_spheres_object_frame
        ]

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
        world_spheres = self._attached_world_spheres(scratch, attached)

        permitted_body = goal_container_body if phase == CollisionPhase.PLACE_APPROACH else None

        violations: list[Contact] = []
        min_clearance = math.inf
        for body_name in self._obstacle_body_names:
            if body_name == attached.object_name or body_name == permitted_body:
                continue
            # include_children=False: `_obstacle_body_names` already lists
            # every body individually (parents and their descendants both
            # get their own entry), so an inclusive per-body query here
            # would recheck a descendant's own geoms a second time under
            # each of its ancestors too. That's only redundant work for an
            # ordinary body -- but for one whose ancestor is *not* excluded
            # while the descendant itself *is* (a real, confirmed case: a
            # fixture's world-parented root can be an empty pass-through
            # frame with a permitted_body match only on its real,
            # geometry-owning child -- see
            # articulation.discover_cabinet_support_bodies's own docstring),
            # the ancestor's inclusive query silently re-admits the exact
            # geoms the child-level exclusion above was supposed to permit.
            # Confirmed via a real rollout: every "place on top of the
            # cabinet" candidate directly on its own measured top surface
            # kept reading as colliding with that same surface, because the
            # empty root body (never excluded, since only its child matched
            # `permitted_body`) still enumerated the child's real geoms.
            for gid in scratch.body_collision_geom_ids(body_name, include_children=False):
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
        finger_joint_names: list[str] | None = None,
        finger_qpos: np.ndarray | None = None,
        permitted_contact_geom_ids: set[int] | None = None,
    ) -> CollisionResult:
        """Validity check at one configuration.

        ``permitted_contact_geom_ids`` are geoms the *gripper* is allowed to
        press against outright -- not merely touch with its pads. Opening a
        drawer is the case that needs it: the handle is the thing the hand
        is deliberately going for, and a hook-style pull works by bearing on
        it rather than by a clean pinch, so contact there is the intent, not
        a failure. Kept to an explicit geom set (the handle assembly) rather
        than the whole target body, so this cannot silently excuse planning
        the hand through a drawer's front panel or side walls.

        ``finger_joint_names``/``finger_qpos`` evaluate the pose with the
        gripper held at a specific opening instead of whatever the live
        scene happened to be at (see ``check_gripper_closure``'s docstring:
        only the arm joints are otherwise set, so the fingers stay wherever
        the clone inherited them -- in practice fully open). A caller that
        intends to *approach* with partly-closed fingers has to be checked
        that way or the check answers a question about a hand shape it will
        never actually be in: measured on the middle drawer, the open
        gripper's ~8cm span cannot fit the 4.0cm gap above its handle at
        all, while a narrowed one has room to spare.

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
        if finger_joint_names is not None and finger_qpos is not None:
            scratch.set_qpos(list(finger_joint_names), np.asarray(finger_qpos, dtype=np.float64))

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
            if permitted_contact_geom_ids and (
                (g1 in permitted_contact_geom_ids and g2 in self.gripper_geom_ids)
                or (g2 in permitted_contact_geom_ids and g1 in self.gripper_geom_ids)
            ):
                continue
            if self._finger_overlap_within_tolerance(g1, g2, phase, target_object_geom_ids, dist):
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
                    if d < margin and not self._finger_overlap_within_tolerance(
                        g1, g2, phase, target_object_geom_ids, d
                    ):
                        violations.append(Contact(scratch.geom_name(g1), scratch.geom_name(g2), d))

        # 1b. STL keep-out zones (LIBERO-Constrained G(Safe(obj))).
        #
        # Checked on the end-effector site and on the carried object's fitted
        # spheres -- the same two signals the benchmark scores (safe_ee /
        # safe_object). The arm's links are deliberately NOT checked: the
        # metric does not score them, and a 6cm cylinder that the elbow must
        # also respect rejects far more of the configuration space than the
        # constraint actually asks for.
        #
        # This runs inside the OMPL state-validity path, so RRTConnect plans
        # around the zone, and inside the live MPC's own check() call, so
        # tracking is held to it too. Cost is one FK-derived site read plus a
        # 2D norm per zone -- negligible next to the contact pass above.
        zones = getattr(self.config, "keepout_zones", None) or []
        if zones:
            ee_xy = scratch.site_pose(self.ee_site)[0][:2]
            attached_name = attached.object_name if attached is not None else None
            for zone in zones:
                if attached_name is not None and attached_name in zone.exempt_bodies:
                    continue
                d = zone.clearance(ee_xy)
                min_clearance = min(min_clearance, d)
                if d < 0.0:
                    violations.append(Contact(self.ee_site, f"keepout:{zone.name}", d))
                if attached is None:
                    continue
                # The carried object sweeps its own footprint through the zone
                # even when the gripper itself stays clear, which is exactly
                # the case safe_object measures.
                for center, radius in self._attached_world_spheres(scratch, attached):
                    d_obj = zone.clearance(center[:2]) - float(radius)
                    min_clearance = min(min_clearance, d_obj)
                    if d_obj < 0.0:
                        violations.append(
                            Contact(attached.object_name, f"keepout:{zone.name}", d_obj)
                        )

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
        finger_joint_names: list[str] | None = None,
        finger_qpos: np.ndarray | None = None,
        permitted_contact_geom_ids: set[int] | None = None,
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
                finger_joint_names=finger_joint_names,
                finger_qpos=finger_qpos,
                permitted_contact_geom_ids=permitted_contact_geom_ids,
            )
            if result.in_collision:
                return result
        return CollisionResult(in_collision=False, contacts=[], min_clearance=math.inf)

    def check_gripper_closure(
        self,
        arm_q: np.ndarray,
        finger_joint_names: list[str],
        finger_closed_qpos: np.ndarray,
        target_object_body: str | None = None,
    ) -> CollisionResult:
        """Simulate fully closing the gripper at a candidate arm
        configuration and check for real, non-permitted contact.

        Neither ``IKSolver``, ``MotionPlanner``, nor ``JointSpaceMPC`` ever
        include the finger joints in their planned state space -- ``check()``/
        ``check_edge()`` only ever set ``self.joint_names`` (the 7 arm
        joints) on the scratch clone, so the fingers sit at whatever qpos the
        clone happened to inherit (the live gripper's actual position at
        snapshot time -- roughly open, since closing only ever happens as a
        final, separate, non-collision-aware action). A candidate grasp
        configuration can therefore pass every existing check -- its *open*-
        finger geometry clears the target fine -- and still be physically
        unreachable: closing the fingers from there can catch the *rigid*
        finger segment (which reaches further than the pad, see
        ``scene_query.finger_reach_below_tcp_m``) on geometry the open
        fingers swept past cleanly. This validates that specific, previously
        unchecked transition, using the exact same permitted-contact rules
        ``check()`` already applies (fingerpad<->target contact and small
        finger-overlap tolerance still allowed; anything else counts).
        """
        scratch = self.base_scene.clone()
        scratch.set_qpos(self.joint_names, arm_q)
        scratch.set_qpos(finger_joint_names, np.asarray(finger_closed_qpos, dtype=np.float64))

        target_object_geom_ids: set[int] = (
            set(scratch.body_geom_ids(target_object_body)) if target_object_body else set()
        )
        violations: list[Contact] = []
        min_clearance = math.inf
        for g1_name, g2_name, dist in scratch.compute_contacts():
            g1, g2 = scratch.geom_id(g1_name), scratch.geom_id(g2_name)
            if g1 not in self.robot_geom_ids and g2 not in self.robot_geom_ids:
                continue
            min_clearance = min(min_clearance, dist)
            if dist >= 0:
                continue
            if self._permitted(g1, g2, CollisionPhase.TERMINAL_APPROACH, target_object_geom_ids):
                continue
            if self._finger_overlap_within_tolerance(
                g1, g2, CollisionPhase.TERMINAL_APPROACH, target_object_geom_ids, dist
            ):
                continue
            violations.append(Contact(g1_name, g2_name, dist))

        return CollisionResult(in_collision=len(violations) > 0, contacts=violations, min_clearance=min_clearance)
