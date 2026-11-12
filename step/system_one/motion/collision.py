"""3D collision checking on a scratch MuJoCo model/data (never the live sim).

Three checks: mj_collision for real penetration (respects the model's own
self-collision exclusions); mj_geomDistance for margin-based clearance
without mutating the shared model.geom_margin; and sphere-approximated
attached-object collision once an object is grasped, since checking its real
(often many-geom) mesh against every environment geom per query is too slow
(see attached_object.py).
"""

from __future__ import annotations

import math
import os
from enum import Enum

import mujoco
import numpy as np

from .config import CollisionConfig

# VLM-MPC-cost baseline: keep the zones visible to the motion layer but stop
# them from being hard validity failures. Read once at import; the benchmark
# driver sets it per process.
_KEEPOUT_SOFT_ONLY = os.environ.get("STEP_KEEPOUT_SOFT_ONLY", "").strip() not in ("", "0")
from .scene_query import SceneQuery, quat_to_rotmat_wxyz
from .types import AttachedObjectState, CollisionResult, Contact


class CollisionPhase(str, Enum):
    FREE_SPACE = "free_space"  # every real obstacle must be avoided, no intentional contact yet
    TERMINAL_APPROACH = "terminal_approach"  # pick: gripper<->target-object contact permitted
    PLACE_APPROACH = "place_approach"  # place: attached-object<->goal-container contact permitted


def _same_body(body_name: str | None, entity: str) -> bool:
    """Whether a MuJoCo body name (akita_black_bowl_1_main) is the scene
    entity a plan names (akita_black_bowl_1)."""
    if not body_name:
        return False
    return body_name == entity or body_name.removesuffix("_main") == entity


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
    """Inverse of compose_world_pose: given the *desired* world pose of a
    rigidly-held object and the fixed hand-from-object grasp transform, solve
    for the hand pose that achieves it. Used by PlaceSkill to turn a
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
        # Only the fingerPADS, matching robosuite's _check_grasp convention --
        # permitting the whole gripper would mask the palm descending into the object.
        self.fingerpad_geom_ids: set[int] = {
            g for g in self.gripper_geom_ids if scene.geom_name(g).endswith("_pad_collision")
        }
        # Every finger geom (pad + rigid segment above it), never the palm --
        # used only for the small compliance tolerance below; real fingers have give, a rigid palm doesn't.
        self.finger_geom_ids: set[int] = {g for g in self.gripper_geom_ids if "finger" in scene.geom_name(g)}
        # The hand/palm shell: every gripper geom that is not part of a finger.
        # Kept separate because contact here means something different -- a
        # finger closing through an object is the grasp, a palm reaching it is
        # the object being pressed into the hand.
        self.palm_geom_ids: set[int] = {
            g for g in self.gripper_geom_ids if g not in self.finger_geom_ids
        }
        self.robot_geom_ids |= self.gripper_geom_ids
        self.robot_body_ids: set[int] = {scene.body_id(b) for b in set(robot_body_names) | set(gripper_body_names)}
        # Candidate obstacles for the attached-object sphere check, cached
        # once since the body set is static (only poses change per query).
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
        """Whether (g1, g2) is an intentional fingerpad<->target contact, not
        a violation. Matches by geom-id membership, not name substrings --
        LIBERO object geoms (e.g. alphabet_soup_1_g7) don't substring-match
        their own body name (alphabet_soup_1_main).
        """
        if (
            phase == CollisionPhase.TERMINAL_APPROACH
            and self.config.permit_gripper_target_contact
            and (
                (g1 in self.finger_geom_ids and g2 in target_object_geom_ids)
                or (g2 in self.finger_geom_ids and g1 in target_object_geom_ids)
            )
        ):
            return True
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
        """Tolerates a small, configurable interpenetration between a FINGER
        geom (pad or rigid segment, never the palm) and the grasp target
        during TERMINAL_APPROACH, mirroring real gripper compliance -- a
        rigid check would reject any object slightly wider than the
        gripper's rated opening."""
        if phase != CollisionPhase.TERMINAL_APPROACH or self.config.finger_overlap_tolerance_m <= 0.0:
            return False
        finger_touch = g1 in self.finger_geom_ids or g2 in self.finger_geom_ids
        if not finger_touch:
            return False
        other = g2 if g1 in self.finger_geom_ids else g1
        if other not in target_object_geom_ids:
            return False
        return dist >= -self.config.finger_overlap_tolerance_m

    def _palm_overlap_within_tolerance(
        self,
        g1: int,
        g2: int,
        phase: CollisionPhase,
        target_object_geom_ids: set[int],
        dist: float,
    ) -> bool:
        """The finger allowance above, for the palm, against the target only.

        Off unless ``palm_overlap_tolerance_m`` is set, so nothing changes for a
        caller that does not ask. Where it is set, it is bounded by what real
        successful grasps do: RoboCasa's own demonstrations close on an eggplant
        with the palm 1-7 mm inside it and lift it, so a pose reproducing that is
        a grasp, not a fault. Still only ever the grasp target -- the palm meeting
        the counter, a neighbouring object or the robot stays a hard collision.
        """
        if phase != CollisionPhase.TERMINAL_APPROACH or self.config.palm_overlap_tolerance_m <= 0.0:
            return False
        palm_touch = g1 in self.palm_geom_ids or g2 in self.palm_geom_ids
        if not palm_touch:
            return False
        other = g2 if g1 in self.palm_geom_ids else g1
        if other not in target_object_geom_ids:
            return False
        return dist >= -self.config.palm_overlap_tolerance_m

    def _attached_world_spheres(
        self, scratch: SceneQuery, attached: AttachedObjectState
    ) -> list[tuple[np.ndarray, float]]:
        """The carried object's fitted spheres placed in the world at whatever
        configuration scratch currently holds -- i.e. carried rigidly by the
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
        other body in the scene. Returns (violations, min_clearance).

        Checked per-geom, not per merged body AABB: a body assembled from
        several geoms (e.g. a basket's floor + walls) has empty space between
        them that a merged box would flatten into solid, falsely flagging a
        hover pose over an open container as colliding."""
        world_spheres = self._attached_world_spheres(scratch, attached)

        permitted_body = goal_container_body if phase == CollisionPhase.PLACE_APPROACH else None

        violations: list[Contact] = []
        min_clearance = math.inf
        for body_name in self._obstacle_body_names:
            if body_name == attached.object_name or body_name == permitted_body:
                continue
            # include_children=False: bodies are already listed individually;
            # an inclusive query would re-admit a child's excluded geoms via its ancestor frame.
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

        permitted_contact_geom_ids lets the *gripper* press directly on
        specific geoms (e.g. a drawer handle during a hook-style pull),
        without excusing contact elsewhere on the same body.

        finger_joint_names/finger_qpos evaluate the gripper at a
        specific opening instead of its default (fully open) qpos.

        check_margin=False (the default; used ~1e3-1e4 times per OMPL
        plan) only enforces actual penetration. The attached-object sphere
        check always applies its margin regardless, since it's cheap.
        """
        scratch = self.base_scene.clone()
        scratch.set_qpos(self.joint_names, q)
        if finger_joint_names is not None and finger_qpos is not None:
            scratch.set_qpos(list(finger_joint_names), np.asarray(finger_qpos, dtype=np.float64))

        target_object_geom_ids: set[int] = set()
        if target_object_body:
            target_object_geom_ids = set(scratch.body_geom_ids(target_object_body))
        # The attached object's real geoms still sit at their live (pre-grasp)
        # pose in the clone, not this candidate q, so they're excluded below.
        attached_real_geom_ids: set[int] = (
            set(scratch.body_geom_ids(attached.object_name)) if attached is not None else set()
        )

        # 1. Hard penetration via mj_collision, restricted to contacts involving
        # the robot/gripper -- other object-vs-object contact is pre-existing scene state.
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
            if self._palm_overlap_within_tolerance(g1, g2, phase, target_object_geom_ids, dist):
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
                    ) and not self._palm_overlap_within_tolerance(
                        g1, g2, phase, target_object_geom_ids, d
                    ):
                        violations.append(Contact(scratch.geom_name(g1), scratch.geom_name(g2), d))

        # 1b. STL keep-out zones (G(Safe(obj))): checked on the end-effector
        # site only (matches the benchmark's safe_ee signal), in both planning and live tracking.
        zones = getattr(self.config, "keepout_zones", None) or []
        carried_pose = None
        if attached is not None and (zones or getattr(self.config, "max_tilt_deg", None) is not None):
            hand_pos, hand_quat = scratch.site_pose(self.ee_site)
            carried_pose = compose_world_pose(hand_pos, hand_quat, attached.grasp_transform_hand_from_object)
        if zones:
            ee_xy = scratch.site_pose(self.ee_site)[0][:2]
            attached_name = attached.object_name if attached is not None else None
            for zone in zones:
                if attached_name is not None and attached_name in zone.exempt_bodies:
                    continue
                if zone.subject is not None:
                    # A zone on a carried object applies only while that object
                    # is the one in the hand; before the grasp it is sitting
                    # where it started, which is clear by construction.
                    if carried_pose is None or not _same_body(attached_name, zone.subject):
                        continue
                    d = zone.clearance(carried_pose[0][:2])
                    if phase != CollisionPhase.FREE_SPACE:
                        # Setting down or picking up: judged on the rule alone.
                        d += zone.buffer_m
                    min_clearance = min(min_clearance, d)
                    if d < 0.0 and not _KEEPOUT_SOFT_ONLY:
                        violations.append(Contact(f"attached:{attached_name}", f"keepout:{zone.name}", d))
                    continue
                d = zone.clearance(ee_xy)
                if phase != CollisionPhase.FREE_SPACE:
                    d += zone.buffer_m
                min_clearance = min(min_clearance, d)
                if d < 0.0 and not _KEEPOUT_SOFT_ONLY:
                    # Soft-only is the VLM-MPC-cost baseline: the zone still
                    # reaches the motion layer (so the MPC's keep-out cost can
                    # bend the trajectory) but is never a hard validity failure,
                    # which is what separates "a cost the optimizer trades off"
                    # from "a specification the planner must satisfy".
                    violations.append(Contact(self.ee_site, f"keepout:{zone.name}", d))
                # The carried object is deliberately not checked here: its
                # true pose vs. nominal grasp transform differ enough that enforcing it made runs worse.

        # 1c. STL G(Upright(obj, deg)): the carried object, placed by the grasp
        # transform at this configuration, may not tip past the limit from how
        # it stood before the pick. Checked here so every plan, shortcut and
        # goal pose is held to it, exactly as the keep-out zones are.
        max_tilt = getattr(self.config, "max_tilt_deg", None)
        up_local = getattr(attached, "up_axis_object_frame", None) if attached is not None else None
        if max_tilt is not None and carried_pose is not None and up_local is not None:
            up_world = quat_to_rotmat_wxyz(carried_pose[1]) @ np.asarray(up_local, dtype=np.float64)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up_world[2])))))
            if tilt > float(max_tilt):
                violations.append(Contact(f"attached:{attached.object_name}", "upright",
                                          math.radians(float(max_tilt) - tilt)))

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

    def attached_exact_contacts(
        self,
        q: np.ndarray,
        attached: AttachedObjectState,
        goal_container_body: str | None = None,
        drop_m: float = 0.0,
        clearance_m: float = 0.0,
    ) -> list[tuple[str, str, float]]:
        """Penetrating contacts of the carried object's real collision geoms
        (not its spheres) at configuration q, optionally lowered drop_m below
        the hand. For static goal-pose checks where packing is tight: the
        spheres make a moka pot ~25% wider than it is, enough to rule out
        two pots side by side on a burner. Gripper and goal-surface contact are
        ignored. clearance_m also tests the object shifted that far each way
        horizontally -- contacts only register on penetration, so touching
        distance would otherwise pass with no slack for set-down drift."""
        scratch = self.base_scene.clone()
        scratch.set_qpos(self.joint_names, q)
        scratch.forward()
        hand_pos, hand_quat = scratch.site_pose(self.ee_site)
        obj_pos, obj_quat = compose_world_pose(hand_pos, hand_quat, attached.grasp_transform_hand_from_object)
        obj_geoms = set(scratch.body_collision_geom_ids(attached.object_name))
        ignored = set(self.robot_geom_ids)
        if goal_container_body is not None:
            ignored |= set(scratch.body_collision_geom_ids(goal_container_body))
        base = obj_pos - np.array([0.0, 0.0, drop_m])
        shifts = [np.zeros(3)]
        if clearance_m > 0.0:
            shifts += [clearance_m * np.array(v, dtype=float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
        out = []
        for shift in shifts:
            scratch.set_free_body_qpos(attached.object_name, base + shift, obj_quat)
            mujoco.mj_collision(scratch.model, scratch.data)
            for i in range(scratch.data.ncon):
                c = scratch.data.contact[i]
                i1, i2, dist = int(c.geom1), int(c.geom2), float(c.dist)
                if (i1 in obj_geoms) == (i2 in obj_geoms):
                    continue
                other = i2 if i1 in obj_geoms else i1
                if other in ignored or dist >= 0.0:
                    continue
                out.append((scratch.geom_name(i1), scratch.geom_name(i2), dist))
            if out:
                break
        return out

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
        """Simulate fully closing the gripper at a candidate arm config and
        check for real, non-permitted contact.

        check()/check_edge() never plan the finger joints, so a
        candidate can pass with open-finger geometry clear yet still catch
        the rigid segment above the pad once closed -- this validates that
        specific transition.
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
            if self._palm_overlap_within_tolerance(
                g1, g2, CollisionPhase.TERMINAL_APPROACH, target_object_geom_ids, dist
            ):
                continue
            violations.append(Contact(g1_name, g2_name, dist))

        return CollisionResult(in_collision=len(violations) > 0, contacts=violations, min_clearance=min_clearance)
