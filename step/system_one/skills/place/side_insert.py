"""Planning for placing into a side-opening container (e.g. a microwave):
the object is set down on the table in front of the opening, re-grasped by
its handle with the hand pointing in, then slid straight in. This picks the
staging spot by checking every pose of that sequence for IK and arm collision
up front, since a neighbor behind the handle can block the re-grasp."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...motion.collision import CollisionChecker, CollisionPhase
from ...motion.kinematics import IKSolver
from ...motion.panda import FINGER_JOINT_NAMES, GRIPPER_BODIES, ROBOT_BODIES
from ...motion.scene_query import SceneQuery, quat_to_rotmat_wxyz
from ...motion.side_opening import SideOpening, support_top_z_below
from ..ik_utils import solve_ik_robust, yaw_quat_wxyz
from ..pick.grasp_generator import HandleSideGraspGenerator, _horizontal_alignment


@dataclass
class SideInsertPlan:
    final_object_xyz: np.ndarray
    insert_distance_m: float
    # Only set when the object first has to be set down and re-grasped by its handle.
    staging_object_xyz: np.ndarray | None = None
    support_body: str | None = None
    support_top_z: float | None = None


def side_insert_targets(scene: SceneQuery, side: SideOpening, object_body: str, obj_quat: np.ndarray) -> SideInsertPlan:
    """Where the object ends up inside (opening's lateral center, region depth,
    just above the floor) and how far it slides in from fully outside."""
    d = np.asarray(side.insert_dir, dtype=float)
    lateral = np.array([-d[1], d[0], 0.0])
    upright = scene.clone()
    upright.set_free_body_qpos(object_body, scene.body_pose(object_body)[0], obj_quat)
    ext = upright.body_collision_extent_world(object_body)
    bottom_local_z = float(ext.center_offset[2] - ext.half_extents[2])
    forward_extent = float(np.dot(ext.center_offset, d) + np.dot(ext.half_extents, np.abs(d)))
    final_xy = side.region_center[:2].copy()
    if side.opening_center_xy is not None:
        final_xy += lateral[:2] * float(np.dot(side.opening_center_xy - side.region_center[:2], lateral[:2]))
    return SideInsertPlan(
        final_object_xyz=np.array([final_xy[0], final_xy[1], side.floor_z + 0.01 - bottom_local_z]),
        insert_distance_m=side.front_offset_m + forward_extent + 0.03,
    )


def _support_body_below(scene: SceneQuery, xy: np.ndarray, top_z: float, exclude: str) -> str | None:
    for gid in range(scene.model.ngeom):
        if scene.model.geom_contype[gid] == 0 and scene.model.geom_conaffinity[gid] == 0:
            continue
        c, h = scene.geom_world_aabb(gid)
        if abs(c[2] + h[2] - top_z) < 1e-6 and abs(xy[0] - c[0]) <= h[0] and abs(xy[1] - c[1]) <= h[1]:
            body = scene.model.body(int(scene.model.geom_bodyid[gid])).name
            if body != exclude:
                return body
    return None


def plan_side_insert(
    scene: SceneQuery,
    side: SideOpening,
    object_body: str,
    joint_names: list[str],
    ee_site: str,
    collision_config,
    standoff_m: float,
    clearances_m: tuple[float, ...] = (0.12, 0.09, 0.15),
    lateral_offsets_m: tuple[float, ...] = (0.0, 0.03, -0.03, 0.06, -0.06),
    rejections: list[str] | None = None,
    home_q: np.ndarray | None = None,
) -> SideInsertPlan | None:
    d = np.asarray(side.insert_dir, dtype=float)
    lateral = np.array([-d[1], d[0], 0.0])
    obj_pos, held_quat = scene.body_pose(object_body)
    # Staged upright: keep only the yaw of however it sits in the hand now.
    held_R = quat_to_rotmat_wxyz(held_quat)
    obj_quat = yaw_quat_wxyz(float(np.arctan2(held_R[1, 0], held_R[0, 0])))
    targets = side_insert_targets(scene, side, object_body, obj_quat)
    final_obj, insert_distance = targets.final_object_xyz, targets.insert_distance_m
    upright = scene.clone()
    upright.set_free_body_qpos(object_body, obj_pos, obj_quat)
    ext = upright.body_collision_extent_world(object_body)
    bottom_local_z = float(ext.center_offset[2] - ext.half_extents[2])
    rng = np.random.default_rng(0)
    # The re-grasp starts from home, so seed IK there, not from the carrying pose.
    seed_q = home_q if home_q is not None else scene.get_qpos(joint_names)

    for clearance in clearances_m:
        for dx in lateral_offsets_m:
            xy = side.region_center[:2] - d[:2] * (side.front_offset_m + clearance) + lateral[:2] * dx
            top = support_top_z_below(scene, xy, side.floor_z, {object_body})
            if top is None:
                continue
            support = _support_body_below(scene, xy, top, object_body)
            staged = np.array([xy[0], xy[1], top - bottom_local_z])
            scratch = scene.clone()
            # Fingers are open for the re-grasp approach, not closed as they are while carrying.
            scratch.set_qpos(FINGER_JOINT_NAMES, np.array([0.04, -0.04]))
            scratch.set_free_body_qpos(object_body, staged, obj_quat)
            checker = CollisionChecker(scratch, joint_names, ee_site, ROBOT_BODIES, GRIPPER_BODIES, collision_config)
            ik = IKSolver(joint_names=joint_names, site_name=ee_site)
            for cand in HandleSideGraspGenerator().generate(scratch, [object_body]):
                if _horizontal_alignment(cand.approach_direction_world, d) < 0.8:
                    continue
                hand_from_obj = cand.grasp_position - staged
                final_hand = final_obj + hand_from_obj
                # (hand position, phase, where the object is at that moment)
                poses = [
                    (cand.grasp_position, CollisionPhase.TERMINAL_APPROACH, staged),
                    (cand.grasp_position - standoff_m * cand.approach_direction_world, CollisionPhase.FREE_SPACE, staged),
                    (final_hand - d * insert_distance, CollisionPhase.FREE_SPACE, final_obj - d * insert_distance),
                    (final_hand, CollisionPhase.FREE_SPACE, final_obj),
                ]
                ok = True
                q = seed_q
                for label, (pos, phase, obj_at) in zip(("grasp", "pre_grasp", "pre_insert", "inserted"), poses):
                    scratch.set_free_body_qpos(object_body, obj_at, obj_quat)
                    checker = CollisionChecker(scratch, joint_names, ee_site, ROBOT_BODIES, GRIPPER_BODIES, collision_config)
                    reason, sol = f"{label}_ik_failed", None
                    # IK only fixes the hand pose; different seeds give different
                    # elbow/wrist placements, so try several before rejecting.
                    for attempt in range(8):
                        seed = q if attempt == 0 else rng.uniform(*scene.joint_qpos_range(joint_names))
                        candidate_sol = solve_ik_robust(ik, scratch, pos, cand.grasp_quat_wxyz, seed, rng, n_restarts=2)
                        if not candidate_sol.success:
                            continue
                        check = checker.check(candidate_sol.q, phase=phase, target_object_body=object_body)
                        if check.in_collision:
                            reason = f"{label}_collision:{check.contacts[0]}"
                            continue
                        reason, sol = None, candidate_sol
                        break
                    if reason is not None:
                        if rejections is not None:
                            rejections.append(f"clearance={clearance} dx={dx}: {reason}")
                        ok = False
                        break
                    q = sol.q
                if ok:
                    return SideInsertPlan(final_obj, insert_distance, staged, support, top)
    return None
