"""Executor-facing adapter for the ``place`` skill.

Deliberately simple: compute the object's desired final pose (goal surface
height + how far the carried object's own collision geometry sits below its
reference point), convert that to a hand target via the fixed grasp
transform, and servo straight there in small steps. No OMPL plan, no
MPC-tracked reference, no separate raw-proportional "settle" phase.

This replaces an earlier OMPL-planned/MPC-tracked/settle pipeline that hit a
real, repeated bug: the carried object separating from the gripper mid-
descent with zero contacts recorded anywhere -- not a collision, not a weak
grip, some kind of abrupt commanded discontinuity buried in that machinery
that resisted several targeted fixes (widening the settle loop's speed cap,
etc.). Going straight to the target in small, bounded steps (the same dense-
ramp pattern already proven safe for PickSkill's post-grasp carry lift) is a
much smaller surface area for that class of bug, at the cost of not routing
around obstacles the way a real motion plan would -- acceptable since
transport already leaves the arm hovering clear above the goal first.
"""

from __future__ import annotations

import numpy as np

from ...orientation_utils import orientation_error_rotvec
from ..collision import required_hand_pose_for_object_pose
from ..config import MotionConfig
from ..panda import EE_SITE, JOINT_NAMES
from .place_skill import _container_floor_and_rim_z, _support_floor_and_rim_z
from ..scene_query import SceneQuery, quat_to_rotmat_wxyz, resolve_body_name
from ..types import AttachedObjectState, SkillFailureReason, SkillResult


class LiberoPlaceAdapter:
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _step_env(self, cart_pos: np.ndarray, cart_ori: np.ndarray, gripper_cmd: float) -> dict:
        full_action = self.env._compose_full_action(cart_pos, cart_ori, gripper_cmd)
        obs, _reward, _terminated, info = self.env._step_backend(full_action)
        self.env.last_raw_obs = obs
        return info

    def _servo_toward_cartesian_target(
        self, target_pos: np.ndarray, target_R: np.ndarray, pos_scale_m: float
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        ee_pos = self.env._eef_pos(self.env.last_raw_obs)
        ee_R = self.env._eef_rotmat(self.env.last_raw_obs)
        pos_err_vec = target_pos - ee_pos
        ori_err_vec = orientation_error_rotvec(ee_R, target_R)
        cart_pos = np.clip(pos_err_vec / pos_scale_m, -1.0, 1.0).astype(np.float32)
        cart_ori = np.clip(2.0 * ori_err_vec, -1.0, 1.0).astype(np.float32)
        return cart_pos, cart_ori, float(np.linalg.norm(pos_err_vec)), float(np.linalg.norm(ori_err_vec))

    def run(
        self,
        goal_container_name: str,
        attached: AttachedObjectState,
        max_steps: int = 400,
        step_callback=None,
        place_surface: str = "inside",
        goal_point_xyz=None,
    ) -> tuple[SkillResult, int]:
        scene = self._build_scene()
        goal_container_body = resolve_body_name(scene, goal_container_name)
        pcfg = self.config.place
        steps = 0

        # Desired final object pose: goal XY (or an explicit point) at the
        # target surface's real height, raised by however far the carried
        # object's own collision geometry (not its visual mesh -- see
        # config.py's settle_pos_scale_m comment for why that distinction
        # keeps recurring) extends below its own reference point.
        container_pos, _ = scene.body_pose(goal_container_body)
        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        floor_z, _rim_z = floor_fn(scene, goal_container_body)
        obj_extent = scene.body_collision_extent_world(attached.object_name)
        obj_bottom_local_z = (
            float(obj_extent.center_offset[2] - obj_extent.half_extents[2]) if obj_extent is not None else -0.02
        )
        place_z = floor_z - obj_bottom_local_z + pcfg.drop_clearance_m

        if goal_point_xyz is not None:
            target_xy = np.asarray(goal_point_xyz, dtype=float)[:2]
        else:
            target_xy = container_pos[:2]
        object_xyz = np.array([target_xy[0], target_xy[1], place_z])

        object_quat_now = scene.body_pose(attached.object_name)[1]
        place_hand_pos, place_hand_quat = required_hand_pose_for_object_pose(
            object_xyz, object_quat_now, attached.grasp_transform_hand_from_object
        )
        place_R = quat_to_rotmat_wxyz(place_hand_quat)

        result = SkillResult(success=True, selected_object=attached.object_name, diagnostics={})
        result.diagnostics["object_xyz"] = object_xyz.tolist()
        result.diagnostics["place_hand_pos"] = place_hand_pos.tolist()

        # Bounded-step descent straight to the target -- see module
        # docstring. Deliberately closed-loop, not a precomputed linspace: a
        # fixed schedule of waypoints assumes the arm covers ~1cm per control
        # step, which it often doesn't (measured real per-step motion as low
        # as 1-2mm under load) -- advancing through the schedule regardless
        # of how far the arm actually got left the *next* waypoint further
        # and further ahead of the real position, compounding into the exact
        # same large, sustained-high-speed residual error (and resulting
        # zero-contact grasp separation) this rewrite was meant to avoid.
        # Recomputing a small step from wherever the arm *actually* is, every
        # iteration, means the commanded action is always small regardless of
        # how slowly or unevenly the real motion tracks it.
        gripper_cmd = self.env.close_gripper_cmd
        max_step_m = 0.01
        descent_tol_m = 0.002
        while steps < max_steps:
            ee_pos = self.env._eef_pos(self.env.last_raw_obs)
            to_target = place_hand_pos - ee_pos
            dist = float(np.linalg.norm(to_target))
            if dist <= descent_tol_m:
                break
            step_target = ee_pos + to_target * (min(max_step_m, dist) / dist)
            cart_pos, cart_ori, _pos_err, _ori_err = self._servo_toward_cartesian_target(
                step_target, place_R, pos_scale_m=0.03
            )
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        # Settle briefly at the true target before releasing -- the dense
        # ramp above ends with a final waypoint equal to the target, but the
        # arm can still be a few mm off from steady-state tracking lag.
        exec_cfg = self.config.execution
        settle_steps = 0
        pos_err = ori_err = float("inf")
        for _ in range(exec_cfg.max_settle_steps):
            cart_pos, cart_ori, pos_err, ori_err = self._servo_toward_cartesian_target(
                place_hand_pos, place_R, pos_scale_m=exec_cfg.settle_pos_scale_m
            )
            if pos_err <= exec_cfg.settle_position_tol_m and ori_err <= exec_cfg.settle_orientation_tol_rad:
                break
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            settle_steps += 1
            if step_callback is not None:
                step_callback()
        result.diagnostics["settle"] = {
            "steps": settle_steps,
            "converged": pos_err <= exec_cfg.settle_position_tol_m and ori_err <= exec_cfg.settle_orientation_tol_rad,
            "position_error_m": pos_err,
            "orientation_error_rad": ori_err,
        }

        # Release and dwell/retreat a touch so the object settles before
        # anything downstream reads its resting pose.
        gripper_cmd = self.env.open_gripper_cmd
        for _ in range(int(self.env.close_steps) * 2):
            cart_pos, cart_ori, _pos_err, _ori_err = self._servo_toward_cartesian_target(
                place_hand_pos, place_R, pos_scale_m=0.03
            )
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        # Withdraw before judging the release, not while still parked at the
        # place pose. _is_grasped() is a *contact* test, so an object that has
        # genuinely been let go still reads as grasped whenever it stays
        # touching the open fingers -- which is the normal case when placing
        # into an already-occupied container, where the object comes to rest
        # higher than the empty-container floor this pose was aimed at.
        # Confirmed on the real two-object libero_10 basket task: placing the
        # second object (onto the first, already in the basket) failed with
        # "gripper opened but object is still reported grasped" while the
        # place itself was fine. Lifting the hand clear first makes the test
        # measure what it is actually trying to measure -- did the object stay
        # behind -- rather than incidental contact.
        withdraw_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        withdraw_from, withdraw_quat = withdraw_scene.site_pose(self.ee_site)
        withdraw_R = quat_to_rotmat_wxyz(withdraw_quat)
        withdraw_to = withdraw_from + np.array([0.0, 0.0, self.config.place.release_withdraw_m])
        for withdraw_waypoint in np.linspace(withdraw_from, withdraw_to, 16)[1:]:
            ee_pos = self.env._eef_pos(self.env.last_raw_obs)
            ee_R = self.env._eef_rotmat(self.env.last_raw_obs)
            cart_pos = np.clip((withdraw_waypoint - ee_pos) / 0.03, -1.0, 1.0).astype(np.float32)
            cart_ori = np.clip(
                2.0 * orientation_error_rotvec(ee_R, withdraw_R), -1.0, 1.0
            ).astype(np.float32)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        released = not bool(self.env._is_grasped())
        if not released:
            result.success = False
            result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            result.diagnostics["detail"] = "gripper opened but object is still reported grasped"
            return result, steps

        # Where the object *actually* came to rest, not where we aimed. Until
        # now this skill verified only that it released the gripper at a pose
        # it had planned -- which reports success just as readily when the
        # object rolls out of a drawer, is nudged off by the retreating hand,
        # or never sat where it was dropped. The same "verify the outcome,
        # not the intent" gap let a drawer skill report the target drawer's
        # own opening fraction while it had actually dragged a different
        # drawer open. Cheap to measure, and the only honest basis for
        # calling a place successful.
        rest_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        object_rest_xyz = np.asarray(rest_scene.body_pose(attached.object_name)[0], dtype=float)
        planned_xyz = np.asarray(result.diagnostics.get("object_xyz", object_rest_xyz), dtype=float)
        goal_pos, _ = rest_scene.body_pose(goal_container_body)
        goal_extent = rest_scene.body_collision_extent_world(goal_container_body)
        drift = float(np.linalg.norm(object_rest_xyz - planned_xyz))
        over_goal_xy = None
        if goal_extent is not None:
            half = np.asarray(goal_extent.half_extents[:2], dtype=float)
            centre = np.asarray(goal_pos[:2], dtype=float) + np.asarray(goal_extent.center_offset[:2], dtype=float)
            over_goal_xy = bool(np.all(np.abs(object_rest_xyz[:2] - centre) <= half))
        result.diagnostics["object_rest_verification"] = {
            "object_rest_xyz": [float(v) for v in object_rest_xyz],
            "planned_xyz": [float(v) for v in planned_xyz],
            "drift_from_planned_m": drift,
            "within_goal_footprint_xy": over_goal_xy,
            "goal_container_body": goal_container_body,
        }

        return result, steps
