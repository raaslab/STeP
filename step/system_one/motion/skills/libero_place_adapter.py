"""Executor-facing adapter for the ``place`` skill: wires ``PlaceSkill``
(grasp-aware OMPL planning that treats the carried object as part of the
robot, via its sphere approximation) into a live ``LiberoManipulationEnv``.

Shares its low-level stepping/servo pattern with ``libero_pick_adapter.py``
(same Cartesian-delta bridge, same reasons -- see that module's docstring);
duplicated here rather than factored into a shared base class to keep this
integration slice small and independently reviewable. Consolidating the two
adapters' shared plumbing is a reasonable follow-up cleanup once both are
proven out.
"""

from __future__ import annotations

import numpy as np

from ...orientation_utils import orientation_error_rotvec
from ..collision import CollisionChecker, CollisionPhase
from ..config import MotionConfig
from ..kinematics import IKSolver
from ..mpc import JointSpaceMPC, MPCContext
from ..ompl_planner import MotionPlanner
from ..panda import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES
from .place_skill import PlaceSkill
from ..scene_query import SceneQuery, quat_to_rotmat_wxyz, resolve_body_name
from ..time_parameterize import TimeParameterizer
from ..types import AttachedObjectState, MPCCostConfig, SkillFailureReason, SkillResult


class LiberoPlaceAdapter:
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _servo_toward_joint_target(
        self, scene: SceneQuery, target_q: np.ndarray, pos_scale_m: float = 0.03, ori_gain: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray]:
        scratch = scene.clone()
        scratch.set_qpos(self.joint_names, target_q)
        target_pos, target_quat = scratch.site_pose(self.ee_site)
        target_R = quat_to_rotmat_wxyz(target_quat)
        ee_pos = self.env._eef_pos(self.env.last_raw_obs)
        ee_R = self.env._eef_rotmat(self.env.last_raw_obs)
        cart_pos = np.clip((target_pos - ee_pos) / pos_scale_m, -1.0, 1.0).astype(np.float32)
        ori_err = orientation_error_rotvec(ee_R, target_R)
        cart_ori = np.clip(ori_gain * ori_err, -1.0, 1.0).astype(np.float32)
        return cart_pos, cart_ori

    def _servo_toward_joint_target_with_error(
        self, scene: SceneQuery, target_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        scratch = scene.clone()
        scratch.set_qpos(self.joint_names, target_q)
        target_pos, target_quat = scratch.site_pose(self.ee_site)
        target_R = quat_to_rotmat_wxyz(target_quat)
        ee_pos = self.env._eef_pos(self.env.last_raw_obs)
        ee_R = self.env._eef_rotmat(self.env.last_raw_obs)
        pos_err = target_pos - ee_pos
        cart_pos = np.clip(pos_err / 0.03, -1.0, 1.0).astype(np.float32)
        ori_err_vec = orientation_error_rotvec(ee_R, target_R)
        cart_ori = np.clip(2.0 * ori_err_vec, -1.0, 1.0).astype(np.float32)
        return cart_pos, cart_ori, float(np.linalg.norm(pos_err)), float(np.linalg.norm(ori_err_vec))

    def _step_env(self, cart_pos: np.ndarray, cart_ori: np.ndarray, gripper_cmd: float) -> dict:
        full_action = self.env._compose_full_action(cart_pos, cart_ori, gripper_cmd)
        obs, _reward, _terminated, info = self.env._step_backend(full_action)
        self.env.last_raw_obs = obs
        return info

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

        checker = CollisionChecker(scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision)
        lo, hi = scene.joint_qpos_range(self.joint_names)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        skill = PlaceSkill(ik, planner, checker, self.config)

        start_q = scene.get_qpos(self.joint_names)
        result = skill.execute(
            scene, attached, goal_container_body, start_q, place_surface=place_surface, goal_point_xyz=goal_point_xyz
        )
        if not result.success:
            return result, 0

        traj = TimeParameterizer(self.config.time_param).parameterize(result.planned_trajectory)
        if traj.positions is None:
            result.success = False
            result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            result.diagnostics["detail"] = "time parameterization produced no reference"
            return result, 0

        hover_q = np.asarray(result.diagnostics["hover_q"])
        place_q = np.asarray(result.diagnostics["place_q"])
        approach_positions = np.asarray(result.diagnostics["approach_positions"])
        full_reference = np.concatenate([traj.positions, approach_positions[1:]], axis=0)
        approach_start_idx = len(traj.positions)

        steps = 0
        # The plan's reference starts from the *escaped* start_q PlaceSkill
        # computed (see escape_start_collision's own docstring), which can
        # differ from wherever the real arm currently sits. Move there
        # directly first -- see LiberoTransportAdapter's own identical
        # comment for why skipping this can leave the MPC reconciling a gap
        # between "where the plan assumes it starts" and "where the arm
        # actually is" via a large, uncontrolled tracking error instead of
        # smoothly following the validated collision-free path.
        plan_start_q = np.asarray(result.planned_trajectory.metadata.get("plan_start_q", start_q))
        if float(np.linalg.norm(plan_start_q - start_q)) > 1e-6:
            escape_reference = np.linspace(start_q, plan_start_q, 20)
            for target_q in escape_reference:
                live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
                cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, target_q)
                self._step_env(cart_pos, cart_ori, self.env.close_gripper_cmd)
                steps += 1
                if step_callback is not None:
                    step_callback()

        mpc = JointSpaceMPC(
            self.joint_names,
            self.ee_site,
            horizon=int(self.config.mpc.get("horizon", 16)),
            cost_config=MPCCostConfig(**{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}),
        )
        ctx = MPCContext(reference=full_reference)
        mpc.reset()

        gripper_cmd = self.env.close_gripper_cmd  # still holding the object throughout transport/descent
        for t0 in range(len(full_reference)):
            phase = CollisionPhase.FREE_SPACE if t0 < approach_start_idx else CollisionPhase.PLACE_APPROACH
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            q_current = live_scene.get_qpos(self.joint_names)
            mpc_result = mpc.step(
                live_scene,
                q_current,
                ctx,
                t0_step=t0,
                collision_checker=checker,
                collision_phase=phase,
                goal_container_body=goal_container_body,
                attached=attached,
                replan_config=self.config.replan,
            )
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, mpc_result.joint_command)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            if steps >= max_steps:
                break

        # Settle at the true place pose before releasing -- same reasoning as
        # the pick adapter's settle phase: the MPC's steady-state tracking lag
        # means the reference trajectory running out doesn't guarantee the
        # arm has actually arrived, and releasing early can drop the object
        # short of (or outside) the intended spot.
        cfg = self.config.execution
        settle_steps = 0
        pos_err = ori_err = float("inf")
        for _ in range(cfg.max_settle_steps):
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            cart_pos, cart_ori, pos_err, ori_err = self._servo_toward_joint_target_with_error(live_scene, place_q)
            if pos_err <= cfg.settle_position_tol_m and ori_err <= cfg.settle_orientation_tol_rad:
                break
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            settle_steps += 1
            if step_callback is not None:
                step_callback()
        result.diagnostics["settle"] = {
            "steps": settle_steps,
            "converged": pos_err <= cfg.settle_position_tol_m and ori_err <= cfg.settle_orientation_tol_rad,
            "position_error_m": pos_err,
            "orientation_error_rad": ori_err,
        }

        # Release and dwell/retreat a touch so the object settles before
        # anything downstream reads its resting pose.
        gripper_cmd = self.env.open_gripper_cmd
        for _ in range(int(self.env.close_steps) * 2):
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, place_q)
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
