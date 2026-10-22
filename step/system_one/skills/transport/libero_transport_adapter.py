"""Executor-facing adapter for the transport skill: wires
TransportSkill's attached-object-aware hover-leg planning into a live
LiberoManipulationEnv.

Passes attached= through to both planning and live MPC collision
checking, so the carried object is checked as part of the robot's own
geometry for the whole leg. Ends holding the object at the hover pose;
release is a subsequent place subtask's job."""

from __future__ import annotations

import numpy as np

from ...motion.attached_object import attached_sphere_infeasible_for_goal, refresh_grasp_transform
from ...motion.collision import CollisionChecker
from ...motion.config import MotionConfig
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from .transport_skill import TransportSkill
from ...motion.panda import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES
from ...motion.scene_query import SceneQuery, resolve_body_name
from ...motion.servo import MotionTraceRecorder, _ServoMixin, track_and_settle
from ...motion.time_parameterize import TimeParameterizer
from ...motion.types import AttachedObjectState, MPCCostConfig, SkillFailureReason, SkillResult


class LiberoTransportAdapter(_ServoMixin):
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _build_skill(self, scene: SceneQuery) -> TransportSkill:
        checker = CollisionChecker(
            scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision
        )
        lo, hi = scene.joint_qpos_range(self.joint_names)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        return TransportSkill(ik, planner, checker, self.config)

    def run(
        self,
        goal_container_name: str,
        attached: AttachedObjectState,
        max_steps: int = 400,
        step_callback=None,
        place_surface: str = "inside",
        goal_point_xyz=None,
        goal_radius_m: float | None = None,
    ) -> tuple[SkillResult, int]:
        scene = self._build_scene()
        goal_container_body = resolve_body_name(scene, goal_container_name)
        skill = self._build_skill(scene)
        start_q = scene.get_qpos(self.joint_names)
        if self.env._is_grasped():
            refresh_grasp_transform(attached, scene, self.ee_site)
        # If the carried object's sphere can never fit the goal's radius
        # regardless of approach, stop weighing it in motion cost -- fighting an unresolvable conflict only worsens tracking.
        ignore_attached_sphere = attached_sphere_infeasible_for_goal(attached, goal_radius_m)
        mpc_attached = None if ignore_attached_sphere else attached

        # Always the real attached, never mpc_attached -- execute() needs
        # the object's real identity/grasp transform throughout, not just for sphere-avoidance.
        plan_result = skill.execute(
            scene, attached, goal_container_body, start_q,
            place_surface=place_surface, goal_point_xyz=goal_point_xyz,
        )
        if not plan_result.success:
            return plan_result, 0

        traj = TimeParameterizer(self.config.time_param).parameterize(plan_result.planned_trajectory)
        if traj.positions is None:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = "time parameterization produced no reference"
            return plan_result, 0

        hover_q = np.asarray(plan_result.planned_trajectory.metadata["hover_q"])
        hover_hand_pos = np.asarray(plan_result.diagnostics["hover_hand_pos"])
        hover_hand_quat = np.asarray(plan_result.diagnostics["hover_hand_quat_wxyz"])

        checker = skill.collision_checker
        steps = 0

        # The plan's reference starts from the escaped start_q, which can be
        # far from where the arm actually sits; move there directly first or the MPC's gap-reconciling drives the object through obstacles.
        plan_start_q = np.asarray(plan_result.planned_trajectory.metadata.get("plan_start_q", start_q))
        if float(np.linalg.norm(plan_start_q - start_q)) > 1e-6:
            escape_reference = np.linspace(start_q, plan_start_q, 20)
            for target_q in escape_reference:
                live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
                cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, target_q)
                self._step_env(cart_pos, cart_ori, self.env.close_gripper_cmd)
                steps += 1
                if step_callback is not None:
                    step_callback()
        cfg = self.config.execution
        gripper_cmd = self.env.close_gripper_cmd  # still holding the object throughout transport
        trace = MotionTraceRecorder()
        # approach_start_idx is set past the whole tracking budget -- transport
        # never enters TERMINAL_APPROACH, since the object is kept clear via attached= instead of a phase change.
        result = track_and_settle(
            self,
            traj.positions,
            joint_names=self.joint_names,
            ee_site=self.ee_site,
            approach_start_idx=len(traj.positions) + cfg.max_settle_steps,
            target_pos=hover_hand_pos,
            target_quat_wxyz=hover_hand_quat,
            collision_checker=checker,
            mpc_horizon=int(self.config.mpc.get("horizon", 16)),
            mpc_cost_config=MPCCostConfig(
                **{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}
            ),
            replan_config=self.config.replan,
            gripper_cmd=gripper_cmd,
            settle_position_tol_m=cfg.settle_position_tol_m,
            settle_orientation_tol_rad=cfg.settle_orientation_tol_rad,
            stall_window_steps=cfg.stall_window_steps,
            stall_position_tol_m=cfg.stall_position_tol_m,
            max_settle_steps=cfg.max_settle_steps,
            mpc_step_kwargs={"goal_container_body": goal_container_body, "attached": mpc_attached},
            steps_so_far=steps,
            max_steps=max_steps,
            step_callback=step_callback,
            target_object_body=goal_container_body,
            trace=trace,
            max_reference_lead_m=cfg.transport_max_reference_lead_m,
        )
        steps += result.steps
        plan_result.diagnostics["settle"] = result.as_diagnostics()
        plan_result.diagnostics["motion_trace"] = {"transport": trace.summary()}

        # Structural check: is the object actually over the goal and still
        # grasped, via measured geometry rather than a contact heuristic.
        final_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        object_pos, _ = final_scene.body_pose(attached.object_name)
        container_pos, _ = final_scene.body_pose(goal_container_body)
        over_goal_xy_err = float(np.linalg.norm(object_pos[:2] - container_pos[:2]))
        still_grasped = bool(self.env._is_grasped()) if hasattr(self.env, "_is_grasped") else None
        plan_result.diagnostics["over_goal_verification"] = {
            "xy_err_m": over_goal_xy_err,
            "still_grasped": still_grasped,
        }
        if still_grasped is False:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = "object no longer grasped after transport"
        hover_xy_err = float(np.linalg.norm(object_pos[:2] - np.asarray(plan_result.diagnostics["hover_xyz"])[:2]))
        plan_result.diagnostics["over_goal_verification"]["hover_xy_err_m"] = hover_xy_err
        if plan_result.success and hover_xy_err > self.config.transport.max_hover_xy_err_m:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = f"object ended {hover_xy_err:.2f}m from the hover target"

        plan_result.selected_object = attached.object_name
        return plan_result, steps
