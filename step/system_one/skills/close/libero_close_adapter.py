"""Executor-facing adapter for the close skill: wires CloseSkill's
push plan (drawer or microwave door) into a live LiberoManipulationEnv.
Simpler than LiberoOpenDrawerAdapter: since closing never grasps
anything, the gripper stays open for the whole motion and the plan tracks as
one continuous MPC reference (FREE_SPACE to the standoff, then
TERMINAL_APPROACH through to the closed target). There's no grip-verification
gate either -- success is purely structural, from how far the joint's qpos moved toward closed."""

from __future__ import annotations

import numpy as np

from ...motion.collision import CollisionChecker
from ...motion.config import MotionConfig
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from ...motion.panda import EE_SITE, FINGER_JOINT_NAMES, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES
from ...motion.scene_query import SceneQuery
from ...motion.servo import MotionTraceRecorder, _ServoMixin, track_and_settle
from ...motion.time_parameterize import TimeParameterizer
from ...motion.types import MotionPlan, MPCCostConfig, SkillFailureReason, SkillResult
from .close_skill import CloseSkill


class LiberoCloseAdapter(_ServoMixin):
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE
        self.hand_site = HAND_SITE

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _build_skill(self, scene: SceneQuery) -> CloseSkill:
        checker = CollisionChecker(
            scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision
        )
        lo, hi = scene.joint_qpos_range(self.joint_names)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        return CloseSkill(ik, planner, checker, self.config)

    def _servo_to_home(self, step_callback=None) -> int:
        return self.servo_to_home(self.config.close.return_home_steps, step_callback)

    def run(
        self,
        target_ref: str,
        constraints: dict | None = None,
        max_steps: int = 600,
        step_callback=None,
    ) -> tuple[SkillResult, int, dict]:
        constraints = constraints or {}

        # Plan the approach from a known, fixed starting configuration
        # rather than wherever the previous subtask left the arm.
        pre_home_steps = self._servo_to_home(step_callback)

        scene = self._build_scene()
        skill = self._build_skill(scene)
        start_q = scene.get_qpos(self.joint_names)

        plan_result = skill.execute(scene, target_ref, start_q, constraints)
        if not plan_result.success:
            return plan_result, pre_home_steps, {}

        target_body = plan_result.diagnostics["target_body"]
        joint_name = plan_result.diagnostics["joint_name"]
        qpos_closed = plan_result.diagnostics["qpos_closed"]
        qpos_open_reference = plan_result.diagnostics["qpos_open_reference"]
        final_target_pos = np.asarray(plan_result.diagnostics["final_target_position"])
        contact_quat_wxyz = np.asarray(plan_result.diagnostics["contact_quat_wxyz"])
        # Same permitted-contact geom set CloseSkill validated the plan
        # against -- a push (palm-first, not fingerpad-first) needs this passed explicitly.
        target_geom_ids = set(scene.body_geom_ids(target_body))

        traj = TimeParameterizer(self.config.time_param).parameterize(plan_result.planned_trajectory)
        if traj.positions is None:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = "time parameterization produced no reference"
            return plan_result, 0, {}

        approach_positions = np.asarray(plan_result.planned_trajectory.metadata["approach_positions"])
        push_positions = np.asarray(plan_result.planned_trajectory.metadata["push_positions"])
        # The raw arc is only ~10 IK solutions over the whole swing -- too sparse
        # to track one per step, so resample it like the FREE_SPACE leg.
        push_plan = TimeParameterizer(self.config.time_param).parameterize(
            MotionPlan(success=True, joint_names=self.joint_names, geometric_path=push_positions)
        )
        if push_plan.positions is not None:
            push_positions = push_plan.positions
        # One continuous reference: FREE_SPACE to standoff, descent, then
        # push to closed -- no gripper-state change partway, so nothing to split into two tracked legs.
        full_reference = np.concatenate([traj.positions, approach_positions[1:], push_positions[1:]], axis=0)
        approach_start_idx = len(traj.positions)

        checker = skill.collision_checker
        cfg = self.config.execution
        steps = pre_home_steps
        # Open the entire time -- contact is made by the palm/wrist, and
        # CloseSkill's collision checks validate against the default/open posture execution must match.
        gripper_cmd = self.env.open_gripper_cmd
        trace = MotionTraceRecorder()
        # terminal_approach_uses_mpc=False: track the planned reference
        # directly with no MPC smoothing -- descent and push are contact BY DESIGN.
        result = track_and_settle(
            self,
            full_reference,
            joint_names=self.joint_names,
            ee_site=self.ee_site,
            approach_start_idx=approach_start_idx,
            target_pos=final_target_pos,
            target_quat_wxyz=contact_quat_wxyz,
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
            mpc_step_kwargs={"target_object_body": target_body},
            steps_so_far=steps,
            max_steps=max_steps,
            step_callback=step_callback,
            terminal_approach_uses_mpc=False,
            target_object_body=target_body,
            permitted_contact_geom_ids=target_geom_ids,
            trace=trace,
        )
        steps += result.steps
        plan_result.diagnostics["settle"] = result.as_diagnostics()
        plan_result.diagnostics["motion_trace"] = {"push": trace.summary()}

        # Structural success check: how far the qpos actually ended up, not
        # whether the settle loop converged (a near-stop stall still counts).
        final_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        qpos_final = float(final_scene.get_qpos([joint_name])[0])
        denom = qpos_open_reference - qpos_closed
        open_frac = (qpos_final - qpos_closed) / denom if abs(denom) > 1e-9 else 0.0
        plan_result.diagnostics["close_verification"] = {
            "joint_name": joint_name,
            "qpos_closed": qpos_closed,
            "qpos_open_reference": qpos_open_reference,
            "qpos_final": qpos_final,
            "open_frac": open_frac,
        }

        # Release and retreat clear of the fixture regardless of outcome, so
        # a following subtask doesn't start with the gripper wedged against it.
        gripper_cmd = self.env.open_gripper_cmd
        push_axis = np.asarray(plan_result.diagnostics["contact_position"]) - final_target_pos
        push_axis_norm = float(np.linalg.norm(push_axis))
        retreat_direction = (
            push_axis / push_axis_norm if push_axis_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        )
        retreat_target = final_target_pos + self.config.close.retreat_m * retreat_direction
        for _ in range(self.config.close.retreat_steps):
            retreat_pos, retreat_ori, _, _ = self._servo_toward_cartesian_pose(retreat_target, contact_quat_wxyz)
            self._step_env(retreat_pos, retreat_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        post_retreat_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        qpos_post_retreat = float(post_retreat_scene.get_qpos([joint_name])[0])
        open_frac_post_retreat = (qpos_post_retreat - qpos_closed) / denom if abs(denom) > 1e-9 else 0.0
        plan_result.diagnostics["close_verification"]["open_frac_post_retreat"] = open_frac_post_retreat

        home_q = np.asarray(self.env.env.robots[0].init_qpos, dtype=np.float64)
        current_q = post_retreat_scene.get_qpos(self.joint_names)
        home_reference = np.linspace(current_q, home_q, self.config.close.return_home_steps)
        for target_q in home_reference:
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, target_q)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        if open_frac_post_retreat > self.config.close.close_threshold_frac:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = (
                f"only reached open_frac={open_frac_post_retreat:.3f} after retreat "
                f"(was {open_frac:.3f} right after the push), "
                f"above close threshold {self.config.close.close_threshold_frac}"
            )
            return plan_result, steps, {}

        plan_result.selected_object = target_body
        return plan_result, steps, plan_result.diagnostics["close_verification"]
