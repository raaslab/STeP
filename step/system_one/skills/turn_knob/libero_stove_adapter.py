"""Executor-facing adapter for the turn_on skill: wires TurnKnobSkill's
grasp-knob + turn plan into a live LiberoManipulationEnv, mirroring
LiberoOpenDrawerAdapter's time-parameterize -> MPC descent ->
close-gripper -> MPC turn -> retreat structure. The turn leg tracks a
rotation about the knob's axis instead of a slide-axis translation, and
success reads the knob's final qpos against KnobConfig.turn_on_qpos_threshold."""

from __future__ import annotations

import numpy as np

from ...motion.articulation import measure_knob_rotation, resolve_knob
from ...motion.collision import CollisionChecker, CollisionPhase
from ...motion.config import MotionConfig
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from .turn_knob_skill import TurnKnobSkill, _axis_angle_to_rotmat
from ...motion.panda import EE_SITE, FINGER_JOINT_NAMES, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES
from ...motion.scene_query import SceneQuery, quat_to_rotmat_wxyz, rotmat_to_quat_wxyz
from ...motion.servo import MotionTraceRecorder, _ServoMixin, real_nonpermitted_contacts, track_and_settle
from ...motion.time_parameterize import TimeParameterizer
from ...motion.types import MPCCostConfig, SkillFailureReason, SkillResult


class LiberoStoveAdapter(_ServoMixin):
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE
        self.hand_site = HAND_SITE

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _build_skill(self, scene: SceneQuery) -> TurnKnobSkill:
        checker = CollisionChecker(
            scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision
        )
        lo, hi = scene.joint_qpos_range(self.joint_names)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        return TurnKnobSkill(ik, planner, checker, self.config)

    def run(
        self,
        knob_ref: str,
        constraints: dict | None = None,
        max_steps: int = 600,
        step_callback=None,
    ) -> tuple[SkillResult, int, dict]:
        constraints = constraints or {}
        scene = self._build_scene()
        skill = self._build_skill(scene)
        start_q = scene.get_qpos(self.joint_names)

        plan_result = skill.execute(scene, knob_ref, start_q, constraints)
        if not plan_result.success:
            return plan_result, 0, {}

        knob = resolve_knob(scene, knob_ref)
        axis_world, center_world, qpos_lo, qpos_hi = measure_knob_rotation(scene, knob)

        traj = TimeParameterizer(self.config.time_param).parameterize(plan_result.planned_trajectory)
        if traj.positions is None:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = "time parameterization produced no reference"
            return plan_result, 0, {}

        grasp_position = np.asarray(plan_result.diagnostics["grasp_position"])
        grasp_quat_wxyz = np.asarray(plan_result.diagnostics["grasp_quat_wxyz"])

        approach_positions = np.asarray(plan_result.planned_trajectory.metadata["approach_positions"])

        checker = skill.collision_checker
        knob_body = knob.knob_body
        cfg = self.config.execution
        steps = 0
        gripper_cmd = self.env.open_gripper_cmd
        descent_trace = MotionTraceRecorder()

        def _track(reference, approach_start_idx, target_pos, trace=None):
            return track_and_settle(
                self,
                reference,
                joint_names=self.joint_names,
                ee_site=self.ee_site,
                approach_start_idx=approach_start_idx,
                target_pos=target_pos,
                target_quat_wxyz=grasp_quat_wxyz,
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
                mpc_step_kwargs={"target_object_body": knob_body},
                steps_so_far=steps,
                max_steps=max_steps,
                step_callback=step_callback,
                target_object_body=knob_body,
                trace=trace,
            )

        # Settle above the knob before descending, as pick does: the wrist
        # rotates far slower than the hand translates, and descending while
        # still ~45 deg off put an open finger on the burner, stalling short.
        standoff_position = grasp_position + self.config.knob.approach_standoff_m * axis_world
        above_result = _track(traj.positions, len(traj.positions), standoff_position)
        steps += above_result.steps
        plan_result.diagnostics["settle_to_above"] = above_result.as_diagnostics()

        descent_result = _track(approach_positions, 0, grasp_position, trace=descent_trace)
        steps += descent_result.steps
        plan_result.diagnostics["settle"] = descent_result.as_diagnostics()

        # Close the gripper on the knob before tracking the turn leg.
        gripper_cmd = self.env.close_gripper_cmd
        close_dwell_steps = max(int(self.env.close_steps) * 2, 25)
        # Then keep squeezing until the fingers stop moving: turning while they
        # are still closing on the handle loses the first ~13 deg to slack.
        openings: list[float] = []
        for i in range(close_dwell_steps + 40):
            hold_pos, hold_ori, _, _ = self._servo_toward_cartesian_pose(grasp_position, grasp_quat_wxyz)
            self._step_env(hold_pos, hold_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            q = SceneQuery.from_robosuite_sim(self.env.env.sim).get_qpos(FINGER_JOINT_NAMES)
            openings.append(abs(float(q[0] - q[1])))
            if i >= close_dwell_steps and max(openings[-5:]) - min(openings[-5:]) < 0.0005:
                break

        # Diagnostic only, not a hard gate -- the same fingerpad<->target
        # contact test CollisionChecker uses to decide permitted contact.
        grip_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        knob_geom_ids = set(grip_scene.body_geom_ids(knob_body))
        gripped_knob = any(
            (g1 in checker.fingerpad_geom_ids and g2 in knob_geom_ids)
            or (g2 in checker.fingerpad_geom_ids and g1 in knob_geom_ids)
            for name1, name2, _dist in grip_scene.compute_contacts()
            for g1, g2 in [(grip_scene.geom_id(name1), grip_scene.geom_id(name2))]
        )
        plan_result.diagnostics["knob_grip_verification"] = {"gripped": gripped_knob}

        # Fresh MPC-tracked reference for the turn leg, not a continuation of
        # the pre-grasp approach -- same settle-to-convergence pattern as the descent phase.
        turned_q = np.asarray(plan_result.planned_trajectory.metadata["turned_q"])
        turn_fraction_used = float(plan_result.diagnostics.get("turn_fraction_used") or 0.0)
        current_qpos = float(plan_result.diagnostics["current_qpos"])
        target_qpos = float(plan_result.diagnostics["target_qpos"])
        delta_angle = turn_fraction_used * (target_qpos - current_qpos)
        r_grasp_to_turned = _axis_angle_to_rotmat(axis_world, delta_angle)
        turned_position = center_world + r_grasp_to_turned @ (grasp_position - center_world)
        # The hand turns with the knob: settle at the turned orientation. The
        # pre-turn one undid the turn once the reference ran out, which only
        # went unnoticed for grasps far from the axis, where position alone
        # dragged the hand (and the knob) round.
        turned_quat_wxyz = rotmat_to_quat_wxyz(r_grasp_to_turned @ quat_to_rotmat_wxyz(grasp_quat_wxyz))

        current_q = SceneQuery.from_robosuite_sim(self.env.env.sim).get_qpos(self.joint_names)
        turn_reference = np.linspace(current_q, turned_q, 20)
        turn_trace = MotionTraceRecorder()
        turn_result = track_and_settle(
            self,
            turn_reference,
            joint_names=self.joint_names,
            ee_site=self.ee_site,
            approach_start_idx=0,  # always TERMINAL_APPROACH -- gripper is already closed on the knob
            target_pos=turned_position,
            target_quat_wxyz=turned_quat_wxyz,
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
            mpc_step_kwargs={"target_object_body": knob_body},
            steps_so_far=steps,
            max_steps=max_steps,
            step_callback=step_callback,
            target_object_body=knob_body,
            trace=turn_trace,
        )
        steps += turn_result.steps
        plan_result.diagnostics["turn_settle"] = {
            "steps": turn_result.settle_steps,
            "converged": turn_result.converged,
            "stalled": turn_result.stalled,
        }

        # Structural success check: how far did the knob's own joint actually
        # rotate, read directly off live qpos -- not a contact heuristic.
        final_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        qpos_final = float(final_scene.get_qpos([knob.joint_name])[0])
        threshold = self.config.knob.turn_on_qpos_threshold
        plan_result.diagnostics["turn_on_verification"] = {
            "joint_name": knob.joint_name,
            "qpos_lo": qpos_lo,
            "qpos_hi": qpos_hi,
            "qpos_final": qpos_final,
            "turn_on_qpos_threshold": threshold,
            "turned_on": qpos_final >= threshold,
        }

        # Retreat along the knob's own axis, not back toward the pre-turn
        # grasp pose -- the hand has rotated with the knob, so "backward" would twist back into it.
        gripper_cmd = self.env.open_gripper_cmd
        retreat_target = turned_position + self.config.knob.retreat_m * axis_world
        retreat_trace = MotionTraceRecorder()
        for _ in range(self.config.knob.retreat_steps):
            retreat_pos, retreat_ori, _, _ = self._servo_toward_cartesian_pose(retreat_target, turned_quat_wxyz)
            self._step_env(retreat_pos, retreat_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            retreat_trace.record_step(
                steps,
                post_scene.site_pose(self.ee_site)[0],
                float("inf"),
                False,
                real_nonpermitted_contacts(post_scene, checker, CollisionPhase.TERMINAL_APPROACH, knob_body),
            )

        # Re-check after retreating: the retreat is itself a contactful
        # motion near the knob, so only post-retreat qpos reflects the true final state.
        post_retreat_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        qpos_post_retreat = float(post_retreat_scene.get_qpos([knob.joint_name])[0])
        plan_result.diagnostics["turn_on_verification"]["qpos_post_retreat"] = qpos_post_retreat
        plan_result.diagnostics["turn_on_verification"]["turned_on_post_retreat"] = qpos_post_retreat >= threshold

        plan_result.diagnostics["motion_trace"] = {
            "descent": descent_trace.summary(),
            "turn": turn_trace.summary(),
            "retreat": retreat_trace.summary(),
        }

        if qpos_post_retreat < threshold:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = (
                f"knob qpos={qpos_post_retreat:.3f} after retreat (was {qpos_final:.3f} right after the turn), "
                f"below turn-on threshold {threshold}"
            )
            return plan_result, steps, {}

        plan_result.selected_object = knob_body
        return plan_result, steps, plan_result.diagnostics["turn_on_verification"]
