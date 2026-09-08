"""Executor-facing adapter for the ``turn_on`` skill: wires ``TurnKnobSkill``'s
grasp-knob + turn plan into a live ``LiberoManipulationEnv``.

Mirrors ``LiberoOpenDrawerAdapter``'s structure almost exactly (same time-
parameterize -> MPC descent -> close-gripper -> MPC turn -> retreat shape,
same structural-not-contact-heuristic success check), swapping only the
physically different part: the second leg tracks a *rotation* about the
knob's measured axis (``turned_q``) instead of a translation along a slide
axis (``pulled_q``), and success reads the knob joint's final qpos against
``KnobConfig.turn_on_qpos_threshold`` instead of an open/closed travel
fraction. See ``turn_knob_skill.py``'s module docstring for why a fixture
gets its own small skill instead of being forced through ``PickSkill``.
"""

from __future__ import annotations

import numpy as np

from ..articulation import measure_knob_rotation, resolve_knob
from ..collision import CollisionChecker, CollisionPhase
from ..config import MotionConfig
from ..kinematics import IKSolver
from ..mpc import JointSpaceMPC, MPCContext
from ..ompl_planner import MotionPlanner
from .turn_knob_skill import TurnKnobSkill, _axis_angle_to_rotmat
from ..panda import EE_SITE, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES
from ..scene_query import SceneQuery
from ..servo import MotionTraceRecorder, _ServoMixin, is_settle_stalled, real_nonpermitted_contacts
from ..time_parameterize import TimeParameterizer
from ..types import MPCCostConfig, SkillFailureReason, SkillResult


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
        full_reference = np.concatenate([traj.positions, approach_positions[1:]], axis=0)
        approach_start_idx = len(traj.positions)

        checker = skill.collision_checker
        knob_body = knob.knob_body
        mpc = JointSpaceMPC(
            self.joint_names,
            self.ee_site,
            horizon=int(self.config.mpc.get("horizon", 16)),
            cost_config=MPCCostConfig(**{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}),
        )
        ctx = MPCContext(reference=full_reference)
        mpc.reset()

        cfg = self.config.execution
        max_hold_steps = max_steps if max_steps is not None else len(full_reference) + cfg.max_settle_steps
        tracking_budget = min(max_hold_steps, len(full_reference) + cfg.max_settle_steps)

        steps = 0
        gripper_cmd = self.env.open_gripper_cmd
        pos_err = ori_err = float("inf")
        converged = False
        stalled = False
        recent_pos_errs: list[float] = []
        t0 = 0
        descent_trace = MotionTraceRecorder()
        while t0 < tracking_budget:
            phase = CollisionPhase.FREE_SPACE if t0 < approach_start_idx else CollisionPhase.TERMINAL_APPROACH
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            q_current = live_scene.get_qpos(self.joint_names)
            mpc_result = mpc.step(
                live_scene,
                q_current,
                ctx,
                t0_step=t0,
                collision_checker=checker,
                collision_phase=phase,
                target_object_body=knob_body,
                replan_config=self.config.replan,
            )
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, mpc_result.joint_command)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            t0 += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            descent_trace.record_step(
                t0,
                post_scene.site_pose(self.ee_site)[0],
                mpc_result.min_clearance,
                mpc_result.replan_requested,
                real_nonpermitted_contacts(post_scene, checker, phase, knob_body),
            )
            if steps >= max_steps:
                break

            if t0 >= len(full_reference):
                _, _, pos_err, ori_err = self._servo_toward_cartesian_pose(grasp_position, grasp_quat_wxyz)
                if pos_err <= cfg.settle_position_tol_m and ori_err <= cfg.settle_orientation_tol_rad:
                    converged = True
                    break
                recent_pos_errs.append(pos_err)
                if is_settle_stalled(recent_pos_errs, cfg.stall_window_steps, cfg.stall_position_tol_m):
                    stalled = True
                    break

        settle_steps = max(0, t0 - len(full_reference))
        plan_result.diagnostics["settle"] = {
            "steps": settle_steps,
            "converged": converged,
            "stalled": stalled,
            "position_error_m": pos_err,
            "orientation_error_rad": ori_err,
        }

        # Close the gripper on the knob -- only *now* does tracking the turn
        # leg make physical sense (see turn_knob_skill.py's execute()-comment,
        # same reasoning as OpenDrawerSkill's pull leg: driving the turn
        # motion before the gripper is closed never actually grips anything).
        gripper_cmd = self.env.close_gripper_cmd
        close_dwell_steps = max(int(self.env.close_steps) * 2, 25)
        for _ in range(close_dwell_steps):
            hold_pos, hold_ori, _, _ = self._servo_toward_cartesian_pose(grasp_position, grasp_quat_wxyz)
            self._step_env(hold_pos, hold_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        # Diagnostic only (not a hard gate) -- same fingerpad<->target contact
        # test CollisionChecker itself uses to decide permitted contact,
        # mirrors LiberoOpenDrawerAdapter's own handle_grip_verification.
        grip_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        knob_geom_ids = set(grip_scene.body_geom_ids(knob_body))
        gripped_knob = any(
            (g1 in checker.fingerpad_geom_ids and g2 in knob_geom_ids)
            or (g2 in checker.fingerpad_geom_ids and g1 in knob_geom_ids)
            for name1, name2, _dist in grip_scene.compute_contacts()
            for g1, g2 in [(grip_scene.geom_id(name1), grip_scene.geom_id(name2))]
        )
        plan_result.diagnostics["knob_grip_verification"] = {"gripped": gripped_knob}

        # Track the turn leg with its own short MPC-tracked reference (fresh
        # context, not a continuation of `ctx`/`full_reference`, which only
        # ever covered the pre-grasp approach) -- same settle-to-convergence
        # pattern as the descent phase, same reasoning as
        # LiberoOpenDrawerAdapter's pull leg (a fixed short step budget
        # measurably under-delivers against the real controller's tracking
        # lag).
        turned_q = np.asarray(plan_result.planned_trajectory.metadata["turned_q"])
        turn_fraction_used = float(plan_result.diagnostics.get("turn_fraction_used") or 0.0)
        current_qpos = float(plan_result.diagnostics["current_qpos"])
        target_qpos = float(plan_result.diagnostics["target_qpos"])
        delta_angle = turn_fraction_used * (target_qpos - current_qpos)
        r_grasp_to_turned = _axis_angle_to_rotmat(axis_world, delta_angle)
        turned_position = center_world + r_grasp_to_turned @ (grasp_position - center_world)

        current_q = SceneQuery.from_robosuite_sim(self.env.env.sim).get_qpos(self.joint_names)
        turn_reference = np.linspace(current_q, turned_q, 20)
        turn_ctx = MPCContext(reference=turn_reference)
        mpc.reset()
        turn_budget = len(turn_reference) + cfg.max_settle_steps
        t0 = 0
        turn_converged = False
        turn_stalled = False
        recent_turn_errs: list[float] = []
        turn_trace = MotionTraceRecorder()
        while t0 < turn_budget and steps < max_steps:
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            q_current = live_scene.get_qpos(self.joint_names)
            mpc_result = mpc.step(
                live_scene,
                q_current,
                turn_ctx,
                t0_step=t0,
                collision_checker=checker,
                collision_phase=CollisionPhase.TERMINAL_APPROACH,
                target_object_body=knob_body,
                replan_config=self.config.replan,
            )
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, mpc_result.joint_command)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            t0 += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            turn_trace.record_step(
                t0,
                post_scene.site_pose(self.ee_site)[0],
                mpc_result.min_clearance,
                mpc_result.replan_requested,
                real_nonpermitted_contacts(post_scene, checker, CollisionPhase.TERMINAL_APPROACH, knob_body),
            )
            if t0 >= len(turn_reference):
                # Orientation drifts with the turn itself (the hand rotates
                # with the knob), so only position convergence gates the
                # settle here -- checking against the fixed pre-turn
                # `grasp_quat_wxyz` would never converge once the hand has
                # actually rotated with the knob.
                _, _, turn_pos_err, _ = self._servo_toward_cartesian_pose(turned_position, grasp_quat_wxyz)
                if turn_pos_err <= cfg.settle_position_tol_m:
                    turn_converged = True
                    break
                recent_turn_errs.append(turn_pos_err)
                if is_settle_stalled(recent_turn_errs, cfg.stall_window_steps, cfg.stall_position_tol_m):
                    turn_stalled = True
                    break
        plan_result.diagnostics["turn_settle"] = {
            "steps": max(0, t0 - len(turn_reference)),
            "converged": turn_converged,
            "stalled": turn_stalled,
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

        # Release and retreat clear of the knob regardless of outcome, so a
        # following subtask doesn't start with the gripper wedged against it.
        # Retreat along the knob's own axis (away from the burner surface),
        # not back toward the pre-turn grasp orientation/position -- the hand
        # has rotated with the knob, so "backward" would mean twisting back
        # into it.
        gripper_cmd = self.env.open_gripper_cmd
        retreat_target = turned_position + self.config.knob.retreat_m * axis_world
        retreat_trace = MotionTraceRecorder()
        for _ in range(self.config.knob.retreat_steps):
            retreat_pos, retreat_ori, _, _ = self._servo_toward_cartesian_pose(retreat_target, grasp_quat_wxyz)
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

        # Re-check after retreating, not before -- mirrors
        # LiberoOpenDrawerAdapter's own post-retreat re-check (the retreat is
        # a real, physically-contactful motion right next to the knob, so
        # it's the true final state that determines whether it was actually
        # left turned on).
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
