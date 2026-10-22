"""Executor-facing adapter for the open skill: wires OpenDrawerSkill's
grasp-handle + pull plan into a live LiberoManipulationEnv, mirroring
LiberoPickAdapter's time-parameterize -> MPC tracking structure. Success
is verified structurally by reading the drawer joint's live qpos after the
pull, since LIBERO fixtures carry no contact-based grasp signal. Nothing is
attached/carried afterward -- the drawer stays connected to the cabinet via
its own joint, and a retreat (not a hold) ends the motion so a following
subtask doesn't start wedged against the handle."""

from __future__ import annotations

import numpy as np

from ...motion.articulation import discover_drawers, measure_pull_axis_and_travel, resolve_drawer
from ...motion.collision import CollisionChecker, CollisionPhase
from ...motion.config import MotionConfig
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from .open_drawer_skill import OpenDrawerSkill
from ...motion.panda import EE_SITE, FINGER_JOINT_NAMES, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES
from ...motion.scene_query import SceneQuery
from ...motion.servo import MotionTraceRecorder, _ServoMixin, real_nonpermitted_contacts, track_and_settle
from ...motion.time_parameterize import TimeParameterizer
from ...motion.types import MPCCostConfig, SkillFailureReason, SkillResult


class LiberoOpenDrawerAdapter(_ServoMixin):
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE
        self.hand_site = HAND_SITE

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _build_skill(self, scene: SceneQuery) -> OpenDrawerSkill:
        checker = CollisionChecker(
            scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision
        )
        lo, hi = scene.joint_qpos_range(self.joint_names)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        return OpenDrawerSkill(ik, planner, checker, self.config)

    def _servo_to_home(self, step_callback=None) -> int:
        return self.servo_to_home(self.config.drawer.return_home_steps, step_callback)

    def run(
        self,
        drawer_ref: str,
        constraints: dict | None = None,
        max_steps: int = 600,
        step_callback=None,
    ) -> tuple[SkillResult, int, dict]:
        constraints = constraints or {}

        # Home first -- planning the handle approach from whatever pose the
        # previous task left was a recurring source of planner timeouts and bad settles.
        pre_home_steps = self._servo_to_home(step_callback)

        scene = self._build_scene()
        skill = self._build_skill(scene)
        start_q = scene.get_qpos(self.joint_names)

        plan_result = skill.execute(scene, drawer_ref, start_q, constraints)
        if not plan_result.success:
            return plan_result, pre_home_steps, {}

        drawer = resolve_drawer(scene, drawer_ref)
        pull_axis_world, travel_m, qpos_closed, qpos_open = measure_pull_axis_and_travel(scene, drawer)

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
        drawer_body = drawer.drawer_body
        cfg = self.config.execution
        steps = pre_home_steps  # the homing move above is real env time too
        gripper_cmd = self.env.open_gripper_cmd
        descent_trace = MotionTraceRecorder()
        descent_result = track_and_settle(
            self,
            full_reference,
            joint_names=self.joint_names,
            ee_site=self.ee_site,
            approach_start_idx=approach_start_idx,
            target_pos=grasp_position,
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
            mpc_step_kwargs={"target_object_body": drawer_body},
            steps_so_far=steps,
            max_steps=max_steps,
            step_callback=step_callback,
            target_object_body=drawer_body,
            trace=descent_trace,
        )
        steps += descent_result.steps
        plan_result.diagnostics["settle"] = descent_result.as_diagnostics()

        # Close the gripper on the handle before tracking the pull leg,
        # which is driven separately once the grasp has actually closed.
        gripper_cmd = self.env.close_gripper_cmd
        close_dwell_steps = max(int(self.env.close_steps) * 2, 25)
        for _ in range(close_dwell_steps):
            hold_pos, hold_ori, _, _ = self._servo_toward_cartesian_pose(grasp_position, grasp_quat_wxyz)
            self._step_env(hold_pos, hold_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        # Diagnostic only, not a hard gate -- reuses the fingerpad<->target
        # contact test so a low open_frac can later be told apart from never-gripped vs. gripped-but-short.
        grip_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        drawer_geom_ids = set(grip_scene.body_geom_ids(drawer_body))
        gripped_handle = any(
            (g1 in checker.fingerpad_geom_ids and g2 in drawer_geom_ids)
            or (g2 in checker.fingerpad_geom_ids and g1 in drawer_geom_ids)
            for name1, name2, _dist in grip_scene.compute_contacts()
            for g1, g2 in [(grip_scene.geom_id(name1), grip_scene.geom_id(name2))]
        )
        # Directional miss, not just distance -- a downward miss can land on
        # the next drawer's handle since they're closely spaced vertically.
        end_hand_pos = grip_scene.site_pose(self.ee_site)[0]
        handle_miss = np.asarray(end_hand_pos) - grasp_position
        plan_result.diagnostics["handle_approach_miss"] = {
            "delta_xyz": [float(v) for v in handle_miss],
            "distance_m": float(np.linalg.norm(handle_miss)),
            "dz_m": float(handle_miss[2]),
        }

        plan_result.diagnostics["handle_grip_verification"] = {"gripped": gripped_handle}

        # Gates on "captured between fingers," not "pads touching," since a
        # valid hook-style pull may never touch the bar.
        captured = gripped_handle or float(np.linalg.norm(handle_miss)) <= self.config.drawer.handle_capture_radius_m
        plan_result.diagnostics["handle_grip_verification"]["captured_between_fingers"] = bool(captured)
        if not captured:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = (
                "aborted before pulling: the handle is not between the fingers after the "
                f"descent (missed {drawer_body}'s handle by {np.linalg.norm(handle_miss):.3f}m, "
                f"dz={handle_miss[2]:+.3f}m). Pulling from here would drag whatever the hand "
                "is resting against."
            )
            steps += self._servo_to_home(step_callback)
            return plan_result, steps, {}

        # Runs to actual Cartesian convergence, not a fixed step count -- a
        # short fixed budget leaves the drawer partially pulled due to tracking lag.
        pulled_q = np.asarray(plan_result.planned_trajectory.metadata["pulled_q"])
        pull_fraction_used = float(plan_result.diagnostics.get("pull_fraction_used") or 0.0)
        pulled_position = grasp_position + pull_fraction_used * travel_m * pull_axis_world
        current_q = SceneQuery.from_robosuite_sim(self.env.env.sim).get_qpos(self.joint_names)
        pull_reference = np.linspace(current_q, pulled_q, 15)
        pull_trace = MotionTraceRecorder()
        pull_result = track_and_settle(
            self,
            pull_reference,
            joint_names=self.joint_names,
            ee_site=self.ee_site,
            approach_start_idx=0,  # always TERMINAL_APPROACH -- gripper is already closed on the handle
            target_pos=pulled_position,
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
            mpc_step_kwargs={"target_object_body": drawer_body},
            steps_so_far=steps,
            max_steps=max_steps,
            step_callback=step_callback,
            target_object_body=drawer_body,
            trace=pull_trace,
        )
        steps += pull_result.steps
        plan_result.diagnostics["pull_settle"] = {
            "steps": pull_result.settle_steps,
            "converged": pull_result.converged,
            "stalled": pull_result.stalled,
        }

        # Structural success check: how far did the joint actually travel,
        # not whether any contact heuristic fired.
        final_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        qpos_final = float(final_scene.get_qpos([drawer.joint_name])[0])
        denom = qpos_open - qpos_closed
        open_frac = (qpos_final - qpos_closed) / denom if abs(denom) > 1e-9 else 0.0
        plan_result.diagnostics["open_verification"] = {
            "joint_name": drawer.joint_name,
            "qpos_closed": qpos_closed,
            "qpos_open": qpos_open,
            "qpos_final": qpos_final,
            "open_frac": open_frac,
        }

        # Retreat from the current pulled position, not the stale pre-pull
        # grasp_position -- the latter would travel back through the handle, shoving the drawer closed again.
        gripper_cmd = self.env.open_gripper_cmd
        retreat_target = pulled_position + self.config.drawer.retreat_m * pull_axis_world
        retreat_trace = MotionTraceRecorder()
        for _ in range(self.config.drawer.retreat_steps):
            retreat_pos, retreat_ori, _, _ = self._servo_toward_cartesian_pose(retreat_target, grasp_quat_wxyz)
            self._step_env(retreat_pos, retreat_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            # Retreat is a plain servo (no MPC step), so there's no
            # min_clearance/replan signal -- still worth recording ee_pos/contacts.
            retreat_trace.record_step(
                steps,
                post_scene.site_pose(self.ee_site)[0],
                float("inf"),
                False,
                real_nonpermitted_contacts(post_scene, checker, CollisionPhase.TERMINAL_APPROACH, drawer_body),
            )

        # Re-check after retreating: the retreat is itself a contactful
        # motion near the handle, so only post-retreat qpos reflects the true final state.
        post_retreat_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        qpos_post_retreat = float(post_retreat_scene.get_qpos([drawer.joint_name])[0])
        open_frac_post_retreat = (qpos_post_retreat - qpos_closed) / denom if abs(denom) > 1e-9 else 0.0
        plan_result.diagnostics["open_verification"]["open_frac_post_retreat"] = open_frac_post_retreat

        # Every drawer's travel, not just the target's -- a hand that stalls
        # and drags along the cabinet face can open the wrong drawer.
        sibling_travel = {}
        for other in discover_drawers(post_retreat_scene):
            lo_o, hi_o = post_retreat_scene.joint_qpos_range([other.joint_name])
            span = float(hi_o[0] - lo_o[0])
            q_now = float(post_retreat_scene.get_qpos([other.joint_name])[0])
            q_closed = float(np.clip(0.0, lo_o[0], hi_o[0]))
            sibling_travel[other.level_label] = {
                "joint": other.joint_name,
                "qpos": q_now,
                "open_fraction_of_range": abs(q_now - q_closed) / span if span > 1e-9 else 0.0,
                "is_target": other.joint_name == drawer.joint_name,
            }
        plan_result.diagnostics["open_verification"]["all_drawers"] = sibling_travel

        # Return to robot.init_qpos regardless of outcome, via dense
        # joint-space interpolation rather than a single Cartesian move.
        home_q = np.asarray(self.env.env.robots[0].init_qpos, dtype=np.float64)
        current_q = post_retreat_scene.get_qpos(self.joint_names)
        home_reference = np.linspace(current_q, home_q, self.config.drawer.return_home_steps)
        return_home_trace = MotionTraceRecorder()
        for target_q in home_reference:
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, target_q)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            return_home_trace.record_step(
                steps,
                post_scene.site_pose(self.ee_site)[0],
                float("inf"),
                False,
                real_nonpermitted_contacts(post_scene, checker, CollisionPhase.FREE_SPACE, drawer_body),
            )

        # Per-phase path smoothness, worst clearance, and real collisions --
        # a stand-in for "what did this look like" without a video.
        plan_result.diagnostics["motion_trace"] = {
            "descent": descent_trace.summary(),
            "pull": pull_trace.summary(),
            "retreat": retreat_trace.summary(),
            "return_home": return_home_trace.summary(),
        }

        if open_frac_post_retreat < self.config.drawer.open_threshold_frac:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = (
                f"drawer only reached open_frac={open_frac_post_retreat:.3f} after retreat "
                f"(was {open_frac:.3f} right after the pull), "
                f"below threshold {self.config.drawer.open_threshold_frac}"
            )
            return plan_result, steps, {}

        plan_result.selected_object = drawer.level_label + "_drawer"
        return plan_result, steps, plan_result.diagnostics["open_verification"]
