"""Executor-facing adapter for the place skill.

Computes the object's desired final pose from the goal surface height and
the carried object's own collision geometry, converts it to a hand target via
the grasp transform, and servos straight there in small bounded steps instead
of planning with OMPL or tracking an MPC reference -- a smaller failure
surface at the cost of not routing around obstacles, acceptable since
transport already leaves the arm hovering clear above the goal."""

from __future__ import annotations

import numpy as np

from ...orientation_utils import orientation_error_rotvec
from ...motion.attached_object import refresh_grasp_transform
from ...motion.collision import required_hand_pose_for_object_pose
from ...motion.config import MotionConfig
from ...motion.panda import EE_SITE, JOINT_NAMES
from .place_skill import _container_floor_and_rim_z, _support_floor_and_rim_z
from ...motion.scene_query import SceneQuery, quat_to_rotmat_wxyz, resolve_body_name
from ...motion.servo import _ServoMixin, is_settle_stalled
from ...motion.types import AttachedObjectState, SkillFailureReason, SkillResult


class LiberoPlaceAdapter(_ServoMixin):
    """Uses only _ServoMixin's _step_env; _servo_toward_cartesian_target
    takes a Cartesian target directly, unlike _ServoMixin's other joint-space helpers."""

    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _ee_force_magnitude(self) -> float:
        # Real wrist F/T sensor (gripper0_right_ft_frame's force_ee), not a
        # proxy -- reads whatever the arm is actually pushing against.
        try:
            return float(np.linalg.norm(self.env.env.robots[0].ee_force))
        except Exception:
            return 0.0

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

    def _servo_line(self, target_pos, target_R, gripper_cmd, max_iters, step_callback, max_step_m=0.01, tol_m=0.003):
        """Closed-loop straight line in small increments; returns (steps, stalled)."""
        steps, errs = 0, []
        for _ in range(max_iters):
            ee_pos = self.env._eef_pos(self.env.last_raw_obs)
            to_target = target_pos - ee_pos
            dist = float(np.linalg.norm(to_target))
            if dist <= tol_m:
                return steps, False
            waypoint = ee_pos + to_target * min(1.0, max_step_m / dist)
            cart_pos, cart_ori, _p, _o = self._servo_toward_cartesian_target(waypoint, target_R, pos_scale_m=0.03)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            errs.append(dist)
            if is_settle_stalled(errs, self.config.place.blocked_stall_window_steps, self.config.execution.stall_position_tol_m):
                return steps, True
        return steps, True

    def run_side_insert(self, attached: AttachedObjectState, side, plan, max_steps: int = 600, step_callback=None):
        """Slide a handle-held object straight into a side-opening container
        (see skills/place/side_insert.py), release, back out and go home."""
        from ...motion.collision import CollisionChecker, CollisionPhase
        from ...motion.kinematics import IKSolver
        from ...motion.ompl_planner import MotionPlanner
        from ...motion.panda import GRIPPER_BODIES, ROBOT_BODIES
        from ...motion.servo import track_and_settle
        from ...motion.time_parameterize import TimeParameterizer
        from ...motion.types import MPCCostConfig
        from ..ik_utils import solve_ik_robust

        result = SkillResult(success=True, selected_object=attached.object_name, diagnostics={"mode": "side_insert"})
        scene = self._build_scene()
        refresh_grasp_transform(attached, scene, self.ee_site)
        obj_quat = scene.body_pose(attached.object_name)[1]
        final_hand, hand_quat = required_hand_pose_for_object_pose(
            plan.final_object_xyz, obj_quat, attached.grasp_transform_hand_from_object
        )
        hand_R = quat_to_rotmat_wxyz(hand_quat)
        d = np.asarray(side.insert_dir, dtype=float)
        pre_hand = final_hand - d * plan.insert_distance_m
        result.diagnostics.update(final_hand=final_hand.tolist(), pre_insert_hand=pre_hand.tolist())

        def _fail(detail: str, steps: int):
            result.success = False
            result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            result.diagnostics["detail"] = detail
            return result, steps

        checker = CollisionChecker(scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision)
        lo, hi = scene.joint_qpos_range(self.joint_names)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        start_q = scene.get_qpos(self.joint_names)
        rng = np.random.default_rng(0)
        pre_q = None
        for attempt in range(8):
            seed = start_q if attempt == 0 else rng.uniform(lo, hi)
            sol = solve_ik_robust(ik, scene, pre_hand, hand_quat, seed, rng, n_restarts=2)
            if sol.success and not checker.check(
                sol.q, phase=CollisionPhase.PLACE_APPROACH, attached=attached, goal_container_body=side.container_body
            ).in_collision:
                pre_q = sol.q
                break
        if pre_q is None:
            return _fail("no collision-free arm pose in front of the opening", 0)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        motion = planner.plan(
            start_q, [pre_q], phase=CollisionPhase.PLACE_APPROACH, attached=attached, goal_container_body=side.container_body
        )
        if not motion.success:
            return _fail(f"planning to the opening failed: {motion.failure_reason}", 0)
        traj = TimeParameterizer(self.config.time_param).parameterize(motion)
        cfg = self.config.execution
        tracked = track_and_settle(
            self,
            traj.positions,
            joint_names=self.joint_names,
            ee_site=self.ee_site,
            approach_start_idx=len(traj.positions) + cfg.max_settle_steps,
            target_pos=pre_hand,
            target_quat_wxyz=hand_quat,
            collision_checker=checker,
            mpc_horizon=int(self.config.mpc.get("horizon", 16)),
            mpc_cost_config=MPCCostConfig(
                **{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}
            ),
            replan_config=self.config.replan,
            gripper_cmd=self.env.close_gripper_cmd,
            settle_position_tol_m=cfg.settle_position_tol_m,
            settle_orientation_tol_rad=cfg.settle_orientation_tol_rad,
            stall_window_steps=cfg.stall_window_steps,
            stall_position_tol_m=cfg.stall_position_tol_m,
            max_settle_steps=cfg.max_settle_steps,
            mpc_step_kwargs={"goal_container_body": side.container_body, "attached": attached},
            max_steps=max_steps,
            step_callback=step_callback,
            target_object_body=side.container_body,
            max_reference_lead_m=cfg.transport_max_reference_lead_m,
        )
        steps = tracked.steps
        result.diagnostics["to_opening"] = tracked.as_diagnostics()

        insert_steps, stalled = self._servo_line(final_hand, hand_R, self.env.close_gripper_cmd, 120, step_callback)
        steps += insert_steps
        result.diagnostics["insert"] = {"steps": insert_steps, "stalled": stalled}

        hold_pos = self.env._eef_pos(self.env.last_raw_obs).copy()
        for _ in range(int(self.env.close_steps) * 2):
            cart_pos, cart_ori, _p, _o = self._servo_toward_cartesian_target(hold_pos, hand_R, pos_scale_m=0.03)
            self._step_env(cart_pos, cart_ori, self.env.open_gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
        back_steps, _ = self._servo_line(
            hold_pos - d * (plan.insert_distance_m + 0.03), hand_R, self.env.open_gripper_cmd, 120, step_callback
        )
        steps += back_steps
        steps += self.servo_to_home(self.config.drawer.return_home_steps, step_callback)

        rest = SceneQuery.from_robosuite_sim(self.env.env.sim)
        object_rest = np.asarray(rest.body_pose(attached.object_name)[0], dtype=float)
        inside = bool(np.all(np.abs(object_rest - side.region_center) <= side.region_half))
        released = not bool(self.env._is_grasped())
        result.diagnostics["object_rest_verification"] = {
            "object_rest_xyz": object_rest.tolist(),
            "inside_region": inside,
            "released": released,
        }
        if not released:
            return _fail("gripper opened but object is still reported grasped", steps)
        if not inside:
            return _fail("object did not end inside the container's region", steps)
        return result, steps

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
        if self.env._is_grasped():
            refresh_grasp_transform(attached, scene, self.ee_site)
        steps = 0

        # Desired final object pose: goal XY at the surface's real height,
        # raised by how far the carried object's collision geometry extends below its reference point.
        container_pos, _ = scene.body_pose(goal_container_body)
        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        floor_z, _rim_z = floor_fn(scene, goal_container_body)
        # A point target's own z overrides the container's: goal_container_body
        # can fall back to a bare parent surface (e.g. "table"), which would aim the descent too low -- mirrors transport_skill's rim_top_z override.
        if goal_point_xyz is not None:
            goal_point_xyz = np.asarray(goal_point_xyz, dtype=float)
            if goal_point_xyz.shape[0] > 2 and np.isfinite(goal_point_xyz[2]):
                floor_z = float(goal_point_xyz[2])
        obj_extent = scene.body_collision_extent_world(attached.object_name)
        obj_bottom_local_z = (
            float(obj_extent.center_offset[2] - obj_extent.half_extents[2]) if obj_extent is not None else -0.02
        )
        place_z = floor_z - obj_bottom_local_z + pcfg.drop_clearance_m

        if goal_point_xyz is not None:
            target_xy = np.asarray(goal_point_xyz, dtype=float)[:2]
        elif place_surface == "top":
            # A large support has no single right default spot; transport
            # already carried the object to a clear spot, so keep that instead of the support's own center.
            target_xy = scene.body_pose(attached.object_name)[0][:2]
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

        # Closed-loop, not a precomputed linspace: recomputing a small step
        # from wherever the arm actually is keeps the commanded action bounded regardless of tracking lag.
        exec_cfg = self.config.execution
        gripper_cmd = self.env.close_gripper_cmd
        max_step_m = 0.01
        descent_tol_m = 0.002
        # Baseline, not an absolute threshold: resting force already includes
        # gravity-compensation residual plus whatever's carried, which varies by object.
        baseline_force_n = self._ee_force_magnitude()
        force_delta_n = pcfg.blocked_force_delta_n
        blocked = False
        blocked_reason = None
        blocked_retreat_dir = np.array([0.0, 0.0, 1.0])
        recent_descent_errs: list[float] = []
        while steps < max_steps:
            ee_pos = self.env._eef_pos(self.env.last_raw_obs)
            to_target = place_hand_pos - ee_pos
            dist = float(np.linalg.norm(to_target))
            if dist <= descent_tol_m:
                break
            force_mag = self._ee_force_magnitude()
            if force_mag - baseline_force_n >= force_delta_n:
                blocked, blocked_reason = True, "force"
                if dist > 1e-6:
                    blocked_retreat_dir = -to_target / dist
                break
            recent_descent_errs.append(dist)
            if is_settle_stalled(recent_descent_errs, pcfg.blocked_stall_window_steps, exec_cfg.stall_position_tol_m):
                blocked, blocked_reason = True, "stall"
                if dist > 1e-6:
                    blocked_retreat_dir = -to_target / dist
                break
            step_target = ee_pos + to_target * (min(max_step_m, dist) / dist)
            cart_pos, cart_ori, _pos_err, _ori_err = self._servo_toward_cartesian_target(
                step_target, place_R, pos_scale_m=0.03
            )
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        # Settle briefly before releasing -- the descent loop above ends near
        # the target, but steady-state tracking lag can leave the arm a few mm off.
        settle_steps = 0
        pos_err = ori_err = float("inf")
        recent_settle_errs: list[float] = []
        if not blocked:
            for _ in range(exec_cfg.max_settle_steps):
                cart_pos, cart_ori, pos_err, ori_err = self._servo_toward_cartesian_target(
                    place_hand_pos, place_R, pos_scale_m=exec_cfg.settle_pos_scale_m
                )
                if pos_err <= exec_cfg.settle_position_tol_m and ori_err <= exec_cfg.settle_orientation_tol_rad:
                    break
                settle_to_target = place_hand_pos - self.env._eef_pos(self.env.last_raw_obs)
                settle_to_target_norm = float(np.linalg.norm(settle_to_target))
                force_mag = self._ee_force_magnitude()
                if force_mag - baseline_force_n >= force_delta_n:
                    blocked, blocked_reason = True, "force"
                    if settle_to_target_norm > 1e-6:
                        blocked_retreat_dir = -settle_to_target / settle_to_target_norm
                    break
                recent_settle_errs.append(pos_err)
                if is_settle_stalled(recent_settle_errs, pcfg.blocked_stall_window_steps, exec_cfg.stall_position_tol_m):
                    blocked, blocked_reason = True, "stall"
                    if settle_to_target_norm > 1e-6:
                        blocked_retreat_dir = -settle_to_target / settle_to_target_norm
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
            "blocked": blocked,
            "blocked_reason": blocked_reason,
        }
        result.diagnostics["blocked"] = blocked
        if blocked:
            result.diagnostics["detail"] = (
                f"released early ({blocked_reason}): the goal pose was blocked before the arm could reach it"
            )
        # Release and dwell/retreat a touch so the object settles before
        # anything downstream reads its resting pose. If blocked, back off
        # along the approach direction first -- opening while still pressed
        # into the obstruction can leave the finger actuators unable to reach
        # fully open against that contact force, no matter how long they're
        # given (confirmed via a real rollout: fingers plateaued at ~72% open
        # and never moved again over 80+ extra steps of just waiting).
        if blocked:
            retreat_from_pos, retreat_from_quat = self._build_scene().site_pose(self.ee_site)
            dwell_R = quat_to_rotmat_wxyz(retreat_from_quat)
            dwell_pos = retreat_from_pos + pcfg.blocked_retreat_m * blocked_retreat_dir
        else:
            dwell_pos, dwell_R = place_hand_pos, place_R
        gripper_cmd = self.env.open_gripper_cmd
        for _ in range(int(self.env.close_steps) * 2):
            cart_pos, cart_ori, _pos_err, _ori_err = self._servo_toward_cartesian_target(
                dwell_pos, dwell_R, pos_scale_m=0.03
            )
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
        if blocked:
            # Confirms the retreat actually freed the fingers rather than
            # just assuming a fixed step count was enough.
            from ...motion.panda import FINGER_JOINT_NAMES

            finger_lo, finger_hi = self._build_scene().joint_qpos_range(FINGER_JOINT_NAMES)
            # Each joint's open extreme is whichever bound sits away from 0
            # (closed) -- joint1 opens toward +hi, joint2 toward -lo.
            finger_open_target = np.where(np.abs(finger_lo) > np.abs(finger_hi), finger_lo, finger_hi)
            finger_open_tol = 0.002
            for _ in range(int(self.env.close_steps) * 4):
                finger_qpos = self._build_scene().get_qpos(FINGER_JOINT_NAMES)
                if np.all(np.abs(finger_qpos - finger_open_target) <= finger_open_tol):
                    break
                cart_pos, cart_ori, _pos_err, _ori_err = self._servo_toward_cartesian_target(
                    dwell_pos, dwell_R, pos_scale_m=0.03
                )
                self._step_env(cart_pos, cart_ori, gripper_cmd)
                steps += 1
                if step_callback is not None:
                    step_callback()

        # Withdraws before judging release: _is_grasped() is a contact test,
        # so a released object touching the open fingers would otherwise still read as grasped.
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

        # Checks where the object actually came to rest, not where it was
        # aimed -- verifying only the release pose would miss it rolling or being nudged off.
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
