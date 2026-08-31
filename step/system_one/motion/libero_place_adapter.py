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

from ..orientation_utils import orientation_error_rotvec
from .collision import CollisionChecker, CollisionPhase
from .config import MotionConfig
from .kinematics import IKSolver
from .mpc import JointSpaceMPC, MPCContext
from .ompl_planner import MotionPlanner
from .panda import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES
from .place_skill import PlaceSkill
from .scene_query import SceneQuery, quat_to_rotmat_wxyz, resolve_body_name
from .time_parameterize import TimeParameterizer
from .types import AttachedObjectState, MPCCostConfig, SkillFailureReason, SkillResult


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
    ) -> tuple[SkillResult, int]:
        scene = self._build_scene()
        goal_container_body = resolve_body_name(scene, goal_container_name)

        checker = CollisionChecker(scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision)
        lo, hi = scene.joint_qpos_range(self.joint_names)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        skill = PlaceSkill(ik, planner, checker, self.config)

        start_q = scene.get_qpos(self.joint_names)
        result = skill.execute(scene, attached, goal_container_body, start_q)
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

        mpc = JointSpaceMPC(
            self.joint_names,
            self.ee_site,
            horizon=int(self.config.mpc.get("horizon", 16)),
            cost_config=MPCCostConfig(**{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}),
        )
        ctx = MPCContext(reference=full_reference)
        mpc.reset()

        steps = 0
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

        released = not bool(self.env._is_grasped())
        if not released:
            result.success = False
            result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            result.diagnostics["detail"] = "gripper opened but object is still reported grasped"
            return result, steps

        return result, steps
