"""Executor-facing adapter: wires the grasp-gen -> IK -> OMPL -> time-param ->
MPC pipeline into a live ``LiberoManipulationEnv`` for the ``pick`` skill.

Scope of this integration slice: **pick only**. ``transport``/``place`` still
run on the legacy stage machine even when ``motion_backend: "ompl_mpc"`` is
set -- see ``System1Executor``'s dispatch in ``executor.py``. This keeps the
integration's blast radius small and independently verifiable: the new
pipeline's grasp/IK/planning/collision logic is exercised against the real
env and gripper on every run, without also needing new transport/place
motion-generation code to be correct for a pick-only smoke test to pass.

Action-space bridge: the MPC outputs joint-position targets, but LIBERO's
robosuite env (like the legacy stage machine) takes Cartesian-delta OSC_POSE
actions -- there's no way to swap in a joint-position controller from here
without changing how ``LiberoManipulationEnv`` constructs its env (out of
scope for this pass). So each joint target is converted to a Cartesian pose
via forward kinematics (on a scratch ``SceneQuery``, never the live sim) and
tracked with the *same* clipped-delta servo law the legacy stage machine
already uses (``BaseManipulationEnv._compose_full_action``) -- this reuses a
proven, already-tuned low-level control primitive instead of inventing a
second one, and means the new pipeline is only responsible for *where* to
go, not the raw actuation.
"""

from __future__ import annotations

import numpy as np

from ..orientation_utils import orientation_error_rotvec
from .attached_object import AttachedObjectManager
from .collision import CollisionChecker, CollisionPhase
from .config import MotionConfig
from .grasp_generator import make_grasp_generator
from .kinematics import IKSolver
from .mpc import JointSpaceMPC, MPCContext
from .ompl_planner import MotionPlanner
from .panda import EE_SITE, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES
from .pick_skill import PickSkill
from .scene_query import SceneQuery, quat_to_rotmat_wxyz, resolve_body_name
from .time_parameterize import TimeParameterizer
from .types import AttachedObjectState, MPCCostConfig, SkillFailureReason, SkillResult


class LiberoPickAdapter:
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE
        self.hand_site = HAND_SITE

    # ------------------------------------------------------------ construction

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _build_pick_skill(self, scene: SceneQuery) -> PickSkill:
        checker = CollisionChecker(
            scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision
        )
        lo, hi = scene.joint_qpos_range(self.joint_names)
        gen = make_grasp_generator(self.config.grasp)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        return PickSkill(gen, ik, planner, checker, self.config)

    # ------------------------------------------------------------ low-level stepping

    def _servo_toward_joint_target(
        self, scene: SceneQuery, target_q: np.ndarray, pos_scale_m: float = 0.03, ori_gain: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a joint-space target into the (cart_pos, cart_ori) delta
        pair the legacy stage machine's action layout expects, via FK on a
        scratch scene (never the live sim)."""
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

    def _servo_toward_cartesian_pose(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        pos_scale_m: float = 0.03,
        ori_gain: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Same servo law as ``_servo_toward_joint_target``, but against a
        fixed Cartesian target directly (no FK needed) -- used by the settle
        phase, which tracks one fixed grasp pose rather than a moving
        receding-horizon reference. Also returns the current (position_error,
        orientation_error) so the caller can check convergence."""
        target_R = quat_to_rotmat_wxyz(target_quat_wxyz)
        ee_pos = self.env._eef_pos(self.env.last_raw_obs)
        ee_R = self.env._eef_rotmat(self.env.last_raw_obs)
        pos_err = target_pos - ee_pos
        cart_pos = np.clip(pos_err / pos_scale_m, -1.0, 1.0).astype(np.float32)
        ori_err_vec = orientation_error_rotvec(ee_R, target_R)
        cart_ori = np.clip(ori_gain * ori_err_vec, -1.0, 1.0).astype(np.float32)
        return cart_pos, cart_ori, float(np.linalg.norm(pos_err)), float(np.linalg.norm(ori_err_vec))

    def _settle_toward_grasp_pose(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        gripper_cmd: float,
        step_callback=None,
    ) -> tuple[int, bool, float, float]:
        """Keep servoing toward the grasp pose (gripper unchanged, typically
        still open) until the end effector actually arrives within tolerance
        or ``max_settle_steps`` is exhausted -- addresses the MPC's nonzero
        steady-state tracking lag (see mpc.py's module docstring) closing the
        gripper before the arm has actually reached the grasp pose, which
        can miss the object or catch it off-center instead of centered
        between the fingers. Returns (steps_taken, converged, final_pos_err,
        final_ori_err)."""
        cfg = self.config.execution
        steps = 0
        pos_err = ori_err = float("inf")
        for _ in range(cfg.max_settle_steps):
            cart_pos, cart_ori, pos_err, ori_err = self._servo_toward_cartesian_pose(
                target_pos, target_quat_wxyz
            )
            if pos_err <= cfg.settle_position_tol_m and ori_err <= cfg.settle_orientation_tol_rad:
                return steps, True, pos_err, ori_err
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
        return steps, (pos_err <= cfg.settle_position_tol_m and ori_err <= cfg.settle_orientation_tol_rad), pos_err, ori_err

    def _step_env(self, cart_pos: np.ndarray, cart_ori: np.ndarray, gripper_cmd: float) -> dict:
        full_action = self.env._compose_full_action(cart_pos, cart_ori, gripper_cmd)
        obs, _reward, _terminated, info = self.env._step_backend(full_action)
        self.env.last_raw_obs = obs
        return info

    def _current_q(self, scene: SceneQuery) -> np.ndarray:
        # Re-sync from the live sim every read -- the scene passed in at
        # construction time is a one-shot snapshot, but the arm moves every
        # step, so joint readings must come from the live obs, not the
        # (now-stale) scratch clone used purely for FK.
        return SceneQuery.from_robosuite_sim(self.env.env.sim).get_qpos(self.joint_names)

    # ------------------------------------------------------------ public entry point

    def run(
        self,
        object_set: list[str],
        constraints: dict | None = None,
        max_steps: int = 600,
        step_callback=None,
    ) -> tuple[SkillResult, int, AttachedObjectState | None]:
        """Returns (SkillResult, steps_executed, AttachedObjectState | None) so the executor can build a
        ``SkillExecutionRecord`` without re-deriving step counts.

        ``step_callback``, if given, is called with no arguments after every
        real env step -- the executor wires this to ``_capture_video_frame``
        so ``--save-video`` works for this backend too (this loop doesn't go
        through ``step_nominal()``, which is where that normally gets called).
        """
        constraints = constraints or {}
        scene = self._build_scene()
        skill = self._build_pick_skill(scene)
        start_q = scene.get_qpos(self.joint_names)

        # object_set comes from SceneResolver-canonical names (e.g.
        # "alphabet_soup_1"); the motion pipeline needs real MuJoCo body names
        # (e.g. "alphabet_soup_1_main") -- see resolve_body_name's docstring.
        body_names = [resolve_body_name(scene, name) for name in object_set]
        canonical_by_body = dict(zip(body_names, object_set))

        plan_result = skill.execute(scene, body_names, start_q, constraints)
        if not plan_result.success:
            return plan_result, 0, None

        traj = TimeParameterizer(self.config.time_param).parameterize(plan_result.planned_trajectory)
        if traj.positions is None:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = "time parameterization produced no reference"
            return plan_result, 0, None

        cand = plan_result.selected_grasp
        object_body_name = plan_result.selected_grasp.object_name  # real MuJoCo body name
        object_canonical_name = canonical_by_body.get(object_body_name, object_body_name)

        # PickSkill already validated (via check_edge, not blind execution --
        # see its module docstring) a short final descent from the standoff
        # point to the grasp pose; append it here, same as the OMPL-planned
        # standoff leg, at the same control rate.
        approach_positions = np.asarray(plan_result.planned_trajectory.metadata["approach_positions"])
        full_reference = np.concatenate([traj.positions, approach_positions[1:]], axis=0)
        approach_start_idx = len(traj.positions)

        checker = skill.collision_checker
        mpc = JointSpaceMPC(
            self.joint_names,
            self.ee_site,
            horizon=int(self.config.mpc.get("horizon", 16)),
            cost_config=MPCCostConfig(**{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}),
        )
        ctx = MPCContext(reference=full_reference)
        mpc.reset()

        steps = 0
        gripper_cmd = self.env.open_gripper_cmd
        for t0 in range(len(full_reference)):
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
                target_object_body=object_body_name,
                replan_config=self.config.replan,
            )
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, mpc_result.joint_command)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            if steps >= max_steps:
                break

        # Settle: the MPC has nonzero steady-state tracking lag, so the arm
        # may not have actually *arrived* at the grasp pose when the
        # reference trajectory runs out -- closing immediately at that point
        # can miss the object or catch it off-center. Keep the gripper open
        # and keep servoing directly toward the true grasp pose (not the
        # trajectory's last waypoint, which is itself only an IK solution for
        # it) until it actually converges.
        settle_steps, converged, settle_pos_err, settle_ori_err = self._settle_toward_grasp_pose(
            cand.grasp_position, cand.grasp_quat_wxyz, gripper_cmd, step_callback=step_callback
        )
        steps += settle_steps
        plan_result.diagnostics["settle"] = {
            "steps": settle_steps,
            "converged": converged,
            "position_error_m": settle_pos_err,
            "orientation_error_rad": settle_ori_err,
        }

        # Close the gripper and dwell, continuing to servo toward the grasp
        # pose every step (not a single stale action) since the arm may still
        # be settling from tracking error when the approach segment ends.
        gripper_cmd = self.env.close_gripper_cmd
        close_dwell_steps = max(int(self.env.close_steps) * 3, 40)
        for _ in range(close_dwell_steps):
            hold_pos, hold_ori, _, _ = self._servo_toward_cartesian_pose(
                cand.grasp_position, cand.grasp_quat_wxyz
            )
            self._step_env(hold_pos, hold_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        grasped = bool(self.env._is_grasped())
        if not grasped:
            import os

            if os.environ.get("STEP_DEBUG_PICK_ADAPTER"):
                dbg_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
                ee_pos, _ = dbg_scene.site_pose(self.ee_site)
                obj_pos, _ = dbg_scene.body_pose(object_body_name)
                print(
                    f"[pick_adapter debug] ee_pos={ee_pos} obj_pos={obj_pos} "
                    f"grasp_target={cand.grasp_position} ee-target_err={np.linalg.norm(ee_pos - cand.grasp_position)} "
                    f"gripper_qpos={dbg_scene.get_qpos(['gripper0_finger_joint1', 'gripper0_finger_joint2'])}"
                )
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.GRASP_VERIFICATION_FAILED
            plan_result.diagnostics["detail"] = "gripper closed but _is_grasped() reported False"
            return plan_result, steps, None

        attached_mgr = AttachedObjectManager()
        attach_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        attached_state = attached_mgr.attach(attach_scene, object_body_name, self.hand_site, step=steps)
        plan_result.diagnostics["attached_object"] = {
            "object_name": attached_state.object_name,
            "collision_radius_m": attached_state.collision_radius_m,
        }
        # Report back in SceneResolver-canonical form -- everything downstream
        # of this adapter (env.active_object_name, later subtask dispatch,
        # the execution-summary record) uses that convention, not raw MuJoCo
        # body names.
        plan_result.selected_object = object_canonical_name
        self.env.set_active_object(object_canonical_name)
        return plan_result, steps, attached_state
