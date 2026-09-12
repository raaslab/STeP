"""Executor-facing adapter for the ``transport`` skill: wires
``TransportSkill``'s attached-object-aware hover-leg planning into a live
``LiberoManipulationEnv``.

Mirrors ``LiberoOpenDrawerAdapter``/``LiberoStoveAdapter``'s own shape (same
``_ServoMixin`` stepping, same MPC-tracked-reference-then-settle pattern),
but tracks with ``attached=`` passed through to both planning and live MPC
collision checking throughout -- the carried object is a real, checked part
of the robot's own collision geometry for this entire leg, not just the
bare gripper. Ends holding the object at the hover pose (gripper still
closed) -- release is a subsequent ``place`` subtask's job, matching the
OverGoal vs Placed/Released predicate split already used in STL formulas.
"""

from __future__ import annotations

import numpy as np

from ..attached_object import attached_sphere_infeasible_for_goal
from ..collision import CollisionChecker, CollisionPhase
from ..config import MotionConfig
from ..kinematics import IKSolver
from ..mpc import JointSpaceMPC, MPCContext
from ..ompl_planner import MotionPlanner
from .transport_skill import TransportSkill
from ..panda import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES
from ..scene_query import SceneQuery, resolve_body_name
from ..servo import MotionTraceRecorder, _ServoMixin, is_settle_stalled, real_nonpermitted_contacts
from ..time_parameterize import TimeParameterizer
from ..types import AttachedObjectState, MPCCostConfig, SkillFailureReason, SkillResult


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
        # If the carried object's own sphere approximation can never fit
        # within the goal's own radius regardless of approach (see
        # attached_sphere_infeasible_for_goal's own docstring for the real
        # rollout this was measured on -- a book whose fitted spheres,
        # ~5.7cm, can't satisfy a ~2.2cm goal clearance no matter what),
        # stop weighing that sphere in the motion cost for this transport:
        # continuing to fight a conflict the approximation can never resolve
        # only produces worse tracking (a wildly wrong final settle) and
        # risks shaking the object loose, for a conflict the real object
        # may not even have. The real robot/gripper geometry is still
        # checked normally either way -- only the carried object's own
        # approximate sphere stops being treated as an obstacle.
        ignore_attached_sphere = attached_sphere_infeasible_for_goal(attached, goal_radius_m)
        mpc_attached = None if ignore_attached_sphere else attached

        # NOTE: always the real `attached`, never `mpc_attached` -- unlike
        # the MPC's own clearance cost below (a pure avoidance detail, safe
        # to disable), TransportSkill.execute() needs the object's real
        # identity/grasp transform throughout (escape_start_collision,
        # hover-pose computation, etc.), not just for sphere-avoidance.
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

        # The plan's reference starts from the *escaped* start_q the skill
        # computed (see TransportSkill.execute()'s own comment on
        # `plan_start_q`), which can be a non-trivial distance from wherever
        # the real arm currently sits (a carried object resting on the
        # table can need a real lift to clear it, unlike PlaceSkill's own,
        # usually-small escape). Move there directly first -- confirmed via
        # a real rollout that skipping this left the MPC reference assuming
        # a starting position the arm was never actually at, and its
        # attempt to reconcile that gap produced a huge (~0.5-0.9 rad)
        # joint-space tracking error for the *entire* subsequent leg,
        # driving the carried object through the table/cabinet the whole
        # way rather than the collision-free path that was actually planned.
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
        mpc = JointSpaceMPC(
            self.joint_names,
            self.ee_site,
            horizon=int(self.config.mpc.get("horizon", 16)),
            cost_config=MPCCostConfig(**{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}),
        )
        ctx = MPCContext(reference=traj.positions)
        mpc.reset()

        cfg = self.config.execution
        gripper_cmd = self.env.close_gripper_cmd  # still holding the object throughout transport
        pos_err = ori_err = float("inf")
        converged = False
        stalled = False
        recent_pos_errs: list[float] = []
        t0 = 0
        trace = MotionTraceRecorder()
        while t0 < len(traj.positions) + cfg.max_settle_steps:
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            q_current = live_scene.get_qpos(self.joint_names)
            mpc_result = mpc.step(
                live_scene,
                q_current,
                ctx,
                t0_step=t0,
                collision_checker=checker,
                collision_phase=CollisionPhase.FREE_SPACE,
                goal_container_body=goal_container_body,
                attached=mpc_attached,
                replan_config=self.config.replan,
            )
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, mpc_result.joint_command)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            t0 += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            trace.record_step(
                t0,
                post_scene.site_pose(self.ee_site)[0],
                mpc_result.min_clearance,
                mpc_result.replan_requested,
                real_nonpermitted_contacts(post_scene, checker, CollisionPhase.FREE_SPACE, goal_container_body),
            )
            if steps >= max_steps:
                break

            if t0 >= len(traj.positions):
                _, _, pos_err, ori_err = self._servo_toward_cartesian_pose(hover_hand_pos, hover_hand_quat)
                if pos_err <= cfg.settle_position_tol_m and ori_err <= cfg.settle_orientation_tol_rad:
                    converged = True
                    break
                recent_pos_errs.append(pos_err)
                if is_settle_stalled(recent_pos_errs, cfg.stall_window_steps, cfg.stall_position_tol_m):
                    stalled = True
                    break

        settle_steps = max(0, t0 - len(traj.positions))
        plan_result.diagnostics["settle"] = {
            "steps": settle_steps,
            "converged": converged,
            "stalled": stalled,
            "position_error_m": pos_err,
            "orientation_error_rad": ori_err,
        }
        plan_result.diagnostics["motion_trace"] = {"transport": trace.summary()}

        # Structural verification: is the carried object actually over the
        # goal now, and still grasped? Same measured-geometry signal
        # PlaceSkill's own success ultimately depends on -- not a contact
        # heuristic, which the legacy backend this replaces had no
        # equivalent of anyway.
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

        plan_result.selected_object = attached.object_name
        return plan_result, steps
