"""Unified ``Pick(object_or_object_set, constraints)`` skill.

Externally this is one atomic operation -- callers never see separate
PreGrasp/Grasp subtasks (see ``step/system_two/stl_plan.py``'s
``normalize_legacy_pregrasp_grasp`` for the compatibility boundary that
guarantees this even for legacy/external data). Internally: generate grasp
candidates -> IK + collision filtering -> plan a collision-free long-range
approach (under strict ``FREE_SPACE``, no contact with the target permitted)
to a standoff point close to the object, then validate a short final descent
from there to the exact grasp pose under ``TERMINAL_APPROACH`` (gripper<->
target contact now expected) via ``CollisionChecker.check_edge`` -- never
executed without that check. (An earlier version planned the *entire* path
under ``TERMINAL_APPROACH``, reasoning that a single planning call is enough
to keep this "one atomic Pick"; that turned out to let RRTConnect bump/shove
the object anywhere along a long approach, since contact was permitted
everywhere rather than just at the very end. Keeping the long-range leg
strict avoids that regression while still fixing the actual bug: the
standoff->grasp segment is small and explicitly validated now, whereas an
even-earlier version executed it as raw, unchecked joint-space interpolation.
See ``GraspCandidate``'s docstring in ``types.py`` for the full history.)
Execution/tracking happens by the caller feeding the returned ``MotionPlan``
to ``JointSpaceMPC``, since ``PickSkill`` itself is planning-only and
backend-agnostic; the caller closes the gripper, verifies the grasp, and
registers the object as attached via ``AttachedObjectManager``.

OR semantics ("pick any cup"): every candidate object in ``object_set`` is
fully evaluated (grasp generation -> IK -> planning -> cost) *before* any
commitment is made. The lowest-cost *feasible* (object, grasp, trajectory)
tuple wins; a candidate is never selected on generator score alone.
"""

from __future__ import annotations

import numpy as np

from .collision import CollisionChecker, CollisionPhase
from .config import MotionConfig
from .grasp_generator import GraspGenerator
from .kinematics import IKSolver
from .ompl_planner import MotionPlanner
from .scene_query import SceneQuery
from .types import (
    CandidateScore,
    GraspCandidate,
    SkillFailureReason,
    SkillResult,
)


class PickSkill:
    def __init__(
        self,
        grasp_generator: GraspGenerator,
        ik_solver: IKSolver,
        planner: MotionPlanner,
        collision_checker: CollisionChecker,
        config: MotionConfig | None = None,
    ) -> None:
        self.grasp_generator = grasp_generator
        self.ik_solver = ik_solver
        self.planner = planner
        self.collision_checker = collision_checker
        self.config = config or MotionConfig()

    def execute(
        self,
        scene: SceneQuery,
        object_set: list[str],
        start_q: np.ndarray,
        constraints: dict | None = None,
    ) -> SkillResult:
        constraints = constraints or {}
        downstream_goal_xy = constraints.get("downstream_goal_xy")  # optional (x, y) reachability hint

        candidates = self.grasp_generator.generate(scene, object_set)
        if not candidates:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": "grasp generator returned no candidates", "object_set": object_set},
            )

        scored: list[CandidateScore] = []
        for cand in candidates:
            scored.append(self._score_candidate(scene, cand, start_q, downstream_goal_xy))

        feasible = [s for s in scored if s.feasible]
        if not feasible:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": "no candidate passed IK/collision/planning/downstream-feasibility filtering",
                    "n_candidates_tried": len(scored),
                    "rejection_reasons": [s.rejection_reason for s in scored],
                },
            )

        best = min(feasible, key=lambda s: s.total_cost)
        return SkillResult(
            success=True,
            selected_object=best.object_name,
            selected_grasp=best.grasp,
            planned_trajectory=best.plan,
            cost_breakdown=best.breakdown,
            failure_reason=SkillFailureReason.NONE,
            diagnostics={
                "n_candidates_tried": len(scored),
                "n_feasible": len(feasible),
                "candidate_costs": [(s.object_name, s.grasp.metadata.get("yaw_rad"), s.total_cost) for s in scored],
            },
        )

    def _score_candidate(
        self,
        scene: SceneQuery,
        cand: GraspCandidate,
        start_q: np.ndarray,
        downstream_goal_xy: tuple[float, float] | None,
    ) -> CandidateScore:
        breakdown: dict[str, float] = {}

        # --- gripper-opening compatibility -------------------------------------------------
        # The AABB-derived width estimate (grasp_generator._estimate_width) is a
        # conservative *overestimate* for round/irregular objects -- it's a bounding
        # box, not the object's true cross-section at the grasp height. A small
        # tolerance avoids hard-rejecting genuinely graspable objects whose real
        # cross-section is a bit smaller than their bounding box; grossly
        # incompatible widths are still rejected.
        width_tolerance = 1.15
        if (
            cand.gripper_width is not None
            and cand.gripper_width > self.grasp_generator.gripper_max_opening_m * width_tolerance
        ):
            return CandidateScore(
                object_name=cand.object_name,
                grasp=cand,
                plan=None,
                feasible=False,
                total_cost=float("inf"),
                rejection_reason="gripper_width_incompatible",
            )

        # --- IK: approach standoff first, then the grasp pose seeded from it -----------------
        # No separate pregrasp *stage* is exposed to callers (this is still one
        # PickSkill.execute() call) -- but internally, planning still goes
        # through a standoff point close to (not at) the object, exactly like
        # PlaceSkill's hover point. An earlier version of this method planned
        # the *entire* path (start -> grasp) under TERMINAL_APPROACH, which
        # permits gripper<->target contact everywhere, not just at the end --
        # that let RRTConnect freely bump/shove the object anywhere along a
        # long approach, since contact was never penalized. Confirmed by a
        # real run: a bowl ended up ~5cm from its recorded scene-discovery
        # position after a "successful" plan, because the arm pushed it
        # before ever closing the gripper. Keeping the long-range leg under
        # strict FREE_SPACE (no target contact permitted at all) avoids that;
        # only the short final descent segment below transitions to
        # TERMINAL_APPROACH, and unlike the old pregrasp->grasp segment, that
        # segment is now explicitly collision-validated via check_edge before
        # ever being trusted for execution -- the actual bug this whole
        # function is fixing.
        standoff_position = cand.grasp_position - self.config.grasp.approach_standoff_m * cand.approach_direction_world
        scratch_standoff = scene.clone()
        standoff_ik = self.ik_solver.solve(scratch_standoff, standoff_position, cand.grasp_quat_wxyz, seed_q=start_q)
        if not standoff_ik.success:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason="approach_standoff_ik_failed"
            )
        scratch_grasp = scene.clone()
        grasp_ik = self.ik_solver.solve(scratch_grasp, cand.grasp_position, cand.grasp_quat_wxyz, seed_q=standoff_ik.q)
        if not grasp_ik.success:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason="grasp_ik_failed"
            )
        breakdown["joint_quality"] = float(np.linalg.norm(standoff_ik.q - start_q))

        # --- standoff / long-range approach collision (no target contact permitted) ---------
        standoff_check = self.collision_checker.check(
            standoff_ik.q, phase=CollisionPhase.FREE_SPACE, target_object_body=cand.object_name
        )
        if standoff_check.in_collision:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason="approach_standoff_in_collision"
            )

        # Terminal grasp contact with the target object is expected/permitted.
        grasp_check = self.collision_checker.check(
            grasp_ik.q, phase=CollisionPhase.TERMINAL_APPROACH, target_object_body=cand.object_name
        )
        if grasp_check.in_collision:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason="grasp_pose_in_collision"
            )

        # --- final descent segment: explicitly validated, never blindly executed ------------
        descent_check = self.collision_checker.check_edge(
            standoff_ik.q,
            grasp_ik.q,
            resolution=self.config.planner.validity_resolution
            * float(np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)),
            phase=CollisionPhase.TERMINAL_APPROACH,
            target_object_body=cand.object_name,
        )
        if descent_check.in_collision:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason="final_descent_in_collision"
            )

        # --- short post-grasp lift feasibility (still IK-reachable a bit higher) ------------
        lift_target = cand.grasp_position + np.array([0.0, 0.0, self.config.grasp.lift_check_m])
        scratch3 = scene.clone()
        lift_ik = self.ik_solver.solve(scratch3, lift_target, cand.grasp_quat_wxyz, seed_q=grasp_ik.q)
        if not lift_ik.success:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason="post_grasp_lift_infeasible"
            )

        # --- downstream transport/place reachability hint (cheap IK-only check) ------------
        if downstream_goal_xy is not None:
            downstream_target = np.array(
                [downstream_goal_xy[0], downstream_goal_xy[1], cand.grasp_position[2] + 0.15]
            )
            scratch4 = scene.clone()
            downstream_ik = self.ik_solver.solve(
                scratch4, downstream_target, cand.grasp_quat_wxyz, seed_q=lift_ik.q
            )
            if not downstream_ik.success:
                return CandidateScore(
                    cand.object_name,
                    cand,
                    None,
                    False,
                    float("inf"),
                    rejection_reason="downstream_target_unreachable_with_this_grasp",
                )

        # --- plan the long-range approach to the standoff point, scored by path + reachability + clearance --
        plan = self.planner.plan(
            start_q, [standoff_ik.q], phase=CollisionPhase.FREE_SPACE, target_object_body=cand.object_name
        )
        if not plan.success:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason=f"planning_failed:{plan.failure_reason}"
            )

        # Already validated above via check_edge -- safe to hand to the
        # executor as-is (see libero_pick_adapter.py, which appends this to
        # the time-parameterized standoff leg and tracks it under
        # TERMINAL_APPROACH).
        approach_positions = np.linspace(standoff_ik.q, grasp_ik.q, 15)
        plan.metadata["standoff_q"] = standoff_ik.q.tolist()
        plan.metadata["grasp_q"] = grasp_ik.q.tolist()
        plan.metadata["approach_positions"] = approach_positions.tolist()

        path_cost = 0.0
        if plan.geometric_path is not None and len(plan.geometric_path) > 1:
            path_cost = float(np.sum(np.linalg.norm(np.diff(plan.geometric_path, axis=0), axis=1)))
        breakdown["path_length"] = path_cost
        breakdown["reachability"] = float(standoff_ik.position_error + standoff_ik.orientation_error)
        breakdown["generator_score_penalty"] = float(1.0 - cand.generator_score)

        total_cost = (
            1.0 * path_cost
            + 2.0 * breakdown["joint_quality"]
            + 5.0 * breakdown["reachability"]
            + 1.0 * breakdown["generator_score_penalty"]
        )

        return CandidateScore(
            object_name=cand.object_name,
            grasp=cand,
            plan=plan,
            feasible=True,
            total_cost=total_cost,
            breakdown=breakdown,
        )
