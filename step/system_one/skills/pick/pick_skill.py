"""Unified Pick(object_or_object_set, constraints) skill: one atomic op,
no separate PreGrasp/Grasp subtasks exposed to callers.

Plans a contact-free FREE_SPACE approach to a standoff near the object,
then a short, explicitly check_edge-validated TERMINAL_APPROACH
descent to the grasp pose -- the long leg stays contact-free so RRTConnect
can't bump the object. Planning-only/backend-agnostic: the caller executes
via JointSpaceMPC, closes the gripper, and attaches the object.

OR semantics ("pick any cup"): every object in object_set is fully
evaluated before any commitment; the lowest-cost *feasible* tuple wins.
"""

from __future__ import annotations

import time

import numpy as np

from ...motion.collision import CollisionChecker, CollisionPhase
from ...motion.config import MotionConfig
from .grasp_generator import GraspGenerator
from ..ik_utils import escape_start_collision, solve_ik_robust as _solve_ik_robust
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from ...motion.panda import FINGER_CLOSED_QPOS, FINGER_JOINT_NAMES
from ...motion.scene_query import SceneQuery
from ..standoff_search import standoff_candidates
from ...motion.types import (
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
        self._rng = np.random.default_rng(self.config.planner.seed)

    def execute(
        self,
        scene: SceneQuery,
        object_set: list[str],
        start_q: np.ndarray,
        constraints: dict | None = None,
        candidates: list[GraspCandidate] | None = None,
    ) -> SkillResult:
        constraints = constraints or {}
        downstream_goal_xy = constraints.get("downstream_goal_xy")  # optional (x, y) reachability hint

        # A pick often follows a place, leaving the arm in a collision-adjacent
        # pose OMPL refuses to plan from; only the "straight up" retreat is tried.
        start_q = escape_start_collision(self.ik_solver, self.collision_checker, scene, None, None, start_q)

        # Callers that already generated candidates (e.g. for a viz save) can
        # pass them through, skipping a second, possibly expensive regenerate().
        if candidates is None:
            candidates = self.grasp_generator.generate(scene, object_set)
        if not candidates:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": "grasp generator returned no candidates", "object_set": object_set},
            )

        # A hard-to-grasp object can make every candidate exhaust the full
        # standoff search before failing; this budget stops scoring once it's
        # spent rather than silently burning the whole rollout timeout. Split
        # adaptively over remaining candidates (not one shared deadline) so a
        # single pathological candidate (e.g. a centered grasp wedged against
        # a neighbor) can't eat the whole budget and starve out untried ones.
        budget_sec = self.config.grasp.candidate_scoring_budget_sec
        deadline = time.time() + budget_sec if budget_sec and budget_sec > 0 else None
        # Score physically-graspable candidates first. Width is the one filter
        # needing no IK, no collision check and no plan, so a candidate the hand
        # can never straddle should not consume the budget ahead of one it can.
        # Measured on a RoboCasa lemon: GraspNet returned 32 candidates with
        # widths to 13.6 cm against an 8 cm gripper, and the two feasible ones
        # were scored last, after the expensive rejections. Ordering only --
        # nothing is dropped, so a run where every candidate is too wide reports
        # exactly what it did before.
        _max_open = self.grasp_generator.gripper_max_opening_m
        candidates = sorted(
            candidates,
            key=lambda c: (c.gripper_width is not None and c.gripper_width > _max_open,
                           c.gripper_width if c.gripper_width is not None else 0.0),
        )
        scored: list[CandidateScore] = []
        scoring_timed_out = False
        for i, cand in enumerate(candidates):
            now = time.time()
            if deadline is not None and now >= deadline:
                scoring_timed_out = True
                break
            cand_deadline = now + (deadline - now) / (len(candidates) - i) if deadline is not None else None
            scored.append(self._score_candidate(scene, cand, start_q, downstream_goal_xy, deadline=cand_deadline))

        feasible = [s for s in scored if s.feasible]
        if not feasible:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": "no candidate passed IK/collision/planning/downstream-feasibility filtering",
                    "n_candidates_tried": len(scored),
                    "n_candidates_total": len(candidates),
                    "scoring_timed_out": scoring_timed_out,
                    "rejection_reasons": [s.rejection_reason for s in scored],
                    # Enough per-candidate context to diagnose why every candidate
                    # failed directly from execution_summary.json, without a re-run.
                    "rejected_candidates": [
                        {
                            "object_name": s.object_name,
                            "strategy": s.grasp.metadata.get("strategy"),
                            "yaw_rad": s.grasp.metadata.get("yaw_rad"),
                            "gripper_width_m": s.grasp.gripper_width,
                            "grasp_position": s.grasp.grasp_position.tolist(),
                            "rejection_reason": s.rejection_reason,
                        }
                        for s in scored
                    ],
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
                "n_candidates_total": len(candidates),
                "scoring_timed_out": scoring_timed_out,
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
        deadline: float | None = None,
    ) -> CandidateScore:
        breakdown: dict[str, float] = {}

        # Small tolerance (capped at the real opening) for round objects'
        # conservative width overestimate, never past what the hand can span.
        width_tolerance = 1.15
        if (
            cand.gripper_width is not None
            and cand.gripper_width
            > min(
                self.grasp_generator.gripper_max_opening_m * width_tolerance,
                self.grasp_generator.gripper_max_opening_m,
            )
        ):
            return CandidateScore(
                object_name=cand.object_name,
                grasp=cand,
                plan=None,
                feasible=False,
                total_cost=float("inf"),
                rejection_reason="gripper_width_incompatible",
            )

        # qpos=[0,0] is zero gap, not "closed on an object" -- target half the
        # object's real width, falling back to the joint limit if unknown.
        if cand.gripper_width is not None:
            half = float(np.clip(cand.gripper_width / 2.0, 0.0, 0.04))
            finger_closed_qpos = np.array([half, -half])
        else:
            finger_closed_qpos = FINGER_CLOSED_QPOS
        descent_resolution = self.config.planner.validity_resolution * float(
            np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)
        )

        # Each standoff seeds a different grasp_ik solve (DLS IK is non-unique),
        # so validation is retried per standoff -- a reachable one can still sweep the forearm into collision at the grasp pose.
        standoff_ik = grasp_ik = above_ik = None
        last_rejection = "approach_standoff_ik_failed"
        # Tries the straight-line standoff first, then wider sideways rings in
        # case the airspace above the object is blocked.
        for standoff_position in standoff_candidates(
            cand.grasp_position,
            cand.approach_direction_world,
            self.config.grasp.approach_standoff_m,
            self.config.grasp.standoff_distance_multipliers,
            self.config.grasp.standoff_lateral_offsets_m,
            self.config.grasp.standoff_lateral_directions,
        ):
            if deadline is not None and time.time() >= deadline:
                last_rejection = "candidate_time_budget_exhausted"
                break
            candidate_standoff_ik = _solve_ik_robust(
                self.ik_solver, scene, standoff_position, cand.grasp_quat_wxyz, start_q, self._rng,
                n_restarts=self.config.grasp.pose_selection_attempts,
            )
            if not candidate_standoff_ik.success:
                continue
            standoff_check = self.collision_checker.check(
                candidate_standoff_ik.q, phase=CollisionPhase.FREE_SPACE, target_object_body=cand.object_name
            )
            if standoff_check.in_collision:
                last_rejection = "approach_standoff_in_collision"
                continue

            candidate_grasp_ik = _solve_ik_robust(
                self.ik_solver, scene, cand.grasp_position, cand.grasp_quat_wxyz, candidate_standoff_ik.q, self._rng,
                n_restarts=self.config.grasp.pose_selection_attempts,
            )
            if not candidate_grasp_ik.success:
                last_rejection = "grasp_ik_failed"
                continue

            # Target contact is expected/permitted here.
            grasp_check = self.collision_checker.check(
                candidate_grasp_ik.q, phase=CollisionPhase.TERMINAL_APPROACH, target_object_body=cand.object_name
            )
            if grasp_check.in_collision:
                last_rejection = "grasp_pose_in_collision"
                continue

            # Forces the final descent straight down; otherwise a lateral
            # standoff offset makes it diagonal, and a side clip wouldn't register as a collision.
            above_position = (
                cand.grasp_position
                - self.config.grasp.approach_standoff_m * cand.approach_direction_world
            )
            candidate_above_ik = _solve_ik_robust(
                self.ik_solver, scene, above_position, cand.grasp_quat_wxyz, candidate_standoff_ik.q, self._rng,
                n_restarts=self.config.grasp.pose_selection_attempts,
            )
            if not candidate_above_ik.success:
                last_rejection = "above_grasp_ik_failed"
                continue
            above_check = self.collision_checker.check(
                candidate_above_ik.q, phase=CollisionPhase.FREE_SPACE, target_object_body=cand.object_name
            )
            if above_check.in_collision:
                last_rejection = "above_grasp_in_collision"
                continue

            # standoff->above forbids target contact; above->grasp is the one
            # segment allowed to touch, always a straight vertical line.
            approach_descent_check = self.collision_checker.check_edge(
                candidate_standoff_ik.q,
                candidate_above_ik.q,
                resolution=descent_resolution,
                phase=CollisionPhase.FREE_SPACE,
                target_object_body=cand.object_name,
            )
            if approach_descent_check.in_collision:
                last_rejection = "standoff_to_above_in_collision"
                continue
            descent_check = self.collision_checker.check_edge(
                candidate_above_ik.q,
                candidate_grasp_ik.q,
                resolution=descent_resolution,
                phase=CollisionPhase.TERMINAL_APPROACH,
                target_object_body=cand.object_name,
            )
            if descent_check.in_collision:
                last_rejection = "final_descent_in_collision"
                continue

            # IK/OMPL/MPC only move the 7 arm joints, so a pose whose open
            # fingers clear the object can still catch it as the fingers close.
            closure_check = self.collision_checker.check_gripper_closure(
                candidate_grasp_ik.q, FINGER_JOINT_NAMES, finger_closed_qpos, cand.object_name
            )
            if closure_check.in_collision:
                last_rejection = "gripper_closure_in_collision"
                continue

            standoff_ik, grasp_ik, above_ik = candidate_standoff_ik, candidate_grasp_ik, candidate_above_ik
            break

        if standoff_ik is None or grasp_ik is None:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason=last_rejection
            )
        breakdown["joint_quality"] = float(np.linalg.norm(standoff_ik.q - start_q))

        # Confirms a short post-grasp lift is still IK-reachable.
        lift_target = cand.grasp_position + np.array([0.0, 0.0, self.config.grasp.lift_check_m])
        scratch3 = scene.clone()
        lift_ik = self.ik_solver.solve(scratch3, lift_target, cand.grasp_quat_wxyz, seed_q=grasp_ik.q)
        if not lift_ik.success:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason="post_grasp_lift_infeasible"
            )

        # Cheap IK-only hint that transport/place will also be reachable from this grasp.
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

        # Single call, not a retry loop: OMPL ignores reseeding, so retrying
        # only helps if the goal itself changes.
        plan = self.planner.plan(
            start_q, [standoff_ik.q], phase=CollisionPhase.FREE_SPACE, target_object_body=cand.object_name
        )
        if not plan.success:
            return CandidateScore(
                cand.object_name, cand, None, False, float("inf"), rejection_reason=f"planning_failed:{plan.failure_reason}"
            )

        # Already validated via check_edge above; two legs so the executor can
        # track standoff->above and above->grasp as separate MPC phases.
        to_above_count = 8
        approach_positions = np.concatenate(
            [
                np.linspace(standoff_ik.q, above_ik.q, to_above_count),
                np.linspace(above_ik.q, grasp_ik.q, 8)[1:],
            ]
        )
        plan.metadata["standoff_q"] = standoff_ik.q.tolist()
        plan.metadata["above_q"] = above_ik.q.tolist()
        plan.metadata["grasp_q"] = grasp_ik.q.tolist()
        plan.metadata["approach_positions"] = approach_positions.tolist()
        # Where approach_positions splits into its two legs; a single
        # continuous reference would let XY and Z blend across the boundary.
        plan.metadata["to_above_count"] = to_above_count

        path_cost = 0.0
        if plan.geometric_path is not None and len(plan.geometric_path) > 1:
            path_cost = float(np.sum(np.linalg.norm(np.diff(plan.geometric_path, axis=0), axis=1)))
        breakdown["path_length"] = path_cost
        breakdown["reachability"] = float(standoff_ik.position_error + standoff_ik.orientation_error)
        breakdown["generator_score_penalty"] = float(1.0 - cand.generator_score)

        # Biases top-down grasps toward the deepest safe height; gated to
        # near-vertical approaches since the palm/floor clamp is a floor, not an attractor.
        extent = scene.body_extent_world(cand.object_name)
        if extent is not None and float(np.dot(cand.approach_direction_world, [0.0, 0.0, -1.0])) > 0.9:
            obj_pos, _ = scene.body_pose(cand.object_name)
            margin = self.config.grasp.grasp_height_margin_m
            object_center_z = float(obj_pos[2] + extent.center_offset[2])
            object_top_z = object_center_z + float(extent.half_extents[2])
            object_bottom_z = object_center_z - float(extent.half_extents[2])
            min_safe_for_palm = object_top_z - scene.palm_clearance_above_tcp_m() + margin
            min_safe_for_floor = object_bottom_z + scene.finger_reach_below_tcp_m() + margin
            ideal_deep_z = min(max(min_safe_for_palm, min_safe_for_floor), object_top_z - margin)
            breakdown["grasp_depth_penalty"] = max(0.0, float(cand.grasp_position[2]) - ideal_deep_z)
        else:
            breakdown["grasp_depth_penalty"] = 0.0

        # Weight 8.0 keeps a low-confidence candidate from winning purely on
        # cheapness; weight 300.0 scales the ~1-2cm grasp_depth_penalty to compete without overriding path cost.
        total_cost = (
            1.0 * path_cost
            + 2.0 * breakdown["joint_quality"]
            + 5.0 * breakdown["reachability"]
            + 8.0 * breakdown["generator_score_penalty"]
            + 300.0 * breakdown["grasp_depth_penalty"]
        )

        return CandidateScore(
            object_name=cand.object_name,
            grasp=cand,
            plan=plan,
            feasible=True,
            total_cost=total_cost,
            breakdown=breakdown,
        )
