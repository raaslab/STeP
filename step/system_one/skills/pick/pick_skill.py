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
executed without that check. The long-range leg is kept contact-free
specifically so RRTConnect can't bump/shove the object while maneuvering;
only the short final descent is allowed to touch it, and only after that
segment is explicitly validated. Execution/tracking happens by the caller
feeding the returned ``MotionPlan`` to ``JointSpaceMPC``, since ``PickSkill``
itself is planning-only and backend-agnostic; the caller closes the gripper,
verifies the grasp, and registers the object as attached via
``AttachedObjectManager``.

OR semantics ("pick any cup"): every candidate object in ``object_set`` is
fully evaluated (grasp generation -> IK -> planning -> cost) *before* any
commitment is made. The lowest-cost *feasible* (object, grasp, trajectory)
tuple wins; a candidate is never selected on generator score alone.
"""

from __future__ import annotations

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

        # A pick often runs right after a place, which can leave the arm in a
        # collision-adjacent resting pose that OMPL refuses to plan from. No
        # single attached object/goal exists yet here, so only the "straight
        # up" retreat is tried, not Place/Transport's "toward goal" option.
        start_q = escape_start_collision(self.ik_solver, self.collision_checker, scene, None, None, start_q)

        # Callers that already generated candidates for their own purposes
        # (e.g. LiberoPickAdapter saving a grasp-pose visualization before
        # executing) can pass them straight through, avoiding a second,
        # possibly expensive regenerate() call (a learned generator re-runs
        # real inference on every call).
        if candidates is None:
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
        # conservative overestimate for round/irregular objects, so a small
        # tolerance avoids hard-rejecting candidates whose real cross-section is
        # a bit smaller than their bounding box. The tolerance is capped at the
        # gripper's own physical opening, not applied past it: letting it admit
        # widths the hand can never straddle just means the fingers bottom out,
        # clip the object, and shove it -- reading as a successful pick that
        # fails downstream. A candidate wider than the hand has nothing to fall
        # back on, unlike an over-estimated round object's narrower yaws.
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

        # --- IK: approach standoff first, then the grasp pose seeded from it -----------------
        # No separate pregrasp *stage* is exposed to callers (still one
        # PickSkill.execute() call), but internally planning goes through a
        # standoff point close to (not at) the object, exactly like PlaceSkill's
        # hover point -- this keeps the long-range approach under strict
        # FREE_SPACE (no target contact permitted), with only the short final
        # descent transitioning to TERMINAL_APPROACH and explicitly
        # collision-validated via check_edge before being trusted for
        # execution. Without this split, RRTConnect is free to bump/shove the
        # object anywhere along a long approach since contact is never
        # penalized.
        #
        # Tries the plain straight-line standoff first (directly opposite the
        # approach direction -- straight up, for a top-down grasp), then
        # increasingly wide rings of sideways-offset alternatives if that one
        # is unreachable or blocked -- a fixed vertical standoff assumes the
        # airspace above the object is always clear, which an earlier subtask
        # (e.g. an opened drawer) can violate. See standoff_search.standoff_candidates.
        #
        # Each finger travels symmetrically from center (qpos=[0,0] is zero
        # gap, not "closed on an object" -- see check_gripper_closure's
        # docstring), so the right closed-finger target is half the object's
        # own real width at this grasp, falling back to the joint's mechanical
        # limit only if the generator couldn't estimate a width.
        if cand.gripper_width is not None:
            half = float(np.clip(cand.gripper_width / 2.0, 0.0, 0.04))
            finger_closed_qpos = np.array([half, -half])
        else:
            finger_closed_qpos = FINGER_CLOSED_QPOS
        descent_resolution = self.config.planner.validity_resolution * float(
            np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)
        )

        # Every standoff candidate seeds a *different* grasp_ik solve (DLS IK is
        # non-unique -- a different standoff direction can converge to a
        # different elbow/wrist configuration for the same Cartesian grasp
        # target), so grasp/descent/closure validation is tried per-standoff-
        # candidate, not just once against whichever standoff was found first:
        # a reachable standoff can still sweep the forearm into a collision at
        # the grasp pose itself.
        standoff_ik = grasp_ik = above_ik = None
        last_rejection = "approach_standoff_ik_failed"
        for standoff_position in standoff_candidates(
            cand.grasp_position,
            cand.approach_direction_world,
            self.config.grasp.approach_standoff_m,
            self.config.grasp.standoff_distance_multipliers,
            self.config.grasp.standoff_lateral_offsets_m,
            self.config.grasp.standoff_lateral_directions,
        ):
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

            # Terminal grasp contact with the target object is expected/permitted.
            grasp_check = self.collision_checker.check(
                candidate_grasp_ik.q, phase=CollisionPhase.TERMINAL_APPROACH, target_object_body=cand.object_name
            )
            if grasp_check.in_collision:
                last_rejection = "grasp_pose_in_collision"
                continue

            # --- above-object waypoint: force the final descent to be a straight drop ---
            # Without this, a standoff needing a lateral offset to dodge some
            # obstacle produces a descent that moves down and sideways at
            # once -- and since this segment runs under TERMINAL_APPROACH
            # (which deliberately permits target contact), the arm clipping
            # the object's side while angling in wouldn't read as a
            # collision at all. Fixed by construction: always pass through a
            # point directly above the grasp (same XY, one standoff distance
            # back along the approach direction) before permitting target
            # contact, so the segment allowed to touch the object is always a
            # pure vertical drop.
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

            # --- final descent segment: explicitly validated, never blindly executed ---
            # Split in two: standoff -> above still forbids target contact
            # (free maneuvering); above -> grasp is the one segment allowed to
            # touch the object, always a straight vertical line by
            # construction (both endpoints share the grasp's XY).
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

            # --- gripper closure: never simulated by IK/OMPL/the MPC, which only move ---
            # the 7 arm joints -- a pose whose open-finger geometry clears the object can
            # still have its rigid finger segment catch something as the fingers close.
            # See check_gripper_closure's docstring.
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
        # A single call, not a retry loop: OMPL seeds its RNG once per process
        # and refuses reseeding (see MotionPlanner.plan()'s own docstring), so
        # a failed RRTConnect search just returns the same result every time.
        # Retrying only helps when the goal itself changes between attempts
        # (see TransportSkill's nearest-configuration hover-IK selection) --
        # not applicable here, since the standoff search above already
        # committed to one fixed goal.
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
        # TERMINAL_APPROACH). Two legs, not one: standoff -> above (still no
        # target contact permitted), then above -> grasp (the one segment
        # allowed to touch the object, always a straight vertical drop by
        # construction).
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
        # Where approach_positions splits into its two legs (index of above_q,
        # shared by both): the executor needs this to track them as two
        # separate, sequential MPC phases with a real convergence check in
        # between -- a single continuous reference lets the MPC's lookahead
        # horizon blend across the standoff->above/above->grasp boundary, so
        # the arm can still be correcting XY while Z is already dropping
        # toward the object.
        plan.metadata["to_above_count"] = to_above_count

        path_cost = 0.0
        if plan.geometric_path is not None and len(plan.geometric_path) > 1:
            path_cost = float(np.sum(np.linalg.norm(np.diff(plan.geometric_path, axis=0), axis=1)))
        breakdown["path_length"] = path_cost
        breakdown["reachability"] = float(standoff_ik.position_error + standoff_ik.orientation_error)
        breakdown["generator_score_penalty"] = float(1.0 - cand.generator_score)

        # --- depth preference: bias top-down grasps toward the deepest safe height ----------
        # Both generators already clamp a candidate's height to the
        # palm/floor-clearance-safe band (see grasp_generator.py's and
        # contact_graspnet_generator.py's own min_safe_z clamp), but that
        # clamp is a floor, not an attractor -- a shallow, near-the-rim
        # candidate that never hits it passes through untouched, and a
        # learned generator's own confidence score has essentially no
        # correlation with depth. This term breaks that tie toward the deep
        # end, applied uniformly to every generator's candidates. Gated to
        # near-vertical approaches only: a side/rim-pinch candidate's z is set
        # by object geometry, not palm depth, so this formula doesn't apply.
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

        # Weight 8.0, not 1.0: generator_score_penalty is bounded to [0, 1], but
        # path_cost/joint_quality routinely land in the 2-12+ range for a real
        # trajectory -- at weight 1.0 a low-confidence candidate could win
        # purely for being cheap to reach despite missing the object entirely.
        # Weight 300.0: grasp_depth_penalty is a raw distance in meters, and the
        # real gap between a shallow and deep candidate is small (~1-2cm) --
        # this weight puts that gap on the same order as generator_score_
        # penalty's own max contribution, enough to flip a close ranking
        # without letting depth override a large path-cost difference.
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
