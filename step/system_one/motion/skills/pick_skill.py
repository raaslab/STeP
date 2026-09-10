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

from ..collision import CollisionChecker, CollisionPhase
from ..config import MotionConfig
from .grasp_generator import GraspGenerator
from ..kinematics import IKSolver
from ..ompl_planner import MotionPlanner
from ..panda import FINGER_CLOSED_QPOS, FINGER_JOINT_NAMES
from .place_skill import _solve_ik_robust, escape_start_collision
from ..scene_query import SceneQuery
from .standoff_search import standoff_candidates
from ..types import (
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

        # A pick is often the very next skill called right after a place
        # released its object -- the arm's resting spot at that instant is
        # wherever the release left it, not necessarily lifted clear (see
        # escape_start_collision's own docstring). Confirmed directly via a
        # real rollout: this start_q registered START_IN_COLLISION for every
        # single pick candidate, since OMPL (correctly) refuses to plan from
        # an invalid start regardless of how good the candidate goal is.
        # PickSkill has no single "attached object" or "goal" the way
        # Place/Transport do (nothing is held yet, and there may be several
        # candidate objects), so this only tries the "straight up" retreat,
        # not the "toward a goal's center" one those two also get.
        start_q = escape_start_collision(self.ik_solver, self.collision_checker, scene, None, None, start_q)

        # Callers that already generated candidates for their own purposes
        # (e.g. LiberoPickAdapter saving a grasp-pose visualization before
        # executing) can pass them straight through -- avoids a second,
        # possibly expensive/non-deterministic regenerate() call (a learned
        # generator like ContactGraspNetGenerator re-runs real CUDA
        # inference on every call).
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
                    # Enough per-candidate context to diagnose *why* every
                    # candidate failed straight from execution_summary.json,
                    # without needing to re-run with ad hoc instrumentation:
                    # which object/strategy/yaw was tried, how wide the
                    # generator thought the grasp was, and where it aimed.
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
        # conservative *overestimate* for round/irregular objects -- it's a bounding
        # box, not the object's true cross-section at the grasp height. A small
        # tolerance avoids hard-rejecting genuinely graspable objects whose real
        # cross-section is a bit smaller than their bounding box; grossly
        # incompatible widths are still rejected.
        #
        # The tolerance is deliberately *not* applied past the gripper's own
        # physical opening. Slack above that admits candidates the hand can
        # never straddle no matter how good the estimate was: at 1.15 this
        # accepted widths up to 9.2cm on an 8cm gripper, which is how
        # cream_cheese_1 (8.3 x 4.4cm) got grasped across its long axis --
        # the fingers bottom out at their limit, clip the box and shove it,
        # which reads as a successful pick and then fails downstream. An
        # over-estimated round object still has its narrower yaw candidates
        # to fall back on; a candidate wider than the hand has nothing.
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
        # Try the plain straight-line standoff first (directly opposite the
        # approach direction -- straight up, for a top-down grasp), then
        # increasingly wide rings of sideways-offset alternatives if that one
        # is unreachable or blocked. A single fixed vertical standoff assumes
        # the airspace directly above the object is always clear, which
        # isn't true once an earlier subtask has moved something else into
        # it -- measured directly: after opening a drawer that ends up
        # hovering above a bowl, every rim-edge candidate at every angle
        # failed identically, because they all shared that same now-blocked
        # straight-up standoff direction. See standoff_search.standoff_candidates.
        # Each finger travels symmetrically from center (qpos=[0,0] is *zero*
        # gap, not "closed on an object" -- see check_gripper_closure's
        # docstring), so the right closed-finger target is half the object's
        # own real width at this grasp, not the joint's mechanical limit;
        # falls back to that limit only if the generator couldn't estimate a
        # width for this candidate.
        if cand.gripper_width is not None:
            half = float(np.clip(cand.gripper_width / 2.0, 0.0, 0.04))
            finger_closed_qpos = np.array([half, -half])
        else:
            finger_closed_qpos = FINGER_CLOSED_QPOS
        descent_resolution = self.config.planner.validity_resolution * float(
            np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)
        )

        # Every standoff candidate seeds a *different* grasp_ik solve (DLS IK
        # is non-unique -- a different standoff direction can converge to a
        # different elbow/wrist configuration even for the exact same
        # Cartesian grasp target), so grasp/descent/closure validation is
        # tried per-standoff-candidate too, not just once against whichever
        # standoff happened to be found first. Measured directly: widening
        # the standoff search alone (approaching a rim point from the side
        # instead of straight down, to route around a drawer hovering
        # overhead) got *past* the standoff check more often, but then
        # revealed that *that* elbow configuration collided at the grasp
        # pose itself -- a different standoff direction sweeps the forearm
        # through a different part of a cluttered scene even when aiming at
        # the identical final target.
        standoff_ik = grasp_ik = None
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

            # --- final descent segment: explicitly validated, never blindly executed ---
            descent_check = self.collision_checker.check_edge(
                candidate_standoff_ik.q,
                candidate_grasp_ik.q,
                resolution=descent_resolution,
                phase=CollisionPhase.TERMINAL_APPROACH,
                target_object_body=cand.object_name,
            )
            if descent_check.in_collision:
                last_rejection = "final_descent_in_collision"
                continue

            # --- gripper closure: never simulated by IK/OMPL/the MPC, which only ever move ---
            # the 7 arm joints -- a pose whose *open*-finger geometry clears the object can
            # still have its rigid finger segment (reaches further than the pad) catch
            # something as the fingers close from there. See check_gripper_closure's docstring.
            closure_check = self.collision_checker.check_gripper_closure(
                candidate_grasp_ik.q, FINGER_JOINT_NAMES, finger_closed_qpos, cand.object_name
            )
            if closure_check.in_collision:
                last_rejection = "gripper_closure_in_collision"
                continue

            standoff_ik, grasp_ik = candidate_standoff_ik, candidate_grasp_ik
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
        # A single call, not a "planning_attempts times" retry loop: this is
        # one fixed (start_q, standoff_ik.q) pair, and MotionPlanner.plan()'s
        # own docstring documents why retrying it is pure waste -- OMPL seeds
        # its RNG once per process and refuses reseeding, so a failed
        # RRTConnect search returns the identical result every time (measured
        # there: 8 back-to-back retries, byte-identical). An earlier version
        # of this method retried anyway, which meant any candidate whose
        # planning genuinely failed (NO_PATH_FOUND/TIMEOUT) burned up to
        # planning_attempts * timeout_sec seconds for zero benefit --
        # confirmed against a real cream_cheese_1 rollout: two separate
        # fallback stages each spent ~100s here alone (6 retries x ~17s
        # average before timing out), the overwhelming majority of a 223s
        # total pick. Retrying only ever helps when the goal itself changes
        # between attempts (see TransportSkill's nearest-configuration
        # hover-IK selection for that pattern) -- not applicable here, since
        # the standoff search above already committed to one fixed goal.
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

        # --- depth preference: bias top-down grasps toward the deepest safe height ----------
        # Both generators already clamp a candidate's height to never go
        # *past* the palm/floor-clearance-safe band (see grasp_generator.py's
        # and contact_graspnet_generator.py's own min_safe_z clamp, which
        # shares this exact formula) -- but that clamp is a floor, not an
        # attractor: a shallow, near-the-rim candidate that never hits the
        # clamp passes through untouched. Confirmed via a real rollout
        # (alphabet_soup_1_main, a can): ContactGraspNet's 32 candidates spanned
        # the full safe band (79%-105% up the can) with essentially zero
        # correlation between height and the network's own confidence score
        # (r=-0.05) -- the network has no notion of "grip margin", so its
        # top-scoring candidate landed at 101% up (right at the rim, a shallow
        # pinch prone to slipping) purely by chance, even though a
        # near-identical-score deep candidate at 79% up (matching a real,
        # reliable can grasp) was sitting right next to it in the pool. This
        # term breaks that tie toward the deep end without touching either
        # generator, so it applies uniformly to CGN and heuristic candidates
        # alike. Gated to near-vertical approaches only: the "grasp near
        # object_top_z minus palm clearance" reasoning only describes a
        # top-down grasp -- a side/rim-pinch candidate's z is set by object
        # geometry, not palm depth, and penalizing it here would just add
        # noise to a strategy this formula doesn't describe.
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

        # Weight 8.0, not 1.0: generator_score_penalty is bounded to [0, 1],
        # but path_cost/joint_quality routinely land in the 2-12+ range for a
        # real arm trajectory -- at weight 1.0 the network's own confidence
        # was almost noise next to how far the arm has to travel, letting a
        # candidate the network itself flagged as low-confidence (a real
        # case: network_score=0.19) win purely for being cheap to reach.
        # Confirmed via a real rollout ("pick up the milk"): that exact
        # candidate's grasp_position was ~3.5cm laterally and ~6cm
        # vertically off the carton's real body -- the gripper closed fully
        # shut on nothing (finger pads touching each other directly, zero
        # contact with the object) despite tracking to that target with
        # near-zero error. It had 15 other feasible candidates to choose
        # from; this scoring is why it picked the one that missed.
        # Weight 300.0: grasp_depth_penalty is a raw distance in meters, and
        # the real-world gap between a shallow rim candidate and the deep
        # target is small in absolute terms (~1-2cm, confirmed against the
        # alphabet-soup-can pool above) -- at weight 1.0 that's a rounding
        # error next to path_cost/reachability. 300.0 puts a 2cm gap
        # (0.02 * 300 = 6.0) on the same order as generator_score_penalty's
        # own max contribution (8.0), enough to flip the ranking between two
        # otherwise-similar candidates without letting depth alone override a
        # large path-cost or reachability difference.
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
