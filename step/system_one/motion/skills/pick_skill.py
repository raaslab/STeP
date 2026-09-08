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
from ..scene_query import SceneQuery
from ..types import (
    CandidateScore,
    GraspCandidate,
    SkillFailureReason,
    SkillResult,
)


def _standoff_candidates(
    grasp_position: np.ndarray,
    approach_direction_world: np.ndarray,
    standoff_m: float,
    standoff_distance_multipliers: tuple[float, ...],
    lateral_offsets_m: tuple[float, ...],
    n_directions: int,
) -> list[np.ndarray]:
    """Candidate standoff positions to try, in order: the plain straight-line
    standoff first (directly opposite the approach direction, e.g. straight
    up for a top-down grasp, at the nominal distance), then increasingly
    wide rings of positions offset sideways in the plane perpendicular to
    the approach, *and* at increasingly larger standoff distances -- see
    ``GraspSampleConfig.standoff_lateral_offsets_m``'s docstring for why a
    single fixed vertical standoff isn't always enough (something else can
    move into that airspace as a side effect of an earlier subtask).

    Distance matters independently of lateral direction: an obstacle
    hovering over the object can block every horizontal direction at the
    *nominal* standoff height simply because reaching that height at all
    requires sweeping an arm link (elbow/forearm, not just the gripper)
    through the obstacle's footprint -- measured directly against a real
    drawer-overhang scene, widening the lateral search alone left most
    angles still blocked at the standoff stage. A taller standoff (further
    back along the approach direction -- higher above the object, for a
    top-down grasp) can clear the obstacle's own height entirely, letting
    the final descent come down past it instead of trying to detour around
    it at the same blocked altitude.
    """
    approach = approach_direction_world / np.linalg.norm(approach_direction_world)

    # Any orthonormal basis for the plane perpendicular to `approach`.
    ref = np.array([0.0, 0.0, 1.0]) if abs(approach[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    lateral1 = np.cross(approach, ref)
    lateral1 /= np.linalg.norm(lateral1)
    lateral2 = np.cross(approach, lateral1)

    angles = np.linspace(0.0, 2.0 * np.pi, max(1, n_directions), endpoint=False)
    candidates: list[np.ndarray] = []
    for multiplier in standoff_distance_multipliers:
        straight = grasp_position - (standoff_m * multiplier) * approach
        candidates.append(straight)
        for radius in lateral_offsets_m:
            for angle in angles:
                offset = radius * (np.cos(angle) * lateral1 + np.sin(angle) * lateral2)
                candidates.append(straight + offset)
    return candidates


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
        candidates: list[GraspCandidate] | None = None,
    ) -> SkillResult:
        constraints = constraints or {}
        downstream_goal_xy = constraints.get("downstream_goal_xy")  # optional (x, y) reachability hint

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
        # Try the plain straight-line standoff first (directly opposite the
        # approach direction -- straight up, for a top-down grasp), then
        # increasingly wide rings of sideways-offset alternatives if that one
        # is unreachable or blocked. A single fixed vertical standoff assumes
        # the airspace directly above the object is always clear, which
        # isn't true once an earlier subtask has moved something else into
        # it -- measured directly: after opening a drawer that ends up
        # hovering above a bowl, every rim-edge candidate at every angle
        # failed identically, because they all shared that same now-blocked
        # straight-up standoff direction. See _standoff_candidates.
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
        for standoff_position in _standoff_candidates(
            cand.grasp_position,
            cand.approach_direction_world,
            self.config.grasp.approach_standoff_m,
            self.config.grasp.standoff_distance_multipliers,
            self.config.grasp.standoff_lateral_offsets_m,
            self.config.grasp.standoff_lateral_directions,
        ):
            candidate_standoff_ik = self.ik_solver.solve(
                scene.clone(), standoff_position, cand.grasp_quat_wxyz, seed_q=start_q
            )
            if not candidate_standoff_ik.success:
                continue
            standoff_check = self.collision_checker.check(
                candidate_standoff_ik.q, phase=CollisionPhase.FREE_SPACE, target_object_body=cand.object_name
            )
            if standoff_check.in_collision:
                last_rejection = "approach_standoff_in_collision"
                continue

            candidate_grasp_ik = self.ik_solver.solve(
                scene.clone(), cand.grasp_position, cand.grasp_quat_wxyz, seed_q=candidate_standoff_ik.q
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
        total_cost = (
            1.0 * path_cost
            + 2.0 * breakdown["joint_quality"]
            + 5.0 * breakdown["reachability"]
            + 8.0 * breakdown["generator_score_penalty"]
        )

        return CandidateScore(
            object_name=cand.object_name,
            grasp=cand,
            plan=plan,
            feasible=True,
            total_cost=total_cost,
            breakdown=breakdown,
        )
