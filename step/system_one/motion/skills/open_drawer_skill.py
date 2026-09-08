"""Planning-only ``OpenDrawerSkill``: grasp a drawer/cabinet handle and plan a
pull along its own slide-joint axis.

Deliberately *not* built on ``GraspCandidate``/``PickSkill``'s OR-object-set
machinery: a fixture is not an interchangeable candidate object like "any cup
on the table" -- it has exactly one physically correct grasp axis (the
handle's long axis) and exactly one physically correct motion after grasping
(a straight pull along the joint's own travel direction, not a free 6-DoF
carry). Forcing it through ``PickSkill`` would mean either faking a
``GraspCandidate.approach_direction_world``/OR-set for a set of one, or
teaching ``PickSkill`` about joint-constrained post-grasp motion it has no
other use for -- both worse than a small, purpose-built skill.

Grasp orientation convention matches ``grasp_generator.py``: local +Z is the
gripper's approach axis, local +X is the closing axis (the direction the
fingers move together). For a top-down grasp that's fixed by
``_TOP_DOWN_QUAT_WXYZ``; here the approach is horizontal (along the measured
pull axis) instead, so the frame is built directly from two measured
directions -- the pull axis (approach) and the handle's long axis (what the
fingers must straddle, so it must map to the *non*-closing axis).
"""

from __future__ import annotations

import numpy as np

from ..articulation import (
    DrawerHandle,
    estimate_handle_closing_width_m,
    locate_handle_position_world,
    measure_pull_axis_and_travel,
    resolve_drawer,
)
from ..collision import CollisionChecker, CollisionPhase
from ..config import MotionConfig
from ..kinematics import IKSolver
from ..ompl_planner import MotionPlanner
from ..panda import FINGER_JOINT_NAMES
from ..scene_query import SceneQuery, rotmat_to_quat_wxyz
from ..types import IKResult, PlannerFailureReason, SkillFailureReason, SkillResult


def _solve_ik_robust(
    ik_solver: IKSolver,
    scene: SceneQuery,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    seed_q: np.ndarray,
    rng: np.random.Generator,
    n_restarts: int = 8,
    perturb_scale: float = 0.6,
) -> IKResult:
    """Retry a failed DLS solve from randomly perturbed seeds before giving
    up. The horizontal handle approach this skill needs sits in a part of
    the arm's configuration space the single-seed solver isn't reliably
    well-conditioned for -- measured directly: solving the exact same grasp
    target from the exact same ``start_q`` succeeded on one env reset and
    hit a joint-limit-violation local minimum on another (LIBERO resets are
    not perfectly reproducible run-to-run), even though a feasible solution
    demonstrably exists nearby. A handful of perturbed restarts is the
    standard fix for a local optimizer landing in a bad basin, and mirrors
    this codebase's existing "try a few, take the first/best feasible"
    pattern (see ``pull_fraction_fallbacks``, yaw sampling in
    ``grasp_generator.py``) rather than trusting one arbitrary seed."""
    result = ik_solver.solve(scene.clone(), target_pos, target_quat_wxyz, seed_q=seed_q)
    if result.success:
        return result
    lo, hi = scene.joint_qpos_range(ik_solver.joint_names)
    for _ in range(n_restarts):
        perturbed = np.clip(seed_q + rng.uniform(-perturb_scale, perturb_scale, size=seed_q.shape), lo, hi)
        result = ik_solver.solve(scene.clone(), target_pos, target_quat_wxyz, seed_q=perturbed)
        if result.success:
            return result
    return result


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def _orientation_from_approach_and_reference(
    approach_world: np.ndarray, reference_axis_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Hand-frame quaternion (WXYZ) whose local +Z is ``approach_world`` and
    whose local +X (closing axis) is perpendicular to both ``approach_world``
    and ``reference_axis_world`` -- i.e. the fingers close across the handle's
    long axis rather than along it. Also returns that closing axis (world
    frame) directly, since callers need it to measure how wide the handle
    actually is along the direction the fingers close over (see
    ``articulation.estimate_handle_closing_width_m``)."""
    z_axis = _unit(np.asarray(approach_world, dtype=np.float64))
    x_axis = _unit(np.cross(np.asarray(reference_axis_world, dtype=np.float64), z_axis))
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return rotmat_to_quat_wxyz(R), x_axis


def handle_grasp_pose(
    scene: SceneQuery, drawer: DrawerHandle
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Returns (grasp_position, grasp_quat_wxyz, pull_axis_world, travel_m, closing_axis_world)."""
    pull_axis_world, travel_m, _, _ = measure_pull_axis_and_travel(scene, drawer)
    handle_pos = locate_handle_position_world(scene, drawer, pull_axis_world)
    reference_axis_world = _unit(np.cross(np.array([0.0, 0.0, 1.0]), pull_axis_world))
    approach_world = -pull_axis_world  # gripper travels from the open side inward to the handle
    grasp_quat, closing_axis_world = _orientation_from_approach_and_reference(approach_world, reference_axis_world)
    return handle_pos, grasp_quat, pull_axis_world, travel_m, closing_axis_world


class OpenDrawerSkill:
    def __init__(
        self,
        ik_solver: IKSolver,
        planner: MotionPlanner,
        collision_checker: CollisionChecker,
        config: MotionConfig | None = None,
    ) -> None:
        self.ik_solver = ik_solver
        self.planner = planner
        self.collision_checker = collision_checker
        self.config = config or MotionConfig()
        self._rng = np.random.default_rng(self.config.planner.seed)

    def execute(
        self,
        scene: SceneQuery,
        drawer_ref: str,
        start_q: np.ndarray,
        constraints: dict | None = None,
    ) -> SkillResult:
        constraints = constraints or {}
        try:
            drawer = resolve_drawer(scene, drawer_ref)
        except KeyError as exc:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": str(exc)},
            )

        grasp_pos, grasp_quat, pull_axis_world, travel_m, closing_axis_world = handle_grasp_pose(scene, drawer)
        cfg = self.config.drawer
        standoff_pos = grasp_pos + cfg.approach_standoff_m * pull_axis_world
        edge_resolution = self.config.planner.validity_resolution * float(
            np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)
        )

        # How far each finger should actually travel to close on the handle
        # -- NOT the joint's own mechanical closing limit. At qpos=[0,0] the
        # two fingers meet at the gripper's centerline (zero gap between
        # them), which only makes sense for "nothing between the fingers";
        # commanding that against a handle of nonzero width drives the rigid
        # finger segment straight through it. Each finger travels
        # symmetrically from center, so half the handle's real width (along
        # the direction the fingers close over) is the right qpos magnitude
        # for each joint -- confirmed against this exact gripper: gap(q) grows
        # at ~2*q, matching the independently-known `gripper_max_opening_m
        # = 0.08` at q = 0.04 (this gripper's joint limit).
        handle_width_m = estimate_handle_closing_width_m(scene, drawer, pull_axis_world, closing_axis_world)
        finger_half_travel = float(np.clip(handle_width_m / 2.0, 0.0, 0.04))
        finger_closed_qpos = np.array([finger_half_travel, -finger_half_travel])

        # IK is not unique -- a DLS solve only ever returns *one* of possibly
        # many arm configurations that reach the same Cartesian grasp pose
        # (elbow-up vs elbow-down, wrist flipped by pi about the approach
        # axis, etc.), and different configurations sweep through different
        # parts of the scene on the way there. Measured directly: the first
        # IK solution found for a real cluttered scene (a bowl sitting on the
        # table between the robot and the cabinet) put the forearm
        # (robot0_link5) through the bowl for the *entire* standoff->grasp
        # segment, including the grasp pose itself -- not a bad
        # interpolation, a genuinely bad configuration choice. So: retry with
        # fresh perturbed seeds (same "try a few, take the first feasible"
        # pattern as ``_solve_ik_robust``/``pull_fraction_fallbacks``), and
        # -- unlike a plain first-feasible-wins loop -- score every feasible
        # pair by joint-space distance from ``start_q`` and keep the closest
        # one (mirrors PickSkill's own ``joint_quality`` cost term). This
        # matters beyond just avoiding collisions: a perturbed seed can land
        # on a collision-clear but *distant* elbow/wrist configuration that
        # the reactive MPC tracker converges to only slowly or not at all
        # within its step budget (measured directly: a random-seed solution
        # requiring a much larger joint-space reach than the closest
        # available one left the arm still ~18cm from the grasp pose after
        # the full settle budget, while the closest feasible configuration
        # converged to within ~1cm) -- so among collision-clear options, the
        # one nearest home is also the one most likely to actually be
        # reached in practice, not just reachable in principle.
        best_grasp_ik = best_standoff_ik = None
        best_distance = float("inf")
        last_rejection = "no_ik_attempted"
        for attempt in range(cfg.pose_selection_attempts):
            seed = start_q if attempt == 0 else np.clip(
                start_q + self._rng.uniform(-0.8, 0.8, size=start_q.shape),
                *scene.joint_qpos_range(self.ik_solver.joint_names),
            )
            candidate_grasp_ik = self.ik_solver.solve(scene.clone(), grasp_pos, grasp_quat, seed_q=seed)
            if not candidate_grasp_ik.success:
                last_rejection = "grasp_ik_failed"
                continue
            distance = float(np.linalg.norm(candidate_grasp_ik.q - start_q))
            if distance >= best_distance:
                continue  # already have a strictly better (closer) feasible candidate
            candidate_standoff_ik = _solve_ik_robust(
                self.ik_solver, scene, standoff_pos, grasp_quat, candidate_grasp_ik.q, self._rng
            )
            if not candidate_standoff_ik.success:
                last_rejection = "approach_standoff_ik_failed"
                continue
            descent_check = self.collision_checker.check_edge(
                candidate_standoff_ik.q,
                candidate_grasp_ik.q,
                resolution=edge_resolution,
                phase=CollisionPhase.TERMINAL_APPROACH,
                target_object_body=drawer.drawer_body,
            )
            if descent_check.in_collision:
                last_rejection = "final_descent_in_collision"
                continue
            # Neither the IK solve above nor `descent_check` ever move the
            # finger joints -- they simulate the *open* gripper reaching this
            # pose, not what happens when it actually closes. A configuration
            # whose open-finger geometry clears the handle can still have its
            # rigid finger segment (which reaches further than the pad) catch
            # a neighboring part of the handle assembly (a mounting stud,
            # say) as the fingers close. Validate that transition explicitly
            # before accepting the candidate, rather than only ever finding
            # out from the real gripper closing on a bad pose during
            # execution.
            closure_check = self.collision_checker.check_gripper_closure(
                candidate_grasp_ik.q, FINGER_JOINT_NAMES, finger_closed_qpos, drawer.drawer_body
            )
            if closure_check.in_collision:
                last_rejection = "gripper_closure_in_collision"
                continue
            best_grasp_ik, best_standoff_ik, best_distance = candidate_grasp_ik, candidate_standoff_ik, distance

        grasp_ik, standoff_ik = best_grasp_ik, best_standoff_ik
        if grasp_ik is None or standoff_ik is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": last_rejection,
                    "drawer": drawer.drawer_body,
                    "pose_selection_attempts": cfg.pose_selection_attempts,
                },
            )

        pulled_q = None
        pull_fraction_used = None
        for fraction in cfg.pull_fraction_fallbacks:
            pulled_pos = grasp_pos + float(fraction) * travel_m * pull_axis_world
            pull_ik = _solve_ik_robust(self.ik_solver, scene, pulled_pos, grasp_quat, grasp_ik.q, self._rng)
            if not pull_ik.success:
                continue
            pull_check = self.collision_checker.check_edge(
                grasp_ik.q,
                pull_ik.q,
                resolution=self.config.planner.validity_resolution
                * float(np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)),
                phase=CollisionPhase.TERMINAL_APPROACH,
                target_object_body=drawer.drawer_body,
            )
            if pull_check.in_collision:
                continue
            pulled_q = pull_ik.q
            pull_fraction_used = float(fraction)
            break

        if pulled_q is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": "no_feasible_pull_target",
                    "drawer": drawer.drawer_body,
                    "tried_fractions": list(cfg.pull_fraction_fallbacks),
                },
            )

        # RRTConnect is stochastic: a start/goal pair with a real feasible
        # path can still occasionally come back as only an "approximate"
        # solution (rejected, see ompl_planner.py's own documented policy on
        # that) or hit its time budget without connecting, purely from an
        # unlucky tree expansion -- not a structural infeasibility (retrying
        # the identical START_IN_COLLISION/GOAL_IN_COLLISION failures
        # wouldn't help, since those are deterministic given a fixed q, so
        # only the stochastic-outcome failure modes are retried).
        plan = None
        for _ in range(cfg.planning_attempts):
            plan = self.planner.plan(
                start_q, [standoff_ik.q], phase=CollisionPhase.FREE_SPACE, target_object_body=drawer.drawer_body
            )
            if plan.success or plan.failure_reason not in (
                PlannerFailureReason.NO_PATH_FOUND,
                PlannerFailureReason.APPROXIMATE_SOLUTION_ONLY,
                PlannerFailureReason.TIMEOUT,
            ):
                break
        if not plan.success:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": f"planning_failed:{plan.failure_reason}",
                    "drawer": drawer.drawer_body,
                },
            )

        # Only the descent (standoff -> grasp) goes into `approach_positions`
        # -- matching PickSkill's exact convention, where the adapter tracks
        # this tail under TERMINAL_APPROACH, settles onto the grasp pose,
        # *then* closes the gripper. The pull leg (`grasp_q` -> `pulled_q`)
        # is deliberately kept separate (tracked by the adapter only *after*
        # the gripper has actually closed on the handle) -- an earlier
        # version concatenated both legs into one pre-grasp reference, which
        # meant the arm drove all the way through the pull motion with the
        # gripper still open before ever closing it, never actually grasping
        # the handle (caught by a real run: the drawer's qpos barely moved
        # and the post-hoc settle-tracking error was ~20x larger than a
        # normal pick's, because the reference asked the gripper-open arm to
        # reach a joint target that assumed it was already rigidly attached
        # to the handle).
        descent_positions = np.linspace(standoff_ik.q, grasp_ik.q, 15)
        plan.metadata["standoff_q"] = standoff_ik.q.tolist()
        plan.metadata["grasp_q"] = grasp_ik.q.tolist()
        plan.metadata["pulled_q"] = pulled_q.tolist()
        plan.metadata["approach_positions"] = descent_positions.tolist()

        return SkillResult(
            success=True,
            selected_object=drawer.drawer_body,
            planned_trajectory=plan,
            diagnostics={
                "drawer_body": drawer.drawer_body,
                "joint_name": drawer.joint_name,
                "pull_axis_world": pull_axis_world.tolist(),
                "travel_m": travel_m,
                "pull_fraction_used": pull_fraction_used,
                "grasp_position": grasp_pos.tolist(),
                "grasp_quat_wxyz": grasp_quat.tolist(),
                "handle_width_m": handle_width_m,
                "finger_closed_qpos": finger_closed_qpos.tolist(),
            },
        )
