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

from dataclasses import dataclass

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
from .standoff_search import standoff_candidates
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
    ``grasp_generator.py``) rather than trusting one arbitrary seed.

    Does NOT return on the first successful restart: a full-range random
    seed (needed below for feasibility -- see the loop's own comment) can
    just as easily land on a distant, unrelated elbow/wrist branch as a
    nearby one, and accepting whichever happens to succeed first sweeps the
    arm through that branch's own huge Cartesian detour to get there --
    measured directly on the toy (clutter-free) drawer-only scene: a 0.27m
    straight-line descent taking a 2.69m actual path (path_efficiency
    0.10). Mirrors TransportSkill's own hover-candidate search: spend the
    whole restart budget, then keep whichever successful solution is
    closest to `seed_q` in joint space -- the one requiring the least, and
    least dramatic, movement to reach -- rather than the first one found."""
    best_result = ik_solver.solve(scene.clone(), target_pos, target_quat_wxyz, seed_q=seed_q)
    best_distance = float(np.linalg.norm(best_result.q - seed_q)) if best_result.success else float("inf")
    lo, hi = scene.joint_qpos_range(ik_solver.joint_names)
    for attempt in range(n_restarts):
        # Perturbing around `seed_q` only searches that one basin, which is
        # useless when the answer isn't in it. That became the dominant
        # failure once this skill started homing the arm before planning:
        # home is a fine place to *start* the approach from but a poor seed
        # for the handle-grasp configuration, and every restart near it
        # failed (approach_standoff_ik_failed, 6/6). The same split fixed the
        # identical symptom in TransportSkill, where the target pose was
        # measured solvable 20/60 times from full-range seeds and 0/6 from
        # perturbations around the arm's current pose.
        if attempt < n_restarts // 2:
            seed = np.clip(seed_q + rng.uniform(-perturb_scale, perturb_scale, size=seed_q.shape), lo, hi)
        else:
            seed = rng.uniform(lo, hi)
        result = ik_solver.solve(scene.clone(), target_pos, target_quat_wxyz, seed_q=seed)
        if result.success:
            distance = float(np.linalg.norm(result.q - seed_q))
            if distance < best_distance:
                best_result, best_distance = result, distance
    return best_result


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def _roll_about(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of ``vec`` about ``axis`` -- used to roll the
    gripper about its own approach axis, which swings the fingers (and with
    them the palm and forearm) around the handle without changing where the
    hand is aimed."""
    a = _unit(np.asarray(axis, dtype=np.float64))
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return vec * c + np.cross(a, vec) * s + a * float(np.dot(a, vec)) * (1.0 - c)


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


@dataclass
class DrawerHandleGraspCandidate:
    """One accepted candidate from ``find_drawer_handle_grasp``: a reachable,
    collision-free, closure-safe standoff->grasp pair for this drawer's
    handle, plus the per-candidate finger/orientation parameters that varied
    with the tilt/roll/offset that won. Shared by ``OpenDrawerSkill`` (pulls
    +``pull_axis_world`` afterward) and ``CloseSkill``'s grasp-based drawer
    closing (retracts -``pull_axis_world`` instead) -- identical search,
    opposite post-grasp motion."""

    standoff_ik: IKResult
    grasp_ik: IKResult
    tilt_rad: float
    roll_rad: float
    grasp_pos: np.ndarray
    grasp_quat: np.ndarray
    closing_axis_world: np.ndarray
    handle_width_m: float
    finger_closed_qpos: np.ndarray
    approach_finger_qpos: np.ndarray


def find_drawer_handle_grasp(
    scene: SceneQuery,
    drawer: DrawerHandle,
    grasp_pos: np.ndarray,
    pull_axis_world: np.ndarray,
    start_q: np.ndarray,
    ik_solver: IKSolver,
    collision_checker: CollisionChecker,
    cfg: "DrawerConfig",
    edge_resolution: float,
    rng: np.random.Generator,
) -> tuple[DrawerHandleGraspCandidate | None, str]:
    """Search ``cfg``'s tilt x roll x offset x standoff grid (``DrawerConfig``,
    the same one ``OpenDrawerSkill`` tunes) for a candidate approach to this
    drawer's handle that is IK-reachable, has a collision-free descent, and
    survives the fingers actually closing on it. Accepts the first candidate
    that passes all three checks (not an exhaustive best-of-all-candidates
    search) -- extracted verbatim from ``OpenDrawerSkill.execute`` so both
    callers search identically; see the loop body's own comments for why
    standoff-first IK ordering, tilted approaches, and narrowed fingers all
    matter here.

    Returns ``(candidate, last_rejection)`` -- ``candidate`` is ``None`` on
    failure, with ``last_rejection`` naming the last rejection reason seen.
    """
    reference_axis_world = _unit(np.cross(np.array([0.0, 0.0, 1.0]), pull_axis_world))
    approach_plan: list[tuple] = []
    for tilt_rad, roll_rad, (depth_off, vert_off) in (
        (t, r, o)
        for t in cfg.approach_tilts_rad
        for r in cfg.approach_rolls_rad
        for o in cfg.grasp_offsets_m
    ):
        approach_world = _unit(
            -pull_axis_world * float(np.cos(tilt_rad))
            + np.array([0.0, 0.0, -1.0]) * float(np.sin(tilt_rad))
        )
        tilt_quat, tilt_closing_axis = _orientation_from_approach_and_reference(
            approach_world, _roll_about(reference_axis_world, approach_world, roll_rad)
        )
        tilt_width_m = estimate_handle_closing_width_m(
            scene, drawer, pull_axis_world, tilt_closing_axis
        )
        tilt_half_travel = float(np.clip(tilt_width_m / 2.0, 0.0, 0.04))
        tilt_finger_qpos = np.array([tilt_half_travel, -tilt_half_travel])
        # Offset the aim point off the bar's exact centre: `depth_off` runs
        # along the pull axis (negative = deeper, toward the drawer front),
        # `vert_off` straight up/down.
        tilt_grasp_pos = (
            grasp_pos + pull_axis_world * float(depth_off) + np.array([0.0, 0.0, float(vert_off)])
        )
        # Approach with the fingers already narrowed to just wider than the
        # handle, rather than fully open -- see OpenDrawerSkill's own module
        # docstring/comment history for why (measured on the middle drawer:
        # only 4.0cm of clear space above the handle, which an open gripper
        # cannot enter).
        tilt_approach_half = float(
            np.clip(tilt_width_m / 2.0 + cfg.approach_finger_clearance_m, 0.0, 0.04)
        )
        tilt_approach_finger_qpos = np.array([tilt_approach_half, -tilt_approach_half])
        for standoff_position in standoff_candidates(
            tilt_grasp_pos,
            approach_world,  # standoff_candidates wants gripper->target (see handle_grasp_pose)
            cfg.approach_standoff_m,
            cfg.standoff_distance_multipliers,
            cfg.standoff_lateral_offsets_m,
            cfg.standoff_lateral_directions,
        ):
            approach_plan.append(
                (
                    float(tilt_rad),
                    float(roll_rad),
                    tilt_grasp_pos,
                    tilt_quat,
                    tilt_closing_axis,
                    tilt_width_m,
                    tilt_finger_qpos,
                    tilt_approach_finger_qpos,
                    standoff_position,
                )
            )

    last_rejection = "no_ik_attempted"
    for (
        tilt_rad,
        roll_rad,
        grasp_pos_candidate,
        grasp_quat,
        closing_axis_world,
        handle_width_m,
        finger_closed_qpos,
        approach_finger_qpos,
        standoff_position,
    ) in approach_plan:
        # Robust solve, not a single seed from start_q -- see
        # OpenDrawerSkill's own comment history: solving each from start_q
        # alone found none of 14/75 genuinely reachable+collision-free
        # candidates, reporting "approach_standoff_ik_failed" as though the
        # drawer were unreachable.
        candidate_standoff_ik = _solve_ik_robust(
            ik_solver, scene, standoff_position, grasp_quat, start_q, rng,
            n_restarts=cfg.pose_selection_attempts,
        )
        if not candidate_standoff_ik.success:
            last_rejection = "approach_standoff_ik_failed"
            continue
        candidate_grasp_ik = _solve_ik_robust(
            ik_solver, scene, grasp_pos_candidate, grasp_quat, candidate_standoff_ik.q, rng,
            n_restarts=cfg.pose_selection_attempts,
        )
        if not candidate_grasp_ik.success:
            last_rejection = "grasp_ik_failed"
            continue
        descent_check = collision_checker.check_edge(
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
        # pose, not what happens when it actually closes. Validate that
        # transition explicitly before accepting the candidate, rather than
        # only ever finding out from the real gripper closing on a bad pose
        # during execution.
        closure_check = collision_checker.check_gripper_closure(
            candidate_grasp_ik.q, FINGER_JOINT_NAMES, finger_closed_qpos, drawer.drawer_body
        )
        if closure_check.in_collision:
            last_rejection = "gripper_closure_in_collision"
            continue
        return (
            DrawerHandleGraspCandidate(
                standoff_ik=candidate_standoff_ik,
                grasp_ik=candidate_grasp_ik,
                tilt_rad=tilt_rad,
                roll_rad=roll_rad,
                grasp_pos=grasp_pos_candidate,
                grasp_quat=grasp_quat,
                closing_axis_world=closing_axis_world,
                handle_width_m=handle_width_m,
                finger_closed_qpos=finger_closed_qpos,
                approach_finger_qpos=approach_finger_qpos,
            ),
            "",
        )
    return None, last_rejection


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
        handle_centre_for_report = np.array(grasp_pos, dtype=np.float64)  # before any candidate offset
        cfg = self.config.drawer
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
        # interpolation, a genuinely bad configuration choice.
        #
        # Solve the *standoff* first and reseed the grasp solve from it
        # (mirrors PickSkill's own order, see its comment on
        # ``standoff_candidates`` for why) -- not the other way around. An
        # earlier version solved grasp first and only reseeded the standoff
        # from that one fixed grasp solution; measured directly against the
        # real cluttered scene above, that fixed the *elbow config at the
        # grasp target* before ever considering the standoff, so no amount
        # of standoff variety could escape it once that grasp configuration
        # happened to sweep through the bowl -- every one of ~75 standoff
        # candidates failed identically (same descent-collision) for every
        # grasp seed that happened to succeed. Standoff-first lets each
        # standoff candidate reseed grasp IK into a genuinely different
        # basin, since a different arm posture approaching from a different
        # direction usually settles into a different elbow config at the
        # same Cartesian grasp target too.
        # Approach the handle along a *tilted* line, not only straight along
        # the drawer's own pull axis. A purely horizontal approach forces the
        # whole hand through the thin layer of space directly in front of the
        # cabinet at handle height, which is exactly where table clutter
        # lives -- measured on the middle drawer (handle only ~11.6cm above
        # the tabletop): 44 of 75 standoff poses were reachable, and *none*
        # had a collision-free descent, blocked by the tabletop itself (13)
        # and by a plate and cream cheese parked in front (15). Tilting the
        # approach downward brings the hand in from above that layer and
        # descends onto the handle instead of driving through the clutter.
        # The pull afterwards is unaffected: it always follows the drawer
        # joint's own measured axis, not however the hand arrived.
        candidate, last_rejection = find_drawer_handle_grasp(
            scene, drawer, grasp_pos, pull_axis_world, start_q,
            self.ik_solver, self.collision_checker, cfg, edge_resolution, self._rng,
        )
        grasp_ik = standoff_ik = None
        if candidate is not None:
            grasp_ik, standoff_ik = candidate.grasp_ik, candidate.standoff_ik
            tilt_rad = candidate.tilt_rad
            roll_rad = candidate.roll_rad
            # Everything downstream -- the pull reference, the diagnostics the
            # adapter tracks -- must use the aim point that actually passed,
            # not the handle's raw centre.
            grasp_pos = candidate.grasp_pos
            approach_finger_qpos = candidate.approach_finger_qpos
            grasp_quat = candidate.grasp_quat
            closing_axis_world = candidate.closing_axis_world
            handle_width_m = candidate.handle_width_m
            finger_closed_qpos = candidate.finger_closed_qpos
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
                    # Which approach the candidate search settled on. A
                    # tilted approach reaches the handle from further above,
                    # so its standoff can be a much harder place to plan a
                    # path to than the straight-on one -- worth knowing here
                    # rather than inferring, since accepting a tilted
                    # candidate is exactly what turns "no candidate" into
                    # "candidate found, path to it timed out".
                    "approach_tilt_rad": tilt_rad,
                    "approach_roll_rad": roll_rad,
                    "standoff_position": standoff_ik.q.tolist(),
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
                "approach_roll_rad": roll_rad,
                "grasp_offset_from_handle_m": (grasp_pos - handle_centre_for_report).tolist(),
                "approach_tilt_rad": tilt_rad,
            },
        )
