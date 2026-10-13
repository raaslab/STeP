"""Planning-only ``OpenDrawerSkill``: grasp a drawer/cabinet handle and plan a
pull along its own slide-joint axis.

Deliberately *not* built on ``GraspCandidate``/``PickSkill``'s OR-object-set
machinery: a fixture has exactly one physically correct grasp axis (the
handle's long axis) and exactly one correct post-grasp motion (a straight
pull along the joint's own travel direction, not a free 6-DoF carry), so a
small purpose-built skill is simpler than bending ``PickSkill`` to fit.

Grasp orientation convention matches ``grasp_generator.py``: local +Z is the
approach axis, local +X is the closing axis. Here the approach is horizontal
(along the measured pull axis) rather than top-down, so the frame is built
from two measured directions -- the pull axis (approach) and the handle's
long axis (which must map to the non-closing axis, since the fingers straddle
it rather than close along it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...motion.articulation import (
    DrawerHandle,
    estimate_handle_closing_width_m,
    locate_handle_position_world,
    measure_pull_axis_and_travel,
    resolve_drawer,
)
from ...motion.collision import CollisionChecker, CollisionPhase
from ...motion.config import MotionConfig
from ..ik_utils import solve_ik_robust as _solve_ik_robust
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from ...motion.panda import FINGER_JOINT_NAMES
from ...motion.scene_query import SceneQuery, rotmat_to_quat_wxyz
from ..standoff_search import standoff_candidates
from ...motion.types import IKResult, PlannerFailureReason, SkillFailureReason, SkillResult


@dataclass
class DrawerConfig:
    """Tuning for ``OpenDrawerSkill``/``LiberoOpenDrawerAdapter`` (slide-joint
    drawers/cabinets only -- see ``motion/articulation.py``)."""

    approach_standoff_m: float = 0.08
    # Downward tilts (radians) tried for the handle approach, in order. 0 is
    # the straight-along-the-pull-axis approach, best when the space in front
    # of the drawer is clear; the tilted ones bring the hand in from above
    # instead, which is the only way past clutter sitting in that space.
    # Opening a drawer doesn't need a clean grasp, so a hook-style tilted
    # approach is an acceptable fallback, not a compromise.
    approach_tilts_rad: tuple[float, ...] = (0.0, 0.35, 0.61, 0.87)  # 0, 20, 35, 50 degrees
    # Wrist roll about the approach axis. Only the plain pinch (fingers
    # above/below the handle bar) is kept -- rolls of pi/2 and +/-45 were
    # tried and measured worse (can't close on the bar, or close but can't
    # clear the tabletop); the real blocker is clutter in front of the
    # cabinet, not the wrist angle.
    approach_rolls_rad: tuple[float, ...] = (0.0,)  # straight pinch only
    # Offsets applied to the handle target, as (along-pull-axis, vertical)
    # metres. Aiming dead centre drives the fingers into the bar itself, so a
    # few offsets around it are tried instead -- opening a drawer doesn't
    # need the bar centred between the pads, just caught when pulled.
    grasp_offsets_m: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.02),
        (-0.02, 0.02),
        (-0.03, 0.02),
        (-0.04, 0.02),
        (-0.03, 0.0),
        (0.0, -0.02),
        (-0.015, 0.0),
    )
    # How close the hand has to end up to the handle for the pull to count as
    # "the handle is between the fingers". Looser than a grasp test (fingers
    # are ~8cm apart open, so some slack is fine), but tight enough to still
    # reject an approach that landed on the cabinet face near a neighbouring
    # handle instead.
    handle_capture_radius_m: float = 0.045
    # Clearance left on each side of the handle when approaching with
    # pre-narrowed fingers (see OpenDrawerSkill's approach_finger_qpos), so
    # the handle passes cleanly between them without needing the full open
    # span -- which wouldn't fit the gap above a lower drawer's handle.
    approach_finger_clearance_m: float = 0.006
    # IK is not unique -- different seeds can converge to different elbow/
    # wrist configurations reaching the same pose while sweeping through
    # different parts of the scene. This many (grasp, standoff) candidate
    # pairs are tried before giving up, taking the first whose connecting
    # segment is actually collision-clear.
    pose_selection_attempts: int = 6
    # Same standoff-candidate search PickSkill uses (see
    # standoff_search.standoff_candidates) -- the nominal straight-back
    # standoff is tried first for the simple case, with these mattering only
    # once that's blocked by clutter sitting directly in that one path.
    standoff_distance_multipliers: tuple[float, ...] = (1.0, 2.5, 4.0)
    standoff_lateral_offsets_m: tuple[float, ...] = (0.05, 0.08, 0.12)
    standoff_lateral_directions: int = 8
    # RRTConnect is stochastic and can occasionally return only an
    # "approximate" (rejected) solution or hit its timeout on a genuinely
    # feasible start/goal pair; this many attempts are retried before
    # reporting NO_FEASIBLE_CANDIDATE.
    planning_attempts: int = 3
    # Tried largest-first; the first fraction of the joint's full measured
    # travel that's both IK-reachable and collision-clear wins, rather than
    # failing outright when the full travel distance isn't reachable.
    pull_fraction_fallbacks: tuple[float, ...] = (0.85, 0.7, 0.5, 0.3)
    # Fraction of the joint's full qpos range travelled (0=closed, 1=open)
    # required to call the drawer "open".
    open_threshold_frac: float = 0.8
    retreat_m: float = 0.08
    retreat_steps: int = 40
    # After retreating, servo back to the robot's home/reset joint config
    # (``robot.init_qpos``) before ending the subtask -- the handle-approach
    # IK solution is often an awkward configuration (elbow twisted, wrist
    # near a limit) that otherwise leaves the next subtask's IK solve cramped.
    # Dense joint-space interpolation rather than a single Cartesian move,
    # since commanding home's Cartesian pose directly can converge to a
    # different elbow config than true home (IK is not unique).
    return_home_steps: int = 60


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def _roll_about(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of ``vec`` about ``axis`` -- rolls the gripper about
    its own approach axis without changing where the hand is aimed."""
    a = _unit(np.asarray(axis, dtype=np.float64))
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return vec * c + np.cross(a, vec) * s + a * float(np.dot(a, vec)) * (1.0 - c)


def _orientation_from_approach_and_reference(
    approach_world: np.ndarray, reference_axis_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Hand-frame quaternion (WXYZ) whose local +Z is ``approach_world`` and
    whose local +X (closing axis) is perpendicular to both ``approach_world``
    and ``reference_axis_world``, so the fingers close across the handle's
    long axis rather than along it. Also returns that closing axis, which
    callers need to measure the handle's width along it (see
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
    handle, plus the per-candidate finger/orientation parameters for the
    tilt/roll/offset that won. Shared by ``OpenDrawerSkill`` (pulls
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
    """Search ``cfg``'s tilt x roll x offset x standoff grid for a candidate
    approach to this drawer's handle that is IK-reachable, has a
    collision-free descent, and survives the fingers actually closing on it.
    Accepts the first candidate that passes all three checks rather than an
    exhaustive best-of-all-candidates search. Shared by ``OpenDrawerSkill``
    and ``CloseSkill`` so both search identically.

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
        # handle, rather than fully open -- a fully-open gripper can't fit
        # through the clearance above some handles.
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
        # Robust (multi-restart) solve, not a single seed from start_q --
        # a single seed missed genuinely reachable+collision-free candidates,
        # reporting them as unreachable instead.
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
        # Neither the IK solve above nor `descent_check` move the finger
        # joints -- they simulate the open gripper reaching this pose, not
        # what happens when it closes. Validate that explicitly here rather
        # than finding out from the real gripper closing on a bad pose.
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

        # How far each finger should travel to close on the handle -- NOT the
        # joint's mechanical closing limit (qpos=[0,0] meets at the
        # centerline, which would drive the rigid finger through a
        # nonzero-width handle). Each finger travels symmetrically from
        # center, so half the handle's width is the right qpos per joint.
        handle_width_m = estimate_handle_closing_width_m(scene, drawer, pull_axis_world, closing_axis_world)
        finger_half_travel = float(np.clip(handle_width_m / 2.0, 0.0, 0.04))
        finger_closed_qpos = np.array([finger_half_travel, -finger_half_travel])

        # IK is not unique -- different arm configurations reach the same
        # Cartesian grasp pose while sweeping through different parts of the
        # scene (e.g. the forearm through clutter sitting on the table).
        # Solving the standoff first and reseeding the grasp solve from it
        # (rather than the reverse) lets each standoff candidate reseed grasp
        # IK into a genuinely different elbow config, instead of locking in
        # one grasp-target configuration before the standoff is even chosen.
        #
        # The handle is also approached along a tilted line, not only
        # straight along the pull axis: a purely horizontal approach forces
        # the hand through the thin band of space in front of the cabinet at
        # handle height, exactly where table clutter tends to sit. Tilting
        # downward brings the hand in from above instead. The pull itself is
        # unaffected, since it always follows the drawer joint's own axis.
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

        # RRTConnect is stochastic: a feasible start/goal pair can still
        # occasionally come back as only an "approximate" solution or hit its
        # time budget from an unlucky tree expansion, so those outcomes are
        # retried (deterministic failures like START_IN_COLLISION are not).
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
                    # Which approach the candidate search settled on -- a
                    # tilted approach's standoff can be a much harder place
                    # to plan a path to than the straight-on one.
                    "approach_tilt_rad": tilt_rad,
                    "approach_roll_rad": roll_rad,
                    "standoff_position": standoff_ik.q.tolist(),
                },
            )

        # Only the descent (standoff -> grasp) goes into `approach_positions`
        # -- matching PickSkill's convention: settle onto the grasp pose,
        # then close the gripper. The pull leg (`grasp_q` -> `pulled_q`) is
        # kept separate and only tracked after the gripper has actually
        # closed on the handle; concatenating both into one reference would
        # drive the pull motion with the gripper still open.
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
