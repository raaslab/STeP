"""Planning-only OpenDrawerSkill: grasps a drawer/cabinet handle and plans
a pull along its own slide-joint axis, instead of PickSkill's generic grasp
search, since a fixture has exactly one correct grasp axis and pull motion.
Orientation follows grasp_generator.py's convention (+Z approach, +X
closing), built from the measured pull axis and the handle's long axis.
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
    """Tuning for OpenDrawerSkill/LiberoOpenDrawerAdapter (slide-joint
    drawers/cabinets only)."""

    approach_standoff_m: float = 0.08
    # Downward tilts tried in order; 0 is straight along the pull axis, the
    # rest bring the hand in from above to clear clutter in front of the drawer.
    approach_tilts_rad: tuple[float, ...] = (0.0, 0.35, 0.61, 0.87)  # 0, 20, 35, 50 degrees
    # Wrist roll about the approach axis; other rolls tested worse for
    # closing on the bar, so only the plain pinch is kept.
    approach_rolls_rad: tuple[float, ...] = (0.0,)  # straight pinch only
    # (along-pull-axis, vertical) offsets from the handle's exact centre, so
    # fingers aren't driven straight into the bar.
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
    # How close the hand must land to count the handle as captured -- loose
    # enough for finger slack, tight enough to reject a neighbouring handle.
    handle_capture_radius_m: float = 0.045
    # Clearance left per side when approaching with pre-narrowed fingers, so
    # the handle fits between them without needing the full open span.
    approach_finger_clearance_m: float = 0.006
    # IK is not unique; this many (grasp, standoff) pairs are tried before
    # giving up, taking the first whose segment is collision-clear.
    pose_selection_attempts: int = 6
    # These matter only once the nominal straight-back standoff is blocked
    # by clutter sitting directly in that one path.
    standoff_distance_multipliers: tuple[float, ...] = (1.0, 2.5, 4.0)
    standoff_lateral_offsets_m: tuple[float, ...] = (0.05, 0.08, 0.12)
    standoff_lateral_directions: int = 8
    # RRTConnect is stochastic and can return only an approximate solution
    # or time out on a feasible pair; retried this many times before giving up.
    planning_attempts: int = 3
    # Tried largest-first; the first IK-reachable, collision-clear fraction
    # of full travel wins, instead of failing when full travel isn't reachable.
    pull_fraction_fallbacks: tuple[float, ...] = (0.85, 0.7, 0.5, 0.3)
    # Fraction of the joint's full qpos range travelled (0=closed, 1=open)
    # required to call the drawer "open".
    open_threshold_frac: float = 0.8
    retreat_m: float = 0.08
    retreat_steps: int = 40
    # Return to robot.init_qpos after retreating so the handle-approach's
    # often-awkward elbow/wrist config doesn't carry into the next subtask's IK.
    return_home_steps: int = 60


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def _roll_about(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of vec about axis -- rolls the gripper about
    its own approach axis without changing where the hand is aimed."""
    a = _unit(np.asarray(axis, dtype=np.float64))
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return vec * c + np.cross(a, vec) * s + a * float(np.dot(a, vec)) * (1.0 - c)


def _orientation_from_approach_and_reference(
    approach_world: np.ndarray, reference_axis_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Hand-frame quaternion (WXYZ) whose local +Z is approach_world and
    whose local +X is perpendicular to both approach_world and
    reference_axis_world, so fingers close across the handle's long axis.
    Also returns that closing axis, which callers need for handle-width estimation."""
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
    """A reachable, collision-free, closure-safe standoff->grasp pair for a
    drawer handle, plus the finger/orientation params for the tilt/roll/offset
    that won. Shared by OpenDrawerSkill (pulls +pull_axis_world) and
    CloseSkill (retracts -pull_axis_world) from an identical search."""

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
    """Searches cfg's tilt x roll x offset x standoff grid for the first
    candidate that is IK-reachable, has a collision-free descent, and
    survives the fingers closing on the handle. Shared by OpenDrawerSkill
    and CloseSkill so both search identically. Returns
    (candidate, last_rejection); candidate is None on failure,
    with last_rejection naming the last rejection reason seen."""
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
        # Offset the aim point off the bar's centre: depth_off runs along the
        # pull axis (negative = deeper), vert_off straight up/down.
        tilt_grasp_pos = (
            grasp_pos + pull_axis_world * float(depth_off) + np.array([0.0, 0.0, float(vert_off)])
        )
        # Approach with fingers narrowed to just wider than the handle --
        # fully open can't fit through the clearance above some handles.
        tilt_approach_half = float(
            np.clip(tilt_width_m / 2.0 + cfg.approach_finger_clearance_m, 0.0, 0.04)
        )
        tilt_approach_finger_qpos = np.array([tilt_approach_half, -tilt_approach_half])
        for standoff_position in standoff_candidates(
            tilt_grasp_pos,
            approach_world,  # standoff_candidates wants gripper->target direction
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
        # Multi-restart solve, not a single seed -- a single seed missed
        # genuinely reachable candidates, reporting them as unreachable.
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
        # IK and descent_check only simulate the open gripper reaching this
        # pose, not closing -- validate closure explicitly instead of finding out live.
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

        # Half the handle's width per joint (fingers travel symmetrically),
        # not qpos=[0,0] -- that would drive rigid fingers through the handle.
        handle_width_m = estimate_handle_closing_width_m(scene, drawer, pull_axis_world, closing_axis_world)
        finger_half_travel = float(np.clip(handle_width_m / 2.0, 0.0, 0.04))
        finger_closed_qpos = np.array([finger_half_travel, -finger_half_travel])

        # Approached along a tilted line, not straight along the pull axis --
        # a horizontal approach forces the hand through clutter at handle height.
        candidate, last_rejection = find_drawer_handle_grasp(
            scene, drawer, grasp_pos, pull_axis_world, start_q,
            self.ik_solver, self.collision_checker, cfg, edge_resolution, self._rng,
        )
        grasp_ik = standoff_ik = None
        if candidate is not None:
            grasp_ik, standoff_ik = candidate.grasp_ik, candidate.standoff_ik
            tilt_rad = candidate.tilt_rad
            roll_rad = candidate.roll_rad
            # Downstream code (pull reference, adapter diagnostics) must use
            # the aim point that passed, not the handle's raw centre.
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

        # RRTConnect is stochastic -- a feasible pair can still come back
        # approximate or time out, so those are retried (unlike deterministic failures).
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
                    # Which approach the search settled on -- a tilted
                    # standoff can be harder to plan a path to than the straight-on one.
                    "approach_tilt_rad": tilt_rad,
                    "approach_roll_rad": roll_rad,
                    "standoff_position": standoff_ik.q.tolist(),
                },
            )

        # Only the descent goes into approach_positions -- the pull leg is
        # tracked separately, once the gripper has actually closed on the handle.
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
