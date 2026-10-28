"""Planning-only TurnKnobSkill: grasp a hinge-jointed knob/dial near its
outer edge and plan a rotation about its own measured axis -- e.g. LIBERO's
flat_stove_1 control ("turn on the stove"). A knob has exactly one
physically correct rotation axis and exactly one correct motion after
grasping, so it gets its own small skill rather than being forced through PickSkill.

**"Turn on" threshold**: matches LIBERO's own FlatStove.turn_on(qpos)
(qpos >= 0.5) so this skill's success check agrees with ground truth;
overridable per-knob via KnobConfig.turn_on_qpos_threshold.

**Grasp geometry**: the hand grips the knob's handle (its raised bar, the
tallest part) at its midpoint, fingers closing across the bar's thin side. Gripping a chord at the knob's outer edge instead left
only the thin base disc to hold, right against the stove: the hand stalled
short and swept the knob round rather than turning it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...motion.articulation import (
    KnobHandle,
    estimate_knob_radius_m,
    measure_knob_rotation,
    resolve_knob,
)
from ...motion.collision import CollisionChecker, CollisionPhase
from ...motion.config import MotionConfig
from ..ik_utils import solve_ik_robust as _solve_ik_robust
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from ...motion.panda import FINGER_JOINT_NAMES
from ...motion.scene_query import SceneQuery, quat_to_rotmat_wxyz, rotmat_to_quat_wxyz
from ...motion.types import PlannerFailureReason, SkillFailureReason, SkillResult


@dataclass
class KnobConfig:
    """Tuning for TurnKnobSkill/LiberoStoveAdapter (hinge-jointed
    knobs/dials only)."""

    approach_standoff_m: float = 0.06
    # How far below the grip's top the TCP sits, tried in order -- only a
    # real contact check finds which depth has material between the pads.
    grasp_depths_below_top_m: tuple[float, ...] = (0.015, 0.02, 0.01, 0.025, 0.03)
    # Hand yaw offsets (rad) from "fingers closing across the grip", for IK
    # room; a grip with no clear long direction sweeps half a turn instead.
    grasp_yaw_offsets_rad: tuple[float, ...] = (0.0, 0.26, -0.26, 0.52, -0.52)
    round_grip_yaw_samples: int = 8
    # Same "try a few, take the first feasible" shape as
    # DrawerConfig.pose_selection_attempts, spent per angle.
    pose_selection_attempts: int = 6
    planning_attempts: int = 3
    # Tried largest-first; the first IK-reachable, collision-clear fraction
    # of full range (toward the "on" endpoint) wins.
    turn_fraction_fallbacks: tuple[float, ...] = (0.75, 0.6, 0.45, 0.3)
    # Absolute qpos (radians) at/above which the knob counts as "turned on",
    # matching LIBERO's own flat_stove.turn_on() threshold.
    turn_on_qpos_threshold: float = 0.5
    # How far past turn_on_qpos_threshold to target, rather than the joint's
    # full range end -- cranking to the mechanical stop drags the arm through the surrounding grate/table.
    turn_on_safety_margin_rad: float = 0.6
    # A grasp counts as turning far enough once its planned turn passes the
    # threshold by this much; beyond that, squareness on the handle matters more.
    turn_reach_margin_rad: float = 0.1
    retreat_m: float = 0.06
    retreat_steps: int = 40


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def _axis_angle_to_rotmat(axis_world: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula: the rotation matrix for turning by
    angle_rad about unit vector axis_world."""
    axis = _unit(axis_world)
    kx, ky, kz = axis
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1.0 - np.cos(angle_rad)) * (k @ k)


def _grip_contact_depth(
    collision_checker: CollisionChecker,
    scene: SceneQuery,
    arm_q: np.ndarray,
    finger_closed_qpos: np.ndarray,
    target_body: str,
) -> float | None:
    """Does closing the fingers here touch real material on target_body,
    not just avoid colliding with anything else? check_gripper_closure
    reports "not in collision" even for a closure that touches nothing,
    which matters here since a knob's non-cylindrical mesh can leave a
    collision-clear closure gripping empty air. Returns the deepest
    fingerpad<->target penetration found, or None if there's no real
    contact -- used only to break ties between candidates that turn the
    knob equally far, since turn reach, not grip depth, determines success."""
    scratch = scene.clone()
    scratch.set_qpos(collision_checker.joint_names, arm_q)
    scratch.set_qpos(FINGER_JOINT_NAMES, np.asarray(finger_closed_qpos, dtype=np.float64))
    target_geom_ids = set(scratch.body_geom_ids(target_body))
    deepest: float | None = None
    for g1_name, g2_name, dist in scratch.compute_contacts():
        g1, g2 = scratch.geom_id(g1_name), scratch.geom_id(g2_name)
        if (g1 in collision_checker.fingerpad_geom_ids and g2 in target_geom_ids) or (
            g2 in collision_checker.fingerpad_geom_ids and g1 in target_geom_ids
        ):
            if deepest is None or dist < deepest:
                deepest = dist
    return deepest


def _orientation_for_grasp(approach_world: np.ndarray, closing_world: np.ndarray) -> np.ndarray:
    """Hand-frame quaternion (WXYZ) whose local +Z is approach_world and
    whose local +X (closing axis) is closing_world."""
    z_axis = _unit(approach_world)
    x_axis = _unit(closing_world - z_axis * float(closing_world @ z_axis))
    y_axis = np.cross(z_axis, x_axis)
    return rotmat_to_quat_wxyz(np.column_stack([x_axis, y_axis, z_axis]))


def _knob_grip(
    scene: SceneQuery, knob_body: str, axis_world: np.ndarray, center_world: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, float, float]:
    """The knob's raised grip, from its own collision geometry: the upper
    half of the knob along its axis. Returns (grip centre in the mounting
    plane, top height above that plane, long direction in the plane,
    half-thickness across it, elongation)."""
    bid = scene.body_id(knob_body)
    R = scene.data.xmat[bid].reshape(3, 3)
    pts = scene.body_surface_points_dense_local(knob_body) @ R.T + scene.data.xpos[bid]
    h = (pts - center_world) @ axis_world
    top_h = float(h.max())
    top = pts[h > 0.5 * (top_h + float(h.min()))]
    rel = (top - center_world) - np.outer((top - center_world) @ axis_world, axis_world)
    evals, evecs = np.linalg.eigh(np.cov((rel - rel.mean(axis=0)).T))
    long_dir = _unit(evecs[:, -1] - axis_world * float(evecs[:, -1] @ axis_world))
    thin_dir = np.cross(axis_world, long_dir)
    # Midpoint of the handle's extent: surface samples are uneven across
    # geoms, which pulls their mean toward the knob's pointer end.
    along, across = rel @ long_dir, rel @ thin_dir
    offset = 0.5 * (along.min() + along.max()) * long_dir + 0.5 * (across.min() + across.max()) * thin_dir
    thin_half = 0.5 * float(across.max() - across.min())
    elongation = float(np.sqrt(evals[-1] / max(evals[-2], 1e-12)))
    return center_world + offset, top_h, long_dir, thin_half, elongation


class TurnKnobSkill:
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
        knob_ref: str,
        start_q: np.ndarray,
        constraints: dict | None = None,
    ) -> SkillResult:
        constraints = constraints or {}
        try:
            knob = resolve_knob(scene, knob_ref)
        except KeyError as exc:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": str(exc)},
            )

        cfg = self.config.knob
        axis_world, center_world, qpos_lo, qpos_hi = measure_knob_rotation(scene, knob)
        radius_m = estimate_knob_radius_m(scene, knob, axis_world)
        current_qpos = float(scene.get_qpos([knob.joint_name])[0])

        # Which range endpoint is "on"? Compare against threshold rather
        # than assuming qpos increases toward "on" -- a different knob could be wired the opposite way.
        threshold = cfg.turn_on_qpos_threshold
        if qpos_hi >= threshold:
            # Stop just past the threshold, not at the joint's full range end.
            target_qpos = min(qpos_hi, threshold + cfg.turn_on_safety_margin_rad)
        elif qpos_lo >= threshold:
            target_qpos = max(qpos_lo, threshold - cfg.turn_on_safety_margin_rad)
        else:
            # Neither endpoint clears the threshold -- shouldn't happen for a
            # correctly configured knob, but turn as far as possible rather than refuse outright.
            target_qpos = qpos_hi if abs(qpos_hi - current_qpos) >= abs(qpos_lo - current_qpos) else qpos_lo

        # Grasp the grip at its centre, fingers closing across its thin side.
        grip_center, grip_top_h, grip_long, grip_thin_half, elongation = _knob_grip(
            scene, knob.knob_body, axis_world, center_world
        )
        closing0 = np.cross(axis_world, grip_long)
        if elongation >= 1.3:
            # Either way round closes across the handle; the flip gives the wrist more room.
            yaws = list(cfg.grasp_yaw_offsets_rad) + [y + np.pi for y in cfg.grasp_yaw_offsets_rad]
        else:
            yaws = [np.pi * i / cfg.round_grip_yaw_samples for i in range(cfg.round_grip_yaw_samples)]
        approach_world = -axis_world  # top-down onto the knob, same convention as every other grasp in this package
        # Close just past the grip's faces so the pads register real contact.
        finger_half_travel = float(np.clip(grip_thin_half - 0.002, 0.0, 0.04))
        finger_closed_qpos = np.array([finger_half_travel, -finger_half_travel])

        # Grip choice and turn feasibility are coupled: a deepest-gripping
        # point can sit next to an obstacle that caps its rotation, so candidates are scored by deliverable turn.
        best_grasp_ik = best_standoff_ik = best_turn_ik = None
        best_turn_fraction = -1.0
        best_depth = 0.0
        best_grasp_pos = best_grasp_quat = None
        best_key = None
        last_rejection = "no_ik_attempted"
        full_delta = target_qpos - current_qpos
        for depth_below_top in cfg.grasp_depths_below_top_m:
            height = max(grip_top_h - depth_below_top, 0.0)
            # The pivot at this height: the turn rotates the grasp about it.
            grasp_center_world = center_world + height * axis_world
            grasp_pos = grip_center + height * axis_world
            for yaw in yaws:
                closing = _axis_angle_to_rotmat(axis_world, yaw) @ closing0
                grasp_quat = _orientation_for_grasp(approach_world, closing)
                candidate_grasp_ik = _solve_ik_robust(
                    self.ik_solver, scene, grasp_pos, grasp_quat, start_q, self._rng, n_restarts=cfg.pose_selection_attempts
                )
                if not candidate_grasp_ik.success:
                    last_rejection = "grasp_ik_failed"
                    continue
                standoff_pos = grasp_pos + cfg.approach_standoff_m * axis_world
                candidate_standoff_ik = _solve_ik_robust(
                    self.ik_solver, scene, standoff_pos, grasp_quat, candidate_grasp_ik.q, self._rng
                )
                if not candidate_standoff_ik.success:
                    last_rejection = "approach_standoff_ik_failed"
                    continue
                descent_check = self.collision_checker.check_edge(
                    candidate_standoff_ik.q,
                    candidate_grasp_ik.q,
                    resolution=self.config.planner.validity_resolution
                    * float(np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)),
                    phase=CollisionPhase.TERMINAL_APPROACH,
                    target_object_body=knob.knob_body,
                )
                if descent_check.in_collision:
                    last_rejection = "final_descent_in_collision"
                    continue
                closure_check = self.collision_checker.check_gripper_closure(
                    candidate_grasp_ik.q, FINGER_JOINT_NAMES, finger_closed_qpos, knob.knob_body
                )
                if closure_check.in_collision:
                    last_rejection = "gripper_closure_in_collision"
                    continue
                depth = _grip_contact_depth(
                    self.collision_checker, scene, candidate_grasp_ik.q, finger_closed_qpos, knob.knob_body
                )
                if depth is None:
                    last_rejection = "no_grip_contact"
                    continue

                # How much of the turn can this grasp point deliver? Tried
                # largest-first; the first IK-reachable, collision-clear fraction wins.
                r_grasp = quat_to_rotmat_wxyz(grasp_quat)
                candidate_turn_ik = None
                candidate_turn_fraction = -1.0
                for fraction in cfg.turn_fraction_fallbacks:
                    delta_angle = float(fraction) * full_delta
                    r_delta = _axis_angle_to_rotmat(axis_world, delta_angle)
                    turned_pos = grasp_center_world + r_delta @ (grasp_pos - grasp_center_world)
                    turned_quat = rotmat_to_quat_wxyz(r_delta @ r_grasp)
                    turn_ik = _solve_ik_robust(self.ik_solver, scene, turned_pos, turned_quat, candidate_grasp_ik.q, self._rng)
                    if not turn_ik.success:
                        continue
                    turn_check = self.collision_checker.check_edge(
                        candidate_grasp_ik.q,
                        turn_ik.q,
                        resolution=self.config.planner.validity_resolution
                        * float(np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)),
                        phase=CollisionPhase.TERMINAL_APPROACH,
                        target_object_body=knob.knob_body,
                    )
                    if turn_check.in_collision:
                        continue
                    candidate_turn_ik = turn_ik
                    candidate_turn_fraction = float(fraction)
                    break
                if candidate_turn_ik is None:
                    last_rejection = "no_feasible_turn_target"
                    continue

                # Once a grasp's turn clears the "on" threshold with margin, the
                # squarest grasp on the handle wins: a diagonal pinch slips
                # ~15 deg before it bites, so its larger planned turn isn't
                # real. Otherwise the largest turn wins. Grip depth breaks ties.
                off_square = abs(float(np.sin(yaw)))
                reaches = current_qpos + candidate_turn_fraction * full_delta >= threshold + cfg.turn_reach_margin_rad
                key = (reaches, -off_square, candidate_turn_fraction, -depth) if reaches else (reaches, candidate_turn_fraction, -off_square, -depth)
                if best_key is not None and key <= best_key:
                    continue
                best_key = key
                best_grasp_ik, best_standoff_ik, best_turn_ik = candidate_grasp_ik, candidate_standoff_ik, candidate_turn_ik
                best_turn_fraction, best_depth = candidate_turn_fraction, depth
                best_grasp_pos, best_grasp_quat = grasp_pos, grasp_quat

        grasp_ik, standoff_ik, turn_ik = best_grasp_ik, best_standoff_ik, best_turn_ik
        if grasp_ik is None or standoff_ik is None or turn_ik is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": last_rejection,
                    "knob": knob.knob_body,
                    "grasp_depths_below_top_m": list(cfg.grasp_depths_below_top_m),
                    "grasp_yaws_rad": [float(y) for y in yaws],
                    "pose_selection_attempts": cfg.pose_selection_attempts,
                    "tried_fractions": list(cfg.turn_fraction_fallbacks),
                },
            )
        grasp_pos, grasp_quat = best_grasp_pos, best_grasp_quat
        turned_q = turn_ik.q
        turn_fraction_used = best_turn_fraction

        plan = None
        for _ in range(cfg.planning_attempts):
            plan = self.planner.plan(
                start_q, [standoff_ik.q], phase=CollisionPhase.FREE_SPACE, target_object_body=knob.knob_body
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
                diagnostics={"detail": f"planning_failed:{plan.failure_reason}", "knob": knob.knob_body},
            )

        # Only the descent goes into approach_positions -- the turn leg is
        # tracked separately, after the gripper has closed.
        descent_positions = np.linspace(standoff_ik.q, grasp_ik.q, 15)
        plan.metadata["standoff_q"] = standoff_ik.q.tolist()
        plan.metadata["grasp_q"] = grasp_ik.q.tolist()
        plan.metadata["turned_q"] = turned_q.tolist()
        plan.metadata["approach_positions"] = descent_positions.tolist()

        return SkillResult(
            success=True,
            selected_object=knob.knob_body,
            planned_trajectory=plan,
            diagnostics={
                "knob_body": knob.knob_body,
                "joint_name": knob.joint_name,
                "axis_world": axis_world.tolist(),
                "center_world": center_world.tolist(),
                "radius_m": radius_m,
                "grasp_position": grasp_pos.tolist(),
                "grasp_quat_wxyz": grasp_quat.tolist(),
                "grip_center": grip_center.tolist(),
                "grip_thin_half_m": grip_thin_half,
                "qpos_lo": qpos_lo,
                "qpos_hi": qpos_hi,
                "current_qpos": current_qpos,
                "target_qpos": target_qpos,
                "turn_on_qpos_threshold": threshold,
                "turn_fraction_used": turn_fraction_used,
                "finger_closed_qpos": finger_closed_qpos.tolist(),
            },
        )
