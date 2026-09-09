"""Planning-only ``TurnKnobSkill``: grasp a hinge-jointed knob/dial near its
outer edge and plan a rotation about its own measured axis -- e.g. LIBERO's
``flat_stove_1`` control ("turn on the stove").

Mirrors ``OpenDrawerSkill``'s own reasoning almost exactly, just for a
rotational joint instead of a prismatic one (see that module's docstring for
why a fixture gets its own small, purpose-built skill instead of being forced
through ``PickSkill``'s free-object OR-set machinery): a knob has exactly one
physically correct rotation axis (measured, see ``articulation.
measure_knob_rotation``) and exactly one physically correct motion after
grasping (a rotation about that axis, not a free 6-DoF carry).

**Real-world grounding for the "turn on" threshold**: LIBERO's own
``FlatStove.turn_on(qpos)`` (``libero/envs/objects/articulated_objects.py``)
returns ``qpos >= 0.5`` -- the object's own joint range is
``[-0.005, 2.1]`` radians (a hinge, confirmed against the real
``flat_stove.xml`` asset, despite the joint itself being named "button").
``KnobConfig.turn_on_qpos_threshold`` defaults to that same 0.5 so this
skill's own internal success check agrees with LIBERO's ground truth, but
it's a plain config value, not hardcoded into the skill logic itself, so a
different knob/threshold can be passed via ``MotionConfig``.

**Grasp geometry**: a knob this size (measured radius ~5.5cm on the real
flat_stove asset) is *wider* than this gripper's max opening (~8cm giving a
usable diameter far short of 11cm) -- gripping through its full diameter
isn't possible. Instead, this grips a narrow chord near the knob's outer
edge: for a circle of radius R, a chord at perpendicular offset ``d`` from
the center has length ``2*sqrt(R^2 - d^2)``; solving for the ``d`` that
produces a target (narrow, comfortably-graspable) chord width places the
grasp point close to the rim without needing to span the center. The
gripper's closing axis at that point is *tangential* (perpendicular to the
radial direction to the chord's midpoint), not radial -- confirmed by
direct derivation, not assumed by analogy with the hollow-rim (bowl) case in
``grasp_generator.py``'s ``_rim_edge_candidates``, which closes *radially*
across a thin wall; a solid knob has no "inside" to straddle.
"""

from __future__ import annotations

import numpy as np

from ..articulation import (
    KnobHandle,
    estimate_knob_radius_m,
    measure_knob_rotation,
    resolve_knob,
)
from ..collision import CollisionChecker, CollisionPhase
from ..config import MotionConfig
from ..kinematics import IKSolver
from ..ompl_planner import MotionPlanner
from ..panda import FINGER_JOINT_NAMES
from ..scene_query import SceneQuery, quat_to_rotmat_wxyz, rotmat_to_quat_wxyz
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
    """Same retry-from-perturbed-seeds shape as open_drawer_skill.py's own
    helper of the same name -- a local DLS solve landing in a bad basin is a
    generic IK-solver risk, not specific to drawers."""
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


def _arbitrary_perpendicular(axis_world: np.ndarray) -> np.ndarray:
    """Any one unit vector perpendicular to ``axis_world`` -- there's no
    single "correct" starting angular position around a knob, so this just
    picks a well-conditioned one (avoids the degenerate cross-product case
    where the reference happens to be parallel to the axis)."""
    axis = _unit(axis_world)
    reference = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(axis, reference))


def _axis_angle_to_rotmat(axis_world: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula: the rotation matrix for turning by
    ``angle_rad`` about unit vector ``axis_world``."""
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
    """Does actually closing the fingers here touch real material on
    ``target_body``, not just "closing doesn't collide with anything else"?

    ``CollisionChecker.check_gripper_closure`` only rejects a *bad* contact
    (anything outside the permitted fingerpad<->target/small-overlap rules)
    -- it happily reports "not in collision" for a closure that touches
    nothing at all, since no contact is trivially not a bad one. That gap is
    invisible for a drawer handle or a normal pick target (their geometry is
    simple enough that a collision-clear closure at the intended width
    reliably means "touching it"), but a real knob's collision mesh is not a
    simple prism -- confirmed directly against the real flat_stove_1 asset,
    whose solid cross-section at a plausible grasp radius varies sharply
    with height (see ``KnobConfig.grasp_height_margins_m``'s own comment):
    several collision-clear (margin, angle) candidates turned out to close
    on pure empty air. Verified with a real rollout: accepting those left
    ``knob_grip_verification.gripped=False`` and the knob barely rotated at
    all. This check closes that gap directly, the same way
    ``LiberoOpenDrawerAdapter``'s own post-execution ``gripped_handle``
    diagnostic already does -- just applied here at *plan* time, as a hard
    acceptance gate instead of an after-the-fact diagnostic.

    Returns the deepest (most negative) fingerpad<->target penetration
    found, or ``None`` if there's no real contact at all -- ``None`` is a
    hard rejection (no material there at all); the actual depth is used
    only to break ties between candidates that can turn the knob equally
    far (see ``TurnKnobSkill.execute``'s own selection comment -- turn
    reach, not raw grip depth, is what actually determined success in a
    real rollout, since the single deepest-gripping point available turned
    out to sit right next to an obstacle that capped its usable rotation).
    """
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


def _orientation_for_tangential_grasp(approach_world: np.ndarray, radial_world: np.ndarray) -> np.ndarray:
    """Hand-frame quaternion (WXYZ) whose local +Z is ``approach_world`` and
    whose local +X (closing axis) is tangential -- perpendicular to both the
    approach direction and the radial direction to the chosen grasp point on
    the knob's rim. See this module's docstring for the chord-geometry
    derivation of why the closing axis is tangential, not radial."""
    z_axis = _unit(approach_world)
    x_axis = _unit(np.cross(radial_world, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return rotmat_to_quat_wxyz(R)


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

        # Which range endpoint is "on"? Compare against the configured
        # threshold rather than assuming qpos increases toward "on" -- a
        # different knob's joint could be wired the opposite way.
        threshold = cfg.turn_on_qpos_threshold
        if qpos_hi >= threshold:
            # Stop just past the threshold, not at the joint's full range
            # end -- see KnobConfig.turn_on_safety_margin_rad's own comment.
            target_qpos = min(qpos_hi, threshold + cfg.turn_on_safety_margin_rad)
        elif qpos_lo >= threshold:
            target_qpos = max(qpos_lo, threshold - cfg.turn_on_safety_margin_rad)
        else:
            # Neither endpoint clears the threshold -- shouldn't happen for a
            # correctly configured knob, but turn as far as possible rather
            # than refuse outright; the real qpos check downstream will
            # honestly report whether this actually turned it "on".
            target_qpos = qpos_hi if abs(qpos_hi - current_qpos) >= abs(qpos_lo - current_qpos) else qpos_lo

        # Grasp point: a chord near the knob's outer edge, narrow enough to
        # fit the gripper's real opening (see module docstring's derivation).
        # d = sqrt(R^2 - (w/2)^2) for target chord width w.
        target_width = min(cfg.grasp_chord_width_m, 2.0 * radius_m * 0.99)
        d = float(np.sqrt(max(radius_m**2 - (target_width / 2.0) ** 2, 0.0)))
        radial0 = _arbitrary_perpendicular(axis_world)
        approach_world = -axis_world  # top-down onto the knob, same convention as every other grasp in this package
        finger_half_travel = float(np.clip(target_width / 2.0, 0.0, 0.04))
        finger_closed_qpos = np.array([finger_half_travel, -finger_half_travel])

        # A knob's grasp point has two free dimensions a drawer handle's
        # doesn't: which angle around the rim to grip, and at what height
        # along the rotation axis (see KnobConfig.grasp_height_margins_m's
        # own comment -- a knob's real collision mesh is not a simple
        # prism, so the solid cross-section at a given radius genuinely
        # varies with height). A point offset along the rotation axis from
        # `center_world` stays on that same axis, so any margin still
        # rotates identically to the true center for this skill's turn-angle
        # math below. Per (margin, angle) pair, IK-seed retries are
        # delegated to `_solve_ik_robust`'s own perturbed-restart search.
        #
        # Grip choice and turn feasibility are coupled, not separable steps:
        # confirmed directly against the real flat_stove_1 asset, the
        # (margin, angle) with the single deepest measured grip sat right
        # next to the surrounding burner grate, tolerating only ~0.2rad of
        # rotation before the open gripper swept into it -- nowhere near
        # enough to cross the "on" threshold -- while a shallower-but-still-
        # real grip elsewhere on the rim allowed a much larger sweep. So
        # each candidate here is scored by how much of the *turn* it can
        # actually deliver (checked right here, not deferred to a later,
        # separate step), with a real (negative) grip-contact depth as a
        # hard prerequisite -- not the other way around. And collision-clear
        # alone isn't a sufficient prerequisite either: several (margin,
        # angle) pairs were collision-clear yet closed on empty air (see
        # `_grip_contact_depth`'s own docstring) -- confirmed real contact
        # is required, not just an extra diagnostic.
        best_grasp_ik = best_standoff_ik = best_turn_ik = None
        best_turn_fraction = -1.0
        best_depth = 0.0
        best_radial = best_grasp_quat = best_grasp_center_world = None
        last_rejection = "no_ik_attempted"
        full_delta = target_qpos - current_qpos
        for margin in cfg.grasp_height_margins_m:
            grasp_center_world = center_world + margin * axis_world
            for angle_idx in range(cfg.radial_angle_samples):
                angle = 2.0 * np.pi * angle_idx / cfg.radial_angle_samples
                radial = _axis_angle_to_rotmat(axis_world, angle) @ radial0
                grasp_pos = grasp_center_world + d * radial
                grasp_quat = _orientation_for_tangential_grasp(approach_world, radial)
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

                # How much of the turn can this specific grasp point actually
                # deliver? Tried largest-first (mirrors
                # DrawerConfig.pull_fraction_fallbacks); the first
                # IK-reachable, collision-clear fraction wins for this
                # candidate.
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

                # Largest deliverable turn wins outright; a real grip depth
                # only breaks ties between candidates that turn equally far
                # (confirmed with a real rollout: turn reach, not raw grip
                # depth, is what determined whether the knob actually
                # crossed the "on" threshold).
                if candidate_turn_fraction < best_turn_fraction:
                    continue
                if candidate_turn_fraction == best_turn_fraction and depth >= best_depth:
                    continue
                best_grasp_ik, best_standoff_ik, best_turn_ik = candidate_grasp_ik, candidate_standoff_ik, candidate_turn_ik
                best_turn_fraction, best_depth = candidate_turn_fraction, depth
                best_radial, best_grasp_quat, best_grasp_center_world = radial, grasp_quat, grasp_center_world

        grasp_ik, standoff_ik, turn_ik = best_grasp_ik, best_standoff_ik, best_turn_ik
        if grasp_ik is None or standoff_ik is None or turn_ik is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": last_rejection,
                    "knob": knob.knob_body,
                    "grasp_height_margins_m": list(cfg.grasp_height_margins_m),
                    "radial_angle_samples": cfg.radial_angle_samples,
                    "pose_selection_attempts": cfg.pose_selection_attempts,
                    "tried_fractions": list(cfg.turn_fraction_fallbacks),
                },
            )
        radial0, grasp_quat, grasp_center_world = best_radial, best_grasp_quat, best_grasp_center_world
        grasp_pos = grasp_center_world + d * radial0
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

        # Only the descent (standoff -> grasp) goes into `approach_positions`
        # -- matching OpenDrawerSkill's exact convention: the turn leg
        # (grasp_q -> turned_q) is tracked separately by the adapter, only
        # after the gripper has actually closed on the knob (see that
        # module's execute()-comment for why -- driving the turn motion
        # before the gripper is closed never actually grips anything).
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
                "grasp_chord_width_m": target_width,
                "qpos_lo": qpos_lo,
                "qpos_hi": qpos_hi,
                "current_qpos": current_qpos,
                "target_qpos": target_qpos,
                "turn_on_qpos_threshold": threshold,
                "turn_fraction_used": turn_fraction_used,
                "finger_closed_qpos": finger_closed_qpos.tolist(),
            },
        )
