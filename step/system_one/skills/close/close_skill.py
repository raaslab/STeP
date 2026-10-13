"""Planning-only ``CloseSkill``: shut a drawer/cabinet level or a microwave
door.

Closing completes on contact -- unlike opening a drawer (which must hook a
handle to pull it), there is nothing to catch and nothing to miss on the way
in. A microwave door closes as an open-palm push (see ``_execute_door`` and
``LiberoCloseAdapter``'s docstring): the gripper stays open, and contact is
made by the palm/wrist pushing straight through the door's swept arc.

A drawer/cabinet level tries **three** strategies, in order, each a fallback
for when the previous one can't even find a reachable approach (a strategy
that reaches but can't fully close reports that failure as-is, rather than
falling back silently past a real contact):

1. **Linear, no tilt, a single predefined point** (``_execute_drawer_linear``):
   keeps the arm's homed approach direction and tilt and goes to one fixed
   standoff/contact/push point, rather than searching many. Tried first
   because a wrist rotated into a tilted search frame tends to sweep the
   forearm into clutter next to the cabinet that a home-tilt approach never
   has to pass. Neither leg runs a collision check: the static checker is
   measurably too conservative both about the descent (IK non-uniqueness
   makes an equally valid posture look blocked) and about contact with
   whatever's resting inside the drawer during the push, which a real
   MPC-tracked push simply nudges aside rather than jamming against.
2. **Grasp-and-retract** (``_execute_drawer_grasp``): reuses
   ``open_drawer_skill.find_drawer_handle_grasp`` verbatim -- the same
   tilt/roll/offset/standoff search and narrowed-finger approach
   ``OpenDrawerSkill`` uses to grasp the handle to pull it open -- then
   retracts along *minus* the measured pull axis instead, dragging the
   drawer shut. A narrowed-finger grasp sweeps a smaller profile than an
   open palm needing flush contact, so it can reach handles the push can't.
3. **Open-palm push, tilted** (``_execute_drawer_push``, final fallback):
   approach with an open gripper along a tilted line and continue straight
   through the handle until the palm/wrist has pushed the drawer's front
   face flush to its closed end. Needs ``CollisionChecker``'s
   ``permitted_contact_geom_ids`` (previously unused by any caller), since
   ``_permitted()`` otherwise only ever allows fingerpad<->target contact,
   never palm<->target.

All three strategies (and the microwave door) return the same diagnostics/
``plan.metadata`` shape (``approach_positions`` standoff->first-contact,
``push_positions`` contact->closed) plus a ``"kind"`` diagnostic
(``"drawer_grasp"``, ``"drawer"``, or ``"door"``) so ``LiberoCloseAdapter``
knows whether to close the gripper on contact and release it before
retreating (grasp) or leave it open throughout (push/door).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...motion.articulation import (
    DrawerHandle,
    MicrowaveDoor,
    measure_door_close_arc,
    measure_pull_axis_and_travel,
    resolve_drawer,
    resolve_microwave_door,
)
from ...motion.collision import CollisionChecker, CollisionPhase
from ...motion.config import MotionConfig
from ..ik_utils import solve_ik_robust as _solve_ik_robust
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from ...motion.panda import FINGER_JOINT_NAMES
from ...motion.scene_query import SceneQuery, quat_mul_wxyz
from ..open_drawer.open_drawer_skill import (
    _orientation_from_approach_and_reference,
    _unit,
    find_drawer_handle_grasp,
    handle_grasp_pose,
)
from ..standoff_search import standoff_candidates
from ...motion.types import PlannerFailureReason, SkillFailureReason, SkillResult


@dataclass
class CloseConfig:
    """Tuning for ``CloseSkill``/``LiberoCloseAdapter`` -- closes a drawer
    (slide joint) or a microwave door (hinge joint) by pushing it shut with
    a closed gripper, never grasping it. Pushing has none of
    ``DrawerConfig``'s grasp-precision machinery (offsets, tilts, per-angle
    finger width) because a closed fist just needs to *contact* the fixture
    somewhere and keep moving through it -- there is no handle to miss."""

    approach_standoff_m: float = 0.08
    # Higher than every other skill's pose_selection_attempts (6) -- needed
    # for the microwave door specifically: its vertical hinge axis means
    # every approach reaches sideways across the robot's own base, and a
    # perturbed-seed DLS solve converges to a self-colliding elbow far more
    # often than not, so more restarts are needed to find a clean one.
    # Drawer closing doesn't need this itself but shares the field.
    pose_selection_attempts: int = 24
    standoff_distance_multipliers: tuple[float, ...] = (1.0, 2.5, 4.0)
    standoff_lateral_offsets_m: tuple[float, ...] = (0.05, 0.08, 0.12)
    standoff_lateral_directions: int = 8
    # Drawer branch only (see DrawerConfig.approach_tilts_rad): a purely
    # horizontal approach sweeps the forearm through whatever sits on the
    # table in front of the cabinet at handle height. Not used by the door
    # branch, whose approach direction comes from the arc's own tangent.
    approach_tilts_rad: tuple[float, ...] = (0.0, 0.35, 0.61, 0.87)
    # Drawer branch only -- mirrors DrawerConfig.pull_fraction_fallbacks
    # (tried largest/most-closed first): pushing all the way to the joint's
    # exact closed end can bring the palm into contact with whatever sits
    # flush around the drawer opening once shut (e.g. the drawer above it in
    # a stacked cabinet). A push that only gets most of the way still counts
    # as closed (see close_threshold_frac) and avoids that tight clearance.
    close_fraction_fallbacks: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.75, 0.6)
    # ``_execute_drawer_linear`` only: a fixed orientation (yaw about world
    # +Z, then roll about the resulting local +Z, then tilt about the
    # resulting local +X -- these three span the full orientation space)
    # plus a fixed height offset relative to the handle. All four are pinned
    # rather than swept, tuned by hand against real physics rollouts rather
    # than the static collision checker, which is too conservative for this
    # leg (see close_fraction_fallbacks). scripts/tune_close_drawer.py
    # exposes all four as CLI flags for further tuning.
    linear_yaws_deg: tuple[float, ...] = (90.0,)
    linear_roll_deg: float = 0.0
    linear_tilt_deg: float = -30.0
    linear_height_offsets_m: tuple[float, ...] = (0.0,)
    planning_attempts: int = 3
    # Number of Cartesian waypoints FK-sampled along a microwave door's
    # swept closing arc (see articulation.measure_door_close_arc) -- each is
    # IK-solved in sequence, warm-started from the previous one, to build the
    # joint-space reference the MPC tracks. Irrelevant for a drawer, whose
    # closing path is a straight line (2 endpoints, IK-solved directly).
    door_arc_waypoints: int = 10
    # Fraction of the joint's full qpos range still open, required to call
    # the fixture "closed" -- the mirror image of
    # DrawerConfig.open_threshold_frac, just checked in the other direction
    # (<= this counts as closed, vs. >= open_threshold_frac counts as open).
    # Looser than 0.0 because a real push rarely drives the joint to its
    # exact mechanical zero.
    close_threshold_frac: float = 0.15
    retreat_m: float = 0.06
    retreat_steps: int = 40
    return_home_steps: int = 60


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def _yaw_about_world_z(quat_wxyz: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a world-frame orientation by ``degrees`` about the world +Z
    axis (applied on the left) -- used by ``_execute_drawer_linear`` to turn
    the fixed predefined-point approach while keeping the arm's home tilt."""
    half = np.radians(float(degrees)) / 2.0
    yaw_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    return quat_mul_wxyz(yaw_quat, np.asarray(quat_wxyz, dtype=np.float64))


def _roll_about_local_z(quat_wxyz: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate an orientation by ``degrees`` about its own local +Z axis (the
    approach direction), applied on the right -- unlike the world-Z yaw
    above, this spins the fingers around the fixed approach direction
    without changing which way the palm faces."""
    half = np.radians(float(degrees)) / 2.0
    roll_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    return quat_mul_wxyz(np.asarray(quat_wxyz, dtype=np.float64), roll_quat)


def _tilt_about_local_x(quat_wxyz: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate an orientation by ``degrees`` about its own (post-yaw,
    post-roll) local +X axis, applied on the right -- the third independent
    rotation; composed after yaw and roll, the three span the full
    orientation space."""
    half = np.radians(float(degrees)) / 2.0
    tilt_quat = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
    return quat_mul_wxyz(np.asarray(quat_wxyz, dtype=np.float64), tilt_quat)


class CloseSkill:
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

    def _resolve_target(self, scene: SceneQuery, target_ref: str):
        """Returns ``("drawer", DrawerHandle)`` or ``("door", MicrowaveDoor)``.
        Tries whichever kind the name hints at first (LIBERO's microwave
        names always contain "micro"), only falling back to the other kind
        on failure -- guards against the resolvers' permissive "only one
        candidate exists" fallback matching the wrong kind of fixture."""
        prefer_door = "micro" in _normalize(target_ref)
        order = (
            [("door", resolve_microwave_door), ("drawer", resolve_drawer)]
            if prefer_door
            else [("drawer", resolve_drawer), ("door", resolve_microwave_door)]
        )
        last_exc: KeyError | None = None
        for kind, resolver in order:
            try:
                return kind, resolver(scene, target_ref)
            except KeyError as exc:
                last_exc = exc
        raise KeyError(str(last_exc))

    def execute(
        self,
        scene: SceneQuery,
        target_ref: str,
        start_q: np.ndarray,
        constraints: dict | None = None,
    ) -> SkillResult:
        constraints = constraints or {}
        try:
            kind, target = self._resolve_target(scene, target_ref)
        except KeyError as exc:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": str(exc)},
            )
        if kind == "drawer":
            return self._execute_drawer(scene, target, start_q)
        return self._execute_door(scene, target, start_q)

    # ------------------------------------------------------------ drawer

    def _execute_drawer(self, scene: SceneQuery, drawer: DrawerHandle, start_q: np.ndarray) -> SkillResult:
        edge_resolution = self.config.planner.validity_resolution * float(
            np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)
        )
        contact_pos, _flat_contact_quat, pull_axis_world, travel_m, _closing_axis = handle_grasp_pose(scene, drawer)
        _, _, qpos_closed, qpos_open = measure_pull_axis_and_travel(scene, drawer)
        qpos_current = float(scene.get_qpos([drawer.joint_name])[0])
        # NOT simply (qpos_current - qpos_closed): `qpos_open` isn't
        # guaranteed to be greater than `qpos_closed` -- a drawer's local
        # joint axis can point opposite to the world-measured
        # `pull_axis_world`. The qpos delta must be normalized into a
        # fraction of the full (qpos_open-qpos_closed) range before scaling
        # by travel_m/pull_axis_world, or this can push the drawer further
        # open instead of closed.
        frac_open = (
            (qpos_current - qpos_closed) / (qpos_open - qpos_closed)
            if abs(qpos_open - qpos_closed) > 1e-9
            else 0.0
        )

        linear_result = self._execute_drawer_linear(
            scene, drawer, start_q, contact_pos, pull_axis_world, travel_m, frac_open,
            qpos_closed, qpos_open, edge_resolution,
        )
        if linear_result is not None:
            return linear_result
        grasp_result = self._execute_drawer_grasp(
            scene, drawer, start_q, contact_pos, pull_axis_world, travel_m, frac_open,
            qpos_closed, qpos_open, edge_resolution,
        )
        if grasp_result is not None:
            return grasp_result
        return self._execute_drawer_push(
            scene, drawer, start_q, contact_pos, pull_axis_world, travel_m, frac_open,
            qpos_closed, qpos_open, edge_resolution,
        )

    def _execute_drawer_linear(
        self,
        scene: SceneQuery,
        drawer: DrawerHandle,
        start_q: np.ndarray,
        contact_pos: np.ndarray,
        pull_axis_world: np.ndarray,
        travel_m: float,
        frac_open: float,
        qpos_closed: float,
        qpos_open: float,
        _edge_resolution: float,
    ) -> SkillResult | None:
        """Approach and push with the SAME orientation the arm already has
        at ``start_q`` (its homed pose) -- no wrist rotation at all, tried
        before both other strategies. A wrist rotated into a tilted search
        frame tends to sweep the forearm into clutter next to the cabinet;
        holding the home orientation sidesteps that at the source.

        Deliberately not a search: a single predefined point in front of the
        drawer and a single push straight through it, at a fixed height
        above the handle. A standoff/lateral/yaw search was tried and made
        this worse -- it kept preferring degenerate points that only cleared
        the checker because they'd drifted clear of the drawer's face
        entirely, over the one point a real rollout confirmed actually
        works. Neither leg runs a collision check, for the same reason the
        static checker is too conservative here: about the descent itself
        (IK non-uniqueness can make an equally valid posture look blocked)
        and about contact with whatever's resting inside the drawer, which a
        real MPC-tracked push simply nudges aside. IK success is still
        required for both legs, just not a static checker's approval.

        Returns ``None`` (not a ``SkillResult``) only when the predefined
        point isn't even IK-reachable at all, telling the caller to fall
        back to the other strategies.
        """
        cfg = self.config.close
        # The arm's actual current orientation, not a recomputed home pose --
        # valid since the adapter always homes the arm before planning.
        _home_pos, home_quat = scene.site_pose(self.ik_solver.site_name)
        yaw_deg = cfg.linear_yaws_deg[0] if cfg.linear_yaws_deg else 0.0
        home_quat = _yaw_about_world_z(home_quat, yaw_deg)
        home_quat = _roll_about_local_z(home_quat, cfg.linear_roll_deg)
        home_quat = _tilt_about_local_x(home_quat, cfg.linear_tilt_deg)
        approach_world = -pull_axis_world  # straight in, no tilt at all
        height_offset_m = cfg.linear_height_offsets_m[0] if cfg.linear_height_offsets_m else 0.0
        aim_pos = contact_pos + np.array([0.0, 0.0, float(height_offset_m)])

        # One candidate: this point, right in front of the drawer. No
        # collision veto (see the method's docstring) -- IK success alone
        # is the requirement, not a static check's approval of the posture.
        standoff_position = aim_pos - cfg.approach_standoff_m * approach_world
        standoff_ik = _solve_ik_robust(
            self.ik_solver, scene, standoff_position, home_quat, start_q, self._rng,
            n_restarts=cfg.pose_selection_attempts,
        )
        if not standoff_ik.success:
            return None
        contact_ik = _solve_ik_robust(
            self.ik_solver, scene, aim_pos, home_quat, standoff_ik.q, self._rng,
            n_restarts=cfg.pose_selection_attempts,
        )
        if not contact_ik.success:
            return None

        # Try the most-closed target first, falling back only on outright
        # IK infeasibility (never on collision -- see the method's own
        # docstring for why that check is dropped for this leg).
        push_ik = None
        push_target_pos = None
        for fraction in cfg.close_fraction_fallbacks:
            push_target_frac = aim_pos - fraction * frac_open * travel_m * pull_axis_world
            candidate_ik = _solve_ik_robust(
                self.ik_solver, scene, push_target_frac, home_quat, contact_ik.q, self._rng,
                n_restarts=cfg.pose_selection_attempts,
            )
            if candidate_ik.success:
                push_ik = candidate_ik
                push_target_pos = push_target_frac
                break
        if push_ik is None:
            return None

        plan = self._plan_free_space(scene, start_q, standoff_ik.q, drawer.drawer_body, cfg.planning_attempts)
        if not plan.success:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": f"planning_failed:{plan.failure_reason}",
                    "drawer": drawer.drawer_body,
                    "kind": "drawer",
                },
            )

        plan.metadata["standoff_q"] = standoff_ik.q.tolist()
        plan.metadata["contact_q"] = contact_ik.q.tolist()
        plan.metadata["approach_positions"] = np.linspace(standoff_ik.q, contact_ik.q, 15).tolist()
        plan.metadata["push_positions"] = np.linspace(contact_ik.q, push_ik.q, 15).tolist()

        return SkillResult(
            success=True,
            selected_object=drawer.drawer_body,
            planned_trajectory=plan,
            diagnostics={
                "kind": "drawer",
                "target_body": drawer.drawer_body,
                "joint_name": drawer.joint_name,
                "qpos_closed": qpos_closed,
                "qpos_open_reference": qpos_open,
                "contact_position": aim_pos.tolist(),
                "contact_quat_wxyz": home_quat.tolist(),
                "final_target_position": push_target_pos.tolist(),
            },
        )

    def _execute_drawer_grasp(
        self,
        scene: SceneQuery,
        drawer: DrawerHandle,
        start_q: np.ndarray,
        contact_pos: np.ndarray,
        pull_axis_world: np.ndarray,
        travel_m: float,
        frac_open: float,
        qpos_closed: float,
        qpos_open: float,
        edge_resolution: float,
    ) -> SkillResult | None:
        """Grasp the handle exactly like ``OpenDrawerSkill`` does, then
        retract along *minus* the measured pull axis instead of plus it --
        dragging the drawer shut instead of pulling it further open.

        Returns ``None`` (not a ``SkillResult``) when no reachable,
        collision-free, closure-safe grasp exists at all, telling the caller
        to fall back to the push strategy untouched. Returns a real
        ``SkillResult`` once a grasp was found, since a grasped-but-can't-
        retract failure is distinct and worth reporting, not falling back
        past.
        """
        candidate, grasp_rejection = find_drawer_handle_grasp(
            scene, drawer, contact_pos, pull_axis_world, start_q,
            self.ik_solver, self.collision_checker, self.config.drawer, edge_resolution, self._rng,
        )
        if candidate is None:
            return None

        cfg = self.config.close
        target_geom_ids = set(scene.body_geom_ids(drawer.drawer_body))
        best = None
        best_fallback = None  # (open_frac_reached, retract_ik, retract_target)
        last_rejection = "no_retract_attempted"
        for fraction in cfg.close_fraction_fallbacks:
            retract_target = candidate.grasp_pos - fraction * frac_open * travel_m * pull_axis_world
            retract_ik = _solve_ik_robust(
                self.ik_solver, scene, retract_target, candidate.grasp_quat, candidate.grasp_ik.q, self._rng,
                n_restarts=cfg.pose_selection_attempts,
            )
            if not retract_ik.success:
                last_rejection = "retract_target_ik_failed"
                continue
            # Validated with the CLOSED-finger geometry, not check_edge's
            # open-gripper default -- the real motion keeps the fingers
            # closed on the handle for the whole retract, so a configuration
            # that looks clear open-handed could still drag it into a
            # neighbor.
            retract_check = self.collision_checker.check_edge(
                candidate.grasp_ik.q, retract_ik.q, resolution=edge_resolution,
                phase=CollisionPhase.TERMINAL_APPROACH, target_object_body=drawer.drawer_body,
                finger_joint_names=FINGER_JOINT_NAMES, finger_qpos=candidate.finger_closed_qpos,
                permitted_contact_geom_ids=target_geom_ids,
            )
            if retract_check.in_collision:
                last_rejection = "retract_path_in_collision"
                continue
            open_frac_reached = frac_open * (1.0 - fraction)
            if best_fallback is None or open_frac_reached < best_fallback[0]:
                best_fallback = (open_frac_reached, retract_ik, retract_target)
            if open_frac_reached <= cfg.close_threshold_frac:
                best = (retract_ik, retract_target)
                break

        if best is None and best_fallback is not None:
            _, retract_ik, retract_target = best_fallback
            best = (retract_ik, retract_target)

        if best is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": last_rejection,
                    "drawer": drawer.drawer_body,
                    "kind": "drawer_grasp",
                    "grasp_rejection": grasp_rejection,
                },
            )
        retract_ik, retract_target = best

        plan = self._plan_free_space(scene, start_q, candidate.standoff_ik.q, drawer.drawer_body, cfg.planning_attempts)
        if not plan.success:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": f"planning_failed:{plan.failure_reason}",
                    "drawer": drawer.drawer_body,
                    "kind": "drawer_grasp",
                },
            )

        plan.metadata["standoff_q"] = candidate.standoff_ik.q.tolist()
        plan.metadata["contact_q"] = candidate.grasp_ik.q.tolist()
        plan.metadata["approach_positions"] = np.linspace(candidate.standoff_ik.q, candidate.grasp_ik.q, 15).tolist()
        plan.metadata["push_positions"] = np.linspace(candidate.grasp_ik.q, retract_ik.q, 15).tolist()

        return SkillResult(
            success=True,
            selected_object=drawer.drawer_body,
            planned_trajectory=plan,
            diagnostics={
                "kind": "drawer_grasp",
                "target_body": drawer.drawer_body,
                "joint_name": drawer.joint_name,
                "qpos_closed": qpos_closed,
                "qpos_open_reference": qpos_open,
                "contact_position": candidate.grasp_pos.tolist(),
                "contact_quat_wxyz": candidate.grasp_quat.tolist(),
                "final_target_position": retract_target.tolist(),
                "finger_closed_qpos": candidate.finger_closed_qpos.tolist(),
                "approach_tilt_rad": candidate.tilt_rad,
                "approach_roll_rad": candidate.roll_rad,
            },
        )

    def _execute_drawer_push(
        self,
        scene: SceneQuery,
        drawer: DrawerHandle,
        start_q: np.ndarray,
        contact_pos: np.ndarray,
        pull_axis_world: np.ndarray,
        travel_m: float,
        frac_open: float,
        qpos_closed: float,
        qpos_open: float,
        edge_resolution: float,
    ) -> SkillResult:
        cfg = self.config.close
        # A push makes contact with the PALM, not the fingerpads --
        # CollisionChecker._permitted() only ever allows fingerpad<->target
        # contact, so permitted_contact_geom_ids explicitly allows the whole
        # gripper to press against the target while the forearm/other robot
        # links still count any contact as a real violation.
        target_geom_ids = set(scene.body_geom_ids(drawer.drawer_body))
        reference_axis_world = _unit(np.cross(np.array([0.0, 0.0, 1.0]), pull_axis_world))

        best = None
        best_fallback = None  # (open_frac_reached, standoff_ik, contact_ik, push_ik, contact_quat)
        last_rejection = "no_ik_attempted"
        for tilt_rad in cfg.approach_tilts_rad:
            # Same direction the grasp approach uses, brought in from above
            # at steeper tilts -- closing continues straight through that
            # direction rather than reversing, unlike opening's pull leg.
            approach_world = _unit(
                -pull_axis_world * float(np.cos(tilt_rad)) + np.array([0.0, 0.0, -1.0]) * float(np.sin(tilt_rad))
            )
            contact_quat, _ = _orientation_from_approach_and_reference(approach_world, reference_axis_world)

            for standoff_position in standoff_candidates(
                contact_pos,
                approach_world,
                cfg.approach_standoff_m,
                cfg.standoff_distance_multipliers,
                cfg.standoff_lateral_offsets_m,
                cfg.standoff_lateral_directions,
            ):
                standoff_ik = _solve_ik_robust(
                    self.ik_solver, scene, standoff_position, contact_quat, start_q, self._rng,
                    n_restarts=cfg.pose_selection_attempts,
                )
                if not standoff_ik.success:
                    last_rejection = "approach_standoff_ik_failed"
                    continue
                contact_ik = _solve_ik_robust(
                    self.ik_solver, scene, contact_pos, contact_quat, standoff_ik.q, self._rng,
                    n_restarts=cfg.pose_selection_attempts,
                )
                if not contact_ik.success:
                    last_rejection = "contact_ik_failed"
                    continue
                descent_check = self.collision_checker.check_edge(
                    standoff_ik.q, contact_ik.q, resolution=edge_resolution,
                    phase=CollisionPhase.TERMINAL_APPROACH, target_object_body=drawer.drawer_body,
                    permitted_contact_geom_ids=target_geom_ids,
                )
                if descent_check.in_collision:
                    last_rejection = "approach_descent_in_collision"
                    continue
                # Tried largest (most-closed) fraction first -- see
                # CloseConfig.close_fraction_fallbacks's own comment for why
                # a full push isn't always the reachable one.
                push_found = False
                for fraction in cfg.close_fraction_fallbacks:
                    push_target_frac = contact_pos - fraction * frac_open * travel_m * pull_axis_world
                    push_ik = _solve_ik_robust(
                        self.ik_solver, scene, push_target_frac, contact_quat, contact_ik.q, self._rng,
                        n_restarts=cfg.pose_selection_attempts,
                    )
                    if not push_ik.success:
                        last_rejection = "push_target_ik_failed"
                        continue
                    push_check = self.collision_checker.check_edge(
                        contact_ik.q, push_ik.q, resolution=edge_resolution,
                        phase=CollisionPhase.TERMINAL_APPROACH, target_object_body=drawer.drawer_body,
                        permitted_contact_geom_ids=target_geom_ids,
                    )
                    if push_check.in_collision:
                        last_rejection = "push_path_in_collision"
                        continue
                    open_frac_reached = frac_open * (1.0 - fraction)
                    if best_fallback is None or open_frac_reached < best_fallback[0]:
                        best_fallback = (
                            open_frac_reached, standoff_ik, contact_ik, push_ik, contact_quat, push_target_frac,
                        )
                    if open_frac_reached <= cfg.close_threshold_frac:
                        best = (standoff_ik, contact_ik, push_ik, contact_quat, push_target_frac)
                        push_found = True
                    break
                if push_found:
                    break
            if best is not None:
                break

        if best is None and best_fallback is not None:
            # Same "best feasible wins" fallback as the door branch -- the
            # adapter's own post-execution structural check still gates on
            # close_threshold_frac, so this never ships a false "success".
            _, standoff_ik, contact_ik, push_ik, contact_quat, push_target_frac = best_fallback
            best = (standoff_ik, contact_ik, push_ik, contact_quat, push_target_frac)

        if best is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": last_rejection, "drawer": drawer.drawer_body},
            )
        standoff_ik, contact_ik, push_ik, contact_quat, push_target_pos = best

        plan = self._plan_free_space(scene, start_q, standoff_ik.q, drawer.drawer_body, cfg.planning_attempts)
        if not plan.success:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": f"planning_failed:{plan.failure_reason}", "drawer": drawer.drawer_body},
            )

        plan.metadata["standoff_q"] = standoff_ik.q.tolist()
        plan.metadata["contact_q"] = contact_ik.q.tolist()
        plan.metadata["approach_positions"] = np.linspace(standoff_ik.q, contact_ik.q, 15).tolist()
        plan.metadata["push_positions"] = np.linspace(contact_ik.q, push_ik.q, 15).tolist()

        return SkillResult(
            success=True,
            selected_object=drawer.drawer_body,
            planned_trajectory=plan,
            diagnostics={
                "kind": "drawer",
                "target_body": drawer.drawer_body,
                "joint_name": drawer.joint_name,
                "qpos_closed": qpos_closed,
                "qpos_open_reference": qpos_open,
                "contact_position": contact_pos.tolist(),
                "contact_quat_wxyz": contact_quat.tolist(),
                "final_target_position": push_target_pos.tolist(),
            },
        )

    # ------------------------------------------------------------ microwave door

    def _execute_door(self, scene: SceneQuery, door: MicrowaveDoor, start_q: np.ndarray) -> SkillResult:
        cfg = self.config.close
        edge_resolution = self.config.planner.validity_resolution * float(
            np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)
        )

        # See _execute_drawer_push's comment: a push contacts with the palm,
        # which CollisionChecker._permitted() never allows on its own.
        target_geom_ids = set(scene.body_geom_ids(door.door_body))

        waypoints_world, qpos_current, qpos_closed, axis_world, _pivot_world = measure_door_close_arc(
            scene, door, num_waypoints=cfg.door_arc_waypoints
        )
        lo, hi = scene.joint_qpos_range([door.joint_name])
        lo, hi = float(lo[0]), float(hi[0])
        qpos_open_reference = hi if qpos_closed == lo else lo
        # Same sampling measure_door_close_arc used internally -- needed
        # here to know how "closed" the door would be if the chain below
        # only makes it partway along the arc.
        qpos_samples = np.linspace(qpos_current, qpos_closed, len(waypoints_world))
        denom = qpos_open_reference - qpos_closed

        # Orientation at each waypoint: approach axis along the arc's own
        # tangent direction (the way the door is being pushed at that point),
        # reference axis fixed to the hinge axis -- mirrors
        # open_drawer_skill's own approach/reference convention.
        quats = []
        for i in range(len(waypoints_world)):
            if i < len(waypoints_world) - 1:
                tangent = waypoints_world[i + 1] - waypoints_world[i]
            else:
                tangent = waypoints_world[i] - waypoints_world[i - 1]
            approach_world = _unit(tangent)
            quat, _ = _orientation_from_approach_and_reference(approach_world, axis_world)
            quats.append(quat)
        # Approach direction for the standoff search: the same tangent the
        # first waypoint uses, so the arm arrives already moving the way it
        # will immediately continue pushing.
        approach_world0 = _unit(waypoints_world[1] - waypoints_world[0]) if len(waypoints_world) > 1 else axis_world

        best = None
        best_fallback = None  # (open_frac_reached, standoff_ik, contact_ik, arc_qs) -- longest prefix seen so far
        last_rejection = "no_ik_attempted"
        for standoff_position in standoff_candidates(
            waypoints_world[0],
            approach_world0,
            cfg.approach_standoff_m,
            cfg.standoff_distance_multipliers,
            cfg.standoff_lateral_offsets_m,
            cfg.standoff_lateral_directions,
        ):
            standoff_ik = _solve_ik_robust(
                self.ik_solver, scene, standoff_position, quats[0], start_q, self._rng,
                n_restarts=cfg.pose_selection_attempts,
            )
            if not standoff_ik.success:
                last_rejection = "approach_standoff_ik_failed"
                continue
            contact_ik = _solve_ik_robust(
                self.ik_solver, scene, waypoints_world[0], quats[0], standoff_ik.q, self._rng,
                n_restarts=cfg.pose_selection_attempts,
            )
            if not contact_ik.success:
                last_rejection = "contact_ik_failed"
                continue
            descent_check = self.collision_checker.check_edge(
                standoff_ik.q, contact_ik.q, resolution=edge_resolution,
                phase=CollisionPhase.TERMINAL_APPROACH, target_object_body=door.door_body,
                permitted_contact_geom_ids=target_geom_ids,
            )
            if descent_check.in_collision:
                last_rejection = "approach_descent_in_collision"
                continue

            # Chain as much of the arc as is actually IK/collision feasible,
            # same "take the best feasible" shape as
            # DrawerConfig.pull_fraction_fallbacks -- requiring the full arc
            # is a much tighter kinematic constraint than a drawer's short
            # straight pull, so a partial prefix that gets the door past
            # `close_threshold_frac` counts as a real close, not a compromise.
            arc_qs = [contact_ik.q]
            reached_idx = 0
            for i in range(1, len(waypoints_world)):
                wp_ik = _solve_ik_robust(
                    self.ik_solver, scene, waypoints_world[i], quats[i], arc_qs[-1], self._rng,
                    n_restarts=cfg.pose_selection_attempts,
                )
                if not wp_ik.success:
                    last_rejection = f"arc_waypoint_{i}_ik_failed"
                    break
                wp_check = self.collision_checker.check_edge(
                    arc_qs[-1], wp_ik.q, resolution=edge_resolution,
                    phase=CollisionPhase.TERMINAL_APPROACH, target_object_body=door.door_body,
                    permitted_contact_geom_ids=target_geom_ids,
                )
                if wp_check.in_collision:
                    last_rejection = f"arc_waypoint_{i}_in_collision"
                    break
                arc_qs.append(wp_ik.q)
                reached_idx = i

            open_frac_reached = (
                (qpos_samples[reached_idx] - qpos_closed) / denom if abs(denom) > 1e-9 else 0.0
            )
            if reached_idx > 0 and (
                best_fallback is None or open_frac_reached < best_fallback[0]
            ):
                best_fallback = (open_frac_reached, standoff_ik, contact_ik, arc_qs)
            ok = open_frac_reached <= cfg.close_threshold_frac
            if not ok:
                continue
            best = (standoff_ik, contact_ik, arc_qs)
            break

        if best is None and best_fallback is not None:
            # No candidate reached the close threshold outright, but some
            # chain got partway there -- not a silent downgrade, since the
            # adapter's structural success check still gates on
            # close_threshold_frac after execution.
            _, standoff_ik, contact_ik, arc_qs = best_fallback
            best = (standoff_ik, contact_ik, arc_qs)

        if best is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": last_rejection, "door": door.door_body},
            )
        standoff_ik, contact_ik, arc_qs = best

        plan = self._plan_free_space(scene, start_q, standoff_ik.q, door.door_body, cfg.planning_attempts)
        if not plan.success:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": f"planning_failed:{plan.failure_reason}", "door": door.door_body},
            )

        plan.metadata["standoff_q"] = standoff_ik.q.tolist()
        plan.metadata["contact_q"] = contact_ik.q.tolist()
        plan.metadata["approach_positions"] = np.linspace(standoff_ik.q, contact_ik.q, 15).tolist()
        plan.metadata["push_positions"] = [q.tolist() for q in arc_qs]

        return SkillResult(
            success=True,
            selected_object=door.door_body,
            planned_trajectory=plan,
            diagnostics={
                "kind": "door",
                "target_body": door.door_body,
                "joint_name": door.joint_name,
                "qpos_closed": qpos_closed,
                "qpos_open_reference": qpos_open_reference,
                "contact_position": waypoints_world[0].tolist(),
                "contact_quat_wxyz": quats[0].tolist(),
                "final_target_position": waypoints_world[-1].tolist(),
            },
        )

    # ------------------------------------------------------------ shared

    def _plan_free_space(self, scene: SceneQuery, start_q, goal_q, target_object_body: str, planning_attempts: int):
        # RRTConnect is stochastic -- see OpenDrawerSkill.execute()'s
        # identical retry loop for why repeated attempts help.
        #
        # The fixture being closed isn't a neutral obstacle to route around:
        # an open drawer's face already sticks out right next to the
        # standoff, and `target_object_body` alone only exempts fingerpad
        # contact, not the palm/wrist grazing it. Permit contact with the
        # fixture's own geoms wholesale here, or FREE_SPACE reach can be
        # blocked by the very fixture the arm is about to press against.
        permitted_contact_geom_ids = set(scene.body_geom_ids(target_object_body))
        plan = None
        for _ in range(planning_attempts):
            plan = self.planner.plan(
                start_q, [goal_q], phase=CollisionPhase.FREE_SPACE, target_object_body=target_object_body,
                permitted_contact_geom_ids=permitted_contact_geom_ids,
            )
            if plan.success or plan.failure_reason not in (
                PlannerFailureReason.NO_PATH_FOUND,
                PlannerFailureReason.APPROXIMATE_SOLUTION_ONLY,
                PlannerFailureReason.TIMEOUT,
            ):
                break
        return plan
