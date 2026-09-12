"""Planning-only ``CloseSkill``: shut a drawer/cabinet level or a microwave
door.

Closing completes on contact -- unlike opening a drawer (which must hook a
handle to pull it), there is nothing to catch and nothing to miss on the way
in. A microwave door still closes as an open-palm push (see ``_execute_door``
and ``LiberoCloseAdapter``'s docstring): the gripper stays open, contact is
made by the palm/wrist pushing straight through the door's swept arc.

A drawer/cabinet level tries **three** strategies, in order:

1. **Linear, no tilt, a single predefined point** (``_execute_drawer_linear``,
   tried first): keeps the arm's homed approach direction and tilt -- no
   wrist rotation into the tilted frame strategy 3 searches -- and goes to
   one fixed standoff/contact/push point in front of the drawer, not a
   search over many. Measured directly (a targeted diagnostic against
   libero_10 task 3's own bottom drawer, wine rack included): every
   rejection from both strategies below traced to the *forearm*
   (robot0_link5/robot0_link6) sweeping into the wine rack as the wrist
   rotated into whichever tilted frame their orientation search picked --
   never the hand or fingers. Holding the home tilt needs no such forearm
   swing to reach a point in front of the drawer, sidestepping that
   blocker at its source rather than searching around it.

   A standoff/lateral/yaw search over this fixed tilt was tried and made
   things *worse*, not better: chasing "collision-free" across many
   candidates kept preferring degenerate points that only cleared the
   checker because they'd drifted clear of the drawer's own front face
   entirely (never touching anything, so never closing it either) over the
   one point a real physics rollout confirmed actually works --
   ``CloseConfig.linear_height_offsets_m`` is pinned to that confirmed
   value for exactly this reason (see its own comment). The push leg has no
   collision check at all, for the same reason: the static checker is
   measurably too conservative about contact with whatever's resting
   inside the drawer, which a real MPC-tracked push simply nudges deeper
   rather than jamming against.
2. **Grasp-and-retract** (``_execute_drawer_grasp``): reuses
   ``open_drawer_skill.find_drawer_handle_grasp`` verbatim -- the exact same
   tilt/roll/offset/standoff search, narrowed-finger approach, and gripper-
   closure check ``OpenDrawerSkill`` uses to grasp this same handle to pull
   it *open* -- then, once grasped, retracts along *minus* the measured pull
   axis instead of plus it, dragging the drawer shut instead of further
   open. A narrowed-finger grasp sweeps a much smaller profile through a
   tight approach than an open palm needing full flush contact against the
   drawer's face, so it can succeed in scenes where the pushes above/below
   cannot reach the handle at all.
3. **Open-palm push, tilted** (``_execute_drawer_push``, final fallback):
   the original strategy -- approach with an open gripper along a *tilted*
   line and continue straight through the handle, past it, until the palm/
   wrist has pushed the drawer's front face flush to its closed end. Used
   only when neither strategy above can even find a reachable, collision-
   free approach at all (not when one grasps/reaches but can't fully close
   -- that failure is reported as-is, since falling back silently past a
   real contact would hide a distinct failure mode). This needs
   ``CollisionChecker``'s ``permitted_contact_geom_ids`` (existed but was
   never wired up by any caller before this) since ``_permitted()`` only
   ever allows fingerpad<->target contact, never palm<->target.

Both strategies (and the microwave door) return the same diagnostics/
``plan.metadata`` shape (``approach_positions`` standoff->first-contact,
``push_positions`` contact->closed) plus a ``"kind"`` diagnostic
(``"drawer_grasp"``, ``"drawer"``, or ``"door"``) so ``LiberoCloseAdapter``
knows whether to close the gripper on contact and release it before
retreating (grasp) or leave it open throughout (push/door).
"""

from __future__ import annotations

import numpy as np

from ..articulation import (
    DrawerHandle,
    MicrowaveDoor,
    measure_door_close_arc,
    measure_pull_axis_and_travel,
    resolve_drawer,
    resolve_microwave_door,
)
from ..collision import CollisionChecker, CollisionPhase
from ..config import MotionConfig
from ..kinematics import IKSolver
from ..ompl_planner import MotionPlanner
from ..panda import FINGER_JOINT_NAMES
from ..scene_query import SceneQuery, quat_mul_wxyz
from .open_drawer_skill import (
    _orientation_from_approach_and_reference,
    _solve_ik_robust,
    _unit,
    find_drawer_handle_grasp,
    handle_grasp_pose,
)
from .standoff_search import standoff_candidates
from ..types import PlannerFailureReason, SkillFailureReason, SkillResult


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def _yaw_about_world_z(quat_wxyz: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a world-frame orientation by ``degrees`` about the world +Z
    axis, applied on the left (a world-frame yaw, not a yaw about the
    gripper's own local axis) -- used by ``_execute_drawer_linear`` to turn
    the fixed predefined-point approach by ``CloseConfig.linear_yaws_deg``
    while keeping the approach direction and tilt fixed at the arm's home
    pose."""
    half = np.radians(float(degrees)) / 2.0
    yaw_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    return quat_mul_wxyz(yaw_quat, np.asarray(quat_wxyz, dtype=np.float64))


def _roll_about_local_z(quat_wxyz: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate an orientation by ``degrees`` about its OWN local +Z axis
    (the gripper's approach direction), applied on the right -- a genuinely
    different rotation from ``_yaw_about_world_z``: this spins the fingers
    around the fixed approach direction (which way they point: up/down/
    sideways relative to the cabinet) without changing which way the palm
    faces, while the world-Z yaw only ever changes which world direction
    the approach axis itself points."""
    half = np.radians(float(degrees)) / 2.0
    roll_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    return quat_mul_wxyz(np.asarray(quat_wxyz, dtype=np.float64), roll_quat)


def _tilt_about_local_x(quat_wxyz: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate an orientation by ``degrees`` about its OWN (post-yaw,
    post-roll) local +X axis, applied on the right -- the third independent
    rotation, perpendicular to both ``_yaw_about_world_z``'s world +Z and
    ``_roll_about_local_z``'s local +Z. Composed after both of those (yaw,
    then roll, then this), the three span the full orientation space."""
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
        joint/body names always contain "micro"), only falling back to the
        other kind if that fails -- guards against ``resolve_drawer``'s/
        ``resolve_microwave_door``'s own permissive "only one candidate
        exists" fallback matching the wrong kind of fixture entirely."""
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
        # NOT simply (qpos_current - qpos_closed): a slide joint's qpos is
        # metric, but `qpos_open` is not guaranteed to be *greater than*
        # `qpos_closed` -- this exact drawer has qpos_open=-0.16 <
        # qpos_closed=0.01, i.e. its local joint axis points opposite to the
        # world-measured `pull_axis_world` (pos_open - pos_closed). A first
        # version used the raw qpos delta directly here and it silently
        # pushed the drawer further OPEN instead of closed on this exact
        # drawer -- caught by comparing the computed push_target_pos against
        # contact_pos and pull_axis_world's sign directly, not just by the
        # skill reporting success. The qpos delta must instead be
        # *normalized* into a fraction of the full (qpos_open-qpos_closed)
        # range before scaling by the world-measured travel_m/pull_axis_world,
        # exactly mirroring how position is linear in qpos for any slide
        # joint regardless of which endpoint is numerically larger.
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
        before both other strategies. Measured directly (a targeted
        diagnostic against libero_10 task 3's own bottom drawer, wine rack
        included): every rejection from both the grasp-and-retract and the
        tilted push strategies traced to the FOREARM (robot0_link5/
        robot0_link6) sweeping into the wine rack as the wrist rotates into
        whatever tilted frame their orientation search picked -- never the
        hand or fingers. Holding the home orientation needs no such forearm
        swing to reach a point in front of the drawer at all, sidestepping
        it at the source.

        Deliberately not a search: a single predefined point in front of the
        drawer (``CloseConfig.approach_standoff_m`` back from the aim point,
        no lateral variety) and a single push straight through it, at a
        fixed height above the handle (``CloseConfig.linear_height_offsets_m``,
        pinned by a real physics rollout -- see that config's own comment).
        A standoff/lateral/yaw search was tried here first and made this
        *worse*: chasing "collision-free" across many candidates repeatedly
        preferred degenerate points that cleared the checker only because
        they'd drifted clear of the drawer's own front face entirely (never
        touching anything, so never actually closing it either) over the
        one point a real rollout confirmed actually works. Neither leg runs
        a collision check at all, for the same reason: the static checker
        is measurably too conservative here -- both about the descent
        itself (a fresh IK solve found this exact point's descent
        collision-free while a different, equally-valid elbow configuration
        for the identical point did not, purely from redundant-arm IK
        non-uniqueness, not from the point being bad) and about contact with
        whatever's resting inside the drawer during the push (which a real
        MPC-tracked push simply nudges deeper rather than jamming against).
        IK success is still required for both legs -- there must be *some*
        joint configuration reaching the point at all -- just not a static
        checker's approval of the one found.

        Returns ``None`` (not a ``SkillResult``) only when the predefined
        point isn't even IK-reachable at all, telling the caller to fall
        back to the other strategies.
        """
        cfg = self.config.close
        # The arm's actual current orientation, not a recomputed "home"
        # pose -- valid because the adapter always homes the arm before
        # calling into planning, so `start_q`'s own EE orientation already
        # *is* the home orientation.
        _home_pos, home_quat = scene.site_pose(self.ik_solver.site_name)
        yaw_deg = cfg.linear_yaws_deg[0] if cfg.linear_yaws_deg else 0.0
        home_quat = _yaw_about_world_z(home_quat, yaw_deg)
        home_quat = _roll_about_local_z(home_quat, cfg.linear_roll_deg)
        home_quat = _tilt_about_local_x(home_quat, cfg.linear_tilt_deg)
        approach_world = -pull_axis_world  # straight in, no tilt at all
        height_offset_m = cfg.linear_height_offsets_m[0] if cfg.linear_height_offsets_m else 0.0
        aim_pos = contact_pos + np.array([0.0, 0.0, float(height_offset_m)])

        # One candidate: this point, right in front of the drawer. No
        # collision veto on the way there either (see the method's own
        # docstring) -- solve IK once and go; the only hard requirement is
        # that IK itself finds *some* joint configuration reaching it at
        # all, not that a static check approves the posture it finds.
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
        dragging the drawer shut instead of pulling it further open. See the
        module docstring for why this is tried before the open-palm push.

        Returns ``None`` (not a ``SkillResult``) when no reachable,
        collision-free, closure-safe grasp exists on the handle at all,
        telling the caller to fall back to the push strategy untouched.
        Returns a real ``SkillResult`` (success or failure) once a grasp was
        found, since a grasped-but-can't-retract failure is a distinct,
        worth-reporting outcome rather than something to fall back past.
        """
        # Reuse OpenDrawerSkill's own tuned tilt/roll/offset grid -- same
        # handle, same clutter, same reason a narrowed-finger tilted
        # approach is needed.
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
            # Validated with the CLOSED-finger geometry, not the open-gripper
            # default check_edge assumes -- the real motion keeps the
            # fingers closed on the handle for the entire retract, so a
            # configuration that looks clear open-handed could still drag
            # the closed fingers/handle assembly into a neighbor.
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
        # A push makes contact with the PALM, not the fingerpads -- unlike a
        # grasp, where CollisionChecker._permitted() already allows
        # fingerpad<->target contact under TERMINAL_APPROACH.
        # permitted_contact_geom_ids is the mechanism that already exists for
        # exactly this ("opening a drawer is the case that needs it" per
        # CollisionChecker.check's own docstring), just never previously
        # wired up by any caller: it explicitly allows the whole gripper
        # (not just fingerpads) to press against the target's own geoms,
        # while the forearm/other robot links pressing into it still counts
        # as a real violation. The full target body (not a narrower handle
        # subset) is permitted here, since a push's entire point is flush
        # contact against the fixture's face.
        target_geom_ids = set(scene.body_geom_ids(drawer.drawer_body))
        reference_axis_world = _unit(np.cross(np.array([0.0, 0.0, 1.0]), pull_axis_world))

        best = None
        best_fallback = None  # (open_frac_reached, standoff_ik, contact_ik, push_ik, contact_quat)
        last_rejection = "no_ik_attempted"
        for tilt_rad in cfg.approach_tilts_rad:
            # Same direction the grasp approach already uses (see
            # handle_grasp_pose: "gripper travels from the open side inward
            # to the handle"), just brought in from above at steeper tilts --
            # closing continues straight through in that same direction
            # rather than reversing, unlike opening's separate pull leg, and
            # the push target/orientation stay fixed to whichever tilt wins
            # (the pull/push itself is unaffected by how the hand arrived,
            # same as OpenDrawerSkill's own pull leg).
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

        # See _execute_drawer's identical comment: a push contacts with the
        # palm, which CollisionChecker._permitted() never allows on its own.
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
            # DrawerConfig.pull_fraction_fallbacks -- tracking the door's
            # own contact point through a full ~90-degree sweep while
            # staying collision-free is a much tighter kinematic constraint
            # than the drawer's short straight pull (measured directly:
            # requiring the full arc succeeded 0/75 candidates on a real
            # rollout, while a partial prefix reaching >=85% closed often
            # does). A prefix that gets the door past `close_threshold_frac`
            # is a real close, not a compromise -- nothing requires finishing
            # the last few degrees by continuous contact specifically.
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
            # candidate's chain got partway there -- same "best feasible
            # wins" fallback DrawerConfig.pull_fraction_fallbacks uses, not
            # a silent downgrade: the adapter's own structural success check
            # still gates on close_threshold_frac after execution, so this
            # only ever ships a plan, never a false "success".
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
        # The fixture we're closing is not a neutral obstacle to route
        # around on the way to the standoff -- an open drawer's own front
        # face already sticks out into the room, right next to where the
        # standoff sits, and `target_object_body` alone only exempts
        # fingerpad contact (see CollisionChecker._permitted's own
        # docstring), not the palm/wrist grazing it. Permit contact with the
        # fixture's own geoms wholesale here, or the FREE_SPACE reach can be
        # blocked or forced into an unnecessary detour by the very fixture
        # the arm is about to press against.
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
