"""Planning-only ``CloseSkill``: push a drawer/cabinet level or a microwave
door shut, never grasping it.

Closing completes on contact -- unlike opening a drawer (which must hook a
handle to pull it) or turning a knob (which must grip its rim to rotate it),
there is nothing to catch and nothing to miss. A closed gripper (a rigid
fist) commanded straight through the fixture's closing path does the job, so
this skill carries none of ``OpenDrawerSkill``'s grasp-precision machinery
(approach tilts/rolls, per-candidate finger width, the "handle captured
between fingers" gate) -- there is no handle to capture.

Dispatches on the target's own joint type, discovered via
``articulation.py``:

- **Drawer/cabinet level** (slide joint): reuses ``open_drawer_skill.
  handle_grasp_pose`` as-is for the approach geometry (same measured pull
  axis, same handle location) -- the push path is simply the *reverse* of
  that approach, continued past the handle all the way to the joint's
  closed end, rather than stopping there to grasp.
- **Microwave door** (hinge joint): the gripper must follow the door's real
  swept arc, not a straight line -- see ``articulation.measure_door_close_
  arc``. Each arc waypoint is IK-solved in sequence, warm-started from the
  previous one, producing a joint-space path the adapter tracks exactly like
  the drawer's linear one.

Both branches return the same diagnostics/``plan.metadata`` shape
(``approach_positions`` standoff->first-contact, ``push_positions``
contact->closed) so ``LiberoCloseAdapter`` never needs to know which kind of
fixture it's driving.
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
from ..scene_query import SceneQuery
from .open_drawer_skill import _orientation_from_approach_and_reference, _solve_ik_robust, _unit, handle_grasp_pose
from .standoff_search import standoff_candidates
from ..types import PlannerFailureReason, SkillFailureReason, SkillResult


def _normalize(name: str) -> str:
    return str(name).strip().lower()


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
        cfg = self.config.close
        edge_resolution = self.config.planner.validity_resolution * float(
            np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)
        )
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

        contact_pos, _flat_contact_quat, pull_axis_world, _travel_m, _closing_axis = handle_grasp_pose(scene, drawer)
        _, _, qpos_closed, qpos_open = measure_pull_axis_and_travel(scene, drawer)
        qpos_current = float(scene.get_qpos([drawer.joint_name])[0])
        # A slide joint's qpos IS its linear displacement in meters (a MuJoCo
        # invariant, not an asset-specific assumption) -- so how far the
        # handle still has to travel to reach fully closed is exactly
        # (qpos_current - qpos_closed) along the measured pull axis, valid
        # regardless of how far open the drawer currently is.
        remaining_m = qpos_current - qpos_closed
        push_target_pos = contact_pos - remaining_m * pull_axis_world
        reference_axis_world = _unit(np.cross(np.array([0.0, 0.0, 1.0]), pull_axis_world))

        best = None
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
                push_ik = _solve_ik_robust(
                    self.ik_solver, scene, push_target_pos, contact_quat, contact_ik.q, self._rng,
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
                best = (standoff_ik, contact_ik, push_ik, contact_quat)
                break
            if best is not None:
                break

        if best is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": last_rejection, "drawer": drawer.drawer_body},
            )
        standoff_ik, contact_ik, push_ik, contact_quat = best

        plan = self._plan_free_space(start_q, standoff_ik.q, drawer.drawer_body, cfg.planning_attempts)
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

        if best is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": last_rejection, "door": door.door_body},
            )
        standoff_ik, contact_ik, arc_qs = best

        plan = self._plan_free_space(start_q, standoff_ik.q, door.door_body, cfg.planning_attempts)
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

    def _plan_free_space(self, start_q, goal_q, target_object_body: str, planning_attempts: int):
        # RRTConnect is stochastic -- see OpenDrawerSkill.execute()'s
        # identical retry loop for why repeated attempts help.
        plan = None
        for _ in range(planning_attempts):
            plan = self.planner.plan(
                start_q, [goal_q], phase=CollisionPhase.FREE_SPACE, target_object_body=target_object_body
            )
            if plan.success or plan.failure_reason not in (
                PlannerFailureReason.NO_PATH_FOUND,
                PlannerFailureReason.APPROXIMATE_SOLUTION_ONLY,
                PlannerFailureReason.TIMEOUT,
            ):
                break
        return plan
