"""Planning-only ``TransportSkill``: move the currently-attached object to a
hover point above the goal container/support, planning a collision-free
path that treats the carried object as part of the robot's own collision
geometry (via its sphere approximation, see ``attached_object.py``/
``collision.py``) -- not just the bare gripper.

This is the same hover-leg motion ``PlaceSkill`` already plans internally on
the way to its own descent, exposed here as its own subtask instead of only
running as a step inside ``place``. It exists because this codebase's
*legacy* stage-machine transport backend has no notion of the carried
object's own footprint -- a single fixed carry height, no attached-object
collision checking -- and can drive a grasped object straight into an
obstacle reaching for a goal above it. ``TransportSkill`` reuses
``PlaceSkill``'s own floor/rim measurement so the two skills agree on where
"over the goal" is, and stops there -- the final descent-and-release is left
to a subsequent ``place`` subtask, matching the OverGoal vs Placed/Released
predicate split already used in STL formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...motion.collision import CollisionChecker, CollisionPhase, required_hand_pose_for_object_pose
from ...motion.config import MotionConfig
from ..ik_utils import escape_start_collision, yaw_quat_wxyz
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from ...motion.panda import ROBOT_BODIES
from ...motion.scene_query import SceneQuery, quat_mul_wxyz
from ...motion.types import AttachedObjectState, PlannerFailureReason, SkillFailureReason, SkillResult
from ..place.place_skill import _container_floor_and_rim_z, _support_floor_and_rim_z


@dataclass
class TransportConfig:
    """Tuning for ``TransportSkill``/``LiberoTransportAdapter`` -- moves the
    currently-attached object to a hover point above the goal container/
    support, the same target ``PlaceSkill``'s own internal hover leg plans
    to, exposed as its own subtask instead of only running as a step inside
    ``place``. Exists because the *legacy* stage-machine transport backend
    has no notion of the carried object's own footprint (a single fixed
    carry height, no attached-object collision checking) and can drive a
    grasped object straight into an obstacle on the way to the goal.
    """

    # IK is not unique -- see OpenDrawerSkill/PlaceSkill's own comments on
    # the same point. Mix of near-start_q perturbation and full-joint-range
    # random seeds (see TransportSkill.execute's own comment on why both are
    # needed), tried before giving up on one hover-pose candidate.
    pose_selection_attempts: int = 8
    # RRTConnect is stochastic; retried more than most other OMPL-backed
    # skills in this package (default 3) since a tight, recessed transport
    # goal (e.g. an already-open drawer) needs more attempts to reliably
    # succeed.
    planning_attempts: int = 8
    # Candidates further than this from the robot's own base (world-frame xy
    # distance, ignoring z) are tried only after every closer one fails,
    # regardless of how "furthest from the object" ranks them -- see
    # TransportSkill.execute()'s own comment on why "furthest first" isn't
    # enough alone. Past this range the arm struggles to actually track to
    # the candidate even when IK/collision accept it. Sits comfortably under
    # the Panda's nominal ~0.85m kinematic reach, leaving room for a
    # demanding hover orientation; a heuristic cap, not a precisely derived
    # limit.
    max_practical_reach_m: float = 0.70


class TransportSkill:
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
        attached: AttachedObjectState,
        goal_container_body: str,
        start_q: np.ndarray,
        place_surface: str = "inside",
        goal_point_xyz: np.ndarray | None = None,
    ) -> SkillResult:
        # A completed pick leaves the object resting wherever it was
        # grasped, not lifted clear, so the first attached-object-aware
        # collision check can legitimately find `start_q` already touching
        # something; escape it first rather than failing outright (same as
        # PlaceSkill's own handoff -- see escape_start_collision's docstring).
        start_q = escape_start_collision(self.ik_solver, self.collision_checker, scene, attached, goal_container_body, start_q)

        cfg = self.config.transport
        container_pos, _ = scene.body_pose(goal_container_body)
        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        _, rim_top_z = floor_fn(scene, goal_container_body)
        # An explicit point goal says exactly where over the surface to carry
        # the object, replacing the "aim at the goal body's center, then try
        # a ring of offsets around it" search below. That search is right for
        # a small named container but close to useless on a large support,
        # where the body center can be nearly indistinguishable from the
        # object's own current position. The surface body is still needed
        # for the real floor height and for permitting contact with it.
        if goal_point_xyz is not None:
            goal_point_xyz = np.asarray(goal_point_xyz, dtype=float)
            container_pos = np.array([goal_point_xyz[0], goal_point_xyz[1], container_pos[2]])
            if goal_point_xyz.shape[0] > 2 and np.isfinite(goal_point_xyz[2]):
                rim_top_z = float(goal_point_xyz[2])

        # The object's orientation *relative to the gripper* never changes in
        # flight (the grip is rigid), but the world-frame yaw we ask the arm
        # to hold that grip at is still ours to choose -- it doesn't have to
        # be whatever yaw the preceding pick happened to leave it at. A
        # grasp yaw picked for picking reasons can leave the required
        # hover-hand orientation in a basin IK can't reach from any seed,
        # even though a different commanded yaw for the same rigid grip
        # solves immediately. So sweep yaw deltas about world Z (0/90/180/270
        # degrees), trying delta=0 (today's behavior) first.
        object_quat_now_base = scene.body_pose(attached.object_name)[1]
        yaw_deltas = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]

        # A named container (a bowl/basket/drawer) is small enough that
        # aiming for its own geometric center is exactly right. A large open
        # support surface (the table, say) is not: its center is just as
        # likely to already be occupied by something else. So this tries the
        # same center + shrunk-ring spread of XY offsets PlaceSkill's own
        # `_generate_candidates` uses (same config fields) -- harmless
        # near-duplicates of the center for a small container, but what
        # actually finds a clear spot on a big open surface.
        pcfg = self.config.place
        container_extent = scene.body_collision_extent_world(goal_container_body)
        xy_offsets = [np.zeros(2)]
        if goal_point_xyz is not None:
            # The exact point first, then a tight ring of near-misses, for
            # the same reason PlaceSkill's own point branch has one: a single
            # candidate makes the subtask all-or-nothing on one IK solve, and
            # a hover pose a few cm over is still the requested spot.
            # Deliberately much tighter than the wide ring used for a bare
            # support entity.
            for ring_m in (0.04, 0.08):
                for i in range(6):
                    theta = 2.0 * np.pi * i / 6
                    xy_offsets.append(ring_m * np.array([np.cos(theta), np.sin(theta)]))
        elif container_extent is not None:
            if place_surface == "top":
                # Same widened, multi-radius search PlaceSkill's own
                # _generate_candidates uses for place_surface="top" -- see
                # top_surface_* fields' own docstring in config.py. A single
                # ring isn't enough once more than one object needs its own
                # clear spot on a shared surface in the same run.
                n_ring = max(0, int(pcfg.top_surface_num_ring_samples))
                for shrink_fraction in pcfg.top_surface_shrink_fractions:
                    interior_half_xy = shrink_fraction * container_extent.half_extents[:2]
                    for i in range(n_ring):
                        theta = 2.0 * np.pi * i / n_ring
                        xy_offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))
                # Try the spots furthest from where the object already is
                # first: the whole reason to transport onto a big support is
                # to put the object somewhere *else*, and the first feasible
                # candidate in ring order is usually close to the surface's
                # own center -- i.e. close to where the object already is.
                # Every candidate is still fully collision-checked regardless
                # of order, so this only affects which feasible one wins.
                #
                # "Furthest from the object" alone isn't enough though: it
                # says nothing about whether the arm can actually *track* to
                # that candidate, only that IK+collision accept it as a final
                # pose -- a candidate too far from the robot's own base can
                # leave the settle loop essentially frozen even though
                # planning accepted it. So candidates beyond
                # cfg.max_practical_reach_m from the base are pushed to the
                # back of the ordering (still tried, never dropped outright).
                object_xy_now = scene.body_pose(attached.object_name)[0][:2]
                robot_base_xy = scene.body_pose(ROBOT_BODIES[0])[0][:2]
                anchor_xy = container_pos[:2]
                xy_offsets.sort(
                    key=lambda off: (
                        float(np.linalg.norm(anchor_xy + off - robot_base_xy)) > cfg.max_practical_reach_m,
                        -float(np.linalg.norm(anchor_xy + off - object_xy_now)),
                    )
                )
            else:
                interior_half_xy = pcfg.interior_shrink_fraction * container_extent.half_extents[:2]
                n_ring = max(0, int(pcfg.num_ring_samples))
                for i in range(n_ring):
                    theta = 2.0 * np.pi * i / n_ring
                    xy_offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))

        # hover_clearance_m alone measures from the object's own reference
        # point (its body origin), not from whichever part of its real
        # collision geometry hangs lowest below that point -- wrong whenever
        # the two aren't close, which is common for a handle-grasped object
        # whose body origin sits well above its own base, and can otherwise
        # leave the object's real base penetrating the goal surface even
        # when xy is correctly centered. Extending clearance by however far
        # the object's geometry overhangs its reference point fixes this
        # without touching the reference point itself. See PlaceSkill's
        # identical fix for the same reasoning in more detail.
        overhang_m = max(
            0.0,
            -min(
                (float(center[2]) - float(radius) for center, radius in attached.collision_spheres_object_frame),
                default=0.0,
            ),
        )

        lo, hi = scene.joint_qpos_range(self.ik_solver.joint_names)
        hover_ik = None
        hover_xyz = hover_hand_pos = hover_hand_quat = None
        last_rejection = "no_ik_attempted"
        n_offsets_tried = 0
        # Counts per rejection kind across every (yaw, offset, seed) attempt.
        # `last_rejection` alone is misleading -- it only reports whichever
        # attempt failed last, which can look like "IK failed" even when
        # collision was the dominant cause.
        rejection_counts: dict[str, int] = {}

        def _note(kind: str) -> None:
            rejection_counts[kind] = rejection_counts.get(kind, 0) + 1

        for yaw_delta in yaw_deltas:
            object_quat_now = quat_mul_wxyz(yaw_quat_wxyz(float(yaw_delta)), object_quat_now_base)
            for offset in xy_offsets:
                n_offsets_tried += 1
                candidate_hover_xyz = np.array(
                    [
                        container_pos[0] + offset[0],
                        container_pos[1] + offset[1],
                        rim_top_z + pcfg.hover_clearance_m + overhang_m,
                    ]
                )
                candidate_hover_hand_pos, candidate_hover_hand_quat = required_hand_pose_for_object_pose(
                    candidate_hover_xyz, object_quat_now, attached.grasp_transform_hand_from_object
                )
                # IK has many solution branches for the same Cartesian pose
                # (elbow up/down, base rotated round), and they are not
                # interchangeable: the arm has to physically travel from its
                # current configuration to whichever one is picked, and a
                # branch far away in joint space sweeps the hand through a
                # correspondingly huge arc to get there even along a direct
                # joint-space path. So don't accept the first valid solution:
                # collect every valid one this budget finds and keep the one
                # closest to start_q, which needs the least movement to reach.
                best_candidate = None
                best_joint_distance = float("inf")
                for attempt in range(cfg.pose_selection_attempts):
                    # Seeding near start_q tends to converge to a nearby
                    # branch, which is what we want, but can fail outright
                    # when no solution exists in that basin. So spend the
                    # first half of the budget near start_q and the rest on
                    # full-range seeds, then let the distance ranking below
                    # decide -- a far branch is still better than none.
                    if attempt == 0:
                        seed = start_q
                    elif attempt <= cfg.pose_selection_attempts // 2:
                        seed = np.clip(start_q + self._rng.uniform(-0.8, 0.8, size=start_q.shape), lo, hi)
                    else:
                        seed = self._rng.uniform(lo, hi)
                    candidate = self.ik_solver.solve(scene.clone(), candidate_hover_hand_pos, candidate_hover_hand_quat, seed_q=seed)
                    if not candidate.success:
                        last_rejection = "hover_ik_failed"
                        _note("hover_ik_failed")
                        continue
                    # PLACE_APPROACH, not FREE_SPACE: this pose holds the
                    # object right above the surface it's being carried to,
                    # so the carried object is necessarily within the
                    # attached-sphere margin of that surface, and only
                    # PLACE_APPROACH exempts the goal body. PlaceSkill's
                    # hover check has the identical fix for the same reason.
                    check = self.collision_checker.check(
                        candidate.q,
                        phase=CollisionPhase.PLACE_APPROACH,
                        attached=attached,
                        goal_container_body=goal_container_body,
                    )
                    if check.in_collision:
                        last_rejection = f"hover_pose_in_collision:{check.contacts[:4]}"
                        first = check.contacts[0] if check.contacts else None
                        _note(f"in_collision:{first.geom1}|{first.geom2}" if first else "in_collision")
                        continue
                    joint_distance = float(np.linalg.norm(candidate.q - start_q))
                    if joint_distance < best_joint_distance:
                        best_candidate, best_joint_distance = candidate, joint_distance
                if best_candidate is not None:
                    hover_ik = best_candidate
                    hover_xyz, hover_hand_pos, hover_hand_quat = (
                        candidate_hover_xyz,
                        candidate_hover_hand_pos,
                        candidate_hover_hand_quat,
                    )
                    break
            if hover_ik is not None:
                break

        if hover_ik is None:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": last_rejection,
                    "pose_selection_attempts": cfg.pose_selection_attempts,
                    "xy_offsets_tried": n_offsets_tried,
                    "rejection_counts": dict(
                        sorted(rejection_counts.items(), key=lambda kv: -kv[1])[:8]
                    ),
                    "hover_target_xy": [float(container_pos[0]), float(container_pos[1])],
                    "hover_target_z": float(rim_top_z + pcfg.hover_clearance_m + overhang_m),
                },
            )

        # Plan with the carried object as a real, checked obstacle for the
        # entire leg -- this is the whole point of this skill over the
        # legacy backend it replaces.
        #
        # RRTConnect is a *feasibility* planner, not a shortest-path one --
        # its path-simplification pass shortcuts through free space but
        # doesn't reliably turn a circuitous solution into a direct one, so
        # the first collision-free plan found can still swing the arm through
        # a wildly inefficient detour. So keep re-planning across the attempt
        # budget and track the shortest one found, stopping early once a
        # plan is already reasonably direct.
        #
        # Score candidate plans by *Cartesian* hand-path length (via forward
        # kinematics on each joint waypoint), not joint-space distance: a
        # "short" move in joint space can still swing the hand through a
        # long, high Cartesian detour depending which joints move, so
        # joint-space distance doesn't reliably track actual arm-swing.
        joint_names = self.collision_checker.joint_names
        ee_site = self.collision_checker.ee_site

        def _ee_pos_for_q(q: np.ndarray) -> np.ndarray:
            scratch = scene.clone()
            scratch.set_qpos(joint_names, q)
            scratch.forward()
            pos, _ = scratch.site_pose(ee_site)
            return pos

        def _cartesian_path_length_m(geometric_path: np.ndarray) -> float:
            positions = np.asarray([_ee_pos_for_q(q) for q in geometric_path])
            return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))

        straight_line_m = float(np.linalg.norm(hover_hand_pos - _ee_pos_for_q(start_q)))
        good_enough_length_m = max(straight_line_m * 1.5, 1e-3)

        plan = None
        best_plan = None
        best_path_length = float("inf")
        for _ in range(cfg.planning_attempts):
            plan = self.planner.plan(
                start_q,
                [hover_ik.q],
                # Same phase the hover pose above was accepted under -- the
                # planner validates its goal state too, so checking the path
                # under a stricter phase would reject the very pose just
                # chosen. The robot's own geometry is still fully checked
                # against the goal surface; only the carried object's sphere
                # is allowed to be near the thing it is being delivered to.
                phase=CollisionPhase.PLACE_APPROACH,
                attached=attached,
                goal_container_body=goal_container_body,
            )
            if plan.success:
                path_length = float("inf")
                if plan.geometric_path is not None and len(plan.geometric_path) > 1:
                    path_length = _cartesian_path_length_m(plan.geometric_path)
                if path_length < best_path_length:
                    best_plan, best_path_length = plan, path_length
                if best_path_length <= good_enough_length_m:
                    break
            elif plan.failure_reason not in (
                PlannerFailureReason.NO_PATH_FOUND,
                PlannerFailureReason.APPROXIMATE_SOLUTION_ONLY,
                PlannerFailureReason.TIMEOUT,
            ):
                break
        plan = best_plan if best_plan is not None else plan
        if not plan.success:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": f"planning_failed:{plan.failure_reason}",
                    # The planner records *which* geoms made a start/goal
                    # state invalid; dropping it here meant a
                    # START_IN_COLLISION told you nothing about what the arm
                    # was actually touching.
                    "planner_metadata": plan.metadata,
                },
            )

        plan.metadata["hover_q"] = hover_ik.q.tolist()
        # The plan's first waypoint is the *escaped* start_q above, which can
        # be a non-trivial distance from where the real arm currently sits
        # (escaping a carried object resting on the table can need up to
        # `max_lift_m`, more than PlaceSkill's own escape typically needs).
        # The adapter needs this explicitly to move the real arm there first,
        # before tracking the reference this plan produces -- otherwise the
        # MPC's attempt to reconcile the gap produces large tracking error
        # and drives the carried object through obstacles instead of around.
        plan.metadata["plan_start_q"] = start_q.tolist()
        return SkillResult(
            success=True,
            selected_object=attached.object_name,
            planned_trajectory=plan,
            failure_reason=SkillFailureReason.NONE,
            diagnostics={
                "hover_xyz": hover_xyz.tolist(),
                "hover_hand_pos": hover_hand_pos.tolist(),
                "hover_hand_quat_wxyz": hover_hand_quat.tolist(),
                "goal_container_body": goal_container_body,
                # The planned hand path vs. the straight line to the same
                # target: the ratio quantifies "the arm swung wide instead of
                # just moving there." Distinct from motion_trace's own
                # (executed, post-tracking) numbers -- a mismatch between the
                # two separates a bad *plan* from bad *tracking*.
                "planned_hand_path_m": best_path_length,
                "planned_straight_line_m": straight_line_m,
                "hover_joint_distance_rad": float(np.linalg.norm(hover_ik.q - start_q)),
                "plan_solve_status": plan.metadata.get("solve_status"),
            },
        )
