"""Planning-only ``TransportSkill``: move the currently-attached object to a
hover point above the goal container/support, planning a collision-free
path that treats the carried object as part of the robot's own collision
geometry (via its sphere approximation, see ``attached_object.py``/
``collision.py``) -- not just the bare gripper.

This is the *same* hover-leg motion ``PlaceSkill`` already plans internally
on the way to its own descent (see that module's own comment: "Plan the
transport-to-hover leg with the container as a real obstacle"), just
exposed here as its own subtask instead of only ever running as a step
inside ``place``. It exists because this codebase's *legacy* stage-machine
transport backend has no notion of the carried object's own footprint at
all -- a single fixed carry height, no attached-object collision checking
whatsoever -- confirmed directly via a real rollout ("move the bowl out of
the drawer's way, onto the cabinet's top") that it drove a grasped bowl
straight into the cabinet's own wall reaching for a goal on top of it,
dropping the grasp partway there. ``TransportSkill`` reuses ``PlaceSkill``'s
own floor/rim measurement (``_container_floor_and_rim_z``/
``_support_floor_and_rim_z``) so the two skills agree on where "over the
goal" actually is, and stops there -- the final descent-and-release is left
to a subsequent ``place`` subtask, matching the OverGoal vs
Placed/Released predicate split already used in STL formulas.
"""

from __future__ import annotations

import numpy as np

from ..collision import CollisionChecker, CollisionPhase, required_hand_pose_for_object_pose
from ..config import MotionConfig
from ..kinematics import IKSolver
from ..ompl_planner import MotionPlanner
from ..scene_query import SceneQuery, quat_mul_wxyz
from .grasp_generator import yaw_quat_wxyz
from ..types import AttachedObjectState, PlannerFailureReason, SkillFailureReason, SkillResult
from .place_skill import _container_floor_and_rim_z, _support_floor_and_rim_z, escape_start_collision


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
        # A real completed pick leaves the object resting wherever it was
        # grasped -- on the table, or right at a container's rim -- not
        # lifted clear. The very first attached-object-aware collision
        # check below can therefore legitimately find `start_q` already
        # touching something; escape it the same way PlaceSkill's own
        # handoff does (see escape_start_collision's own docstring) rather
        # than failing outright on a collision that's about to be resolved
        # by the very next step anyway.
        start_q = escape_start_collision(self.ik_solver, self.collision_checker, scene, attached, goal_container_body, start_q)

        cfg = self.config.transport
        container_pos, _ = scene.body_pose(goal_container_body)
        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        _, rim_top_z = floor_fn(scene, goal_container_body)
        # An explicit point goal says exactly where over the surface to carry
        # the object, replacing the "aim at the goal body's center, then try
        # a ring of offsets around it" search below. That search is right for
        # a small named container but close to useless on a large support:
        # measured on the real table, its own center sits ~6cm from where the
        # bowl already was, so a "successful" transport left the object
        # essentially where it started. The surface body is still needed (and
        # still passed) for the real floor height and for permitting contact
        # with the thing being placed onto.
        if goal_point_xyz is not None:
            goal_point_xyz = np.asarray(goal_point_xyz, dtype=float)
            container_pos = np.array([goal_point_xyz[0], goal_point_xyz[1], container_pos[2]])
            if goal_point_xyz.shape[0] > 2 and np.isfinite(goal_point_xyz[2]):
                rim_top_z = float(goal_point_xyz[2])

        # The object's orientation *relative to the gripper* never changes
        # in flight (the grip is rigid) -- but the world-frame yaw we ask
        # the arm to hold that grip at is still ours to choose, and doesn't
        # have to be exactly whatever yaw the preceding pick happened to
        # leave it at. Confirmed directly via a real rollout: a plate's
        # rim-edge grasp yaw (picked for reasons entirely about *picking*,
        # not about what's convenient to carry) sometimes left the required
        # hover-hand orientation in a basin the arm's IK solver essentially
        # never reaches from any seed (0/8 across every hover XY offset
        # tried) -- not because the pose was truly infeasible, a *different*
        # commanded yaw for the same rigid grip solved immediately. So
        # sweep a handful of yaw deltas about world Z (0/90/180/270 degrees)
        # applied on top of the object's current orientation, trying delta=0
        # (today's behavior -- a no-op when it already works, e.g. the bowl
        # case) before the others.
        object_quat_now_base = scene.body_pose(attached.object_name)[1]
        yaw_deltas = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]

        # A named container (a bowl/basket/drawer) is small enough that
        # aiming for its own geometric center is exactly right. A large,
        # open support surface (the table, say) is not: its center is just
        # as likely to already be occupied by something else already
        # sitting there. Confirmed via a real rollout: transporting toward
        # the table's own body center failed outright with the target
        # overlapping an unrelated plate already on the table. So this
        # tries the same center + shrunk-ring spread of XY offsets
        # PlaceSkill's own `_generate_candidates` uses (same config fields)
        # -- for a small container the ring candidates are harmless
        # near-duplicates of the center; for a big open surface they're
        # what actually finds a clear spot.
        pcfg = self.config.place
        container_extent = scene.body_collision_extent_world(goal_container_body)
        xy_offsets = [np.zeros(2)]
        if goal_point_xyz is not None:
            # The exact point first, then a tight ring of near-misses, for
            # the same reason PlaceSkill's own point branch has one: a single
            # candidate makes the subtask all-or-nothing on one IK solve, and
            # a hover pose a few cm over is still the requested spot. Leaving
            # this out here (while place had it) was a real regression -- one
            # rollout failed outright with hover_ik_failed on the lone
            # candidate. Deliberately much tighter than the wide ring used
            # for a bare support entity, which is what makes "aim at the
            # table" miss by half a table.
            for ring_m in (0.04, 0.08):
                for i in range(6):
                    theta = 2.0 * np.pi * i / 6
                    xy_offsets.append(ring_m * np.array([np.cos(theta), np.sin(theta)]))
        elif container_extent is not None:
            if place_surface == "top":
                # Same widened, multi-radius search PlaceSkill's own
                # _generate_candidates uses for place_surface="top" -- see
                # top_surface_* fields' own docstring in config.py. A single
                # ring wasn't enough once more than one object needed its
                # own clear spot on the table in the same run (one candidate
                # landed almost exactly on the cabinet's own footprint).
                n_ring = max(0, int(pcfg.top_surface_num_ring_samples))
                for shrink_fraction in pcfg.top_surface_shrink_fractions:
                    interior_half_xy = shrink_fraction * container_extent.half_extents[:2]
                    for i in range(n_ring):
                        theta = 2.0 * np.pi * i / n_ring
                        xy_offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))
                # Try the spots furthest from where the object already is
                # first. The whole reason to transport onto a big support is
                # to put the object somewhere *else*; taking the first
                # feasible candidate in ring order means aiming at the
                # surface's own center, which on this table sits ~6cm from
                # where the bowl started -- a "successful" transport that
                # left it exactly as much in the way as before. Ordering
                # only changes which feasible candidate wins; every one of
                # them is still fully collision-checked, which is what keeps
                # this honest about obstacles nothing else knows about (a
                # real rollout had 206 candidates correctly rejected for
                # putting the carried bowl inside a stove that isn't even a
                # named entity).
                object_xy_now = scene.body_pose(attached.object_name)[0][:2]
                anchor_xy = container_pos[:2]
                xy_offsets.sort(
                    key=lambda off: -float(np.linalg.norm(anchor_xy + off - object_xy_now))
                )
            else:
                interior_half_xy = pcfg.interior_shrink_fraction * container_extent.half_extents[:2]
                n_ring = max(0, int(pcfg.num_ring_samples))
                for i in range(n_ring):
                    theta = 2.0 * np.pi * i / n_ring
                    xy_offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))

        lo, hi = scene.joint_qpos_range(self.ik_solver.joint_names)
        hover_ik = None
        hover_xyz = hover_hand_pos = hover_hand_quat = None
        last_rejection = "no_ik_attempted"
        n_offsets_tried = 0
        # Counts per rejection kind across every (yaw, offset, seed) attempt.
        # `last_rejection` alone is actively misleading -- it reports only
        # whichever attempt happened to fail last, so a run whose candidates
        # were overwhelmingly rejected for collision still reads as
        # "hover_ik_failed" if the final seed happened to miss. That cost
        # real debugging time; the counts are cheap and settle it directly.
        rejection_counts: dict[str, int] = {}

        def _note(kind: str) -> None:
            rejection_counts[kind] = rejection_counts.get(kind, 0) + 1

        for yaw_delta in yaw_deltas:
            object_quat_now = quat_mul_wxyz(yaw_quat_wxyz(float(yaw_delta)), object_quat_now_base)
            for offset in xy_offsets:
                n_offsets_tried += 1
                candidate_hover_xyz = np.array(
                    [container_pos[0] + offset[0], container_pos[1] + offset[1], rim_top_z + pcfg.hover_clearance_m]
                )
                candidate_hover_hand_pos, candidate_hover_hand_quat = required_hand_pose_for_object_pose(
                    candidate_hover_xyz, object_quat_now, attached.grasp_transform_hand_from_object
                )
                # IK has many solution branches for the same Cartesian pose
                # (elbow up/down, base rotated round), and they are *not*
                # interchangeable here: the arm has to physically travel from
                # its current configuration to whichever one is picked, and a
                # branch far away in joint space sweeps the hand through a
                # correspondingly huge arc in world space to get there --
                # even along a perfectly direct joint-space path. That is
                # exactly the "arm shot up in the air" the user reported
                # watching a real rollout: measured 1.5m of hand travel for a
                # 0.18m hover move. So don't accept the first valid solution:
                # collect every valid one this budget finds and keep the one
                # *closest to start_q*, which is the one the arm can reach
                # with the least (and least dramatic) movement.
                best_candidate = None
                best_joint_distance = float("inf")
                for attempt in range(cfg.pose_selection_attempts):
                    # Seeding a DLS solve near start_q tends to converge to a
                    # nearby branch, which is what we want -- but it can also
                    # fail outright when no solution exists in that basin
                    # (measured: a plate-hover pose solvable 20/60 times from
                    # full-range seeds, 0/6 from start_q perturbations). So
                    # spend the first half of the budget near start_q and the
                    # rest on full-range seeds, then let the distance ranking
                    # below decide -- a far branch is still better than none.
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
                    # PLACE_APPROACH, not FREE_SPACE: this is the pose that
                    # holds the object right above the surface it is being
                    # carried to, so the carried object is necessarily within
                    # the attached-sphere margin of that surface -- and only
                    # PLACE_APPROACH exempts the goal body. Measured carrying
                    # a plate to the table: 356 candidates rejected for the
                    # plate's own sphere touching table_collision, the very
                    # thing it was being brought to. PlaceSkill's hover check
                    # had the identical bug and the identical fix.
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
                    "hover_target_z": float(rim_top_z + pcfg.hover_clearance_m),
                },
            )

        # Plan with the carried object as a real, checked obstacle for the
        # entire leg -- this is the whole point of this skill over the
        # legacy backend it replaces.
        #
        # RRTConnect is a *feasibility* planner, not a shortest-path one --
        # its own path-simplification pass shortcuts through free space but
        # doesn't reliably turn a circuitous solution into a direct one, so
        # "the first collision-free plan found" can still be a wildly
        # inefficient one. Confirmed directly via a real rollout (visible in
        # its saved video, and independently reported by the user watching
        # it: "the arm shot up in the air"): a successful bowl transport
        # took a 2.6m joint-space-equivalent path for a 0.8m straight-line
        # hover move (path_efficiency 0.31), swinging the arm up and out of
        # the camera's view entirely for dozens of frames although a much
        # shorter path existed. Rather than accepting the first feasible
        # plan outright, keep re-planning across the attempt budget and
        # track the shortest one found, only stopping early once a plan is
        # already reasonably direct (no need to spend the whole budget once
        # a good path is in hand).
        # Score candidate plans by *Cartesian* hand-path length (via forward
        # kinematics on each joint waypoint), not joint-space distance --
        # confirmed directly that the two don't reliably agree: an earlier
        # version of this retry scored by joint-space path length alone and
        # still accepted an equally-inefficient plan (path_efficiency barely
        # changed, 0.31 -> 0.37), since a "short" move in joint space can
        # still swing the hand through a long, high Cartesian detour
        # depending which joints move. Cartesian length is what actually
        # corresponds to "the arm visibly swinging out of frame."
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
        # The plan's own first waypoint is the *escaped* start_q above, not
        # necessarily wherever the real arm currently sits (escaping a real
        # start-in-collision can move it by a non-trivial distance -- unlike
        # PlaceSkill's own escape, which only ever needs a few cm to clear a
        # container wall, this one can need up to `max_lift_m` to clear a
        # carried object still resting on the table). The adapter needs
        # this explicitly so it can move the real arm there *first*, before
        # tracking the reference this plan produces -- confirmed via a real
        # rollout that skipping this left the reference assuming a start
        # position the arm was never actually at, and the MPC's attempt to
        # reconcile that gap produced a huge (~0.5-0.9 rad) joint-space
        # tracking error for the entire leg, driving the carried object
        # through the table/cabinet the whole way rather than around it.
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
                # The planned hand path vs the straight line to the same
                # target -- the ratio is what "the arm swung wide instead of
                # just moving there" looks like numerically, so it's worth
                # having on every run rather than only under ad hoc
                # instrumentation. Distinct from motion_trace's own
                # (executed, post-tracking) numbers: a mismatch between the
                # two separates a bad *plan* from bad *tracking*.
                "planned_hand_path_m": best_path_length,
                "planned_straight_line_m": straight_line_m,
                "hover_joint_distance_rad": float(np.linalg.norm(hover_ik.q - start_q)),
                "plan_solve_status": plan.metadata.get("solve_status"),
            },
        )
