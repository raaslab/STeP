"""Planning-only TransportSkill: moves the currently-attached object to a
hover point above the goal container/support, planning a collision-free path
that treats the carried object as part of the robot's own collision geometry,
not just the bare gripper.

Exists because this codebase's legacy stage-machine transport backend has no
notion of the carried object's footprint and can drive it straight into an
obstacle. Stops at the hover point; the descent and release are a separate
place subtask, matching the OverGoal vs Placed/Released predicate split
used in STL formulas."""

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
from ..place.place_skill import _container_floor_and_rim_z, _support_floor_and_rim_z, support_top_surface


@dataclass
class TransportConfig:
    """Tuning for TransportSkill/LiberoTransportAdapter."""

    # IK is not unique; mixes near-start_q perturbation with full-joint-range
    # random seeds before giving up on one hover-pose candidate.
    pose_selection_attempts: int = 8
    # RRTConnect is stochastic; retried more than most OMPL-backed skills
    # here since a tight, recessed goal (e.g. an open drawer) needs more attempts.
    planning_attempts: int = 8
    # Candidates beyond this (world-frame xy distance from the base) are
    # tried last -- the arm struggles to track there even when IK/collision accept it.
    max_practical_reach_m: float = 0.70
    # Beyond this the object never reached its hover target; reporting
    # success let place release a moka pot 0.5m off the stove.
    max_hover_xy_err_m: float = 0.06
    # The hand servo can't follow a larger reconfiguration to the hover pose: a
    # 180 deg wrist flip (4.4 rad) flung a moka pot, a 2.8 rad one jammed the arm.
    max_hover_joint_distance_rad: float = 2.5
    # Heights above the hover pose to plan to when the carried object's spheres
    # (wider than the object) collide with a close neighbor at the hover pose itself.
    pre_hover_lifts_m: tuple[float, ...] = (0.10, 0.15)


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
        # A completed pick can leave the object touching whatever it was
        # grasped near, so the first collision check can find start_q already in contact; escape it first.
        start_q = escape_start_collision(self.ik_solver, self.collision_checker, scene, attached, goal_container_body, start_q)

        cfg = self.config.transport
        container_pos, _ = scene.body_pose(goal_container_body)
        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        _, rim_top_z = floor_fn(scene, goal_container_body)
        # An explicit point goal replaces the center+ring search below, which
        # is close to useless on a large support where the center is near the object's own position.
        if goal_point_xyz is not None:
            goal_point_xyz = np.asarray(goal_point_xyz, dtype=float)
            container_pos = np.array([goal_point_xyz[0], goal_point_xyz[1], container_pos[2]])
            if goal_point_xyz.shape[0] > 2 and np.isfinite(goal_point_xyz[2]):
                rim_top_z = float(goal_point_xyz[2])

        # The grip is rigid, but the world-frame yaw to hold it at is still
        # free to choose; sweeps yaw deltas about world Z since the pick's own yaw can leave IK unreachable.
        object_quat_now_base = scene.body_pose(attached.object_name)[1]
        # Smallest rotation first: the hand servo can't follow a large wrist
        # turn (a 90 deg jump drove the arm into a stuck full extension).
        yaw_deltas = [0.0, np.pi / 4.0, -np.pi / 4.0, np.pi / 2.0, -np.pi / 2.0, np.pi]

        # A small named container's own center is a fine target, but a large
        # open support's center is just as likely occupied, so a shrunk-ring spread of XY offsets is tried too.
        pcfg = self.config.place
        container_extent = scene.body_collision_extent_world(goal_container_body)
        xy_offsets = [np.zeros(2)]
        if goal_point_xyz is not None:
            # The exact point first, then a tight ring of near-misses, so a
            # single rejected IK solve doesn't fail the whole subtask.
            for ring_m in (0.04, 0.08):
                for i in range(6):
                    theta = 2.0 * np.pi * i / 6
                    xy_offsets.append(ring_m * np.array([np.cos(theta), np.sin(theta)]))
        elif container_extent is not None:
            if place_surface == "top":
                surface_xy, surface_half_xy, _top_z = support_top_surface(scene, goal_container_body)
                container_pos = np.array([surface_xy[0], surface_xy[1], container_pos[2]])
                # A wider, multi-radius search: a single ring isn't enough
                # once more than one object needs its own clear spot on the same shared surface.
                n_ring = max(0, int(pcfg.top_surface_num_ring_samples))
                for shrink_fraction in pcfg.top_surface_shrink_fractions:
                    interior_half_xy = shrink_fraction * surface_half_xy
                    for i in range(n_ring):
                        theta = 2.0 * np.pi * i / n_ring
                        xy_offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))
                # Moving within the same support (e.g. across a table):
                # furthest-from-current-position first, since the point is to
                # move it. Arriving from elsewhere: only spots that keep the
                # object's whole footprint on the surface, outermost first, so
                # a second object still has room -- unconstrained, a pot went
                # to the burner's far edge and hung off it; centered, it left
                # no room for the second pot. Beyond-reach candidates go last.
                object_xy_now = scene.body_pose(attached.object_name)[0][:2]
                robot_base_xy = scene.body_pose(ROBOT_BODIES[0])[0][:2]
                anchor_xy = container_pos[:2]
                already_on_support = bool(np.all(np.abs(object_xy_now - anchor_xy) <= surface_half_xy))
                if not already_on_support:
                    obj_extent = scene.body_collision_extent_world(attached.object_name)
                    obj_half_xy = obj_extent.half_extents[:2] if obj_extent is not None else np.zeros(2)
                    room_xy = np.maximum(surface_half_xy - obj_half_xy - pcfg.top_surface_footprint_margin_m, 0.0)
                    # The flush-to-edge spots themselves, which the rings rarely land on exactly.
                    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                        xy_offsets.append(room_xy * np.array([sx, sy], dtype=float))
                    fits = [off for off in xy_offsets if np.all(np.abs(off) <= room_xy + 1e-9)]
                    xy_offsets = fits or [np.zeros(2)]

                def _preference(off: np.ndarray) -> float:
                    if already_on_support:
                        return -float(np.linalg.norm(anchor_xy + off - object_xy_now))
                    return -float(np.linalg.norm(off))

                xy_offsets.sort(
                    key=lambda off: (
                        float(np.linalg.norm(anchor_xy + off - robot_base_xy)) > cfg.max_practical_reach_m,
                        _preference(off),
                    )
                )
            else:
                interior_half_xy = pcfg.interior_shrink_fraction * container_extent.half_extents[:2]
                n_ring = max(0, int(pcfg.num_ring_samples))
                for i in range(n_ring):
                    theta = 2.0 * np.pi * i / n_ring
                    xy_offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))

        # hover_clearance_m is measured from the object's reference point,
        # not its lowest geometry; extended by the overhang so the object's real base doesn't penetrate the surface.
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
        # Counts rejections per kind across every attempt -- last_rejection
        # alone only reports whichever failed last, hiding when collision was the dominant cause.
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
                # IK branches (elbow up/down, etc.) aren't interchangeable --
                # a far branch sweeps the arm through a huge arc, so keep the valid solution closest to start_q.
                best_candidate = None
                best_joint_distance = float("inf")
                for attempt in range(cfg.pose_selection_attempts):
                    # Seeding near start_q favors a nearby branch but can
                    # fail outright; spend half the budget there and the rest on full-range seeds.
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
                    # PLACE_APPROACH, not FREE_SPACE: the hover pose holds
                    # the object within the attached-sphere margin of the goal surface, which only PLACE_APPROACH exempts.
                    joint_distance = float(np.linalg.norm(candidate.q - start_q))
                    if joint_distance > cfg.max_hover_joint_distance_rad:
                        last_rejection = "hover_joint_distance_too_large"
                        _note("hover_joint_distance_too_large")
                        continue
                    # Arm checked as usual; the carried object against its exact
                    # geometry at the hover and set-down heights, not its spheres
                    # (too wide to pack objects next to each other on a surface).
                    check = self.collision_checker.check(
                        candidate.q,
                        phase=CollisionPhase.PLACE_APPROACH,
                        goal_container_body=goal_container_body,
                    )
                    if check.in_collision:
                        last_rejection = f"hover_pose_in_collision:{check.contacts[:4]}"
                        first = check.contacts[0] if check.contacts else None
                        _note(f"in_collision:{first.geom1}|{first.geom2}" if first else "in_collision")
                        continue
                    object_contacts = []
                    for drop_m in (0.0, pcfg.hover_clearance_m):
                        object_contacts += self.collision_checker.attached_exact_contacts(
                            candidate.q, attached, goal_container_body, drop_m=drop_m
                        )
                    if object_contacts:
                        last_rejection = f"carried_object_in_collision:{object_contacts[:4]}"
                        _note(f"object_in_collision:{object_contacts[0][0]}|{object_contacts[0][1]}")
                        continue
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

        # RRTConnect's first solution can be a wildly inefficient detour, so
        # re-plan across the budget, scoring by Cartesian hand-path length rather than joint-space distance.
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

        # The hover pose was accepted on the carried object's exact geometry,
        # but the planner validates its goal with the (wider) spheres. When
        # those collide, plan to a point straight above instead and finish
        # with a vertical descent checked on the exact geometry.
        plan_goal_q = hover_ik.q
        descent_path = None
        sphere_goal = self.collision_checker.check(
            hover_ik.q, phase=CollisionPhase.PLACE_APPROACH, attached=attached, goal_container_body=goal_container_body
        )
        if sphere_goal.in_collision:
            descent_resolution = self.config.planner.validity_resolution * float(np.max(hi - lo))
            for lift_m in cfg.pre_hover_lifts_m:
                pre_ik = self.ik_solver.solve(
                    scene.clone(), hover_hand_pos + np.array([0.0, 0.0, lift_m]), hover_hand_quat, seed_q=hover_ik.q
                )
                if not pre_ik.success:
                    continue
                if self.collision_checker.check(
                    pre_ik.q, phase=CollisionPhase.PLACE_APPROACH, attached=attached, goal_container_body=goal_container_body
                ).in_collision:
                    continue
                if self.collision_checker.check_edge(
                    pre_ik.q, hover_ik.q, resolution=descent_resolution,
                    phase=CollisionPhase.PLACE_APPROACH, goal_container_body=goal_container_body,
                ).in_collision:
                    continue
                candidate_descent = np.linspace(pre_ik.q, hover_ik.q, 10)
                if any(
                    self.collision_checker.attached_exact_contacts(q, attached, goal_container_body)
                    for q in candidate_descent
                ):
                    continue
                plan_goal_q, descent_path = pre_ik.q, candidate_descent
                break
            if descent_path is None:
                return SkillResult(
                    success=False,
                    failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                    diagnostics={"detail": "no collision-free vertical approach to the hover pose"},
                )

        straight_line_m = float(np.linalg.norm(hover_hand_pos - _ee_pos_for_q(start_q)))
        good_enough_length_m = max(straight_line_m * 1.5, 1e-3)

        plan = None
        best_plan = None
        best_path_length = float("inf")
        for _ in range(cfg.planning_attempts):
            plan = self.planner.plan(
                start_q,
                [plan_goal_q],
                # Same phase the hover pose was accepted under, since the
                # planner validates its goal state too; only the carried object's sphere is exempted near the goal.
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
                    # Includes which geoms made a start/goal state invalid, so
                    # a START_IN_COLLISION diagnostic says what the arm was actually touching.
                    "planner_metadata": plan.metadata,
                },
            )

        if descent_path is not None:
            plan.geometric_path = np.concatenate([np.asarray(plan.geometric_path), descent_path[1:]], axis=0)
        plan.metadata["hover_q"] = hover_ik.q.tolist()
        # The plan's first waypoint is the escaped start_q, which can be far
        # from where the arm sits now; the adapter must move there first or the MPC drives the object through obstacles.
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
                # Planned hand path vs. straight line to the target; a
                # mismatch against motion_trace's executed numbers separates a bad plan from bad tracking.
                "planned_hand_path_m": best_path_length,
                "planned_straight_line_m": straight_line_m,
                "hover_joint_distance_rad": float(np.linalg.norm(hover_ik.q - start_q)),
                "plan_solve_status": plan.metadata.get("solve_status"),
            },
        )
