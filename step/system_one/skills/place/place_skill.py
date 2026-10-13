"""Place a currently-attached object into/onto a goal container, planning a
collision-free trajectory that treats the carried object as part of the
robot's own collision geometry (see ``attached_object.py``/``collision.py``)
rather than moving the gripper as a bare point.

Mirrors ``PickSkill``'s shape: generate candidate placements, filter by IK +
collision, plan each with OMPL, and score to pick the best feasible one --
the same "propose several candidates, let planning decide" principle used
for rim grasps in ``grasp_generator.py``, applied here to where inside the
container to place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...motion.collision import CollisionChecker, CollisionPhase, required_hand_pose_for_object_pose
from ...motion.config import MotionConfig
from ..ik_utils import escape_start_collision, solve_ik_robust as _solve_ik_robust
from ...motion.kinematics import IKSolver
from ...motion.ompl_planner import MotionPlanner
from ...motion.scene_query import SceneQuery
from ...motion.types import AttachedObjectState, MotionPlan, SkillFailureReason, SkillResult


@dataclass
class PlaceSampleConfig:
    # Candidate placement points are sampled within a shrunk copy of the goal
    # container's interior footprint (center + a ring), mirroring how the
    # grasp generator shrinks inward from an object's rim.
    interior_shrink_fraction: float = 0.35
    num_ring_samples: int = 4
    # place_surface="top" candidate search (e.g. the table) is deliberately
    # wider/denser than the container-interior defaults above: a shared open
    # surface tends to have other objects and fixed obstacles (e.g. a
    # cabinet) already on it, so more radii/angles are needed to find a
    # clear spot.
    top_surface_shrink_fractions: tuple[float, ...] = (0.2, 0.35, 0.5)
    top_surface_num_ring_samples: int = 10
    hover_clearance_m: float = 0.06  # above the container's rim before descending
    drop_clearance_m: float = 0.02  # above the container's floor (or the next object already in it)
    # How far the hand lifts away after opening, before the release is
    # verified. _is_grasped() is a contact test, so a genuinely released
    # object still reads as grasped while it rests against the open fingers
    # -- e.g. when placing into an already-occupied container, where the
    # object settles higher than the empty-container floor the place pose
    # aimed at.
    release_withdraw_m: float = 0.06
    # IK is not unique -- a single DLS solve from one seed can land in a bad
    # basin even when a feasible solution exists nearby, same as the other
    # skills in this package. Retry from multiple seeds instead.
    pose_selection_attempts: int = 6


def _container_floor_and_rim_z(scene: SceneQuery, container_body: str) -> tuple[float, float]:
    """Interior floor height (top of the container's lowest solid piece,
    typically its base) and rim height (top of its tallest piece, typically
    a wall), derived from the container's collision geoms individually
    rather than one merged bounding box -- a merged box's bottom edge is the
    container's *outer* bottom, not its floor piece's top surface, which
    sits higher by the floor piece's own thickness. Using the merged box's
    bottom edge placed objects at or below actual ground level instead of on
    the container's real floor.
    """
    gids = scene.body_collision_geom_ids(container_body)
    tops = []
    bottoms = []
    for g in gids:
        center, half = scene.geom_world_aabb(g)
        tops.append(center[2] + half[2])
        bottoms.append(center[2] - half[2])
    tops_arr = np.asarray(tops)
    bottoms_arr = np.asarray(bottoms)
    floor_top_z = float(tops_arr[int(np.argmin(bottoms_arr))])
    rim_top_z = float(np.max(tops_arr))
    return floor_top_z, rim_top_z


def _support_floor_and_rim_z(scene: SceneQuery, support_body: str) -> tuple[float, float]:
    """Like ``_container_floor_and_rim_z``, but for placing on top of a
    solid support (e.g. a cabinet's own frame body) rather than inside a
    concave container: there's no floor-vs-wall distinction, so the "floor"
    and "rim" are the same surface -- the support's own highest point."""
    gids = scene.body_collision_geom_ids(support_body)
    top_z = float(max(scene.geom_world_aabb(g)[0][2] + scene.geom_world_aabb(g)[1][2] for g in gids))
    return top_z, top_z


@dataclass
class _PlaceCandidateScore:
    object_xyz: np.ndarray
    hover_q: np.ndarray | None
    place_q: np.ndarray | None
    approach_positions: np.ndarray | None
    plan: MotionPlan | None
    feasible: bool
    total_cost: float
    breakdown: dict[str, float] = field(default_factory=dict)
    rejection_reason: str | None = None


class PlaceSkill:
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
        start_q = escape_start_collision(self.ik_solver, self.collision_checker, scene, attached, goal_container_body, start_q)
        candidates = self._generate_candidates(scene, attached, goal_container_body, place_surface, goal_point_xyz)
        if not candidates:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": "goal container has no measurable geometry"},
            )

        scored = [
            self._score_candidate(scene, obj_xyz, attached, goal_container_body, start_q, place_surface)
            for obj_xyz in candidates
        ]
        feasible = [s for s in scored if s.feasible]
        if not feasible:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={
                    "detail": "no placement candidate passed IK/collision/planning",
                    "n_candidates_tried": len(scored),
                    "rejection_reasons": [s.rejection_reason for s in scored],
                },
            )

        best = min(feasible, key=lambda s: s.total_cost)
        # The plan's first waypoint is the *escaped* start_q above, not
        # necessarily where the real arm currently sits -- the adapter needs
        # this to move the real arm there first, before tracking the plan.
        if best.plan is not None:
            best.plan.metadata["plan_start_q"] = start_q.tolist()
        return SkillResult(
            success=True,
            selected_object=attached.object_name,
            planned_trajectory=best.plan,
            cost_breakdown=best.breakdown,
            failure_reason=SkillFailureReason.NONE,
            diagnostics={
                "n_candidates_tried": len(scored),
                "n_feasible": len(feasible),
                "object_xyz": best.object_xyz.tolist(),
                "hover_q": best.hover_q.tolist() if best.hover_q is not None else None,
                "place_q": best.place_q.tolist() if best.place_q is not None else None,
                "approach_positions": best.approach_positions.tolist()
                if best.approach_positions is not None
                else None,
            },
        )

    def _generate_candidates(
        self,
        scene: SceneQuery,
        attached: AttachedObjectState,
        goal_container_body: str,
        place_surface: str = "inside",
        goal_point_xyz: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Candidate object *positions* (XYZ) inside/on-top-of the goal
        body: its center plus a shrunk-in ring, mirroring the rim-grasp
        generator's "center + ring, shrunk inward from the wall/edge"
        pattern. ``place_surface="top"`` targets a solid support's own top
        surface (e.g. a cabinet's frame) instead of a container's interior
        floor -- see ``_support_floor_and_rim_z``."""
        cfg = self.config.place
        container_pos, _ = scene.body_pose(goal_container_body)
        container_extent = scene.body_collision_extent_world(goal_container_body)
        if container_extent is None:
            return []
        obj_extent = scene.body_collision_extent_world(attached.object_name)
        obj_half_h = float(obj_extent.half_extents[2]) if obj_extent is not None else 0.02
        # How far the object's real collision geometry extends below its own
        # reference point (body origin). half_extents alone is wrong when
        # the collision box's center_offset isn't zero, since it silently
        # assumes the origin is at the object's vertical center -- this was
        # the root cause of objects being released noticeably too high.
        #
        # Deliberately the tight AABB bound here, not the fitted-sphere
        # bound _score_candidate's hover fix uses: a sphere's radius mixes
        # in lateral spread, so for a laterally-asymmetric shape it
        # overshoots the box's true bottom. That's fine for a hover margin
        # that should err generous, but it would make the release itself
        # too high again. The AABB's Z-extent is exact, which a tight
        # release height needs.
        obj_bottom_local_z = float(obj_extent.center_offset[2] - obj_extent.half_extents[2]) if obj_extent is not None else -obj_half_h

        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        floor_z, rim_z = floor_fn(scene, goal_container_body)
        place_z = floor_z - obj_bottom_local_z + cfg.drop_clearance_m

        # An explicit point goal names the exact spot to set the object down
        # on; only its XY is taken, since the resting height is better
        # measured from the support's own real surface than trusted from the
        # caller.
        if goal_point_xyz is not None:
            goal_point_xyz = np.asarray(goal_point_xyz, dtype=float)
            # The exact point first, then a tight ring of near-misses around
            # it, so a single rejected IK/collision solve doesn't fail the
            # whole subtask when "a few cm over" is still a faithful match
            # for the caller's intent.
            out = [np.array([goal_point_xyz[0], goal_point_xyz[1], place_z])]
            for ring_m in (0.04, 0.08):
                for i in range(6):
                    theta = 2.0 * np.pi * i / 6
                    out.append(
                        np.array(
                            [
                                goal_point_xyz[0] + ring_m * np.cos(theta),
                                goal_point_xyz[1] + ring_m * np.sin(theta),
                                place_z,
                            ]
                        )
                    )
            return out

        offsets = [np.zeros(2)]
        if place_surface == "top":
            object_xy_now = scene.body_pose(attached.object_name)[0][:2]
            anchor_xy = container_pos[:2]
            # Try "exactly where the preceding transport already carried it"
            # as its own candidate, not just as a sort tiebreaker on a
            # container-centered ring -- the dead-center offset is usually
            # numerically closer to any sort key than this current position
            # is, so without an explicit candidate it tends to win even when
            # it's occupied by something else.
            offsets.append(object_xy_now - anchor_xy)
            # A large open support (the table): several ring radii, denser
            # angular sampling -- see top_surface_* fields' own docstring.
            n = max(0, int(cfg.top_surface_num_ring_samples))
            for shrink_fraction in cfg.top_surface_shrink_fractions:
                interior_half_xy = shrink_fraction * container_extent.half_extents[:2]
                for i in range(n):
                    theta = 2.0 * np.pi * i / n
                    offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))
            # Nearest-to-current-position first, the mirror of
            # TransportSkill's furthest-first ordering for the same surface
            # -- otherwise the two search the same ring independently and
            # can settle on opposite sides of a large support.
            offsets.sort(key=lambda off: float(np.linalg.norm(anchor_xy + off - object_xy_now)))
        else:
            interior_half_xy = cfg.interior_shrink_fraction * container_extent.half_extents[:2]
            n = max(0, int(cfg.num_ring_samples))
            for i in range(n):
                theta = 2.0 * np.pi * i / n
                offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))

        candidates = [
            np.array([container_pos[0] + off[0], container_pos[1] + off[1], place_z]) for off in offsets
        ]

        # Fallback "drop it in" candidates, above the container's own rim
        # rather than its floor: every candidate above assumes the container
        # is empty at floor height, which is false once something is already
        # resting inside it. Rather than nest neatly (which needs knowing
        # exactly how tall the existing contents are at this xy), release
        # from progressively higher above the rim and let gravity settle it
        # -- still fully validated like every other candidate, so these are
        # only selected when nothing lower is feasible.
        if place_surface != "top":
            obj_height_m = 2.0 * obj_half_h
            for level in range(1, 4):
                drop_z = rim_z - obj_bottom_local_z + cfg.drop_clearance_m + (level - 1) * obj_height_m
                candidates.append(np.array([container_pos[0], container_pos[1], drop_z]))

        return candidates

    def _score_candidate(
        self,
        scene: SceneQuery,
        object_xyz: np.ndarray,
        attached: AttachedObjectState,
        goal_container_body: str,
        start_q: np.ndarray,
        place_surface: str = "inside",
    ) -> _PlaceCandidateScore:
        breakdown: dict[str, float] = {}
        cfg = self.config.place

        # Keep the object's current orientation throughout -- this skill
        # places by lowering the carried object straight down, it doesn't
        # attempt to re-orient it in flight.
        object_pos_now, object_quat_now = scene.body_pose(attached.object_name)
        container_pos, _ = scene.body_pose(goal_container_body)
        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        _, rim_top_z = floor_fn(scene, goal_container_body)
        # max(), not the rim height alone: one of the "drop it in" fallback
        # candidates (for a container with no room left at floor height) can
        # itself sit at or above the normal rim-plus-clearance hover height,
        # which would otherwise make the "descent" go up instead of down.
        # Anchoring to whichever is higher keeps the approach a real descent.
        #
        # hover_clearance_m alone measures from the object's own reference
        # point (its body origin), not from whichever part of its real
        # collision geometry hangs lowest below that point -- wrong whenever
        # the two aren't close, which is common for a handle-grasped object
        # whose body origin sits well above its own base; a flat clearance
        # can then reject a hover pose as colliding with the surface even
        # though its xy is correctly centered on the goal. Extending the
        # clearance by however far the object's geometry overhangs its
        # reference point fixes this without touching the reference point
        # itself (used unchanged everywhere else).
        lowest_local_z = min(
            (float(center[2]) - float(radius) for center, radius in attached.collision_spheres_object_frame),
            default=0.0,
        )
        overhang_m = max(0.0, -lowest_local_z)
        hover_z = max(
            rim_top_z + cfg.hover_clearance_m + overhang_m,
            float(object_xyz[2]) + cfg.hover_clearance_m + overhang_m,
        )
        hover_xyz = np.array([object_xyz[0], object_xyz[1], hover_z])

        hover_hand_pos, hover_hand_quat = required_hand_pose_for_object_pose(
            hover_xyz, object_quat_now, attached.grasp_transform_hand_from_object
        )
        place_hand_pos, place_hand_quat = required_hand_pose_for_object_pose(
            object_xyz, object_quat_now, attached.grasp_transform_hand_from_object
        )

        hover_ik = _solve_ik_robust(
            self.ik_solver, scene, hover_hand_pos, hover_hand_quat, start_q, self._rng,
            n_restarts=self.config.place.pose_selection_attempts,
        )
        if not hover_ik.success:
            return _PlaceCandidateScore(
                object_xyz,
                None,
                None,
                None,
                None,
                False,
                float("inf"),
                rejection_reason=f"hover_ik_failed:pos_err={hover_ik.position_error:.4f},ori_err={hover_ik.orientation_error:.4f}",
            )
        place_ik = _solve_ik_robust(
            self.ik_solver, scene, place_hand_pos, place_hand_quat, hover_ik.q, self._rng,
            n_restarts=self.config.place.pose_selection_attempts,
        )
        if not place_ik.success:
            return _PlaceCandidateScore(
                object_xyz,
                hover_ik.q,
                None,
                None,
                None,
                False,
                float("inf"),
                rejection_reason=f"place_ik_failed:pos_err={place_ik.position_error:.4f},ori_err={place_ik.orientation_error:.4f}",
            )

        # PLACE_APPROACH, not FREE_SPACE: the hover pose sits only
        # `hover_clearance_m` above the goal, so the carried object is
        # necessarily within the attached-sphere safety margin of the
        # surface it's about to be set on, and FREE_SPACE doesn't permit the
        # goal body. Placing onto a flat top trips this every time (placing
        # inside a container doesn't, since the hover is over open interior).
        # The descent check below uses the same phase for the same reason.
        hover_check = self.collision_checker.check(
            hover_ik.q,
            phase=CollisionPhase.PLACE_APPROACH,
            attached=attached,
            goal_container_body=goal_container_body,
        )
        if hover_check.in_collision:
            return _PlaceCandidateScore(
                object_xyz,
                hover_ik.q,
                place_ik.q,
                None,
                None,
                False,
                float("inf"),
                rejection_reason=f"hover_pose_in_collision:{hover_check.contacts[:4]}",
            )
        place_check = self.collision_checker.check(
            place_ik.q,
            phase=CollisionPhase.PLACE_APPROACH,
            attached=attached,
            goal_container_body=goal_container_body,
        )
        if place_check.in_collision:
            return _PlaceCandidateScore(
                object_xyz,
                hover_ik.q,
                place_ik.q,
                None,
                None,
                False,
                float("inf"),
                rejection_reason=(
                    f"place_pose_in_collision:ik_pos_err={place_ik.position_error:.4f},"
                    f"target_object_xyz={object_xyz.tolist()},{place_check.contacts[:4]}"
                ),
            )

        # Plan the transport-to-hover leg with the container as a real
        # obstacle (attached-object sphere check, FREE_SPACE phase); the
        # final descent is a short direct segment checked separately under
        # PLACE_APPROACH, mirroring PickSkill's pregrasp-plan +
        # direct-final-approach split.
        plan = self.planner.plan(
            start_q,
            [hover_ik.q],
            phase=CollisionPhase.FREE_SPACE,
            attached=attached,
            goal_container_body=goal_container_body,
        )
        if not plan.success:
            return _PlaceCandidateScore(
                object_xyz,
                hover_ik.q,
                place_ik.q,
                None,
                None,
                False,
                float("inf"),
                rejection_reason=f"planning_failed:{plan.failure_reason}",
            )

        approach_positions = np.linspace(hover_ik.q, place_ik.q, 15)
        descent_check = self.collision_checker.check_edge(
            hover_ik.q,
            place_ik.q,
            resolution=self.config.planner.validity_resolution
            * float(np.max(self.planner.joint_limits_hi - self.planner.joint_limits_lo)),
            phase=CollisionPhase.PLACE_APPROACH,
            attached=attached,
            goal_container_body=goal_container_body,
        )
        if descent_check.in_collision:
            return _PlaceCandidateScore(
                object_xyz,
                hover_ik.q,
                place_ik.q,
                None,
                None,
                False,
                float("inf"),
                rejection_reason="descent_in_collision",
            )

        path_cost = 0.0
        if plan.geometric_path is not None and len(plan.geometric_path) > 1:
            path_cost = float(np.sum(np.linalg.norm(np.diff(plan.geometric_path, axis=0), axis=1)))
        breakdown["path_length"] = path_cost
        breakdown["reachability"] = float(hover_ik.position_error + hover_ik.orientation_error)
        # Measured from the object's current position, NOT the goal
        # container's geometric center: `execute()` picks the global
        # min-cost candidate regardless of list order, so a center-anchored
        # cost would silently override _generate_candidates' own
        # nearest-to-current-position ordering for place_surface="top" and
        # keep picking the (possibly occupied) dead center instead.
        breakdown["offset_from_current_position"] = float(np.linalg.norm(object_xyz[:2] - object_pos_now[:2]))
        total_cost = 1.0 * path_cost + 5.0 * breakdown["reachability"] + 0.5 * breakdown["offset_from_current_position"]

        return _PlaceCandidateScore(
            object_xyz=object_xyz,
            hover_q=hover_ik.q,
            place_q=place_ik.q,
            approach_positions=approach_positions,
            plan=plan,
            feasible=True,
            total_cost=total_cost,
            breakdown=breakdown,
        )
