"""Places a currently-attached object into/onto a goal container, treating the carried
object as part of the robot's own collision geometry rather than a bare gripper point.

Mirrors PickSkill: generate candidate placements, filter by IK/collision, plan each
with OMPL, and score to pick the best feasible one.
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
    # Sampled within a shrunk copy of the goal container's interior (center + a ring).
    interior_shrink_fraction: float = 0.35
    num_ring_samples: int = 4
    # Wider/denser than the interior defaults: a shared open surface (e.g. a table) usually
    # has other objects already on it.
    top_surface_shrink_fractions: tuple[float, ...] = (0.2, 0.35, 0.5)
    top_surface_num_ring_samples: int = 10
    # Extra inset beyond "footprint fully on the surface" when picking a top-surface
    # spot. Zero: two moka pots only fit a 19cm burner when each sits flush to an edge.
    top_surface_footprint_margin_m: float = 0.0
    hover_clearance_m: float = 0.06  # above the container's rim before descending
    drop_clearance_m: float = 0.02  # above the container's floor (or the next object already in it)
    # Withdraw distance before checking release; a just-released object can still read as
    # grasped from contact alone while resting against the open fingers.
    release_withdraw_m: float = 0.06
    # IK is not unique; retry from multiple seeds rather than trust one.
    pose_selection_attempts: int = 6
    # Release early if the carried object is pushing against something (e.g.
    # a basket rim) hard enough to block the goal pose, rather than grinding
    # the full step budget against it. Measured above the pre-descent
    # baseline (gripper weight + whatever's carried), not an absolute force,
    # since that baseline varies by object. Needs real-rollout calibration.
    blocked_force_delta_n: float = 10.0
    # Fallback when force sensing doesn't catch it: no progress for this many
    # steps. Longer than ExecutionConfig's stall_window_steps (15) -- squeezing
    # an object past a tight rim can stall for a while before genuinely getting
    # through, and this should only fire on a stall that lasts (confirmed via a
    # real rollout: 15 steps falsely flagged a place that succeeded by step ~40).
    blocked_stall_window_steps: int = 60
    # Backed off along the approach direction before opening the gripper when
    # blocked -- opening while still pressed into the obstruction can leave the
    # finger actuators unable to reach fully open against that contact force
    # (confirmed via a real rollout: fingers plateaued at ~72% open and stayed
    # there for 80+ extra steps with zero progress -- giving more time alone
    # does not help, since nothing relieves the force holding them there).
    blocked_retreat_m: float = 0.03


def _container_floor_and_rim_z(scene: SceneQuery, container_body: str) -> tuple[float, float]:
    """Interior floor height (top of the lowest geom) and rim height (top of the tallest),
    measured per-geom since a merged bounding box's bottom sits below the real floor."""
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


def support_top_surface(scene: SceneQuery, support_body: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(center_xy, half_xy, top_z) of a solid support's main horizontal
    surface: the largest-footprint collision geom, highest among near-ties.
    Not the body origin or the highest geom -- a stove's origin sits at its
    knob end and its knob is taller than the burner a pot belongs on."""
    gids = scene.body_collision_geom_ids(support_body)
    boxes = [scene.geom_world_aabb(g) for g in gids]
    areas = [4.0 * float(half[0]) * float(half[1]) for _center, half in boxes]
    max_area = max(areas)
    near_ties = [i for i, a in enumerate(areas) if a >= 0.8 * max_area]
    best = max(near_ties, key=lambda i: float(boxes[i][0][2] + boxes[i][1][2]))
    center, half = boxes[best]
    return (
        np.asarray(center[:2], dtype=float),
        np.asarray(half[:2], dtype=float),
        float(center[2] + half[2]),
    )


def _support_floor_and_rim_z(scene: SceneQuery, support_body: str) -> tuple[float, float]:
    """Like _container_floor_and_rim_z but for a solid support (e.g. a cabinet's own
    frame): floor and rim are the same surface, its main top surface."""
    _center_xy, _half_xy, top_z = support_top_surface(scene, support_body)
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
        # The plan starts from the escaped start_q, not necessarily where the
        # real arm sits now -- the adapter must move there first before tracking the plan.
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
        """Candidate object positions: the goal's center plus a shrunk-in ring.
        place_surface="top" targets a support's own top surface instead of a container's interior floor."""
        cfg = self.config.place
        container_pos, _ = scene.body_pose(goal_container_body)
        container_extent = scene.body_collision_extent_world(goal_container_body)
        if container_extent is None:
            return []
        obj_extent = scene.body_collision_extent_world(attached.object_name)
        obj_half_h = float(obj_extent.half_extents[2]) if obj_extent is not None else 0.02
        # How far the object's geometry extends below its reference point;
        # uses the exact AABB bound here, not the fitted-sphere bound (which overestimates for release height).
        obj_bottom_local_z = float(obj_extent.center_offset[2] - obj_extent.half_extents[2]) if obj_extent is not None else -obj_half_h

        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        floor_z, rim_z = floor_fn(scene, goal_container_body)
        place_z = floor_z - obj_bottom_local_z + cfg.drop_clearance_m

        # An explicit point goal gives only XY; resting height is measured
        # from the support's real surface rather than trusted from the caller.
        if goal_point_xyz is not None:
            goal_point_xyz = np.asarray(goal_point_xyz, dtype=float)
            # The exact point first, then a tight ring of near-misses, so one
            # rejected IK/collision solve doesn't fail the whole subtask.
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
            surface_xy, surface_half_xy, _top_z = support_top_surface(scene, goal_container_body)
            container_pos = np.array([surface_xy[0], surface_xy[1], container_pos[2]])
            object_xy_now = scene.body_pose(attached.object_name)[0][:2]
            anchor_xy = container_pos[:2]
            # Tries the object's current position as its own candidate, not
            # just a sort tiebreaker -- otherwise the dead-center offset tends to win even when occupied.
            offsets.append(object_xy_now - anchor_xy)
            # A large open support (e.g. a table) gets several ring radii and
            # denser angular sampling than a small container.
            n = max(0, int(cfg.top_surface_num_ring_samples))
            for shrink_fraction in cfg.top_surface_shrink_fractions:
                interior_half_xy = shrink_fraction * surface_half_xy
                for i in range(n):
                    theta = 2.0 * np.pi * i / n
                    offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))
            # Sorted nearest-to-current-position first so this and
            # TransportSkill's furthest-first search don't settle on opposite sides of a large support.
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

        # Fallback candidates above the rim (not the floor), for when the
        # container already has something in it; release progressively higher and let gravity settle it.
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

        # Keeps the object's current orientation throughout -- this skill
        # lowers it straight down, it doesn't re-orient in flight.
        object_pos_now, object_quat_now = scene.body_pose(attached.object_name)
        container_pos, _ = scene.body_pose(goal_container_body)
        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        _, rim_top_z = floor_fn(scene, goal_container_body)
        # max(), not rim height alone: a fallback "drop it in" candidate can
        # sit at or above the normal hover height, turning the descent into a climb.
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

        # PLACE_APPROACH, not FREE_SPACE: the hover pose sits close enough to
        # the goal that FREE_SPACE would reject it as colliding with the goal body.
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

        # Plans the transport-to-hover leg with the container as a real
        # obstacle (FREE_SPACE); the final descent is checked separately under PLACE_APPROACH.
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
        # Measured from the object's current position, not the container's
        # center -- a center-anchored cost would override the nearest-position ordering above and pick the (possibly occupied) dead center.
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
