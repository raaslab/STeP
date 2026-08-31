"""Place a currently-attached object into/onto a goal container, planning a
collision-free trajectory that treats the carried object as part of the
robot (via its sphere approximation, see ``attached_object.py``/
``collision.py``) rather than moving the gripper as a bare point and hoping
the object clears the container's rim.

Mirrors ``PickSkill``'s shape: generate candidate placements, filter by IK +
collision, plan each with OMPL, score, and let the same downstream
feasibility/cost logic pick a working one -- the same "propose several
candidates, let planning decide" principle used for rim grasps in
``grasp_generator.py``, now applied to *where inside the container* to place,
since a container's interior is a region, not a single point, in the same
way a rim grasp needed multiple points around a circumference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .collision import CollisionChecker, CollisionPhase, required_hand_pose_for_object_pose
from .config import MotionConfig
from .kinematics import IKSolver
from .ompl_planner import MotionPlanner
from .scene_query import SceneQuery
from .types import AttachedObjectState, MotionPlan, SkillFailureReason, SkillResult


def _container_floor_and_rim_z(scene: SceneQuery, container_body: str) -> tuple[float, float]:
    """Interior floor height (top surface of the container's lowest solid
    piece, typically its base) and rim height (top of its tallest solid
    piece, typically a wall), derived from the container's real collision
    geoms individually rather than one merged bounding box.

    A merged box's own bottom edge is the *outer* bottom of the whole
    container -- where it rests on the table/floor -- not the top surface of
    its floor piece, which sits above that outer bottom by the floor
    piece's own thickness. Using the merged box's bottom edge as the resting
    surface (what an earlier version of this function did) placed objects
    below where they should land: with a merged box computed over both the
    container's visual mesh and its real collision boxes, the "floor" came
    out low enough that placement targets ended up at or below actual
    ground level, reading as a collision with the ground itself instead of
    a successful placement onto the container's own floor.
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

    def _escape_start_collision(
        self,
        scene: SceneQuery,
        attached: AttachedObjectState,
        goal_container_body: str,
        start_q: np.ndarray,
        max_lift_m: float = 0.12,
        step_m: float = 0.01,
    ) -> np.ndarray:
        """If ``start_q`` is already in collision, back straight up in small
        collision-checked increments until clear (or the lift budget runs
        out). Standard "escape from initial contact" practice for a planner
        that (correctly) refuses to plan from an invalid start state.

        Needed at this specific handoff because legacy ``transport`` has no
        notion of the carried object's real footprint against the goal
        container -- it aims for "roughly over the goal" using its own 2D
        keepout logic, which can leave the object still a few centimeters
        into a container wall by the time ``place`` takes over. Retreating
        first, rather than failing outright, means an imprecise transport
        handoff doesn't sink the whole placement.
        """
        check = self.collision_checker.check(
            start_q, phase=CollisionPhase.FREE_SPACE, attached=attached, goal_container_body=goal_container_body
        )
        if not check.in_collision:
            return start_q

        joint_names = self.collision_checker.joint_names
        ee_site = self.collision_checker.ee_site
        scratch = scene.clone()
        scratch.set_qpos(joint_names, start_q)
        hand_pos, hand_quat = scratch.site_pose(ee_site)

        q = start_q
        n_steps = max(1, int(round(max_lift_m / step_m)))
        for i in range(1, n_steps + 1):
            target_pos = hand_pos + np.array([0.0, 0.0, step_m * i])
            ik = self.ik_solver.solve(scratch, target_pos, hand_quat, seed_q=q)
            if not ik.success:
                continue
            q = ik.q
            c = self.collision_checker.check(
                q, phase=CollisionPhase.FREE_SPACE, attached=attached, goal_container_body=goal_container_body
            )
            if not c.in_collision:
                return q
        return start_q  # couldn't escape within budget; planner will report start_in_collision honestly

    def execute(
        self,
        scene: SceneQuery,
        attached: AttachedObjectState,
        goal_container_body: str,
        start_q: np.ndarray,
    ) -> SkillResult:
        start_q = self._escape_start_collision(scene, attached, goal_container_body, start_q)
        candidates = self._generate_candidates(scene, attached, goal_container_body)
        if not candidates:
            return SkillResult(
                success=False,
                failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                diagnostics={"detail": "goal container has no measurable geometry"},
            )

        scored = [
            self._score_candidate(scene, obj_xyz, attached, goal_container_body, start_q)
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
        self, scene: SceneQuery, attached: AttachedObjectState, goal_container_body: str
    ) -> list[np.ndarray]:
        """Candidate object *positions* (XYZ) inside the container: its
        center plus a shrunk-in ring, mirroring the rim-grasp generator's
        "center + ring, shrunk inward from the wall" pattern."""
        cfg = self.config.place
        container_pos, _ = scene.body_pose(goal_container_body)
        container_extent = scene.body_collision_extent_world(goal_container_body)
        if container_extent is None:
            return []
        obj_extent = scene.body_extent_world(attached.object_name)
        obj_half_h = float(obj_extent.half_extents[2]) if obj_extent is not None else 0.02

        floor_z, _ = _container_floor_and_rim_z(scene, goal_container_body)
        place_z = floor_z + obj_half_h + cfg.drop_clearance_m
        interior_half_xy = cfg.interior_shrink_fraction * container_extent.half_extents[:2]

        offsets = [np.zeros(2)]
        n = max(0, int(cfg.num_ring_samples))
        for i in range(n):
            theta = 2.0 * np.pi * i / n
            offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))

        return [
            np.array([container_pos[0] + off[0], container_pos[1] + off[1], place_z]) for off in offsets
        ]

    def _score_candidate(
        self,
        scene: SceneQuery,
        object_xyz: np.ndarray,
        attached: AttachedObjectState,
        goal_container_body: str,
        start_q: np.ndarray,
    ) -> _PlaceCandidateScore:
        breakdown: dict[str, float] = {}
        cfg = self.config.place

        # Keep the object's current orientation throughout -- this skill
        # places by lowering the carried object straight down, it doesn't
        # attempt to re-orient it in flight.
        object_pos_now, object_quat_now = scene.body_pose(attached.object_name)
        container_pos, _ = scene.body_pose(goal_container_body)
        _, rim_top_z = _container_floor_and_rim_z(scene, goal_container_body)
        hover_xyz = np.array([object_xyz[0], object_xyz[1], rim_top_z + cfg.hover_clearance_m])

        hover_hand_pos, hover_hand_quat = required_hand_pose_for_object_pose(
            hover_xyz, object_quat_now, attached.grasp_transform_hand_from_object
        )
        place_hand_pos, place_hand_quat = required_hand_pose_for_object_pose(
            object_xyz, object_quat_now, attached.grasp_transform_hand_from_object
        )

        scratch = scene.clone()
        hover_ik = self.ik_solver.solve(scratch, hover_hand_pos, hover_hand_quat, seed_q=start_q)
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
        scratch2 = scene.clone()
        place_ik = self.ik_solver.solve(scratch2, place_hand_pos, place_hand_quat, seed_q=hover_ik.q)
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

        hover_check = self.collision_checker.check(
            hover_ik.q,
            phase=CollisionPhase.FREE_SPACE,
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
        breakdown["offset_from_center"] = float(np.linalg.norm(object_xyz[:2] - container_pos[:2]))
        total_cost = 1.0 * path_cost + 5.0 * breakdown["reachability"] + 0.5 * breakdown["offset_from_center"]

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
