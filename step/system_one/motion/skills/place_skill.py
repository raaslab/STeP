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

from ..collision import CollisionChecker, CollisionPhase, required_hand_pose_for_object_pose
from ..config import MotionConfig
from ..kinematics import IKSolver
from ..ompl_planner import MotionPlanner
from ..scene_query import SceneQuery
from ..types import AttachedObjectState, IKResult, MotionPlan, SkillFailureReason, SkillResult


def _solve_ik_robust(
    ik_solver: IKSolver,
    scene: SceneQuery,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    seed_q: np.ndarray,
    rng: np.random.Generator,
    n_restarts: int = 6,
    perturb_scale: float = 0.6,
) -> IKResult:
    """Retry a failed DLS solve from randomly perturbed seeds before giving
    up -- same reasoning as every other skill in this package (see
    OpenDrawerSkill/TurnKnobSkill/TransportSkill's own copies of this exact
    pattern): a local optimizer landing in a bad basin from one arbitrary
    seed doesn't mean the target is actually unreachable."""
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


def _support_floor_and_rim_z(scene: SceneQuery, support_body: str) -> tuple[float, float]:
    """Like ``_container_floor_and_rim_z``, but for placing something *on
    top of* a solid support (e.g. a cabinet's own frame body) rather than
    *inside* a concave container. There's no separate floor-vs-wall
    distinction for a solid support -- the "floor" an object rests on and
    the "rim" clearance is measured above are the same surface: its own
    highest point (e.g. the cabinet's top slab), found the same
    per-geom-not-merged-box way as the container case, for the same reason
    (a merged box's bottom edge sits below the actual resting surface)."""
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


def escape_start_collision(
    ik_solver: IKSolver,
    collision_checker: CollisionChecker,
    scene: SceneQuery,
    attached: AttachedObjectState | None,
    goal_container_body: str | None,
    start_q: np.ndarray,
    max_lift_m: float = 0.20,
    step_m: float = 0.01,
) -> np.ndarray:
    """If ``start_q`` is already in collision, retreat in small
    collision-checked increments until clear (or the retreat budget runs
    out). Standard "escape from initial contact" practice for a planner
    that (correctly) refuses to plan from an invalid start state.

    Shared by ``PlaceSkill`` and ``TransportSkill`` (whichever of the two
    runs first after a pick needs it): a real completed pick leaves the
    carried object still resting at/near wherever it was grasped -- on the
    table, or right at a container's rim -- not lifted clear, so the very
    first attached-object-aware collision check either skill runs can
    legitimately find `start_q` already touching something. Retreating
    first, rather than failing outright, means that imprecise handoff
    doesn't sink the whole subtask.

    Tries straight-up lift *and* a lift-while-moving-toward-the-goal's-own-
    center direction, not just straight up -- confirmed via a real
    rollout's per-step contact distances that a purely vertical retreat can
    leave a penetration depth completely unchanged across the entire lift
    budget (e.g. a carried object embedded sideways in a basket's wall
    stays embedded in that same wall at every height, since the overlap is
    horizontal, not vertical -- lifting straight up just slides along
    inside the wall instead of escaping it). Moving toward the goal's
    center pulls a laterally-embedded object back into clear space instead.
    """
    check = collision_checker.check(
        start_q, phase=CollisionPhase.FREE_SPACE, attached=attached, goal_container_body=goal_container_body
    )
    if not check.in_collision:
        return start_q

    joint_names = collision_checker.joint_names
    ee_site = collision_checker.ee_site
    scratch = scene.clone()
    scratch.set_qpos(joint_names, start_q)
    hand_pos, hand_quat = scratch.site_pose(ee_site)

    directions = {"up": np.array([0.0, 0.0, 1.0])}
    # "toward the goal's own center" only makes sense when there is a single
    # known goal body -- PickSkill calls this with none (it's not escaping
    # toward any one of several candidate objects), so this second retreat
    # direction is skipped rather than guessing one.
    if goal_container_body is not None:
        container_pos, _ = scene.body_pose(goal_container_body)
        toward_center_xy = container_pos[:2] - hand_pos[:2]
        norm = float(np.linalg.norm(toward_center_xy))
        toward_center_xy = toward_center_xy / norm if norm > 1e-6 else np.zeros(2)
        directions["up_toward_center"] = np.array([toward_center_xy[0], toward_center_xy[1], 1.0])
    n_steps = max(1, int(round(max_lift_m / step_m)))
    for direction in directions.values():
        unit = direction / float(np.linalg.norm(direction))
        q = start_q
        for i in range(1, n_steps + 1):
            target_pos = hand_pos + unit * step_m * i
            ik = ik_solver.solve(scratch, target_pos, hand_quat, seed_q=q)
            if not ik.success:
                continue
            q = ik.q
            c = collision_checker.check(
                q, phase=CollisionPhase.FREE_SPACE, attached=attached, goal_container_body=goal_container_body
            )
            if not c.in_collision:
                return q
    return start_q  # couldn't escape within budget; planner will report start_in_collision honestly


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
        # The plan's own first waypoint is the *escaped* start_q above, not
        # necessarily wherever the real arm currently sits -- see
        # TransportSkill's own identical comment on why the adapter needs
        # this explicitly to move the real arm there *first*, before
        # tracking the reference this plan produces.
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
        surface (e.g. a cabinet's frame body) instead of a container's
        interior floor -- see ``_support_floor_and_rim_z``."""
        cfg = self.config.place
        container_pos, _ = scene.body_pose(goal_container_body)
        container_extent = scene.body_collision_extent_world(goal_container_body)
        if container_extent is None:
            return []
        obj_extent = scene.body_collision_extent_world(attached.object_name)
        obj_half_h = float(obj_extent.half_extents[2]) if obj_extent is not None else 0.02
        # How far the object's real collision geometry extends below its own
        # reference point (body origin). half_extents alone is wrong here
        # whenever the collision box's center_offset isn't zero (true for
        # moka_pot_1: its box center sits 0.98cm above the body origin),
        # since it silently assumes the origin is at the object's vertical
        # center. Confirmed against moka_pot_1: this assumption alone
        # accounted for essentially the entire 11.45cm "released too high"
        # gap the user reported (predicted target 1.0449m vs. the real
        # burner top 0.9305m, matching the observed bug almost exactly).
        #
        # Deliberately the tight AABB bound here, not the fitted-sphere
        # bound _score_candidate's hover fix uses: a sphere's radius mixes
        # in lateral (XY) spread, so for a laterally-asymmetric shape like
        # a handled pot, center_z - radius overshoots well past the box's
        # true bottom (measured 10.85cm below origin for moka_pot_1, vs.
        # the AABB's exact 6.61cm) -- fine for a hover margin that should
        # err generous, but it makes the release itself even higher than
        # this bug already was. The AABB's Z-extent is exact (not a
        # conservative approximation), which is what a tight release height
        # needs.
        obj_bottom_local_z = float(obj_extent.center_offset[2] - obj_extent.half_extents[2]) if obj_extent is not None else -obj_half_h

        floor_fn = _support_floor_and_rim_z if place_surface == "top" else _container_floor_and_rim_z
        floor_z, rim_z = floor_fn(scene, goal_container_body)
        place_z = floor_z - obj_bottom_local_z + cfg.drop_clearance_m

        # An explicit point goal names the exact spot to set the object down
        # on; only its XY is taken, since the resting height is better
        # measured from the support's own real surface than trusted from the
        # caller. See TransportSkill's own comment for why the ring search
        # below is the wrong tool on a large support surface.
        if goal_point_xyz is not None:
            goal_point_xyz = np.asarray(goal_point_xyz, dtype=float)
            # The exact point first, then a tight ring of near-misses around
            # it. A single candidate makes the whole subtask all-or-nothing
            # on one IK/collision solve (measured: one rejection, one
            # candidate tried, subtask failed), which is needlessly brittle
            # when "a few cm over" is still an entirely faithful reading of
            # the caller's intent -- unlike the wide ring used for a bare
            # support entity, this stays close enough to still be the
            # requested spot.
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
            # A large open support (the table): several ring radii, denser
            # angular sampling -- see top_surface_* fields' own docstring.
            n = max(0, int(cfg.top_surface_num_ring_samples))
            for shrink_fraction in cfg.top_surface_shrink_fractions:
                interior_half_xy = shrink_fraction * container_extent.half_extents[:2]
                for i in range(n):
                    theta = 2.0 * np.pi * i / n
                    offsets.append(interior_half_xy * np.array([np.cos(theta), np.sin(theta)]))
            # Set the object down where the preceding transport actually
            # carried it: nearest-to-current-position first, the mirror of
            # TransportSkill's furthest-first ordering for the same surface.
            # Without this the two search the same ring independently and
            # can settle on opposite sides of a large support, turning a
            # short descent into another full traverse.
            object_xy_now = scene.body_pose(attached.object_name)[0][:2]
            anchor_xy = container_pos[:2]
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
        # is empty at floor height, which is false the moment something is
        # already resting inside it -- confirmed via a real two-object
        # basket task, where a second can's every ring position at floor
        # height collided with the first can already occupying that same
        # footprint, and the place subtask failed outright with no fallback.
        # Rather than nest neatly (which needs knowing exactly how tall
        # whatever's already inside is, at this exact xy), release from
        # progressively higher above the rim instead and let gravity settle
        # it -- these still go through the exact same IK/collision/planning
        # validation as every candidate above (never a blind release), so
        # they're only ever selected when genuinely nothing lower is
        # feasible, and a clean nested placement still always wins on cost
        # when one exists.
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
        # max(), not the rim height alone: a normal floor-level candidate is
        # always well below the rim, so this is the same fixed
        # rim-plus-clearance hover it's always been -- but one of the
        # "drop it in" fallback candidates (_generate_candidates, for a
        # container with no room left at floor height) can itself sit at or
        # above that fixed hover height, which would otherwise make the
        # "descent" from hover to place go *up* instead of down. Anchoring
        # hover to whichever is higher keeps the approach a real descent for
        # every candidate this method is ever called with.
        # hover_clearance_m alone measures from the object's own reference
        # point (its body origin), not from whichever part of its real
        # collision geometry actually hangs lowest below that point -- wrong
        # whenever the two aren't close, which is common for a handle-grasped
        # object whose body origin sits well above its own base. Confirmed
        # against the real moka_pot_1 asset: its lowest fitted collision
        # sphere reaches 10.85cm below the body origin, so a flat 6cm
        # clearance above the target surface still left the pot's real base
        # ~1.8cm *below* the table -- rejected every hover candidate as
        # hover_pose_in_collision against the table, even though the
        # candidate's own xy was correctly centered on the goal. Extending
        # clearance by however far the object's own geometry overhangs its
        # reference point fixes this without touching the reference point
        # itself (used unchanged everywhere else -- IK, the final place pose,
        # the descent path).
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

        # PLACE_APPROACH, not FREE_SPACE: the hover pose sits directly over
        # the goal by only `hover_clearance_m`, so the carried object is
        # necessarily within the attached-sphere safety margin of the very
        # surface it is about to be set down on -- and FREE_SPACE doesn't
        # permit the goal body, so that reads as a collision. Placing
        # *inside* a container hid this (the hover is over the open interior,
        # with no geom directly underneath), but placing onto a flat top
        # always trips it: measured on the real table, a candidate rejected
        # at +2.8mm of actual clearance -- not touching at all, merely inside
        # the margin. The descent check immediately below already uses this
        # same phase for the same reason.
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
