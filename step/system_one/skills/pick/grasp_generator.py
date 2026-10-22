"""Grasp-candidate generation.

Starting point is intentionally simple, per Cap-X's privileged-state generator
(capx/integrations/franka/libero_privileged.py, MIT License,
https://github.com/capgym/cap-x): FrankaLiberoPrivilegedApi.
sample_grasp_pose() there is a single hardcoded top-down grasp at the
object's privileged pose with quaternion **WXYZ** [0, 1, 0, 0] (gripper
pointing straight down). This module keeps that same top-down, oracle-pose
starting point but extends it in two ways:

1. **Center-grasp candidates**: multiple wrist-yaw samples, a real pregrasp
   offset, and a documented, model-derived fingertip/hand transform instead
   of Cap-X's hardcoded FRANKA_HAND_TO_FINGERTIP_Z_M constant (see
   scene_query.SceneQuery.hand_to_tcp_transform). This is the right
   strategy for anything narrower than the gripper's opening at *some* yaw
   (cans, boxes, bottles).

2. **Rim-edge candidates**: Cap-X has no object-specific grasp logic to
   borrow here -- its privileged generator is a single fixed pose, and its
   vision path delegates entirely to Contact-GraspNet, which this generator
   doesn't have available. Instead it adds an explicit strategy for objects
   whose *whole* cross-section is too wide to encompass at any yaw (bowls,
   mugs, cups): grasp a single point around the object's rim, with the wrist
   yawed so the gripper's closing axis is *radial* at that point (one finger
   just outside the rim, one just inside) rather than spanning the full
   diameter. This generalizes the legacy stage machine's fixed
   grasp_rim_offset (base_env.py/modes.py's
   RIM_GRASP_KEYWORDS) from one hardcoded direction to every object,
   sampled at rim_num_angle_samples points around the full circumference
   so there's always a candidate on a reachable, collision-free side. The
   default rim_offset_fraction is calibrated against a real bowl mesh
   with a known-working legacy offset, not guessed.

Multiple orientations across *both* strategies are handed to PickSkill,
which is responsible for picking a working one via IK/collision/OMPL
feasibility and cost -- this generator's job is only to make sure a feasible
candidate exists somewhere in the set it proposes, not to guess which one
will work.

Quaternion convention: **WXYZ** everywhere (position of components:
[qw, qx, qy, qz]), matching Cap-X and MuJoCo's own native body/site quaternion
layout -- see scene_query.py's module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ...modes import FORCE_MANUAL_KEYWORDS, MANUAL_YAW_OVERRIDE_KEYWORDS, RIM_GRASP_KEYWORDS
from ..ik_utils import yaw_quat_wxyz
from ...motion.scene_query import SceneQuery, quat_mul_wxyz, quat_to_rotmat_wxyz, rotmat_to_quat_wxyz
from ...motion.types import GraspCandidate


@dataclass
class GraspSampleConfig:
    # Which generator proposes this pick's candidates.
    #
    # "object_class" (default) decides per object via a keyword table over
    # MuJoCo body names -- a LIBERO artefact: it matches akita_black_bowl_1 and
    # nothing in a backend that names every task object obj_main, so the manual
    # pose silently wins every object there. "graspnet_first" and
    # "demo:<task>[:<mesh>]" name a generator outright and skip both the table
    # and the manual pre-attempt.
    routing: str = "object_class"
    num_yaw_samples: int = 8
    # Centered on the object's own best-fit orientation, not swept blindly.
    yaw_search_half_range_rad: float = 0.7853981633974483  # pi/4, i.e. +-45 degrees
    lift_check_m: float = 0.05
    # Extra lift after the test lift so the next skill has clearance to plan.
    post_grasp_carry_lift_m: float = 0.10
    # Approach plans under FREE_SPACE to this point, then descends to the
    # grasp pose under TERMINAL_APPROACH (contact now expected).
    approach_standoff_m: float = 0.08
    # Standoff distance multiples tried if the nominal (1.0x) one is blocked.
    standoff_distance_multipliers: tuple[float, ...] = (1.0, 2.5, 4.0)
    # Sideways offsets tried if the straight-up standoff is blocked.
    standoff_lateral_offsets_m: tuple[float, ...] = (0.05, 0.08, 0.12)
    standoff_lateral_directions: int = 8  # tried around a full circle at each offset magnitude
    # Rim/edge grasp (bowls, mugs: too wide to fit the gripper at any yaw).
    rim_num_angle_samples: int = 8
    rim_offset_fraction: float = 0.6
    rim_grasp_width_m: float = 0.02
    # Edge-pinch grasp (plates, lids: flat, no hollow rim to straddle).
    edge_pinch_num_angle_samples: int = 8
    edge_pinch_offset_fraction: float = 0.92
    edge_pinch_width_m: float = 0.02
    # Perturbed-seed IK retries -- the resting pose seeds a top-down grasp
    # well but not every strategy.
    pose_selection_attempts: int = 6
    # Margin above the gripper's real reach so the palm/fingers clear a short
    # object's top or a thin one's underlying surface.
    grasp_height_margin_m: float = 0.01
    # RRTConnect is stochastic; retries avoid losing a valid candidate to one
    # unlucky NO_PATH_FOUND.
    planning_attempts: int = 6
    # Caps total wall-clock time scoring candidates: a hard-to-grasp object
    # can make every candidate exhaust the full standoff search (~75 positions
    # x solve_ik_robust's full restart budget) before failing, which can cost
    # 10+ seconds per candidate and blow the rollout timeout across 32 of them.
    candidate_scoring_budget_sec: float = 60.0

# World-frame quaternion pointing the gripper's approach axis straight down.
_TOP_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# Maps the closing axis to world +Z, so a yaw sweep rotates the approach
# heading while keeping the closing axis vertical.
_EDGE_PINCH_BASE_QUAT_WXYZ = np.array([0.5, 0.5, 0.5, 0.5])


def _is_rim_graspable_object_class(object_name: str) -> bool:
    """Object classes genuinely hollow/thin-walled enough for a rim grasp to
    mean anything -- shared with the legacy stage machine's own check."""
    name = object_name.lower()
    return any(kw in name for kw in RIM_GRASP_KEYWORDS)


def _is_forced_manual_object_class(object_name: str) -> bool:
    """Object classes forced onto the fixed manual pose even when its axis is
    reported infeasible -- no generator currently finds a working grasp for these."""
    name = object_name.lower()
    return any(kw in name for kw in FORCE_MANUAL_KEYWORDS)


def _manual_yaw_offset_rad(object_name: str) -> float:
    """Extra yaw (radians) added to MANUAL_DEEP_CENTER_YAW_RAD for object classes in MANUAL_YAW_OVERRIDE_KEYWORDS."""
    name = object_name.lower()
    for kw, offset in MANUAL_YAW_OVERRIDE_KEYWORDS.items():
        if kw in name:
            return offset
    return 0.0


def _object_xy_obb_world(scene: SceneQuery, object_name: str, obj_quat_wxyz: np.ndarray) -> tuple[float, float, float] | None:
    """PCA-based oriented bounding box of the object's collision geometry,
    projected onto the world-horizontal plane after rotating into its current
    orientation. Returns (angle, short_half, long_half), or None if there's no geometry to sample."""
    points_local = scene.body_surface_points_local(object_name)
    if len(points_local) < 3:
        return None
    R = quat_to_rotmat_wxyz(obj_quat_wxyz)
    points_world_oriented = points_local @ R.T  # rotate (not translate) into the object's current world orientation
    xy = points_world_oriented[:, :2]
    centered = xy - xy.mean(axis=0)
    cov = (centered.T @ centered) / len(xy)
    eigvals, eigvecs = np.linalg.eigh(cov)  # eigvecs columns are the principal axes
    proj = centered @ eigvecs
    half_extents = np.abs(proj).max(axis=0)
    short_idx = int(np.argmin(half_extents))
    long_idx = 1 - short_idx
    short_axis = eigvecs[:, short_idx]
    angle = float(np.arctan2(short_axis[1], short_axis[0]))
    return angle, float(half_extents[short_idx]), float(half_extents[long_idx])


def _object_width_at_yaw(
    scene: SceneQuery, object_name: str, obj_quat_wxyz: np.ndarray, yaw_rad: float
) -> float | None:
    """Real object width spanned when closing along the world-frame axis a
    given yaw implies, measured directly from rotated surface points rather
    than the OBB's own short axis (which is wrong whenever the caller's fixed yaw isn't the object's real short axis)."""
    points_local = scene.body_surface_points_local(object_name)
    if len(points_local) < 3:
        return None
    R = quat_to_rotmat_wxyz(obj_quat_wxyz)
    points_world_oriented = points_local @ R.T
    xy = points_world_oriented[:, :2]
    # [cos, sin], not [-sin, cos] as elsewhere -- this measures an absolute
    # yaw directly, where the other convention's 90-degree offset doesn't cancel.
    close_axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
    proj = xy @ close_axis
    return float(proj.max() - proj.min())


def _measure_local_wall_width_m(
    scene: SceneQuery,
    object_name: str,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
    point_center_world: np.ndarray,
    closing_axis_world: np.ndarray,
    fallback_width_m: float,
    search_radius_m: float = 0.02,
) -> float:
    """Real wall thickness at one rim-grasp point, measured from nearby
    surface points rather than assumed constant -- a container's wall
    thickness varies enough to trip check_gripper_closure. Falls back to fallback_width_m if too few points are found nearby."""
    points_local = scene.body_surface_points_local(object_name)
    if len(points_local) < 2:
        return fallback_width_m
    R = quat_to_rotmat_wxyz(obj_quat_wxyz)
    points_world = points_local @ R.T + obj_pos
    nearby = points_world[np.linalg.norm(points_world - point_center_world, axis=1) <= search_radius_m]
    if len(nearby) < 2:
        return fallback_width_m
    axis = closing_axis_world / np.linalg.norm(closing_axis_world)
    proj = nearby @ axis
    return float(proj.max() - proj.min())


class GraspGenerator(ABC):
    @abstractmethod
    def generate(self, scene: SceneQuery, object_names: list[str]) -> list[GraspCandidate]:
        """Return candidate grasps for every name in object_names. Order is
        not meaningful -- PickSkill scores and ranks candidates itself."""


class CompositeGraspGenerator(GraspGenerator):
    """Concatenates candidates from multiple generators into one OR-set, so
    adding a learned detector is purely additive -- its failure never removes
    the heuristic fallback's candidates. Wrapped generators must agree on
    gripper_max_opening_m, or construction raises rather than silently picking one.
    """

    def __init__(self, generators: list[GraspGenerator]) -> None:
        if not generators:
            raise ValueError("CompositeGraspGenerator needs at least one wrapped generator.")
        openings = {round(float(g.gripper_max_opening_m), 6) for g in generators}
        if len(openings) > 1:
            raise ValueError(
                f"Wrapped generators disagree on gripper_max_opening_m: {sorted(openings)!r}."
            )
        self.generators = generators
        self.gripper_max_opening_m = generators[0].gripper_max_opening_m

    def generate(self, scene: SceneQuery, object_names: list[str]) -> list[GraspCandidate]:
        candidates: list[GraspCandidate] = []
        for gen in self.generators:
            candidates.extend(gen.generate(scene, object_names))
        return candidates


class PrivilegedTopDownGraspGenerator(GraspGenerator):
    """Oracle-pose, top-down, parallel-jaw grasp sampler: fits an oriented
    bounding box to each object's collision geometry, aligns the gripper's
    closing axis to its short axis, and samples wrist yaws around that best
    angle. Grasps at the object's geometric mid-height, with no separate pregrasp waypoint.
    """

    def __init__(
        self,
        config: GraspSampleConfig | None = None,
        hand_site: str = "gripper0_right_grip_site",
        gripper_max_opening_m: float = 0.08,
    ) -> None:
        self.config = config or GraspSampleConfig()
        self.hand_site = hand_site
        self.gripper_max_opening_m = gripper_max_opening_m

    def generate(self, scene: SceneQuery, object_names: list[str]) -> list[GraspCandidate]:
        candidates: list[GraspCandidate] = []
        for name in object_names:
            candidates.extend(self._generate_for_object(scene, name))
        return candidates

    def _generate_for_object(self, scene: SceneQuery, object_name: str) -> list[GraspCandidate]:
        obj_pos, obj_quat = scene.body_pose(object_name)
        extent = scene.body_extent_world(object_name)
        # Mid-height in the object's own AABB, not body origin + half extent --
        # collision geoms aren't always centered on the body's frame origin.
        height_offset = float(extent.center_offset[2]) if extent is not None else 0.0
        grasp_z = obj_pos[2] + height_offset
        if extent is not None:
            margin = self.config.grasp_height_margin_m
            # The palm is a real collision mesh, not a point at the TCP -- for a
            # short object, mid-height can drive the palm into its top even though the TCP clears it.
            object_top_z = obj_pos[2] + extent.center_offset[2] + extent.half_extents[2]
            min_safe_for_palm = object_top_z - scene.palm_clearance_above_tcp_m() + margin
            # Mirror of the palm case: fingers extend past the TCP, so a thin
            # object's mid-height can drive fingertips below its resting surface.
            object_bottom_z = obj_pos[2] + extent.center_offset[2] - extent.half_extents[2]
            min_safe_for_floor = object_bottom_z + scene.finger_reach_below_tcp_m() + margin
            # Caps above the object's own top with a margin -- for a very thin
            # object, min_safe_for_floor can otherwise exceed it and fail at the opposite extreme.
            grasp_z = min(max(grasp_z, min_safe_for_palm, min_safe_for_floor), object_top_z - margin)
        grasp_center = np.array([obj_pos[0], obj_pos[1], grasp_z])
        approach_dir_world = np.array([0.0, 0.0, -1.0])  # top-down: approach from above

        out: list[GraspCandidate] = []
        out.extend(
            self._center_grasp_candidates(scene, object_name, grasp_center, approach_dir_world, obj_quat, extent)
        )
        # Rim-edge only makes sense for a hollow, thin-walled container -- for
        # a solid object that point is interior material, guaranteed to collide.
        if extent is not None and _is_rim_graspable_object_class(object_name):
            out.extend(
                self._rim_edge_candidates(scene, object_name, obj_pos, obj_quat, grasp_center, approach_dir_world, extent)
            )
        # Edge-pinch is a geometric trigger, not a keyword list: fires when the
        # object is too thin for safe vertical clearance and too wide for any center-grasp yaw.
        if extent is not None:
            real_height_m = 2.0 * float(extent.half_extents[2])
            widest_span_m = 2.0 * float(max(extent.half_extents[0], extent.half_extents[1]))
            if real_height_m < self.gripper_max_opening_m and widest_span_m > self.gripper_max_opening_m:
                out.extend(
                    self._edge_pinch_candidates(scene, object_name, obj_pos, obj_quat, extent)
                )
        return out

    def _center_grasp_candidates(
        self,
        scene: SceneQuery,
        object_name: str,
        grasp_center: np.ndarray,
        approach_dir_world: np.ndarray,
        obj_quat_wxyz: np.ndarray,
        extent,
    ) -> list[GraspCandidate]:
        n = max(1, int(self.config.num_yaw_samples))
        half_range = self.config.yaw_search_half_range_rad
        if n > 1:
            offsets = np.linspace(-half_range, half_range, n)
            # Force offset 0 (aligned with the object's short axis) into the set --
            # np.linspace with an even n skips the midpoint, trying only worse angles.
            offsets[np.argmin(np.abs(offsets))] = 0.0
        else:
            offsets = np.array([0.0])

        obb = _object_xy_obb_world(scene, object_name, obj_quat_wxyz)
        if obb is not None:
            world_angle, short_half, long_half = obb
            # close_axis(yaw) points at world angle yaw (confirmed against the
            # real simulated finger pads, not assumed): yaw=world_angle aligns
            # the closing axis with the object's short axis.
            best_yaw = world_angle
            yaws = best_yaw + offsets
            # Exact for a rectangle: 2*short_half at offset 0, 2*long_half at +-pi/2.
            widths = [
                float(2.0 * short_half * abs(np.cos(o)) + 2.0 * long_half * abs(np.sin(o))) for o in offsets
            ]
        else:
            # Fallback for a body with no collision geoms to sample: uniformly
            # sampled yaws with an AABB-projection width estimate.
            yaws = np.linspace(0.0, np.pi, n, endpoint=False)
            widths = [self._estimate_width(extent, float(yaw)) if extent is not None else None for yaw in yaws]

        out: list[GraspCandidate] = []
        for i, (yaw, width) in enumerate(zip(yaws, widths)):
            grasp_quat = quat_mul_wxyz(yaw_quat_wxyz(float(yaw)), _TOP_DOWN_QUAT_WXYZ)
            score = self._score(width)

            out.append(
                GraspCandidate(
                    object_name=object_name,
                    grasp_pose_world=np.concatenate([grasp_center, grasp_quat]),
                    approach_direction_world=approach_dir_world,
                    gripper_width=width,
                    generator_score=score,
                    metadata={
                        "generator": "PrivilegedTopDownGraspGenerator",
                        "strategy": "center",
                        "yaw_rad": float(yaw),
                        "yaw_index": i,
                    },
                )
            )
        return out

    def _rim_edge_candidates(
        self,
        scene: SceneQuery,
        object_name: str,
        obj_pos: np.ndarray,
        obj_quat_wxyz: np.ndarray,
        grasp_center: np.ndarray,
        approach_dir_world: np.ndarray,
        extent,
    ) -> list[GraspCandidate]:
        """Grasp a single point around the object's rim, wrist yawed so the
        gripper closes radially -- one finger outside the rim, one inside."""
        n = max(0, int(self.config.rim_num_angle_samples))
        if n == 0:
            return []
        radius = self.config.rim_offset_fraction * min(extent.half_extents[0], extent.half_extents[1])
        fallback_width = float(self.config.rim_grasp_width_m)

        out: list[GraspCandidate] = []
        for i in range(n):
            theta = 2.0 * np.pi * i / n
            radial_dir = np.array([np.cos(theta), np.sin(theta)])
            point_center = grasp_center + np.array([radius * radial_dir[0], radius * radial_dir[1], 0.0])
            # close_axis(yaw) == [cos(yaw), sin(yaw)] (confirmed against the
            # real simulated finger pads), so yaw=theta closes the fingers
            # through the object's center, straddling the rim wall rather than tangent to it.
            yaw = theta
            grasp_quat = quat_mul_wxyz(yaw_quat_wxyz(float(yaw)), _TOP_DOWN_QUAT_WXYZ)

            # Measured wall thickness at this point, not a fixed constant --
            # a mismatch here reads to check_gripper_closure as the fingers catching the wall.
            width = _measure_local_wall_width_m(
                scene, object_name, obj_pos, obj_quat_wxyz, point_center, np.array([radial_dir[0], radial_dir[1], 0.0]),
                fallback_width,
            )
            score = self._score(width)

            out.append(
                GraspCandidate(
                    object_name=object_name,
                    grasp_pose_world=np.concatenate([point_center, grasp_quat]),
                    approach_direction_world=approach_dir_world,
                    gripper_width=width,
                    generator_score=score,
                    metadata={
                        "generator": "PrivilegedTopDownGraspGenerator",
                        "strategy": "rim_edge",
                        "yaw_rad": float(yaw),
                        "rim_angle_rad": float(theta),
                        "rim_radius_m": float(radius),
                    },
                )
            )
        return out

    def _edge_pinch_candidates(
        self,
        scene: SceneQuery,
        object_name: str,
        obj_pos: np.ndarray,
        obj_quat_wxyz: np.ndarray,
        extent,
    ) -> list[GraspCandidate]:
        """Grasp a flat, wide object (plate, lid) by its outer edge: approach
        horizontally, close vertically, pinching its thickness instead of
        spanning its diameter. Uses true mid-height (no descent-clearance
        adjustment) and a radius near the true edge, since there's no rim cavity to aim inside."""
        n = max(0, int(self.config.edge_pinch_num_angle_samples))
        if n == 0:
            return []
        grasp_z = obj_pos[2] + float(extent.center_offset[2])
        radius = self.config.edge_pinch_offset_fraction * min(extent.half_extents[0], extent.half_extents[1])
        fallback_width = float(self.config.edge_pinch_width_m)

        out: list[GraspCandidate] = []
        for i in range(n):
            theta = 2.0 * np.pi * i / n
            radial_dir = np.array([np.cos(theta), np.sin(theta)])
            point_center = np.array(
                [obj_pos[0] + radius * radial_dir[0], obj_pos[1] + radius * radial_dir[1], grasp_z]
            )
            approach_dir_world = np.array([-radial_dir[0], -radial_dir[1], 0.0])  # horizontal, radially inward
            # Sweeping heading by (theta+pi) about world Z points the approach
            # axis at approach_dir_world while keeping the closing axis vertical.
            grasp_quat = quat_mul_wxyz(yaw_quat_wxyz(float(theta + np.pi)), _EDGE_PINCH_BASE_QUAT_WXYZ)

            width = _measure_local_wall_width_m(
                scene, object_name, obj_pos, obj_quat_wxyz, point_center, np.array([0.0, 0.0, 1.0]),
                fallback_width,
            )
            score = self._score(width)

            out.append(
                GraspCandidate(
                    object_name=object_name,
                    grasp_pose_world=np.concatenate([point_center, grasp_quat]),
                    approach_direction_world=approach_dir_world,
                    gripper_width=width,
                    generator_score=score,
                    metadata={
                        "generator": "PrivilegedTopDownGraspGenerator",
                        "strategy": "edge_pinch",
                        "edge_angle_rad": float(theta),
                        "edge_radius_m": float(radius),
                    },
                )
            )
        return out

    def _estimate_width(self, extent, yaw_rad: float) -> float:
        # AABB extent projected onto the closing axis; exact for a box, left
        # uncapped since capping can't distinguish a round object from a square one with the same AABB.
        close_axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
        return float(
            2.0 * abs(close_axis[0]) * extent.half_extents[0]
            + 2.0 * abs(close_axis[1]) * extent.half_extents[1]
        )

    def _score(self, width: float | None) -> float:
        # Higher = preferred before PickSkill folds in IK/collision feasibility;
        # rewards comfortable clearance inside the gripper's max opening.
        if width is None:
            return 0.5
        if width >= self.gripper_max_opening_m:
            return 0.0
        return float(np.clip(1.0 - width / self.gripper_max_opening_m, 0.0, 1.0))


def _horizontal_alignment(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two directions' horizontal parts (pitch ignored)."""
    a2, b2 = np.asarray(a, dtype=float)[:2], np.asarray(b, dtype=float)[:2]
    na, nb = float(np.linalg.norm(a2)), float(np.linalg.norm(b2))
    return 0.0 if na < 1e-9 or nb < 1e-9 else float(np.dot(a2, b2) / (na * nb))


class HandleSideGraspGenerator(GraspGenerator):
    """Horizontal grasp on a handle's outer bar, gripper pointing at the
    object's body (z = approach, x = closing, same frame convention as
    _TOP_DOWN_QUAT_WXYZ). For side-opening containers like a microwave: the
    hand stays behind the handle, outside the opening, while the object goes
    all the way in -- a top-down hand can't fit under the roof."""

    def __init__(
        self,
        gripper_max_opening_m: float = 0.08,
        z_offsets_m: tuple[float, ...] = (0.0, 0.01, -0.01),
        pitches_deg: tuple[float, ...] = (0.0, 20.0, 35.0, 50.0),
    ) -> None:
        self.gripper_max_opening_m = gripper_max_opening_m
        self.z_offsets_m = z_offsets_m
        # Tilting the approach down raises the wrist: level, it trails ~20cm
        # behind the handle at table height and hit a neighboring mug.
        self.pitches_deg = pitches_deg

    def generate(self, scene: SceneQuery, object_names: list[str]) -> list[GraspCandidate]:
        out: list[GraspCandidate] = []
        for name in object_names:
            boxes = [scene.geom_world_aabb(g) for g in scene.body_collision_geom_ids(name)]
            if len(boxes) < 4:
                continue
            centers = np.asarray([c for c, _h in boxes])
            body_xy = np.median(centers[:, :2], axis=0)
            radial = np.linalg.norm(centers[:, :2] - body_xy, axis=1)
            # A handle is the few geoms well outside the body's own wall ring.
            outer = int(np.argmax(radial))
            if radial[outer] < 1.4 * float(np.median(radial)):
                continue
            bar_center, bar_half = boxes[outer]
            outward = (bar_center[:2] - body_xy) / radial[outer]
            closing = np.array([-outward[1], outward[0], 0.0])
            width = 2.0 * float(np.dot(np.abs(closing), bar_half))
            for pitch in np.radians(self.pitches_deg):
                approach = np.array([-outward[0] * np.cos(pitch), -outward[1] * np.cos(pitch), -np.sin(pitch)])
                rot = np.column_stack([closing, np.cross(approach, closing), approach])
                quat = rotmat_to_quat_wxyz(rot)
                for dz in self.z_offsets_m:
                    position = np.array([bar_center[0], bar_center[1], bar_center[2] + dz])
                    out.append(
                        GraspCandidate(
                            object_name=name,
                            grasp_pose_world=np.concatenate([position, quat]),
                            approach_direction_world=approach,
                            gripper_width=width,
                            generator_score=1.0,
                            metadata={
                                "generator": "HandleSideGraspGenerator",
                                "strategy": "handle_side",
                                "pitch_deg": float(np.degrees(pitch)),
                                "yaw_rad": None,
                            },
                        )
                    )
        return out


MANUAL_DEEP_CENTER_YAW_RAD = np.pi / 2.0


def manual_pose_width_feasible(scene: SceneQuery, object_name: str, gripper_max_opening_m: float) -> bool:
    """Whether ManualDeepCenterGraspGenerator's one fixed-yaw pose could ever
    close on object_name, checked before any IK/collision runs. Uses the real
    width at that specific fixed axis, not the OBB's short axis, which can be
    a different (narrower) number for an elongated object."""
    obj_pos, obj_quat = scene.body_pose(object_name)
    width = _object_width_at_yaw(scene, object_name, obj_quat, MANUAL_DEEP_CENTER_YAW_RAD)
    if width is None:
        return True  # unknown -- don't preemptively exclude; let the generator's own checks decide
    return width <= gripper_max_opening_m


class ManualDeepCenterGraspGenerator(GraspGenerator):
    """Single hardcoded top-down grasp per object: centered on its (x, y), at
    the fixed orientation matching the arm's home pose, with no per-object
    rotation ever -- an object whose short axis doesn't align with this is an
    open problem to solve some other way, not by rotating. Exists because a
    learned generator's candidates can land anywhere in the safe depth band,
    letting a shallow grasp beat a reliable deep one by chance. Gated to
    non-rim-graspable objects, which have no solid center to pinch.
    """

    def __init__(
        self,
        gripper_max_opening_m: float = 0.08,
        grasp_height_margin_m: float = 0.01,
    ) -> None:
        self.gripper_max_opening_m = gripper_max_opening_m
        self.grasp_height_margin_m = grasp_height_margin_m

    def generate(self, scene: SceneQuery, object_names: list[str]) -> list[GraspCandidate]:
        out: list[GraspCandidate] = []
        for name in object_names:
            if _is_rim_graspable_object_class(name):
                continue
            obj_pos, obj_quat = scene.body_pose(name)
            # Collision-only extent, not merged visual+collision -- that's the
            # geometry that determines contact, and the two can disagree.
            extent = scene.body_collision_extent_world(name)
            if extent is None:
                continue
            margin = self.grasp_height_margin_m
            # XY center, not raw obj_pos -- collision geometry isn't always
            # centered on the body's own frame origin.
            object_center_xy = np.array([obj_pos[0], obj_pos[1]]) + extent.center_offset[:2]
            object_center_z = float(obj_pos[2] + extent.center_offset[2])
            object_top_z = object_center_z + float(extent.half_extents[2])
            object_bottom_z = object_center_z - float(extent.half_extents[2])
            min_safe_for_palm = object_top_z - scene.palm_clearance_above_tcp_m() + margin
            min_safe_for_floor = object_bottom_z + scene.finger_reach_below_tcp_m() + margin
            # Floor clearance is a hard bound the "stay below object top" clamp
            # must not override -- for a thin object that clamp can otherwise push fingertips through the table.
            ideal_deep_z = max(
                min(max(min_safe_for_palm, min_safe_for_floor), object_top_z - margin),
                min_safe_for_floor,
            )

            grasp_center = np.array([object_center_xy[0], object_center_xy[1], ideal_deep_z])
            # Fixed orientation matching the arm's home pose, non-negotiable --
            # an elongated object that fails width here fails cleanly rather than silently switching axes.
            manual_yaw_rad = MANUAL_DEEP_CENTER_YAW_RAD + _manual_yaw_offset_rad(name)
            grasp_quat = quat_mul_wxyz(yaw_quat_wxyz(manual_yaw_rad), _TOP_DOWN_QUAT_WXYZ)

            # Width at this specific fixed axis, not the OBB's short_half --
            # those only agree when the axis happens to be the object's real short axis.
            width = _object_width_at_yaw(scene, name, obj_quat, manual_yaw_rad)

            # Slide the pinch point along the object's long axis to dodge a
            # neighbor; offsets scale with object length so they never walk off a short object.
            slide_axis = np.array([-np.sin(manual_yaw_rad), np.cos(manual_yaw_rad)])
            long_span = _object_width_at_yaw(scene, name, obj_quat, manual_yaw_rad + np.pi / 2.0)
            half_long = 0.5 * long_span if long_span is not None else 0.0
            slide_offsets = [0.0]
            for frac in (0.25, 0.5):
                slide_offsets.extend([frac * half_long, -frac * half_long])

            for slide in slide_offsets:
                center = grasp_center.copy()
                center[:2] = center[:2] + slide * slide_axis
                out.append(
                    GraspCandidate(
                        object_name=name,
                        grasp_pose_world=np.concatenate([center, grasp_quat]),
                        approach_direction_world=np.array([0.0, 0.0, -1.0]),
                        gripper_width=width,
                        generator_score=self._score(width),
                        metadata={
                            "generator": "ManualDeepCenterGraspGenerator",
                            "strategy": "manual_deep_center",
                            "yaw_rad": manual_yaw_rad,
                            "slide_offset_m": float(slide),
                        },
                    )
                )
        return out

    def _score(self, width: float | None) -> float:
        if width is None:
            return 0.5
        if width >= self.gripper_max_opening_m:
            return 0.0
        return float(np.clip(1.0 - width / self.gripper_max_opening_m, 0.0, 1.0))


def hand_from_fingertip_pose(
    scene: SceneQuery,
    fingertip_pos: np.ndarray,
    fingertip_quat_wxyz: np.ndarray,
    hand_site: str = "gripper0_right_ft_frame",
    tcp_site: str = "gripper0_right_grip_site",
) -> np.ndarray:
    """Convert a grasp pose in the fingertip/TCP frame (what most grasp
    detectors output) into the panda_hand/flange frame a motion planner's IK
    targets. Reads the hand-to-TCP offset live from the model rather than a
    hardcoded constant. Returns a (7,) [x, y, z, qw, qx, qy, qz] pose.
    """
    offset_z = scene.hand_to_tcp_transform(hand_site, tcp_site)

    R = quat_to_rotmat_wxyz(fingertip_quat_wxyz)
    hand_pos = np.asarray(fingertip_pos, dtype=np.float64) - R @ np.array([0.0, 0.0, offset_z])
    return np.concatenate([hand_pos, fingertip_quat_wxyz])
