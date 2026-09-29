"""Grasp-candidate generation.

Starting point is intentionally simple, per Cap-X's privileged-state generator
(``capx/integrations/franka/libero_privileged.py``, MIT License, Copyright (c)
2026 Max Fu, https://github.com/capgym/cap-x): ``FrankaLiberoPrivilegedApi.
sample_grasp_pose()`` there is a single hardcoded top-down grasp at the
object's privileged pose with quaternion **WXYZ** ``[0, 1, 0, 0]`` (gripper
pointing straight down). This module keeps that same top-down, oracle-pose
starting point but extends it in two ways:

1. **Center-grasp candidates**: multiple wrist-yaw samples, a real pregrasp
   offset, and a documented, model-derived fingertip/hand transform instead
   of Cap-X's hardcoded ``FRANKA_HAND_TO_FINGERTIP_Z_M`` constant (see
   ``scene_query.SceneQuery.hand_to_tcp_transform``). This is the right
   strategy for anything narrower than the gripper's opening at *some* yaw
   (cans, boxes, bottles).

2. **Rim-edge candidates**: Cap-X has no object-specific grasp logic to
   borrow here -- its privileged generator is a single fixed pose, and its
   vision path (``libero.py``) delegates entirely to Contact-GraspNet, which
   finds rim/wall grasps only by sampling antipodal contacts over real
   geometry, not via a hand-coded rule. Since we don't have Contact-GraspNet
   available, this generator adds an explicit strategy for objects whose
   *whole* cross-section is too wide to encompass at any yaw (bowls, mugs,
   cups): grasp a single point around the object's rim instead, with the
   wrist yawed so the gripper's closing axis is *radial* at that point (one
   finger lands just outside the rim, the other just inside) rather than
   trying to span the object's full diameter. This is the same geometric
   idea as the legacy stage machine's fixed ``grasp_rim_offset`` (see
   ``base_env.py``'s ``_effective_grasp_rim_offset``/``modes.py``'s
   ``RIM_GRASP_KEYWORDS``) -- generalized from one hardcoded direction/object
   to *every* object, sampled at ``rim_num_angle_samples`` points around the
   full circumference instead of one fixed direction, so there's always a
   candidate on the side that's actually IK-reachable and collision-free.
   Verified against the real ``akita_black_bowl`` mesh: the legacy system's
   ``[0.0, 0.045, 0.05]`` offset (a Y-offset of 4.5cm) corresponds almost
   exactly to ``rim_offset_fraction=0.6`` of this generator's AABB-derived
   radius for that same bowl (measured rim ring radius ~0.047m vs `0.6 *
   min(half_extents[:2])` = ~0.047m) -- the default wasn't guessed, it was
   calibrated against the one object we already know a working offset for.

Multiple orientations across *both* strategies are handed to ``PickSkill``,
which is responsible for picking a working one via IK/collision/OMPL
feasibility and cost -- this generator's job is only to make sure a feasible
candidate exists somewhere in the set it proposes, not to guess which one
will work.

Quaternion convention: **WXYZ** everywhere (position of components:
``[qw, qx, qy, qz]``), matching Cap-X and MuJoCo's own native body/site quaternion
layout -- see ``scene_query.py``'s module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...modes import FORCE_MANUAL_KEYWORDS, MANUAL_YAW_OVERRIDE_KEYWORDS, RIM_GRASP_KEYWORDS
from ..config import GraspSampleConfig
from ..scene_query import SceneQuery, quat_mul_wxyz, quat_to_rotmat_wxyz
from ..types import GraspCandidate

# Top-down grasp orientation in the *world* frame, expressed as a quaternion
# that rotates the gripper's canonical +Z (approach axis) to point straight
# down (-world Z). This is the same convention as Cap-X's [0, 1, 0, 0]: a 180
# degree rotation about the world X axis.
_TOP_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# Base orientation for the edge-pinch strategy (see _edge_pinch_candidates):
# a 120-degree rotation about the (1,1,1) world axis that cyclically permutes
# the local frame -- local +Z (approach) -> world +X, local +Y (closing
# axis) -> world +Z, local +X -> world +Y. Composing a further world-frame
# yaw_quat_wxyz(heading) rotation (about world Z, the same axis the closing
# direction already maps to) sweeps the horizontal approach heading around
# the full circle while leaving the closing axis exactly vertical throughout
# -- unlike the top-down base above, where yaw sweeps the *closing* axis and
# leaves the (vertical) approach direction fixed. Same quaternion the
# grasp_generator module docstring already references for milk_1 (a
# clean 90-degree-family rotation mapping local Y to world-vertical).
_EDGE_PINCH_BASE_QUAT_WXYZ = np.array([0.5, 0.5, 0.5, 0.5])


def yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    """Quaternion for a rotation of ``yaw_rad`` about the world +Z axis."""
    half = yaw_rad / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def _is_rim_graspable_object_class(object_name: str) -> bool:
    """Same object-class keyword match ``base_env.py``'s ``_is_rim_object``
    already uses to decide the legacy stage machine's grasp mode -- reused
    here rather than re-invented so both backends agree on which objects are
    genuinely hollow/thin-walled enough for a rim grasp to mean anything."""
    name = object_name.lower()
    return any(kw in name for kw in RIM_GRASP_KEYWORDS)


def _is_forced_manual_object_class(object_name: str) -> bool:
    """Object classes routed to ``ManualDeepCenterGraspGenerator`` even when
    ``manual_pose_width_feasible`` reports the fixed axis infeasible -- see
    ``FORCE_MANUAL_KEYWORDS``'s own comment for why (a class of object where
    no generator currently available, learned or heuristic, has found a
    gripper-width-compatible candidate at its real resting geometry)."""
    name = object_name.lower()
    return any(kw in name for kw in FORCE_MANUAL_KEYWORDS)


def _manual_yaw_offset_rad(object_name: str) -> float:
    """Extra yaw (radians) added to MANUAL_DEEP_CENTER_YAW_RAD for object
    classes in MANUAL_YAW_OVERRIDE_KEYWORDS -- see that dict's own comment."""
    name = object_name.lower()
    for kw, offset in MANUAL_YAW_OVERRIDE_KEYWORDS.items():
        if kw in name:
            return offset
    return 0.0


def _object_xy_obb_world(scene: SceneQuery, object_name: str, obj_quat_wxyz: np.ndarray) -> tuple[float, float, float] | None:
    """PCA-based oriented bounding box of the object's real collision
    geometry, restricted to the WORLD-horizontal (XY) plane a top-down
    grasp's closing axis lives in. Returns ``(short_axis_world_angle_rad,
    short_half_m, long_half_m)`` -- the angle is already a world-frame
    direction a gripper should close along to need the least opening, no
    further rotation needed. ``None`` if the body has no real collision
    geometry to sample.

    Matches Cap-X's own privileged (non-learned) fallback for "regular
    objects like boxes, cylinders" -- ``get_oriented_bounding_box_from_3d_
    points`` in their ``integrations/franka/libero.py``, backed by Open3D's
    ``get_oriented_bounding_box``, which is itself PCA-based -- rather than
    their Contact-GraspNet path, which needs a separately-deployed CUDA
    inference server this environment doesn't have.

    Fixes a real bug in an earlier version of this generator: sampling
    wrist yaws in *world* frame and measuring width off a *world-axis-
    aligned* AABB, neither of which has any relationship to the object's
    real shape once it's rotated away from world-axis-aligned.

    Rotates the object's local-frame surface points (see
    ``SceneQuery.body_surface_points_local``) into the object's *current*
    world orientation before dropping Z and projecting -- an earlier
    version of this function projected onto the body's own local XY plane
    directly, which is only the true horizontal footprint if the body's
    local Z axis happens to already point world-up. Verified against
    milk_1 (quaternion ``[0.5,0.5,0.5,0.5]``, a clean 90-degree rotation
    that maps local *Y*, not Z, to world-vertical): projecting onto local
    XY there measured a vertical slice through the carton, not its real
    footprint, and picked a correspondingly wrong closing axis."""
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
    """Real object width the gripper must span when closing along the
    world-frame axis a given ``yaw_rad`` implies (``close_axis(yaw) =
    [-sin(yaw), cos(yaw)]``, the same convention used throughout this
    module) -- computed by projecting the object's real, rotated surface
    points (the same source ``_object_xy_obb_world`` uses) onto that axis
    directly, not assumed to be the object's own OBB-fitted short axis.

    Needed whenever a caller isn't free to choose the minimal-width axis --
    e.g. a fixed, non-object-dependent orientation. Reusing the OBB's own
    short_half there silently reports the *wrong* number whenever that fixed
    axis doesn't happen to line up with the object's real short axis.
    Confirmed against a real cream_cheese_1 rollout: a fixed yaw that
    actually closes along the object's *long* axis (~8.1cm) was reported as
    needing only the OBB's short_half (4.27cm), so a genuinely
    width-infeasible candidate passed the cheap width filter and only
    failed later on the real geometric collision check -- hiding the true
    rejection reason behind a misleading one.
    """
    points_local = scene.body_surface_points_local(object_name)
    if len(points_local) < 3:
        return None
    R = quat_to_rotmat_wxyz(obj_quat_wxyz)
    points_world_oriented = points_local @ R.T
    xy = points_world_oriented[:, :2]
    # [cos, sin], NOT the [-sin, cos] form used elsewhere in this module.
    # Verified against ground truth rather than assumed: with the arm at its
    # home pose (which is exactly yaw=pi/2 composed with _TOP_DOWN_QUAT_WXYZ),
    # the two finger pads' real world positions differ by [~0, 0.065, ~0] --
    # they separate along world *Y*, while [-sin, cos] at that same yaw gives
    # world X. The other convention is self-consistent where it's used
    # (PrivilegedTopDownGraspGenerator derives its yaw from the same offset
    # geometry it then measures against, so the 90 degrees cancels), but it is
    # wrong for measuring the span at an *absolute* yaw the caller fixed
    # itself. Getting this backwards reported an elongated box's long-axis
    # width (8.12cm) for a pose that actually spans its short axis (4.27cm),
    # which then fed a wildly-too-wide opening into the collision/closure
    # checks and rejected a perfectly feasible grasp.
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
    """Real wall thickness at one specific rim-grasp point, measured from the
    object's actual surface geometry near that point, instead of assuming a
    single fixed width for the entire rim. A container's wall isn't
    perfectly uniform (draft angle, a handle, a spout, an asymmetric mesh
    decomposition) -- even a small mismatch between an assumed constant and
    the real thickness *at this particular point* is exactly what
    ``CollisionChecker.check_gripper_closure`` now correctly flags as the
    fingers closing too far and catching the rigid segment on the wall
    (verified directly: several rim angles around a real bowl started
    failing gripper-closure validation specifically once that check existed,
    for a bowl whose rim thickness plausibly does vary around its
    circumference). Projects every real surface point within
    ``search_radius_m`` of ``point_center_world`` onto the closing axis --
    the same "project real geometry onto the closing axis" measurement
    ``estimate_handle_closing_width_m`` (articulation.py) uses for the
    drawer handle, just localized to one point on this object instead of a
    whole small assembly. Falls back to ``fallback_width_m`` if too few real
    points fall in that neighborhood to measure anything meaningful.
    """
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
        """Return candidate grasps for every name in ``object_names``. Order is
        not meaningful -- ``PickSkill`` scores and ranks candidates itself."""


class CompositeGraspGenerator(GraspGenerator):
    """Concatenates candidates from multiple generators into one OR-set.

    ``PickSkill`` doesn't care which generator produced a candidate -- it
    scores/filters the whole list via IK/collision/OMPL feasibility itself
    (see this module's docstring). That makes adding a second, independent
    candidate source (e.g. a learned grasp detector) purely additive: wrap it
    with the existing heuristic generator here instead of replacing it, so a
    learned detector's failure/unavailability never removes the heuristic
    fallback's candidates.

    ``PickSkill`` also reads ``grasp_generator.gripper_max_opening_m``
    directly (a property of the physical gripper, not of any one generator's
    algorithm) -- every wrapped generator must agree on it, or this raises at
    construction rather than silently picking one.
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
    """Oracle-pose, top-down, parallel-jaw grasp sampler.

    For each object: read its privileged world pose from MuJoCo, fit an
    oriented bounding box to its real collision geometry (see
    ``_object_xy_obb_world``) to find the world-frame yaw that aligns the
    gripper's closing axis with the object's own short axis, and sample
    ``config.num_yaw_samples`` wrist yaws across
    ``[best-half_range, best+half_range]`` around that (a parallel-jaw
    gripper is symmetric under a pi rotation about its approach axis, so the
    "best" yaw and "best + pi" are the same candidate). Grasp at the
    object's geometric center (offset by half its Z half-extent so the TCP
    lands at the object's mid-height rather than its base -- see the
    ``Grasped(object, height)`` mid-height convention already used by
    System 2's prompt, `step/system_two/prompts.py` rule 7b). ``PickSkill``
    plans directly to this grasp pose -- there is no separate pregrasp
    waypoint (see ``GraspCandidate``'s docstring).
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
        # Mid-height in the object's own AABB, not just the body origin + half
        # its Z extent -- a body's collision geoms aren't always centered on
        # its own frame origin (this bowl's AABB center sits 2.8cm above its
        # body origin), and using center_offset fixes both strategies' grasp
        # height instead of just approximating it off the origin.
        height_offset = float(extent.center_offset[2]) if extent is not None else 0.0
        grasp_z = obj_pos[2] + height_offset
        if extent is not None:
            margin = self.config.grasp_height_margin_m
            # The palm/hand is a real, sizeable collision mesh, not a point
            # at the TCP -- for a short/squat object, aiming for the exact
            # geometric mid-height can put the palm's own lowest point
            # *into* the object's top even though the TCP itself clears it
            # by several cm (verified against the real alphabet-soup-can
            # asset: TCP cleared by 4.3cm, but the palm's near face was only
            # 2.2cm above the TCP, so it ended up ~2cm inside the can).
            # Raise the target height just enough to keep the palm's real
            # geometry above the object's top, rather than assuming
            # "hand-to-tcp distance" alone describes how deep the gripper
            # can safely reach.
            object_top_z = obj_pos[2] + extent.center_offset[2] + extent.half_extents[2]
            min_safe_for_palm = object_top_z - scene.palm_clearance_above_tcp_m() + margin
            # Mirror image of the palm case: the rigid fingers extend *past*
            # the TCP, so for a thin/flat object, the same exact-mid-height
            # target can put the fingertips *below* the surface it rests on
            # (verified against cream_cheese_1, 2cm tall: the finger body
            # ended up ~1cm into the floor even though the TCP itself was
            # still comfortably above it). Use the object's own AABB *bottom*
            # as the floor reference, not raw obj_pos[2] -- that assumes a
            # body's origin coincides with its base, which is false for
            # cream_cheese_1 (origin sits at its geometric center instead,
            # confirmed: center_offset[2] is ~0). Using obj_pos[2] directly
            # silently assumed a floor ~1cm too high, and -- combined with
            # finger_reach_below_tcp_m() alone already exceeding this
            # object's full ~2cm height -- pushed grasp_z to 1.3cm *above*
            # the object's own top surface. Confirmed via a real rollout:
            # the arm tracked to that target with near-zero error and closed
            # on nothing (object_displacement_m: 0.0).
            object_bottom_z = obj_pos[2] + extent.center_offset[2] - extent.half_extents[2]
            min_safe_for_floor = object_bottom_z + scene.finger_reach_below_tcp_m() + margin
            # Never push the target above the object's own body -- for an
            # object shorter than finger_reach_below_tcp_m() + margin, even
            # the corrected min_safe_for_floor above can still exceed the
            # object's own top, which just relocates the same failure from
            # "1.3cm above" to "pinned exactly at the boundary with zero
            # collision-checker margin" (also confirmed by a real rollout:
            # every candidate for a too-thin object landed on the identical
            # height and was still rejected as in-collision). Back off by
            # the same margin instead of sitting on the exact boundary.
            grasp_z = min(max(grasp_z, min_safe_for_palm, min_safe_for_floor), object_top_z - margin)
        grasp_center = np.array([obj_pos[0], obj_pos[1], grasp_z])
        approach_dir_world = np.array([0.0, 0.0, -1.0])  # top-down: approach from above

        out: list[GraspCandidate] = []
        out.extend(
            self._center_grasp_candidates(scene, object_name, grasp_center, approach_dir_world, obj_quat, extent)
        )
        # Rim-edge only makes physical sense for a hollow, thin-walled
        # container -- it deliberately straddles a *wall*, positioning the
        # closing point at a fraction of the object's radius rather than at
        # its true outer edge. For a solid object (a can, a box, most
        # LIBERO groceries) there is no wall there to straddle: that point
        # is just interior solid material, so the candidate is geometrically
        # guaranteed to have the gripper collide no matter what height is
        # tried. Verified directly against the alphabet-soup-can asset: its
        # own decomposed collision geoms include one sitting almost exactly
        # on the object's central axis (2.3mm off-center) -- solid, not
        # hollow. Reuses the same object-class keyword list the legacy stage
        # machine already uses for this exact distinction (modes.py's
        # RIM_GRASP_KEYWORDS) rather than inventing a second classification.
        if extent is not None and _is_rim_graspable_object_class(object_name):
            out.extend(
                self._rim_edge_candidates(scene, object_name, obj_pos, obj_quat, grasp_center, approach_dir_world, extent)
            )
        # Edge-pinch: a *geometric* trigger, not a keyword list, deliberately
        # -- unlike rim-edge above, "is this object flat and wide" doesn't
        # need object-class knowledge to answer, and a keyword list would
        # only repeat the exact fragility already hit once this session (a
        # real "plate" object silently missing from RIM_GRASP_KEYWORDS).
        # Triggers whenever the object is both too *thin* for the top-down
        # strategies' vertical clearance logic to find a safe height (real
        # height less than the gripper's max opening) and too *wide* for any
        # center-grasp yaw to span (confirmed empirically against the real
        # plate_1 LIBERO asset: every rim-edge candidate there also failed,
        # 1-1.5cm penetrations into solid material -- a flat disc has no
        # hollow wall to straddle at any radius, so that strategy alone
        # isn't sufficient here the way it is for a bowl).
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
            # Guarantee the true optimum (offset 0, aligned with the
            # object's own short axis) is always one of the candidates.
            # np.linspace with an even n skips the midpoint entirely (e.g.
            # n=2 returns only the two range endpoints), which would mean
            # only the *worst*-explored angles in the range ever get tried.
            offsets[np.argmin(np.abs(offsets))] = 0.0
        else:
            offsets = np.array([0.0])

        obb = _object_xy_obb_world(scene, object_name, obj_quat_wxyz)
        if obb is not None:
            world_angle, short_half, long_half = obb
            # close_axis(yaw) = [-sin(yaw), cos(yaw)] points at world angle
            # (yaw + pi/2); solve for the yaw whose closing axis aligns with
            # the object's own short axis -- offsets are then explored
            # *around* that, not blindly across the whole [0, pi) range.
            best_yaw = world_angle - np.pi / 2.0
            yaws = best_yaw + offsets
            # Exact for a true rectangle with these half-extents: at
            # offset=0 (aligned with the short axis) this is exactly
            # 2*short_half; at offset=+-pi/2 it's 2*long_half.
            widths = [
                float(2.0 * short_half * abs(np.cos(o)) + 2.0 * long_half * abs(np.sin(o))) for o in offsets
            ]
        else:
            # Fallback for a body with no discrete collision geoms to sample
            # (shouldn't happen for a real LIBERO object): the old
            # world-frame AABB-projection estimate, uniformly sampled.
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
        """Grasp a single point around the object's rim/perimeter, wrist
        yawed so the gripper closes radially (one finger just outside the
        rim, one just inside) -- see module docstring for why this exists and
        how the default offset fraction was calibrated."""
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
            # _estimate_width's close_axis for yaw is [-sin(yaw), cos(yaw)];
            # solving close_axis == radial_dir gives yaw = theta - pi/2, so the
            # fingers close along the line through the object's center (i.e.
            # straddle the rim wall) instead of tangent to it.
            yaw = theta - np.pi / 2.0
            grasp_quat = quat_mul_wxyz(yaw_quat_wxyz(float(yaw)), _TOP_DOWN_QUAT_WXYZ)

            # The wall's real thickness at *this specific point*, not a
            # single fixed constant for the whole rim -- a container's wall
            # isn't perfectly uniform (draft angle, a handle, a spout), and
            # even a small mismatch between the assumed and real thickness
            # is exactly what CollisionChecker.check_gripper_closure now
            # catches (correctly) as the fingers closing too far and
            # catching the rigid segment on the wall. See
            # _measure_local_wall_width_m's docstring.
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
        """Grasp a flat, wide object (a plate, saucer, lid -- see the
        geometric trigger in ``_generate_for_object``) by its outer edge,
        approaching *horizontally* and closing *vertically*: one finger
        lands above the rim, one below, pinching the object's own thickness
        at that point (a couple cm) instead of trying to span its full
        diameter (the center-grasp strategy, which can't -- that's the whole
        reason this strategy exists) or straddle a hollow wall that, for a
        solid disc, simply isn't there at any radius (the rim-edge strategy
        above, which for the same reason is geometrically guaranteed to
        fail here too).

        Height is the object's own true geometric mid-height, not the
        top-down strategies' palm/finger-clearance-adjusted height -- that
        adjustment is about vertical descent clearance, which doesn't apply
        to a horizontal approach. Radius is close to (not a fraction well
        inside) the object's true outer edge, since there's no rim-cavity
        depth to aim inside of the way there is for a bowl."""
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
            # Sweeping the approach heading by (theta + pi) about world Z on
            # top of _EDGE_PINCH_BASE_QUAT_WXYZ points the approach axis at
            # `approach_dir_world` above while leaving the closing axis
            # (already aligned with world Z, the rotation axis) vertical
            # regardless of heading -- see that constant's own docstring.
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
        # Width the gripper must span across is the object's AABB extent
        # projected onto the axis the fingers close along (gripper local X
        # after the yaw rotation). This is exact for a box-shaped object.
        #
        # An earlier version capped this at the object's narrowest overall
        # horizontal extent, reasoning that a rotated-square-AABB projection
        # *overestimates* a round object's true (yaw-invariant) cross-section
        # at oblique angles (e.g. predicting the AABB's diagonal instead of
        # the actual diameter for a cylinder with near-equal XY half-extents).
        # That's true for genuinely round objects, but the cap can't tell a
        # round object from a square one with the same AABB -- and applying
        # it unconditionally made every yaw report the *same* optimistic
        # width regardless of the object's real shape. Verified against a
        # real (square-ish, not round) soup-can asset: capping reported
        # ~8.6cm at every sampled yaw, hiding that the true projected width
        # actually ranged from ~9.0cm to ~12.7cm depending on yaw (a ~1.41x
        # spread matching a square cross-section's diagonal-to-side ratio).
        # PickSkill's own IK/collision check then had no way to prefer the
        # near-feasible yaw over the badly-infeasible ones, since the width
        # pre-check saw them as identical.
        #
        # Leaving this uncapped is still safe for genuinely round objects:
        # multiple yaws are already sampled, so the axis-aligned ones (where
        # this formula is exact even for a circle) remain correctly
        # estimated and available -- an oblique yaw being pessimistically
        # over-estimated just means PickSkill prefers one of those instead,
        # not that round objects become ungraspable.
        close_axis = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)])
        return float(
            2.0 * abs(close_axis[0]) * extent.half_extents[0]
            + 2.0 * abs(close_axis[1]) * extent.half_extents[1]
        )

    def _score(self, width: float | None) -> float:
        # Higher score = more preferred by the generator alone (before IK/
        # collision/downstream feasibility are folded in by PickSkill). Prefer
        # grasps whose required opening leaves comfortable clearance inside the
        # gripper's max opening; penalize widths that are too close to (or
        # exceed) the physical limit.
        if width is None:
            return 0.5
        if width >= self.gripper_max_opening_m:
            return 0.0
        return float(np.clip(1.0 - width / self.gripper_max_opening_m, 0.0, 1.0))


MANUAL_DEEP_CENTER_YAW_RAD = np.pi / 2.0


def manual_pose_width_feasible(scene: SceneQuery, object_name: str, gripper_max_opening_m: float) -> bool:
    """Whether ``ManualDeepCenterGraspGenerator``'s one fixed-yaw pose could
    ever physically close on ``object_name`` at all, before any IK/collision
    check runs.

    The generator's own fixed axis is non-negotiable (see its class
    docstring), so an object whose real width *at that specific axis* -- not
    its OBB-fitted short axis, which can be a completely different,
    narrower number for an elongated object -- already exceeds the
    gripper's opening has no candidate this generator could ever accept,
    regardless of how many it emits or how they're scored. Confirmed on a
    real book (STUDY_SCENE1's black_book_1): every one of 5 manual
    candidates was rejected as ``gripper_width_incompatible`` at ~9.3cm
    against an 8cm gripper -- a flat, elongated object where the fixed axis
    happens to span its long dimension, the same failure mode already
    documented for cream_cheese_1/butter_1 in ``generate()``'s own comment,
    just never checked for *before* committing to manual-only (see
    ``libero_pick_adapter.py``'s own ``manual_only`` classification, which
    this exists to correct)."""
    obj_pos, obj_quat = scene.body_pose(object_name)
    width = _object_width_at_yaw(scene, object_name, obj_quat, MANUAL_DEEP_CENTER_YAW_RAD)
    if width is None:
        return True  # unknown -- don't preemptively exclude; let the generator's own checks decide
    return width <= gripper_max_opening_m


class ManualDeepCenterGraspGenerator(GraspGenerator):
    """Single hardcoded top-down grasp per object: centered on the object's
    own (x, y), the fixed orientation that matches the arm's own home/resting
    pose exactly (see ``manual_yaw_rad``'s own comment in ``generate()``) --
    no per-object rotation, no search, no short-axis alignment, and no
    reorientation away from home at all, for any object. This is a hard
    constraint, not a default: an object whose real short axis doesn't line
    up with this fixed orientation (e.g. butter_1/cream_cheese_1) is a known
    open problem to solve by other means (grasp position, width handling,
    approach), never by rotating. Aimed at the same palm/floor-clearance-safe
    *deep* target both other generators already use only as a safety
    *ceiling* (see ``ContactGraspNetGenerator._generate_for_object``'s
    ``min_safe_z`` clamp).

    Exists because that ceiling is a floor, not an attractor: a learned
    generator's own candidates can legally sit anywhere in the safe band with
    no preference for the deep end, and a real rollout confirmed this
    matters -- ContactGraspNet's 32 candidates for the alphabet-soup-can
    spanned 79%-105% up the can with ~zero correlation between height and
    the network's own confidence (r=-0.05), so a shallow, rim-adjacent grasp
    could (and did) win purely by chance over a deep, reliable one sitting
    right next to it in the pool. Rather than only reweighting the OR-set's
    cost function to *prefer* depth (see ``PickSkill._score_candidate``'s
    ``grasp_depth_penalty``, which still helps), this generator adds one
    candidate that is already exactly where a reliable grasp on a solid,
    center-graspable object should be -- confirmed 5/5 deterministic
    successes replacing GraspNet/heuristic entirely for the can, vs. the
    combined pipeline needing up to 5 retries against the same object.

    Gated to non-rim-graspable objects only (the same ``_is_rim_graspable_object_class``
    check ``PrivilegedTopDownGraspGenerator`` uses before adding its own
    rim-edge candidates): a hollow object (bowl, mug, cup) has no solid
    material at its center for a straight-through pinch to close on, so a
    candidate like this one is guaranteed wrong there. This exact failure
    mode already burned once when the heuristic's own center candidate --
    cheap to reach, unscored by any generator confidence -- won the OR-set
    over every real GraspNet candidate for a ramekin. This generator's
    ``generator_score`` still reflects real gripper-opening margin (same
    formula as ``PrivilegedTopDownGraspGenerator._score``), not an
    artificial maximum, so it competes on the same terms as every other
    candidate rather than being hand-favored.
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
            # Collision-only, not the merged visual+collision extent: this is
            # the geometry that actually determines contact, and the two can
            # disagree (see body_collision_extent_world's own docstring).
            extent = scene.body_collision_extent_world(name)
            if extent is None:
                continue
            margin = self.grasp_height_margin_m
            # XY center, not just obj_pos[0]/[1] directly -- a body's real
            # collision geometry isn't always centered on its own frame
            # origin (same reasoning already applied to Z below, just never
            # extended to XY). Confirmed against the real alphabet_soup_1
            # can: its collision box's own Y center sits 3.7mm off the body
            # origin, meaning every manual-pose grasp on it targeted a point
            # 3.7mm off the can's true center -- small on its own, but a real,
            # avoidable bias on top of ordinary IK/servo tracking error,
            # compounding into a visibly off-center grasp.
            object_center_xy = np.array([obj_pos[0], obj_pos[1]]) + extent.center_offset[:2]
            object_center_z = float(obj_pos[2] + extent.center_offset[2])
            object_top_z = object_center_z + float(extent.half_extents[2])
            object_bottom_z = object_center_z - float(extent.half_extents[2])
            min_safe_for_palm = object_top_z - scene.palm_clearance_above_tcp_m() + margin
            min_safe_for_floor = object_bottom_z + scene.finger_reach_below_tcp_m() + margin
            # Floor clearance is a hard bound the "stay below the object's own
            # top" clamp must not override. For anything thinner than the
            # fingers' own reach below the TCP, that clamp used to win and
            # push the fingertips straight through the table -- confirmed
            # against the real cream_cheese_1/butter_1 assets (both ~1.8cm
            # tall): the clamp discarded a floor bound sitting 1.2cm higher,
            # and the resulting pose put both finger geoms ~1.9mm into the
            # floor, rejecting every candidate as grasp_pose_in_collision.
            # Grasping a thin object nearer its top edge still closes on it
            # (the pads reach below the TCP, so they straddle its upper half);
            # driving them into the table never works.
            ideal_deep_z = max(
                min(max(min_safe_for_palm, min_safe_for_floor), object_top_z - margin),
                min_safe_for_floor,
            )

            grasp_center = np.array([object_center_xy[0], object_center_xy[1], ideal_deep_z])
            # A single fixed orientation for every object, no per-object
            # search or computed alignment, and -- non-negotiable, per
            # explicit instruction -- no rotation away from the arm's own
            # home/resting orientation, ever, for any object. Confirmed via
            # a real rollout (home_ee_quat measured directly) that home's
            # own orientation is exactly this +90-degree-from-base pose, so
            # this candidate requires zero wrist reorientation to reach.
            #
            # For an elongated box (butter_1/cream_cheese_1) this closes
            # along the object's *long* axis, which the real mesh doesn't
            # fit inside an 8cm gripper at this width (confirmed: ~1.3mm
            # real contact penetration on both pads) -- a genuine open
            # problem to solve *without* rotating, not a reason to rotate.
            # Left as a known-infeasible-for-now case rather than silently
            # switching axes: rejected candidates fail cleanly through the
            # ordinary width/collision checks below, same as any other
            # infeasible candidate.
            manual_yaw_rad = MANUAL_DEEP_CENTER_YAW_RAD + _manual_yaw_offset_rad(name)
            grasp_quat = quat_mul_wxyz(yaw_quat_wxyz(manual_yaw_rad), _TOP_DOWN_QUAT_WXYZ)

            # The width the gripper needs to span *at this specific fixed
            # axis* -- not the OBB's short_half, which is the object's
            # minimum possible width across ALL orientations. Those two only
            # agree when the fixed axis happens to be the object's real short
            # axis (true for a round can, false for an elongated box).
            # Confirmed against a real cream_cheese_1 rollout: reporting the
            # OBB short_half (4.27cm) here for a fixed axis that actually
            # spans the object's long dimension (~8.1cm) let a genuinely
            # width-infeasible candidate pass the cheap width filter and
            # fail only later on the real collision check, with a rejection
            # reason that didn't say why.
            width = _object_width_at_yaw(scene, name, obj_quat, manual_yaw_rad)

            # Slide the pinch point along the axis the fingers do NOT close on
            # (perpendicular to the closing axis, i.e. the object's long
            # dimension for an elongated box). This is the one degree of
            # freedom available when the centered grasp is blocked by a
            # neighbor that rotating would otherwise be used to dodge: the
            # orientation, depth and gripper opening are all unchanged, only
            # *where along the bar* the pads pinch moves. Confirmed against
            # the real libero_10 cream-cheese scene, where a milk carton sits
            # ~8cm away and put itself between the gripper's own hand body and
            # the target -- `milk_1_g1 <-> gripper0_hand_collision` at the
            # centered pose. Raising the grasp does not help there (the carton
            # is 13cm tall, its top well above any grasp height), but sliding
            # 2cm along the long axis clears it at every height.
            #
            # Centered stays first and is what PickSkill's cost function
            # normally selects; the offsets only matter when the centered pose
            # is rejected outright. Offsets scale with the object's own length
            # rather than being fixed, so they never walk off the end of a
            # short object.
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
    """Convert a grasp pose expressed in the fingertip/TCP frame (the
    convention most grasp detectors -- including Contact-GraspNet, per Cap-X's
    ``graspnet.py`` -- output) into the ``panda_hand``/flange frame that a
    motion planner's IK typically targets.

    Mirrors Cap-X's ``curobo_api._grasp_pose_fingertip_to_hand`` in spirit, but
    reads the offset live from the model (``SceneQuery.hand_to_tcp_transform``)
    instead of hardcoding ``FRANKA_HAND_TO_FINGERTIP_Z_M``. Returns a (7,)
    ``[x, y, z, qw, qx, qy, qz]`` pose.
    """
    offset_z = scene.hand_to_tcp_transform(hand_site, tcp_site)
    from ..scene_query import quat_to_rotmat_wxyz

    R = quat_to_rotmat_wxyz(fingertip_quat_wxyz)
    hand_pos = np.asarray(fingertip_pos, dtype=np.float64) - R @ np.array([0.0, 0.0, offset_z])
    return np.concatenate([hand_pos, fingertip_quat_wxyz])
