"""Grasp-candidate generation.

Starting point is intentionally simple, per Cap-X's privileged-state generator
(``capx/integrations/franka/libero_privileged.py``, MIT License,
https://github.com/capgym/cap-x): ``FrankaLiberoPrivilegedApi.
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
   vision path delegates entirely to Contact-GraspNet, which this generator
   doesn't have available. Instead it adds an explicit strategy for objects
   whose *whole* cross-section is too wide to encompass at any yaw (bowls,
   mugs, cups): grasp a single point around the object's rim, with the wrist
   yawed so the gripper's closing axis is *radial* at that point (one finger
   just outside the rim, one just inside) rather than spanning the full
   diameter. This generalizes the legacy stage machine's fixed
   ``grasp_rim_offset`` (``base_env.py``/``modes.py``'s
   ``RIM_GRASP_KEYWORDS``) from one hardcoded direction to every object,
   sampled at ``rim_num_angle_samples`` points around the full circumference
   so there's always a candidate on a reachable, collision-free side. The
   default ``rim_offset_fraction`` is calibrated against a real bowl mesh
   with a known-working legacy offset, not guessed.

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
from dataclasses import dataclass

import numpy as np

from ...modes import FORCE_MANUAL_KEYWORDS, MANUAL_YAW_OVERRIDE_KEYWORDS, RIM_GRASP_KEYWORDS
from ..ik_utils import yaw_quat_wxyz
from ...motion.scene_query import SceneQuery, quat_mul_wxyz, quat_to_rotmat_wxyz
from ...motion.types import GraspCandidate


@dataclass
class GraspSampleConfig:
    num_yaw_samples: int = 8
    # Center-grasp yaws are sampled across [best-half_range, best+half_range],
    # where "best" is computed from the object's own real geometry (an
    # oriented-bounding-box fit -- see grasp_generator._object_xy_obb_world),
    # not blindly across the full [0, pi) world-frame range: the object's
    # true shape and current orientation already tell us where the good
    # candidates are, so samples are concentrated there instead of mostly
    # landing on angles we already know are worse.
    yaw_search_half_range_rad: float = 0.7853981633974483  # pi/4, i.e. +-45 degrees
    lift_check_m: float = 0.05
    # Extra lift after the grasp-verification test lift so the next skill
    # (usually transport) has enough clearance to plan a path -- starting low
    # and table-adjacent routinely caused NO_PATH_FOUND. 0 disables it.
    post_grasp_carry_lift_m: float = 0.10
    # Standoff distance for the final approach segment (see PickSkill's
    # module docstring): the long-range approach plans under strict
    # FREE_SPACE to a point this far from the grasp pose along the approach
    # direction, then a short final descent from there to the grasp pose is
    # explicitly collision-validated (never blindly executed) under
    # TERMINAL_APPROACH. Not a separate caller-visible stage -- PickSkill
    # is still one atomic plan call -- just where the "no contact yet" ->
    # "contact with the target is now expected" transition happens.
    approach_standoff_m: float = 0.08
    # Lateral offsets (in the plane perpendicular to the approach direction)
    # tried for the standoff position if the straight-line one (directly
    # above the object, for a top-down grasp) is unreachable or in collision.
    # A fixed vertical standoff assumes the airspace above the object is
    # always clear, which an earlier subtask (e.g. an opened drawer) can
    # violate; offsetting sideways lets the same validation find a standoff
    # approaching from a shallow angle instead. `(0.0, 0.0)` (straight up) is
    # tried implicitly first by the caller before any of these.
    # Multiples of `approach_standoff_m` tried for the standoff distance
    # itself (1.0 = nominal), before/alongside the lateral search below: an
    # obstacle hovering over the object can block every lateral direction at
    # the nominal standoff height (a non-gripper arm link sweeps through its
    # footprint regardless of heading), so a taller standoff can clear the
    # obstacle's height entirely instead. See standoff_search.standoff_candidates.
    standoff_distance_multipliers: tuple[float, ...] = (1.0, 2.5, 4.0)
    # More radii and finer angular resolution than a narrower first pass,
    # which left some candidates blocked when a wide overhanging obstacle
    # wasn't reliably dodged by a few evenly-spaced directions. Costs a few
    # extra IK/collision checks per candidate (cheap next to a full OMPL
    # plan) for meaningfully better odds of a clear approach angle.
    standoff_lateral_offsets_m: tuple[float, ...] = (0.05, 0.08, 0.12)
    standoff_lateral_directions: int = 8  # tried around a full circle at each offset magnitude
    # Rim/edge grasp strategy (bowls, mugs, cups: objects whose full
    # cross-section doesn't fit the gripper at any yaw). See
    # grasp_generator.PrivilegedTopDownGraspGenerator's module docstring.
    rim_num_angle_samples: int = 8
    rim_offset_fraction: float = 0.6
    rim_grasp_width_m: float = 0.02
    # Edge-pinch grasp strategy (plates, saucers, lids: flat/thin objects
    # whose *height*, not just cross-section, makes the rim-edge wall-
    # straddle strategy above physically meaningless -- there's no hollow
    # wall to straddle at any radius, just uniformly solid material all the
    # way to the edge. See grasp_generator._edge_pinch_candidates).
    edge_pinch_num_angle_samples: int = 8
    edge_pinch_offset_fraction: float = 0.92
    edge_pinch_width_m: float = 0.02
    # Randomized-seed IK retries for the standoff/grasp solves below (same
    # pattern as PlaceSampleConfig.pose_selection_attempts): the arm's current
    # resting pose is a good seed for a top-down grasp but not always for
    # others, e.g. a rim-edge candidate.
    pose_selection_attempts: int = 6
    # Safety margin (m) added on top of the gripper's real measured reach
    # (see SceneQuery.palm_clearance_above_tcp_m / finger_reach_below_tcp_m)
    # when raising a grasp height to keep the palm off a short object's top,
    # or the fingertips off the floor under a thin one.
    grasp_height_margin_m: float = 0.01
    # RRTConnect is stochastic (see TransportConfig's own comment on this) --
    # without a retry, an otherwise-valid candidate that passed IK/collision
    # can still be rejected outright on a single unlucky NO_PATH_FOUND.
    planning_attempts: int = 6

# Top-down grasp orientation in the *world* frame, expressed as a quaternion
# that rotates the gripper's canonical +Z (approach axis) to point straight
# down (-world Z). This is the same convention as Cap-X's [0, 1, 0, 0]: a 180
# degree rotation about the world X axis.
_TOP_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# Base orientation for the edge-pinch strategy (see _edge_pinch_candidates):
# a 120-degree rotation about the (1,1,1) world axis that cyclically permutes
# the local frame -- local +Z (approach) -> world +X, local +Y (closing
# axis) -> world +Z, local +X -> world +Y. Composing a further world-frame
# yaw_quat_wxyz(heading) rotation (about world Z, the axis the closing
# direction already maps to) sweeps the horizontal approach heading around
# the full circle while leaving the closing axis exactly vertical -- unlike
# the top-down base above, where yaw sweeps the closing axis instead and
# leaves the (vertical) approach direction fixed.
_EDGE_PINCH_BASE_QUAT_WXYZ = np.array([0.5, 0.5, 0.5, 0.5])


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

    Rotates the object's local-frame surface points (see
    ``SceneQuery.body_surface_points_local``) into the object's *current*
    world orientation before dropping Z and projecting -- projecting onto the
    body's own local XY plane directly is only the true horizontal footprint
    if the body's local Z axis already points world-up, which isn't true for
    every asset's resting orientation."""
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
    short_half there silently reports the *wrong* (too-small) number
    whenever that fixed axis doesn't line up with the object's real short
    axis, letting a genuinely width-infeasible candidate pass the cheap
    width filter and fail later with a misleading rejection reason.
    """
    points_local = scene.body_surface_points_local(object_name)
    if len(points_local) < 3:
        return None
    R = quat_to_rotmat_wxyz(obj_quat_wxyz)
    points_world_oriented = points_local @ R.T
    xy = points_world_oriented[:, :2]
    # [cos, sin], NOT the [-sin, cos] form used elsewhere in this module --
    # verified against the real finger pad separation at the arm's home pose.
    # The other convention is self-consistent where it's used
    # (PrivilegedTopDownGraspGenerator derives its yaw from the same offset
    # geometry it then measures against, so the 90 degrees cancels), but it's
    # wrong for measuring the span at an absolute yaw the caller fixed itself
    # -- getting this backwards reports the long-axis width for a pose that
    # actually spans the short axis, hiding a width-infeasible grasp.
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
    single fixed width for the entire rim. A container's wall isn't perfectly
    uniform (draft angle, a handle, a spout, an asymmetric mesh
    decomposition), and even a small mismatch between an assumed constant and
    the real thickness there is enough for ``CollisionChecker.
    check_gripper_closure`` to flag the fingers as catching the rigid segment
    on the wall. Projects every real surface point within ``search_radius_m``
    of ``point_center_world`` onto the closing axis -- the same technique
    ``estimate_handle_closing_width_m`` (articulation.py) uses for the drawer
    handle, localized to one point instead of a whole assembly. Falls back to
    ``fallback_width_m`` if too few real points fall in that neighborhood.
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
        # its own frame origin, and using center_offset fixes both
        # strategies' grasp height instead of just approximating it.
        height_offset = float(extent.center_offset[2]) if extent is not None else 0.0
        grasp_z = obj_pos[2] + height_offset
        if extent is not None:
            margin = self.config.grasp_height_margin_m
            # The palm is a real, sizeable collision mesh, not a point at the
            # TCP -- for a short/squat object, the exact geometric mid-height
            # can put the palm's lowest point into the object's top even
            # though the TCP itself clears it by several cm. Raise the target
            # height just enough to keep the palm's real geometry above the
            # object's top.
            object_top_z = obj_pos[2] + extent.center_offset[2] + extent.half_extents[2]
            min_safe_for_palm = object_top_z - scene.palm_clearance_above_tcp_m() + margin
            # Mirror image of the palm case: the rigid fingers extend past the
            # TCP, so for a thin/flat object the same mid-height target can
            # put the fingertips below the surface it rests on. Uses the
            # object's own AABB bottom as the floor reference, not raw
            # obj_pos[2] -- that assumes a body's origin coincides with its
            # base, which isn't true for every asset.
            object_bottom_z = obj_pos[2] + extent.center_offset[2] - extent.half_extents[2]
            min_safe_for_floor = object_bottom_z + scene.finger_reach_below_tcp_m() + margin
            # Never push the target above the object's own body -- for a very
            # thin object, even the corrected min_safe_for_floor above can
            # still exceed the object's own top, which just relocates the
            # failure to the opposite extreme (pinned exactly on the
            # collision checker's own boundary with zero margin). Back off by
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
        # closing point at a fraction of the object's radius rather than its
        # true outer edge. For a solid object there's no wall to straddle:
        # that point is just interior solid material, geometrically
        # guaranteed to collide. Reuses the same object-class keyword list
        # the legacy stage machine uses for this distinction (modes.py's
        # RIM_GRASP_KEYWORDS) rather than inventing a second classification.
        if extent is not None and _is_rim_graspable_object_class(object_name):
            out.extend(
                self._rim_edge_candidates(scene, object_name, obj_pos, obj_quat, grasp_center, approach_dir_world, extent)
            )
        # Edge-pinch: a *geometric* trigger, not a keyword list -- unlike
        # rim-edge above, "is this object flat and wide" doesn't need
        # object-class knowledge to answer, and a keyword list risks silently
        # missing an object class. Triggers whenever the object is both too
        # thin for the top-down strategies' vertical clearance logic to find
        # a safe height, and too wide for any center-grasp yaw to span -- a
        # flat disc has no hollow wall to straddle at any radius, so rim-edge
        # alone isn't sufficient here the way it is for a bowl.
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

            # The wall's real thickness at this specific point, not a single
            # fixed constant for the whole rim -- a container's wall isn't
            # perfectly uniform, and CollisionChecker.check_gripper_closure
            # will flag even a small mismatch as the fingers catching the
            # rigid segment on the wall. See _measure_local_wall_width_m.
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
        # Width the gripper must span is the object's AABB extent projected
        # onto the axis the fingers close along (gripper local X after the
        # yaw rotation). Exact for a box; deliberately left uncapped rather
        # than clamped to the object's narrowest horizontal extent -- a cap
        # can't distinguish a round object (where it would correctly avoid
        # overestimating) from a square one with the same AABB (where it
        # would hide genuinely different widths at different yaws behind one
        # identical number). Multiple yaws are already sampled, so a round
        # object's axis-aligned yaws remain correctly estimated regardless.
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
    its OBB-fitted short axis, which can be a different, narrower number for
    an elongated object -- already exceeds the gripper's opening has no
    candidate this generator could ever accept. Checked up front here rather
    than discovered only after every manual candidate is rejected (see
    ``libero_pick_adapter.py``'s ``manual_only`` classification, which this
    exists to correct)."""
    obj_pos, obj_quat = scene.body_pose(object_name)
    width = _object_width_at_yaw(scene, object_name, obj_quat, MANUAL_DEEP_CENTER_YAW_RAD)
    if width is None:
        return True  # unknown -- don't preemptively exclude; let the generator's own checks decide
    return width <= gripper_max_opening_m


class ManualDeepCenterGraspGenerator(GraspGenerator):
    """Single hardcoded top-down grasp per object: centered on the object's
    own (x, y), at the fixed orientation that matches the arm's own
    home/resting pose exactly -- no per-object rotation, search, short-axis
    alignment, or reorientation away from home, for any object. This is a
    hard constraint, not a default: an object whose real short axis doesn't
    line up with this fixed orientation is a known open problem to solve by
    other means (grasp position, width handling, approach), never by
    rotating. Aimed at the same palm/floor-clearance-safe deep target the
    other generators use only as a safety ceiling (see
    ``ContactGraspNetGenerator._generate_for_object``'s ``min_safe_z`` clamp).

    Exists because that ceiling is a floor, not an attractor: a learned
    generator's own candidates can legally sit anywhere in the safe band with
    no preference for the deep end, which real testing showed costs reliability
    -- a shallow, rim-adjacent grasp can win purely by chance over a deep,
    reliable one sitting right next to it in the pool. Rather than only
    reweighting the OR-set's cost function to prefer depth (see
    ``PickSkill._score_candidate``'s ``grasp_depth_penalty``, which still
    helps), this generator adds one candidate that is already exactly where a
    reliable grasp on a solid, center-graspable object should be.

    Gated to non-rim-graspable objects only (the same
    ``_is_rim_graspable_object_class`` check ``PrivilegedTopDownGraspGenerator``
    uses before adding its own rim-edge candidates): a hollow object has no
    solid material at its center for a straight-through pinch to close on, so
    a candidate like this one is guaranteed wrong there -- the heuristic's own
    cheap, unscored center candidate has previously won the OR-set over a
    real GraspNet candidate for exactly this kind of object. This generator's
    ``generator_score`` still reflects real gripper-opening margin (same
    formula as ``PrivilegedTopDownGraspGenerator._score``), not an artificial
    maximum, so it competes on the same terms as every other candidate.
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
            # extended to XY).
            object_center_xy = np.array([obj_pos[0], obj_pos[1]]) + extent.center_offset[:2]
            object_center_z = float(obj_pos[2] + extent.center_offset[2])
            object_top_z = object_center_z + float(extent.half_extents[2])
            object_bottom_z = object_center_z - float(extent.half_extents[2])
            min_safe_for_palm = object_top_z - scene.palm_clearance_above_tcp_m() + margin
            min_safe_for_floor = object_bottom_z + scene.finger_reach_below_tcp_m() + margin
            # Floor clearance is a hard bound the "stay below the object's own
            # top" clamp must not override -- for anything thinner than the
            # fingers' own reach below the TCP, that clamp can otherwise win
            # and push the fingertips through the table. Grasping a thin
            # object nearer its top edge still closes on it (the pads reach
            # below the TCP, straddling its upper half); driving them into
            # the table never works.
            ideal_deep_z = max(
                min(max(min_safe_for_palm, min_safe_for_floor), object_top_z - margin),
                min_safe_for_floor,
            )

            grasp_center = np.array([object_center_xy[0], object_center_xy[1], ideal_deep_z])
            # A single fixed orientation for every object -- non-negotiable,
            # per explicit instruction -- that matches the arm's home/resting
            # orientation exactly, so this candidate requires zero wrist
            # reorientation to reach. For an elongated object this can close
            # along the object's long axis and fail the gripper's real
            # opening; that's a genuine open problem to solve without
            # rotating, so such candidates are left to fail cleanly through
            # the ordinary width/collision checks below rather than silently
            # switching axes.
            manual_yaw_rad = MANUAL_DEEP_CENTER_YAW_RAD + _manual_yaw_offset_rad(name)
            grasp_quat = quat_mul_wxyz(yaw_quat_wxyz(manual_yaw_rad), _TOP_DOWN_QUAT_WXYZ)

            # The width the gripper needs to span *at this specific fixed
            # axis* -- not the OBB's short_half, which is the object's
            # minimum possible width across ALL orientations. Those two only
            # agree when the fixed axis happens to be the object's real short
            # axis (true for a round can, false for an elongated box); using
            # the OBB value here would let a genuinely width-infeasible
            # candidate pass the cheap filter and fail later with a
            # misleading rejection reason.
            width = _object_width_at_yaw(scene, name, obj_quat, manual_yaw_rad)

            # Slide the pinch point along the axis the fingers do NOT close on
            # (perpendicular to the closing axis, i.e. the object's long
            # dimension for an elongated box). This is the one degree of
            # freedom available when the centered grasp is blocked by a
            # neighbor that rotating would otherwise be used to dodge: the
            # orientation, depth and gripper opening are all unchanged, only
            # where along the bar the pads pinch moves. Raising the grasp
            # doesn't help when the obstacle is tall; sliding along the long
            # axis can clear it at every height instead.
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

    R = quat_to_rotmat_wxyz(fingertip_quat_wxyz)
    hand_pos = np.asarray(fingertip_pos, dtype=np.float64) - R @ np.array([0.0, 0.0, offset_z])
    return np.concatenate([hand_pos, fingertip_quat_wxyz])
