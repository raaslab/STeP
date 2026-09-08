import numpy as np
import pytest

from conftest import HAND_SITE, requires_libero
from step.system_one.motion.config import GraspSampleConfig
from step.system_one.motion.skills.grasp_generator import (
    CompositeGraspGenerator,
    GraspGenerator,
    PrivilegedTopDownGraspGenerator,
    hand_from_fingertip_pose,
    yaw_quat_wxyz,
)
from step.system_one.motion.types import GraspCandidate


class _StubGraspGenerator(GraspGenerator):
    """Plain, non-libero stub for exercising CompositeGraspGenerator's own
    logic in isolation from any real grasp-sampling strategy."""

    def __init__(self, candidates, gripper_max_opening_m=0.08):
        self._candidates = candidates
        self.gripper_max_opening_m = gripper_max_opening_m

    def generate(self, scene, object_names):
        return list(self._candidates)


def _dummy_candidate(name="obj", score=0.5):
    return GraspCandidate(
        object_name=name,
        grasp_pose_world=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        approach_direction_world=np.array([0.0, 0.0, -1.0]),
        gripper_width=None,
        generator_score=score,
    )


def test_composite_grasp_generator_concatenates_candidates():
    a = _dummy_candidate("a", 0.1)
    b = _dummy_candidate("b", 0.2)
    c = _dummy_candidate("c", 0.3)
    gen = CompositeGraspGenerator([_StubGraspGenerator([a, b]), _StubGraspGenerator([c])])
    out = gen.generate(scene=None, object_names=["a", "b", "c"])
    assert out == [a, b, c]


def test_composite_grasp_generator_delegates_gripper_max_opening_m():
    gen = CompositeGraspGenerator([_StubGraspGenerator([], 0.08), _StubGraspGenerator([], 0.08)])
    assert gen.gripper_max_opening_m == 0.08


def test_composite_grasp_generator_raises_on_mismatched_gripper_opening():
    with pytest.raises(ValueError):
        CompositeGraspGenerator([_StubGraspGenerator([], 0.08), _StubGraspGenerator([], 0.05)])


def test_composite_grasp_generator_requires_at_least_one_generator():
    with pytest.raises(ValueError):
        CompositeGraspGenerator([])


def test_yaw_quat_is_unit_and_about_z():
    q = yaw_quat_wxyz(np.pi / 2)
    assert np.isclose(np.linalg.norm(q), 1.0)
    # Rotating [1, 0, 0] by 90 degrees about Z should give [0, 1, 0].
    from step.system_one.motion.scene_query import quat_to_rotmat_wxyz

    R = quat_to_rotmat_wxyz(q)
    v = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v, [0.0, 1.0, 0.0], atol=1e-6)


@requires_libero
def test_generates_candidates_for_every_object(libero_scene):
    # rim_num_angle_samples=0 isolates the center-grasp strategy so the count
    # is a simple function of num_yaw_samples -- rim-strategy candidate count
    # is covered separately by test_rim_edge_candidate_mechanics.
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=8, rim_num_angle_samples=0))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main", "basket_1_main"])
    assert len(cands) == 16
    objects_seen = {c.object_name for c in cands}
    assert objects_seen == {"alphabet_soup_1_main", "basket_1_main"}


@requires_libero
def test_yaw_samples_span_search_range_around_obb_optimum(libero_scene):
    """Center-grasp yaws are no longer sampled blindly across world-frame
    [0, pi) -- they're sampled across
    [best-half_range, best+half_range], where "best" comes from an oriented
    bounding box fit to the object's real geometry (see
    grasp_generator._object_xy_obb_world), and offset 0 (the fitted optimum)
    is always included exactly, regardless of num_yaw_samples's parity."""
    half_range = 0.4
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=8, yaw_search_half_range_rad=half_range))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main"])
    center_yaws = sorted(c.metadata["yaw_rad"] for c in cands if c.metadata["strategy"] == "center")
    span = center_yaws[-1] - center_yaws[0]
    assert np.isclose(span, 2 * half_range, atol=1e-6)
    # The fitted-optimal yaw (some data-dependent value, not necessarily 0
    # now) must be exactly one of the samples.
    best_yaw = center_yaws[0] + half_range
    assert any(np.isclose(y, best_yaw, atol=1e-6) for y in center_yaws)


@requires_libero
def test_rim_edge_candidate_mechanics(libero_scene):
    """Exercises the rim-edge strategy's own candidate math directly (angles,
    positions, width) using alphabet_soup_1_main's real geometry as a
    convenient stand-in -- this is about whether the sampling math is
    correct, not whether rim-edge is the *right strategy* for a soup can
    (it isn't; see test_rim_edge_gated_by_object_class below), so it calls
    _rim_edge_candidates directly rather than going through generate()'s
    object-class gate."""
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=8, rim_num_angle_samples=8))
    obj_pos, obj_quat = libero_scene.body_pose("alphabet_soup_1_main")
    extent = libero_scene.body_extent_world("alphabet_soup_1_main")
    grasp_center = obj_pos + np.array([0.0, 0.0, float(extent.center_offset[2])])
    rim = gen._rim_edge_candidates(
        libero_scene, "alphabet_soup_1_main", obj_pos, obj_quat, grasp_center, np.array([0.0, 0.0, -1.0]), extent
    )
    assert len(rim) == 8
    # Rim candidates must actually be narrow enough to pass the gripper-width
    # gate regardless of the object's real size.
    assert all(c.gripper_width < 0.08 for c in rim)
    # Rim candidates must be positioned off the object's center (that's the
    # whole point) and at 8 distinct angles around it.
    offsets = np.array([c.grasp_position[:2] - obj_pos[:2] for c in rim])
    assert np.all(np.linalg.norm(offsets, axis=1) > 0.01)
    angles = sorted(c.metadata["rim_angle_rad"] for c in rim)
    assert len(set(np.round(angles, 6))) == 8


@requires_libero
def test_rim_edge_gated_by_object_class(libero_scene):
    """Rim-edge only makes sense for a hollow, thin-walled container -- it
    deliberately straddles a wall, positioning the closing point at a
    fraction of the object's radius. For a solid object (verified directly
    against alphabet_soup_1_main: one of its own decomposed collision geoms
    sits almost exactly on the object's central axis, i.e. solid, not
    hollow) that point is just interior material, so the candidate is
    geometrically guaranteed to collide no matter the height. generate()
    must not offer rim-edge candidates for object classes outside the same
    RIM_GRASP_KEYWORDS list the legacy stage machine already uses for this
    exact distinction (modes.py)."""
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=1, rim_num_angle_samples=8))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main"])
    assert not any(c.metadata["strategy"] == "rim_edge" for c in cands)

    from step.system_one.motion.skills.grasp_generator import _is_rim_graspable_object_class

    assert _is_rim_graspable_object_class("akita_black_bowl_1_main")
    assert _is_rim_graspable_object_class("glazed_rim_porcelain_ramekin_1_main")
    assert not _is_rim_graspable_object_class("alphabet_soup_1_main")
    assert not _is_rim_graspable_object_class("milk_1_main")


@requires_libero
def test_rim_edge_yaw_closes_radially(libero_scene):
    """The gripper's closing axis at a rim point must point through the
    object's center (radially), not tangent to the rim -- otherwise the
    fingers would try to span the object's circumference instead of
    straddling the rim wall at one point."""
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=1, rim_num_angle_samples=4))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main"])
    rim = [c for c in cands if c.metadata["strategy"] == "rim_edge"]
    for c in rim:
        theta = c.metadata["rim_angle_rad"]
        radial_dir = np.array([np.cos(theta), np.sin(theta)])
        yaw = c.metadata["yaw_rad"]
        close_axis = np.array([-np.sin(yaw), np.cos(yaw)])
        # Parallel or anti-parallel to the radial direction (a parallel-jaw
        # gripper's closing axis is symmetric, so either sign is correct).
        assert abs(np.dot(close_axis, radial_dir)) > 0.999


def test_zero_yaw_top_down_quat_is_wxyz_0100():
    """quat_wxyz = [0, 1, 0, 0] (yaw=0) is a 180-degree rotation about world X,
    i.e. gripper approach axis (+Z) points straight down (-world Z). A pure
    math check on the underlying quaternion composition -- unlike an earlier
    version, this no longer goes through the full generator, since the
    actual candidate yaw is now fit to each object's real geometry (see
    grasp_generator._object_xy_obb_world) and generally isn't 0.
    ``test_generates_top_down_approach`` already covers that every generated
    candidate points straight down regardless of its yaw."""
    from step.system_one.motion.skills.grasp_generator import _TOP_DOWN_QUAT_WXYZ
    from step.system_one.motion.scene_query import quat_mul_wxyz, quat_to_rotmat_wxyz

    q = quat_mul_wxyz(yaw_quat_wxyz(0.0), _TOP_DOWN_QUAT_WXYZ)
    assert np.allclose(q, [0.0, 1.0, 0.0, 0.0], atol=1e-9)
    R = quat_to_rotmat_wxyz(q)
    approach_world = R @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(approach_world, [0.0, 0.0, -1.0], atol=1e-9)


@requires_libero
def test_grasp_height_uses_object_aabb_center_not_body_origin(libero_scene):
    """Grasp height should be based on the object's actual AABB center
    (origin + center_offset), not just its body-frame origin -- a body's
    collision geometry isn't always centered on its own frame. Every real
    grocery-item asset in this fixture scene ends up needing one of the two
    palm/floor clearance clamps (see test_grasp_height_clamped_above_
    palm_clearance and test_grasp_height_clamped_above_floor_for_thin_
    object), so there's no "plain, unclamped" example left to test the raw
    center formula against directly -- this instead checks that
    center_offset enters the *palm-clamped* height's own object_top_z term
    (i.e. is not silently dropped), using alphabet_soup_1_main. If the
    implementation dropped center_offset (used only the body origin), this
    would fail whenever a reset happens to give the object a non-negligible
    offset; checking full equality against the formula (rather than a "must
    differ from origin" assertion, which would be flaky given LIBERO's
    non-deterministic resets) catches that regardless."""
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=1))
    cand = gen.generate(libero_scene, ["alphabet_soup_1_main"])[0]
    origin_pos, _ = libero_scene.body_pose("alphabet_soup_1_main")
    extent = libero_scene.body_extent_world("alphabet_soup_1_main")
    object_top_z = origin_pos[2] + extent.center_offset[2] + extent.half_extents[2]
    expected_z = object_top_z - libero_scene.palm_clearance_above_tcp_m() + gen.config.grasp_height_margin_m
    assert np.isclose(cand.grasp_position[2], expected_z, atol=1e-6)


@requires_libero
def test_grasp_height_clamped_above_palm_clearance(libero_scene):
    """For a short/squat object, the plain AABB-center height would put the
    palm's own collision geometry into the object's top (the palm is a real,
    sizeable mesh, not a point at the TCP -- see
    ``SceneQuery.palm_clearance_above_tcp_m``'s docstring). The generator
    must raise the grasp height above that plain center formula in this
    case, by exactly the configured clamp."""
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=1))
    cand = gen.generate(libero_scene, ["alphabet_soup_1_main"])[0]
    origin_pos, _ = libero_scene.body_pose("alphabet_soup_1_main")
    extent = libero_scene.body_extent_world("alphabet_soup_1_main")
    naive_center_z = origin_pos[2] + extent.center_offset[2]
    object_top_z = origin_pos[2] + extent.center_offset[2] + extent.half_extents[2]
    min_safe_z = object_top_z - libero_scene.palm_clearance_above_tcp_m() + gen.config.grasp_height_margin_m
    assert min_safe_z > naive_center_z, "fixture object must actually need the clamp for this test to be meaningful"
    assert np.isclose(cand.grasp_position[2], min_safe_z, atol=1e-6)


@requires_libero
def test_grasp_height_clamped_above_floor_for_thin_object(libero_scene):
    """Mirror image of the palm-clearance clamp: for a thin/flat object, the
    plain AABB-center height would put the rigid fingers *below* the surface
    the object rests on (verified against the real cream_cheese_1 asset,
    2cm tall: the finger body ended up ~1cm into the floor). The generator
    must raise the grasp height above that plain center formula.

    The floor reference is the object's own AABB *bottom*
    (object_bottom_z), not its raw body-origin z -- cream_cheese_1's origin
    sits at its geometric center (center_offset[2] is ~0), so using the raw
    origin directly would assume a floor ~1cm too high. And the result is
    capped at object_top_z - margin, never allowed to exceed the object's
    own body: for cream_cheese_1 specifically, finger_reach_below_tcp_m()
    alone already exceeds the object's full height, so the "ideal" floor
    clearance is infeasible and the cap is what actually determines the
    final height -- confirmed via a real rollout that an uncapped clamp
    landed 1.3cm *above* the object's own top surface, and the gripper
    closed on nothing despite tracking to that target with near-zero error.
    """
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=1))
    cand = gen.generate(libero_scene, ["cream_cheese_1_main"])[0]
    origin_pos, _ = libero_scene.body_pose("cream_cheese_1_main")
    extent = libero_scene.body_extent_world("cream_cheese_1_main")
    naive_center_z = origin_pos[2] + extent.center_offset[2]
    object_top_z = origin_pos[2] + extent.center_offset[2] + extent.half_extents[2]
    object_bottom_z = origin_pos[2] + extent.center_offset[2] - extent.half_extents[2]
    margin = gen.config.grasp_height_margin_m
    min_safe_for_floor = object_bottom_z + libero_scene.finger_reach_below_tcp_m() + margin
    assert min_safe_for_floor > naive_center_z, "fixture object must actually need the clamp for this test to be meaningful"
    expected_z = min(min_safe_for_floor, object_top_z - margin)
    assert np.isclose(cand.grasp_position[2], expected_z, atol=1e-6)
    # And, regardless of the exact clamp formula, the result must never
    # float above the object's own body -- that's the actual bug this test
    # (and the clamp itself) exists to prevent.
    assert cand.grasp_position[2] <= object_top_z + 1e-6


@requires_libero
def test_hand_from_fingertip_roundtrip(libero_scene):
    tcp_pos, tcp_quat = libero_scene.site_pose("gripper0_right_grip_site")
    hand_pos, _ = libero_scene.site_pose(HAND_SITE)
    converted = hand_from_fingertip_pose(libero_scene, tcp_pos, tcp_quat, hand_site=HAND_SITE)
    assert np.linalg.norm(converted[:3] - hand_pos) < 1e-9


@requires_libero
def test_width_estimate_matches_real_obb_and_is_minimal_at_optimum(libero_scene):
    """Regression test for a real bug: an earlier version sampled wrist yaws
    in *world* frame and measured width off a *world-axis-aligned* AABB --
    neither has any relationship to the object's real shape once it's
    rotated away from world-axis-aligned, and a naive rotated-box projection
    also overestimates at oblique angles regardless. The generator now fits
    an oriented bounding box to the object's real geometry (see
    grasp_generator._object_xy_obb_world) and computes width *exactly* from
    that true rectangle, so it must (a) genuinely differ across sampled
    yaws, (b) never be less than the OBB's true short side (the physical
    minimum), and (c) hit exactly that minimum at the fitted-optimal yaw
    (offset 0, always one of the samples)."""
    from step.system_one.motion.skills.grasp_generator import _object_xy_obb_world

    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=8, yaw_search_half_range_rad=0.6))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main"])
    center = [c for c in cands if c.metadata["strategy"] == "center"]
    widths = [c.gripper_width for c in center]

    _, obj_quat = libero_scene.body_pose("alphabet_soup_1_main")
    _, short_half, _ = _object_xy_obb_world(libero_scene, "alphabet_soup_1_main", obj_quat)
    true_min_width = 2.0 * short_half

    assert len(set(np.round(widths, 6))) > 1, "widths must genuinely differ across sampled yaws"
    assert all(w >= true_min_width - 1e-6 for w in widths)
    assert np.isclose(min(widths), true_min_width, atol=1e-6)
