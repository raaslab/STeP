import numpy as np

from conftest import HAND_SITE, requires_libero
from step.system_one.motion.config import GraspSampleConfig
from step.system_one.motion.grasp_generator import (
    PrivilegedTopDownGraspGenerator,
    hand_from_fingertip_pose,
    yaw_quat_wxyz,
)


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
    # is covered separately by test_rim_edge_candidates_generated_for_wide_object.
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=8, rim_num_angle_samples=0))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main", "basket_1_main"])
    assert len(cands) == 16
    objects_seen = {c.object_name for c in cands}
    assert objects_seen == {"alphabet_soup_1_main", "basket_1_main"}


@requires_libero
def test_yaw_samples_span_zero_to_pi_exclusive(libero_scene):
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=8))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main"])
    center_yaws = sorted(c.metadata["yaw_rad"] for c in cands if c.metadata["strategy"] == "center")
    assert center_yaws[0] == 0.0
    assert center_yaws[-1] < np.pi


@requires_libero
def test_rim_edge_candidates_generated_for_wide_object(libero_scene):
    """A bowl's cross-section is wider than the gripper's max opening at
    every yaw -- the generator must still offer a *feasible-looking*
    candidate (the rim-edge strategy) rather than only rejected center
    grasps. This is the fix for the real failure this was written against:
    every akita_black_bowl_2 candidate used to come back
    gripper_width_incompatible with no alternative strategy offered."""
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=8, rim_num_angle_samples=8))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main"])
    rim = [c for c in cands if c.metadata["strategy"] == "rim_edge"]
    center = [c for c in cands if c.metadata["strategy"] == "center"]
    assert len(rim) == 8
    assert len(center) == 8
    # Rim candidates must actually be narrow enough to pass the gripper-width
    # gate regardless of the object's real size.
    assert all(c.gripper_width < 0.08 for c in rim)
    # Rim candidates must be positioned off the object's center (that's the
    # whole point) and at 8 distinct angles around it.
    obj_pos, _ = libero_scene.body_pose("alphabet_soup_1_main")
    offsets = np.array([c.grasp_position[:2] - obj_pos[:2] for c in rim])
    assert np.all(np.linalg.norm(offsets, axis=1) > 0.01)
    angles = sorted(c.metadata["rim_angle_rad"] for c in rim)
    assert len(set(np.round(angles, 6))) == 8


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


@requires_libero
def test_grasp_quaternion_is_top_down_wxyz(libero_scene):
    """quat_wxyz = [0, 1, 0, 0] (yaw=0) is a 180-degree rotation about world X,
    i.e. gripper approach axis (+Z) points straight down (-world Z)."""
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=1))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main"])
    q = cands[0].grasp_quat_wxyz
    assert np.allclose(q, [0.0, 1.0, 0.0, 0.0], atol=1e-9)
    from step.system_one.motion.scene_query import quat_to_rotmat_wxyz

    R = quat_to_rotmat_wxyz(q)
    approach_world = R @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(approach_world, [0.0, 0.0, -1.0], atol=1e-9)


@requires_libero
def test_grasp_height_is_object_aabb_center_not_body_origin(libero_scene):
    """Grasp height should be the object's actual AABB center (origin +
    center_offset), not just its body-frame origin -- a body's collision
    geometry isn't always centered on its own frame (this regression-tests
    that ``center_offset`` is actually applied, rather than asserting
    "higher than origin", which isn't true for every object: some objects'
    origins already sit near their own AABB's vertical middle, so a correct
    mid-height formula can legitimately land at ~the same height as the
    origin for those, not strictly above it)."""
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=1))
    cand = gen.generate(libero_scene, ["alphabet_soup_1_main"])[0]
    origin_pos, _ = libero_scene.body_pose("alphabet_soup_1_main")
    extent = libero_scene.body_extent_world("alphabet_soup_1_main")
    expected_z = origin_pos[2] + extent.center_offset[2]
    assert np.isclose(cand.grasp_position[2], expected_z, atol=1e-6)


@requires_libero
def test_hand_from_fingertip_roundtrip(libero_scene):
    tcp_pos, tcp_quat = libero_scene.site_pose("gripper0_grip_site")
    hand_pos, _ = libero_scene.site_pose(HAND_SITE)
    converted = hand_from_fingertip_pose(libero_scene, tcp_pos, tcp_quat, hand_site=HAND_SITE)
    assert np.linalg.norm(converted[:3] - hand_pos) < 1e-9


@requires_libero
def test_width_estimate_never_exceeds_narrowest_extent(libero_scene):
    """Regression test: a naive rotated-AABB projection overestimates a round
    object's cross-section at oblique yaws (predicts the AABB diagonal instead
    of the true diameter). The estimate must be capped at the object's
    narrowest overall horizontal extent."""
    gen = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=8))
    cands = gen.generate(libero_scene, ["alphabet_soup_1_main"])
    extent = libero_scene.body_extent_world("alphabet_soup_1_main")
    narrowest = 2.0 * min(extent.half_extents[0], extent.half_extents[1])
    for c in cands:
        assert c.gripper_width <= narrowest + 1e-9
