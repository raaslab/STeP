import numpy as np

from conftest import EE_SITE, HAND_SITE, JOINT_NAMES, requires_libero
from step.system_one.motion.scene_query import (
    quat_mul_wxyz,
    quat_to_rotmat_wxyz,
    rotmat_to_quat_wxyz,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def test_wxyz_xyzw_roundtrip():
    q_wxyz = np.array([0.7071, 0.0, 0.7071, 0.0])
    q_wxyz /= np.linalg.norm(q_wxyz)
    q_xyzw = wxyz_to_xyzw(q_wxyz)
    assert np.allclose(xyzw_to_wxyz(q_xyzw), q_wxyz)
    assert np.allclose(q_xyzw, [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def test_quat_rotmat_roundtrip():
    q = np.array([0.9238795, 0.3826834, 0.0, 0.0])  # 45 deg about X
    R = quat_to_rotmat_wxyz(q)
    q2 = rotmat_to_quat_wxyz(R)
    # Quaternion double-cover: q and -q represent the same rotation.
    assert np.allclose(q2, q, atol=1e-6) or np.allclose(q2, -q, atol=1e-6)


def test_identity_quat_mul():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.7071, 0.7071, 0.0, 0.0])
    assert np.allclose(quat_mul_wxyz(identity, q), q)


@requires_libero
def test_scene_query_from_robosuite_sim_matches_live_qpos(libero_env):
    from step.system_one.motion.scene_query import SceneQuery

    sim = libero_env.env.sim
    sq = SceneQuery.from_robosuite_sim(sim)
    # Resolved through SceneQuery.site_id (not sim.model.site_name2id directly)
    # so this passes under either robosuite gripper-naming convention -- see
    # SceneQuery._gripper_name_alt.
    live_pos = sim.data.site_xpos[sq.site_id(EE_SITE)]
    sq_pos, _ = sq.site_pose(EE_SITE)
    assert np.allclose(sq_pos, live_pos, atol=1e-9)
    assert np.allclose(sq.get_qpos(JOINT_NAMES), sim.data.qpos[: len(sim.data.qpos)][sq.joint_qpos_addrs(JOINT_NAMES)])


@requires_libero
def test_clone_is_independent_of_source(libero_scene):
    clone = libero_scene.clone()
    q = libero_scene.get_qpos(JOINT_NAMES)
    clone.set_qpos(JOINT_NAMES, q + 0.5)
    assert not np.allclose(libero_scene.get_qpos(JOINT_NAMES), clone.get_qpos(JOINT_NAMES))


@requires_libero
def test_body_extent_world_is_positive_and_reasonable(libero_scene):
    extent = libero_scene.body_extent_world("alphabet_soup_1_main")
    assert extent is not None
    assert np.all(extent.half_extents > 0)
    assert np.all(extent.half_extents < 0.5)  # sanity: not a room-sized object


@requires_libero
def test_hand_to_tcp_transform_is_positive_and_small(libero_scene):
    offset = libero_scene.hand_to_tcp_transform(HAND_SITE, "gripper0_right_grip_site")
    assert 0.0 < offset < 0.3


@requires_libero
def test_nearby_body_names_respects_radius_and_exclude(libero_scene):
    obj_pos, _ = libero_scene.body_pose("alphabet_soup_1_main")
    # A huge radius must find at least one other free-jointed object in this
    # fixture scene (several LIBERO grocery items share the table); a
    # microscopic radius must find none (nothing else sits exactly on top of
    # this object's own origin).
    far = libero_scene.nearby_body_names(obj_pos, radius_m=5.0, exclude=("alphabet_soup_1_main",))
    assert len(far) > 0
    assert "alphabet_soup_1_main" not in far
    near = libero_scene.nearby_body_names(obj_pos, radius_m=1e-6, exclude=("alphabet_soup_1_main",))
    assert near == []


@requires_libero
def test_nearby_body_names_excludes_fixed_scene(libero_scene):
    """Only free-jointed (movable) bodies should ever be returned -- the
    fixed robot/table/world bodies have no free joint at all, so this is what
    keeps pc_full scene context (see ContactGraspNetGenerator) from being
    swamped by the entire fixed scene graph."""
    obj_pos, _ = libero_scene.body_pose("alphabet_soup_1_main")
    names = libero_scene.nearby_body_names(obj_pos, radius_m=5.0)
    assert "table" not in names
    assert not any(n.startswith("robot0") for n in names)
    assert not any(n.startswith("gripper0") for n in names)


@requires_libero
def test_body_surface_points_dense_local_is_denser_than_sparse(libero_scene):
    sparse = libero_scene.body_surface_points_local("alphabet_soup_1_main")
    dense = libero_scene.body_surface_points_dense_local("alphabet_soup_1_main")
    assert len(dense) > len(sparse) * 3
    # Dense sampling must still bound the same real geometry -- no point can
    # fall meaningfully outside the body's own world-frame AABB.
    obj_pos, obj_quat = libero_scene.body_pose("alphabet_soup_1_main")
    extent = libero_scene.body_extent_world("alphabet_soup_1_main")
    R = quat_to_rotmat_wxyz(obj_quat)
    dense_world = dense @ R.T + obj_pos
    center_world = obj_pos + extent.center_offset
    assert np.all(np.abs(dense_world - center_world) <= extent.half_extents + 1e-6)
