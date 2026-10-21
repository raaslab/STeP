import numpy as np

from conftest import EE_SITE, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES, requires_libero
from step.system_one.motion.attached_object import AttachedObjectManager, fit_spheres_to_extent
from step.system_one.motion.collision import CollisionChecker, CollisionPhase


@requires_libero
def test_ground_plane_is_not_a_giant_obstacle(libero_scene):
    """Regression test for a real bug: MuJoCo plane geoms' geom_size is a
    rendering hint (visual quad half-width, grid spacing), not a physical
    half-extent. Treating it literally made the floor read as a slab thick
    enough that any object resting near table height was already "inside"
    it, and (via body_geom_ids' include_children walk) treating the world
    body itself as an obstacle made this even worse -- an aggregated AABB of
    the *entire scene*. An object carried at a normal height must not read
    as colliding with the ground plane."""
    checker = CollisionChecker(libero_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)
    mgr = AttachedObjectManager()
    state = mgr.attach(libero_scene, "alphabet_soup_1_main", HAND_SITE, step=0)
    q0 = libero_scene.get_qpos(JOINT_NAMES)
    result = checker.check(q0, phase=CollisionPhase.FREE_SPACE, attached=state)
    # Real clearance to the floor, not a margin verdict: the slab bug showed up
    # as large negative distances, while an object resting close to the floor
    # can legitimately sit inside the clearance margin.
    assert all(c.distance >= 0.0 for c in result.contacts if c.geom2 == "floor")
    assert 0.0 < result.min_clearance < 1.0


@requires_libero
def test_place_approach_permits_container_contact(libero_scene):
    checker = CollisionChecker(libero_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)
    mgr = AttachedObjectManager()
    state = mgr.attach(libero_scene, "alphabet_soup_1_main", HAND_SITE, step=0)
    q0 = libero_scene.get_qpos(JOINT_NAMES)
    free = checker.check(q0, phase=CollisionPhase.FREE_SPACE, attached=state, goal_container_body="basket_1_main")
    place = checker.check(q0, phase=CollisionPhase.PLACE_APPROACH, attached=state, goal_container_body="basket_1_main")
    # Contact with the basket specifically must never be *more* restrictive
    # under PLACE_APPROACH than FREE_SPACE.
    if free.in_collision:
        assert len(place.contacts) <= len(free.contacts)


@requires_libero
def test_attached_check_never_mutates_scene(libero_scene):
    checker = CollisionChecker(libero_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)
    mgr = AttachedObjectManager()
    state = mgr.attach(libero_scene, "alphabet_soup_1_main", HAND_SITE, step=0)
    q0 = libero_scene.get_qpos(JOINT_NAMES).copy()
    checker.check(q0 + 0.2, phase=CollisionPhase.FREE_SPACE, attached=state)
    assert np.allclose(libero_scene.get_qpos(JOINT_NAMES), q0)


def test_fit_spheres_never_underestimates_cross_section():
    from step.system_one.motion.scene_query import BodyExtent

    extent = BodyExtent(half_extents=np.array([0.1, 0.02, 0.02]), center_offset=np.zeros(3))
    spheres = fit_spheres_to_extent(extent, n_spheres=3)
    assert len(spheres) == 3
    for offset, radius in spheres:
        assert radius >= np.hypot(0.02, 0.02) - 1e-9
    # centers must span (not exceed) the object's longest axis
    centers_x = sorted(o[0] for o, _ in spheres)
    assert centers_x[0] >= -0.1 - 1e-9
    assert centers_x[-1] <= 0.1 + 1e-9


def test_fit_spheres_collapses_to_one_for_compact_object():
    from step.system_one.motion.scene_query import BodyExtent

    extent = BodyExtent(half_extents=np.array([0.04, 0.045, 0.042]), center_offset=np.zeros(3))
    spheres = fit_spheres_to_extent(extent, n_spheres=3)
    offsets = np.array([o for o, _ in spheres])
    assert np.allclose(offsets, offsets[0])  # all identical -- object too compact to spread spheres
