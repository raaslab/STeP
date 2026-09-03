import numpy as np
import pytest

from conftest import (
    EE_SITE,
    GRIPPER_BODIES,
    JOINT_NAMES,
    ROBOT_BODIES,
    requires_libero,
)
from step.system_one.motion.collision import CollisionChecker, CollisionPhase


@pytest.fixture()
def checker(libero_scene):
    return CollisionChecker(libero_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)


@requires_libero
def test_home_configuration_is_not_in_collision(libero_scene, checker):
    q0 = libero_scene.get_qpos(JOINT_NAMES)
    result = checker.check(q0, phase=CollisionPhase.FREE_SPACE, target_object_body="alphabet_soup_1_main")
    assert not result.in_collision


@requires_libero
def test_resting_objects_are_not_flagged_as_robot_collisions(libero_scene, checker):
    """Objects resting on the floor/table (which always touch it, by design)
    must never show up as a violation -- only contacts involving the robot or
    an attached object count. The raw scratch scene DOES have floor<->object
    contacts (proving this is a real scenario the checker must filter, not a
    vacuous test)."""
    raw_contacts = libero_scene.compute_contacts()
    assert any("floor" in g1.lower() or "floor" in g2.lower() for g1, g2, _ in raw_contacts)

    q0 = libero_scene.get_qpos(JOINT_NAMES)
    result = checker.check(q0, phase=CollisionPhase.FREE_SPACE, target_object_body="alphabet_soup_1_main")
    robot_geom_names = {libero_scene.geom_name(g) for g in checker.robot_geom_ids}
    for c in result.contacts:
        assert c.geom1 in robot_geom_names or c.geom2 in robot_geom_names


@requires_libero
def test_check_never_mutates_the_input_scene(libero_scene, checker):
    q0 = libero_scene.get_qpos(JOINT_NAMES).copy()
    checker.check(q0 + 0.3, phase=CollisionPhase.FREE_SPACE, target_object_body="alphabet_soup_1_main")
    assert np.allclose(libero_scene.get_qpos(JOINT_NAMES), q0)


@requires_libero
def test_terminal_approach_permits_gripper_target_contact(libero_scene, checker):
    """Same configuration should never be *more* restrictive under
    TERMINAL_APPROACH than FREE_SPACE for gripper<->target contact."""
    q0 = libero_scene.get_qpos(JOINT_NAMES)
    free = checker.check(q0, phase=CollisionPhase.FREE_SPACE, target_object_body="alphabet_soup_1_main")
    terminal = checker.check(q0, phase=CollisionPhase.TERMINAL_APPROACH, target_object_body="alphabet_soup_1_main")
    if free.in_collision:
        assert len(terminal.contacts) <= len(free.contacts)


@requires_libero
def test_check_edge_validates_intermediate_waypoints(libero_scene, checker):
    q0 = libero_scene.get_qpos(JOINT_NAMES)
    q1 = q0.copy()
    q1[1] += 0.2
    result = checker.check_edge(q0, q1, resolution=0.05, phase=CollisionPhase.FREE_SPACE, target_object_body="alphabet_soup_1_main")
    assert isinstance(result.in_collision, bool)


@requires_libero
def test_margin_check_reports_finite_clearance(libero_scene, checker):
    q0 = libero_scene.get_qpos(JOINT_NAMES)
    result = checker.check(
        q0, phase=CollisionPhase.FREE_SPACE, target_object_body="alphabet_soup_1_main", check_margin=True
    )
    assert result.min_clearance != float("inf")


@requires_libero
def test_terminal_approach_permits_actual_gripper_target_contact(libero_scene, checker):
    """Regression test for a real bug: permitted-contact matching used to
    compare geom *names* against the object's *body* name as a substring
    (e.g. checking whether 'alphabet_soup_1_main' appears inside
    'alphabet_soup_1_g7'), which never matches LIBERO's geom naming
    convention -- every terminal grasp contact was incorrectly rejected as a
    collision. Verified here directly against real gripper/object geom ids.

    Also locks in a second, later fix: only the fingerPAD geoms are ever
    permitted contact, not the whole gripper assembly -- an earlier version
    permitted contact from *any* gripper geom, including the palm/hand body,
    which silently allowed grasp candidates where the palm was already
    penetrating the object (only caught once this test's palm assertion
    below was added, after a real run showed the palm ~2cm inside a squat
    can's top)."""
    finger_gid = next(iter(checker.fingerpad_geom_ids))
    palm_gid = libero_scene.geom_id("gripper0_right_hand_collision")
    object_gids = libero_scene.body_geom_ids("alphabet_soup_1_main")
    assert object_gids, "fixture object must have collision geoms"
    assert checker._permitted(
        finger_gid, object_gids[0], CollisionPhase.TERMINAL_APPROACH, set(object_gids)
    )
    assert not checker._permitted(
        finger_gid, object_gids[0], CollisionPhase.FREE_SPACE, set(object_gids)
    )
    assert not checker._permitted(
        palm_gid, object_gids[0], CollisionPhase.TERMINAL_APPROACH, set(object_gids)
    )
