import numpy as np
import pytest

from conftest import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES, requires_libero
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.config import PlannerConfig
from step.system_one.motion.ompl_planner import OMPL_AVAILABLE, MotionPlanner
from step.system_one.motion.types import CollisionResult, PlannerFailureReason

requires_ompl = pytest.mark.skipif(not OMPL_AVAILABLE, reason="ompl is not installed in this environment")


@pytest.fixture()
def checker(libero_scene):
    return CollisionChecker(libero_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)


@requires_libero
def test_unavailable_plan_shape_is_stable_regardless_of_ompl_presence():
    """Whether or not OMPL happens to be installed, the typed
    'unavailable' result must always be constructible and shaped the same way
    -- this is what a caller falls back to if OMPL is missing, instead of a
    silent straight-line interpolation."""
    plan = MotionPlanner.unavailable_plan("test")
    assert not plan.success
    assert plan.failure_reason == PlannerFailureReason.PLANNER_UNAVAILABLE


@requires_libero
@requires_ompl
def test_plan_finds_a_path_to_a_nearby_ik_goal(libero_scene, checker):
    from step.system_one.motion.kinematics import IKSolver

    lo, hi = libero_scene.joint_qpos_range(JOINT_NAMES)
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    start_q = libero_scene.get_qpos(JOINT_NAMES)
    tcp_pos, tcp_quat = libero_scene.site_pose(EE_SITE)
    scratch = libero_scene.clone()
    ik_result = ik.solve(scratch, tcp_pos + np.array([0.08, 0.0, -0.05]), tcp_quat, seed_q=start_q)
    assert ik_result.success

    planner = MotionPlanner(JOINT_NAMES, lo, hi, checker, PlannerConfig(timeout_sec=5.0, range=0.1, interpolation_density=40))
    plan = planner.plan(start_q, [ik_result.q], phase=CollisionPhase.FREE_SPACE, target_object_body="alphabet_soup_1_main")

    assert plan.success, plan.metadata
    assert plan.geometric_path is not None
    assert np.allclose(plan.geometric_path[0], start_q, atol=1e-6)
    assert np.allclose(plan.geometric_path[-1], ik_result.q, atol=1e-6)


@requires_libero
@requires_ompl
def test_plan_rejects_a_start_configuration_already_in_collision(libero_scene, checker):
    lo, hi = libero_scene.joint_qpos_range(JOINT_NAMES)
    planner = MotionPlanner(JOINT_NAMES, lo, hi, checker, PlannerConfig(timeout_sec=1.0))
    # Fabricate a start colliding by monkeypatching the checker to always
    # report collision -- isolates the "start-in-collision short circuit"
    # behavior from needing a genuinely self-colliding 7-DoF pose.
    checker.check = lambda *a, **k: CollisionResult(in_collision=True)
    start_q = libero_scene.get_qpos(JOINT_NAMES)
    plan = planner.plan(start_q, [start_q], phase=CollisionPhase.FREE_SPACE)
    assert not plan.success
    assert plan.failure_reason == PlannerFailureReason.START_IN_COLLISION


@requires_libero
@requires_ompl
def test_plan_never_mutates_the_live_scene(libero_scene, checker):
    lo, hi = libero_scene.joint_qpos_range(JOINT_NAMES)
    from step.system_one.motion.kinematics import IKSolver

    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    start_q = libero_scene.get_qpos(JOINT_NAMES).copy()
    tcp_pos, tcp_quat = libero_scene.site_pose(EE_SITE)
    scratch = libero_scene.clone()
    ik_result = ik.solve(scratch, tcp_pos + np.array([0.05, 0.0, -0.02]), tcp_quat, seed_q=start_q)
    planner = MotionPlanner(JOINT_NAMES, lo, hi, checker, PlannerConfig(timeout_sec=3.0))
    planner.plan(start_q, [ik_result.q] if ik_result.success else [start_q], target_object_body="alphabet_soup_1_main")
    assert np.allclose(libero_scene.get_qpos(JOINT_NAMES), start_q)
