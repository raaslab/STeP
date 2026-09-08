import numpy as np

from conftest import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES, requires_libero_drawer
from step.system_one.motion.collision import CollisionChecker
from step.system_one.motion.config import MotionConfig, PlannerConfig
from step.system_one.motion.kinematics import IKSolver
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.skills.open_drawer_skill import (
    _orientation_from_approach_and_reference,
    _solve_ik_robust,
    handle_grasp_pose,
)
from step.system_one.motion.articulation import resolve_drawer
from step.system_one.motion.skills.open_drawer_skill import OpenDrawerSkill
from step.system_one.motion.scene_query import quat_to_rotmat_wxyz
from step.system_one.motion.types import SkillFailureReason


def test_orientation_closing_axis_perpendicular_to_approach_and_reference():
    approach = np.array([0.0, -1.0, 0.0])  # pull axis points +y; approach = -pull_axis
    reference = np.array([1.0, 0.0, 0.0])  # handle's long axis
    quat, returned_x_axis = _orientation_from_approach_and_reference(approach, reference)
    R = quat_to_rotmat_wxyz(quat)
    x_axis, y_axis, z_axis = R[:, 0], R[:, 1], R[:, 2]

    assert np.allclose(x_axis, returned_x_axis, atol=1e-9)
    assert np.allclose(z_axis, approach, atol=1e-9)
    assert np.isclose(np.dot(x_axis, approach), 0.0, atol=1e-9)
    assert np.isclose(np.dot(x_axis, reference), 0.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(x_axis), 1.0, atol=1e-9)
    # Right-handed orthonormal frame.
    assert np.allclose(np.cross(x_axis, y_axis), z_axis, atol=1e-9)


def _build_skill(libero_drawer_scene, timeout_sec=10.0):
    checker = CollisionChecker(libero_drawer_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)
    lo, hi = libero_drawer_scene.joint_qpos_range(JOINT_NAMES)
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    # range=0.2 (matching PlannerConfig's production default -- see its own
    # docstring for the benchmark): this scene's free-space leg is a long
    # joint-space reach (home pose -> the drawer handle, ~3.5-5.6 rad
    # combined across the 7 joints), and a tighter range was measurably
    # *not* fixed by a longer timeout -- RRTConnect returns an "approximate"
    # (rejected) solution in under a second either way. Timeout here is
    # generous mainly for headroom, not because this was ever a timeout
    # problem.
    planner = MotionPlanner(
        JOINT_NAMES, lo, hi, checker, PlannerConfig(timeout_sec=timeout_sec, range=0.2, interpolation_density=15)
    )
    return OpenDrawerSkill(ik, planner, checker, MotionConfig())


@requires_libero_drawer
def test_handle_grasp_pose_is_reachable_by_ik(libero_drawer_scene):
    drawer = resolve_drawer(libero_drawer_scene, "top_drawer")
    grasp_pos, grasp_quat, pull_axis_world, travel_m, closing_axis_world = handle_grasp_pose(libero_drawer_scene, drawer)
    assert np.isclose(np.linalg.norm(pull_axis_world), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(closing_axis_world), 1.0, atol=1e-6)
    assert travel_m > 0.0
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    start_q = libero_drawer_scene.get_qpos(JOINT_NAMES)
    # A handful of restarts, not a single-seed solve: the DLS solver isn't
    # reliably well-conditioned for this horizontal approach from every
    # env-reset starting configuration (see _solve_ik_robust's docstring) --
    # this test checks the target is reachable at all, which is what
    # OpenDrawerSkill itself actually relies on.
    result = _solve_ik_robust(
        ik, libero_drawer_scene, grasp_pos, grasp_quat, start_q, np.random.default_rng(0)
    )
    assert result.success


@requires_libero_drawer
def test_open_drawer_skill_produces_feasible_plan(libero_drawer_scene):
    skill = _build_skill(libero_drawer_scene)
    start_q = libero_drawer_scene.get_qpos(JOINT_NAMES)
    result = skill.execute(libero_drawer_scene, "top_drawer", start_q, {})

    assert result.success, result.diagnostics
    assert result.planned_trajectory is not None
    assert result.planned_trajectory.success
    assert result.diagnostics["pull_fraction_used"] is not None
    assert result.diagnostics["travel_m"] > 0.0
    # approach_positions is the descent (standoff -> grasp) only; the pull
    # leg (grasp -> pulled_q) is tracked separately by the adapter, only
    # after the gripper has actually closed on the handle.
    approach_positions = np.asarray(result.planned_trajectory.metadata["approach_positions"])
    grasp_q = np.asarray(result.planned_trajectory.metadata["grasp_q"])
    pulled_q = np.asarray(result.planned_trajectory.metadata["pulled_q"])
    assert np.allclose(approach_positions[-1], grasp_q)
    assert not np.allclose(grasp_q, pulled_q)


@requires_libero_drawer
def test_open_drawer_skill_unknown_drawer_reports_structured_failure(libero_drawer_scene):
    skill = _build_skill(libero_drawer_scene)
    start_q = libero_drawer_scene.get_qpos(JOINT_NAMES)
    result = skill.execute(libero_drawer_scene, "nonexistent_fixture_xyz", start_q, {})
    assert not result.success
    assert result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE


@requires_libero_drawer
def test_scene_state_untouched_after_plan_evaluation(libero_drawer_scene):
    skill = _build_skill(libero_drawer_scene)
    start_q = libero_drawer_scene.get_qpos(JOINT_NAMES).copy()
    skill.execute(libero_drawer_scene, "top_drawer", start_q, {})
    assert np.allclose(libero_drawer_scene.get_qpos(JOINT_NAMES), start_q)
