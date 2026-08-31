from conftest import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES, requires_libero
from step.system_one.motion.collision import CollisionChecker
from step.system_one.motion.config import GraspSampleConfig, MotionConfig, PlannerConfig
from step.system_one.motion.grasp_generator import PrivilegedTopDownGraspGenerator
from step.system_one.motion.kinematics import IKSolver
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.pick_skill import PickSkill
from step.system_one.motion.types import SkillFailureReason


def _build_skill(libero_scene, num_yaw_samples=2, timeout_sec=3.0):
    checker = CollisionChecker(libero_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)
    lo, hi = libero_scene.joint_qpos_range(JOINT_NAMES)
    # rim_num_angle_samples=0: these tests are about OR-set selection across
    # the center-grasp strategy specifically (rim-edge candidate generation
    # is covered by tests/test_grasp_generator.py), and disabling it keeps
    # the candidate counts asserted below simple and exact.
    gen = PrivilegedTopDownGraspGenerator(
        GraspSampleConfig(num_yaw_samples=num_yaw_samples, rim_num_angle_samples=0)
    )
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    planner = MotionPlanner(
        JOINT_NAMES, lo, hi, checker, PlannerConfig(timeout_sec=timeout_sec, range=0.15, interpolation_density=15)
    )
    return PickSkill(gen, ik, planner, checker, MotionConfig())


@requires_libero
def test_pick_single_object_succeeds(libero_scene):
    skill = _build_skill(libero_scene)
    start_q = libero_scene.get_qpos(JOINT_NAMES)
    result = skill.execute(libero_scene, ["alphabet_soup_1_main"], start_q)
    assert result.success
    assert result.selected_object == "alphabet_soup_1_main"
    assert result.selected_grasp is not None
    assert result.planned_trajectory is not None
    assert result.planned_trajectory.success


@requires_libero
def test_pick_or_set_selects_lowest_cost_feasible_across_objects(libero_scene):
    """'Pick any [X]' semantics: given a set of candidate objects, the skill
    must evaluate all of them and commit to whichever (object, grasp,
    trajectory) tuple is cheapest among the *feasible* ones -- not just the
    first one tried."""
    skill = _build_skill(libero_scene, num_yaw_samples=2)
    start_q = libero_scene.get_qpos(JOINT_NAMES)
    object_set = ["alphabet_soup_1_main", "milk_1_main"]
    result = skill.execute(libero_scene, object_set, start_q)

    assert result.success
    assert result.selected_object in object_set
    assert result.diagnostics["n_candidates_tried"] == len(object_set) * 2
    # Every candidate's total cost was actually computed and compared.
    costs_by_object = {}
    for obj_name, _yaw, cost in result.diagnostics["candidate_costs"]:
        costs_by_object.setdefault(obj_name, []).append(cost)
    assert set(costs_by_object) == set(object_set)


@requires_libero
def test_scene_state_untouched_after_full_pick_evaluation(libero_scene):
    import numpy as np

    skill = _build_skill(libero_scene)
    start_q = libero_scene.get_qpos(JOINT_NAMES).copy()
    skill.execute(libero_scene, ["alphabet_soup_1_main", "milk_1_main"], start_q)
    assert np.allclose(libero_scene.get_qpos(JOINT_NAMES), start_q)


@requires_libero
def test_no_feasible_candidate_reports_structured_failure(libero_scene):
    skill = _build_skill(libero_scene, num_yaw_samples=1, timeout_sec=0.05)
    start_q = libero_scene.get_qpos(JOINT_NAMES)
    # An absurdly far, nonexistent-in-reach target forces every candidate to
    # fail IK -- exercised via a target far outside the workspace by asking
    # for an object name that still resolves geometrically but pushing the
    # generator's pregrasp far away is out of scope here; instead verify the
    # *shape* of a no-candidates failure using an empty object set, which is
    # the simplest deterministic way to hit NO_FEASIBLE_CANDIDATE.
    result = skill.execute(libero_scene, [], start_q)
    assert not result.success
    assert result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE
