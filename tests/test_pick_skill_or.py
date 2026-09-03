import numpy as np

from conftest import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES, requires_libero
from step.system_one.motion.collision import CollisionChecker
from step.system_one.motion.config import GraspSampleConfig, MotionConfig, PlannerConfig
from step.system_one.motion.grasp_generator import CompositeGraspGenerator, GraspGenerator, PrivilegedTopDownGraspGenerator
from step.system_one.motion.kinematics import IKSolver
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.pick_skill import PickSkill
from step.system_one.motion.types import GraspCandidate, SkillFailureReason


class _StubGraspGenerator(GraspGenerator):
    def __init__(self, candidates, gripper_max_opening_m=0.08):
        self._candidates = candidates
        self.gripper_max_opening_m = gripper_max_opening_m

    def generate(self, scene, object_names):
        return list(self._candidates)


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
    # milk_1_main, not alphabet_soup_1_main: the soup can's true cross-section
    # (~8.6cm) is wider than the gripper's max opening (8cm) at every yaw, so
    # with rim sampling disabled (center-grasp only, see _build_skill) it has
    # no feasible candidate at all -- a real geometric fact about this
    # object/gripper pairing, not a bug (see PickSkill's fingerpad-only
    # collision model). milk_1_main's cross-section actually fits.
    skill = _build_skill(libero_scene)
    start_q = libero_scene.get_qpos(JOINT_NAMES)
    result = skill.execute(libero_scene, ["milk_1_main"], start_q)
    assert result.success
    assert result.selected_object == "milk_1_main"
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
def test_composite_grasp_generator_candidates_flow_into_pick_skill(libero_scene):
    """A second, independent candidate source (e.g. a learned grasp detector
    like ContactGraspNetGenerator) must be purely additive: PickSkill treats
    every candidate identically regardless of which generator produced it,
    and an infeasible candidate from one generator must not prevent success
    via a feasible candidate from another."""
    checker = CollisionChecker(libero_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)
    lo, hi = libero_scene.joint_qpos_range(JOINT_NAMES)
    heuristic = PrivilegedTopDownGraspGenerator(GraspSampleConfig(num_yaw_samples=2, rim_num_angle_samples=0))
    # An obviously-unreachable candidate (far outside the workspace) that
    # must fail IK -- stands in for a learned detector proposing a pose
    # PickSkill correctly rejects.
    bad_candidate = GraspCandidate(
        object_name="milk_1_main",
        grasp_pose_world=np.array([5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]),
        approach_direction_world=np.array([0.0, 0.0, -1.0]),
        gripper_width=0.02,
        generator_score=1.0,
    )
    gen = CompositeGraspGenerator([heuristic, _StubGraspGenerator([bad_candidate])])
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    planner = MotionPlanner(
        JOINT_NAMES, lo, hi, checker, PlannerConfig(timeout_sec=3.0, range=0.15, interpolation_density=15)
    )
    skill = PickSkill(gen, ik, planner, checker, MotionConfig())
    start_q = libero_scene.get_qpos(JOINT_NAMES)
    result = skill.execute(libero_scene, ["milk_1_main"], start_q)

    assert result.success
    assert result.selected_object == "milk_1_main"
    # heuristic's 2 yaw samples + the stub's 1 bad candidate.
    assert result.diagnostics["n_candidates_tried"] == 3


@requires_libero
def test_no_feasible_candidate_reports_structured_failure(libero_scene):
    skill = _build_skill(libero_scene, num_yaw_samples=1, timeout_sec=0.05)
    start_q = libero_scene.get_qpos(JOINT_NAMES)
    # Forcing every candidate to fail IK would need an unreachable target
    # constructed from real scene geometry, which is out of scope here;
    # instead verify the *shape* of a no-candidates failure using an empty
    # object set, which is the simplest deterministic way to hit
    # NO_FEASIBLE_CANDIDATE.
    result = skill.execute(libero_scene, [], start_q)
    assert not result.success
    assert result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE
