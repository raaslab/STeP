import numpy as np

from conftest import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES, requires_libero_stove
from step.system_one.motion.articulation import resolve_knob
from step.system_one.motion.collision import CollisionChecker
from step.system_one.motion.config import MotionConfig, PlannerConfig
from step.system_one.motion.kinematics import IKSolver
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.scene_query import quat_to_rotmat_wxyz
from step.system_one.motion.skills.turn_knob_skill import (
    TurnKnobSkill,
    _arbitrary_perpendicular,
    _axis_angle_to_rotmat,
    _orientation_for_tangential_grasp,
    _solve_ik_robust,
)
from step.system_one.motion.types import SkillFailureReason


def test_orientation_closing_axis_perpendicular_to_approach_and_radial():
    approach = np.array([0.0, 0.0, -1.0])  # top-down onto the knob
    radial = np.array([1.0, 0.0, 0.0])  # direction from center to the grasp chord
    quat = _orientation_for_tangential_grasp(approach, radial)
    R = quat_to_rotmat_wxyz(quat)
    x_axis, y_axis, z_axis = R[:, 0], R[:, 1], R[:, 2]

    assert np.allclose(z_axis, approach, atol=1e-9)
    # Closing axis is tangential -- perpendicular to both approach and radial.
    assert np.isclose(np.dot(x_axis, approach), 0.0, atol=1e-9)
    assert np.isclose(np.dot(x_axis, radial), 0.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(x_axis), 1.0, atol=1e-9)
    assert np.allclose(np.cross(x_axis, y_axis), z_axis, atol=1e-9)


def test_axis_angle_to_rotmat_matches_known_rotations():
    # 90 degrees about +Z: (1,0,0) -> (0,1,0)
    R = _axis_angle_to_rotmat(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-9)
    # Identity at angle 0.
    R0 = _axis_angle_to_rotmat(np.array([0.0, 0.0, 1.0]), 0.0)
    assert np.allclose(R0, np.eye(3), atol=1e-9)


def test_arbitrary_perpendicular_is_unit_and_orthogonal():
    for axis in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.3, 0.4, 0.5])):
        perp = _arbitrary_perpendicular(axis)
        assert np.isclose(np.linalg.norm(perp), 1.0, atol=1e-9)
        assert np.isclose(np.dot(perp, axis / np.linalg.norm(axis)), 0.0, atol=1e-9)


def _build_skill(libero_stove_scene, timeout_sec=10.0):
    checker = CollisionChecker(libero_stove_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)
    lo, hi = libero_stove_scene.joint_qpos_range(JOINT_NAMES)
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    planner = MotionPlanner(
        JOINT_NAMES, lo, hi, checker, PlannerConfig(timeout_sec=timeout_sec, range=0.2, interpolation_density=15)
    )
    return TurnKnobSkill(ik, planner, checker, MotionConfig())


@requires_libero_stove
def test_knob_grasp_pose_is_reachable_by_ik(libero_stove_scene):
    from step.system_one.motion.articulation import estimate_knob_radius_m, measure_knob_rotation

    knob = resolve_knob(libero_stove_scene, "flat_stove_1")
    axis_world, center_world, qpos_lo, qpos_hi = measure_knob_rotation(libero_stove_scene, knob)
    assert np.isclose(np.linalg.norm(axis_world), 1.0, atol=1e-6)
    assert qpos_hi > qpos_lo

    radius_m = estimate_knob_radius_m(libero_stove_scene, knob, axis_world)
    assert radius_m > 0.0

    cfg = MotionConfig().knob
    target_width = min(cfg.grasp_chord_width_m, 2.0 * radius_m * 0.99)
    d = float(np.sqrt(max(radius_m**2 - (target_width / 2.0) ** 2, 0.0)))
    radial0 = _arbitrary_perpendicular(axis_world)
    grasp_pos = center_world + d * radial0
    grasp_quat = _orientation_for_tangential_grasp(-axis_world, radial0)

    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    start_q = libero_stove_scene.get_qpos(JOINT_NAMES)
    result = _solve_ik_robust(ik, libero_stove_scene, grasp_pos, grasp_quat, start_q, np.random.default_rng(0))
    assert result.success


@requires_libero_stove
def test_turn_knob_skill_produces_feasible_plan(libero_stove_scene):
    skill = _build_skill(libero_stove_scene)
    start_q = libero_stove_scene.get_qpos(JOINT_NAMES)
    result = skill.execute(libero_stove_scene, "flat_stove_1", start_q, {})

    assert result.success, result.diagnostics
    assert result.planned_trajectory is not None
    assert result.planned_trajectory.success
    assert result.diagnostics["turn_fraction_used"] is not None
    assert result.diagnostics["target_qpos"] > result.diagnostics["current_qpos"]
    # approach_positions is the descent (standoff -> grasp) only; the turn
    # leg (grasp -> turned_q) is tracked separately by the adapter, only
    # after the gripper has actually closed on the knob.
    approach_positions = np.asarray(result.planned_trajectory.metadata["approach_positions"])
    grasp_q = np.asarray(result.planned_trajectory.metadata["grasp_q"])
    turned_q = np.asarray(result.planned_trajectory.metadata["turned_q"])
    assert np.allclose(approach_positions[-1], grasp_q)
    assert not np.allclose(grasp_q, turned_q)


@requires_libero_stove
def test_turn_knob_skill_unresolvable_knob_reports_structured_failure(libero_stove_scene, monkeypatch):
    # A single-knob scene resolves any reference to that one knob (same
    # fallback as resolve_drawer's own single-fixture case) -- so an
    # unmatched name only actually fails to resolve when more than one
    # candidate exists. Fake a second knob to exercise that ambiguous path.
    import step.system_one.motion.articulation as articulation_module

    real_knob = resolve_knob(libero_stove_scene, "flat_stove_1")
    fake_second = articulation_module.KnobHandle(knob_body="other_knob", joint_name="other_knob_joint", label="other_knob")
    monkeypatch.setattr(articulation_module, "discover_knobs", lambda scene: [real_knob, fake_second])

    skill = _build_skill(libero_stove_scene)
    start_q = libero_stove_scene.get_qpos(JOINT_NAMES)
    result = skill.execute(libero_stove_scene, "nonexistent_knob_xyz", start_q, {})
    assert not result.success
    assert result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE


@requires_libero_stove
def test_scene_state_untouched_after_plan_evaluation(libero_stove_scene):
    skill = _build_skill(libero_stove_scene)
    start_q = libero_stove_scene.get_qpos(JOINT_NAMES).copy()
    skill.execute(libero_stove_scene, "flat_stove_1", start_q, {})
    assert np.allclose(libero_stove_scene.get_qpos(JOINT_NAMES), start_q)
