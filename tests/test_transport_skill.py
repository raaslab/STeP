import numpy as np

from conftest import EE_SITE, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES, requires_libero_drawer
from step.system_one.motion.attached_object import AttachedObjectManager
from step.system_one.motion.collision import CollisionChecker
from step.system_one.motion.config import MotionConfig, PlannerConfig
from step.system_one.motion.kinematics import IKSolver
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.scene_query import resolve_body_name
from step.system_one.motion.skills.transport_skill import TransportSkill


def _attach_object_near_gripper(scene, ik, object_body, joint_names):
    """Move the arm's own scratch qpos to a plausible top-down pose just
    above ``object_body`` and attach it there -- a realistic (small)
    hand-to-object transform, not an arbitrary/far-apart one, so the
    resulting AttachedObjectState is geometrically sensible for
    TransportSkill's own hover-pose math to build on."""
    obj_pos, _ = scene.body_pose(object_body)
    target_pos = obj_pos + np.array([0.0, 0.0, 0.02])
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing straight down
    start_q = scene.get_qpos(joint_names)
    ik_result = ik.solve(scene.clone(), target_pos, target_quat, seed_q=start_q)
    assert ik_result.success, "setup failed: could not pose the arm near the object"
    attach_scene = scene.clone()
    attach_scene.set_qpos(joint_names, ik_result.q)
    manager = AttachedObjectManager()
    attached = manager.attach(attach_scene, object_body, HAND_SITE)

    # Lift a few cm clear of the object before treating this as the
    # transport leg's own `start_q` -- a real completed pick always leaves
    # the gripper closed and lifted off the object by the time transport
    # begins, never sitting right at the grasp descent pose itself (which
    # would trivially read as FREE_SPACE-in-collision with the fingers
    # still open around it).
    lifted_pos = target_pos + np.array([0.0, 0.0, 0.15])
    lift_result = ik.solve(attach_scene.clone(), lifted_pos, target_quat, seed_q=ik_result.q)
    assert lift_result.success, "setup failed: could not lift clear of the object"
    return attach_scene, lift_result.q, attached


@requires_libero_drawer
def test_transport_skill_produces_feasible_plan(libero_drawer_scene):
    scene = libero_drawer_scene
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    obj_body = resolve_body_name(scene, "akita_black_bowl_1")
    attach_scene, start_q, attached = _attach_object_near_gripper(scene, ik, obj_body, JOINT_NAMES)

    checker = CollisionChecker(attach_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)
    lo, hi = attach_scene.joint_qpos_range(JOINT_NAMES)
    planner = MotionPlanner(
        JOINT_NAMES, lo, hi, checker, PlannerConfig(timeout_sec=20.0, range=0.2, interpolation_density=15)
    )
    config = MotionConfig()
    config.transport.planning_attempts = 6  # RRTConnect is stochastic; the real cabinet geometry is a harder plan than an empty placeholder body
    skill = TransportSkill(ik, planner, checker, config)

    # "wooden_cabinet_1_base" is the cabinet's own static frame body (see
    # libero_scene.infer_cabinet_support_entities) -- the same canonical
    # name a real executor run resolves goal_container_name from. Not
    # "wooden_cabinet_1_main": that's a real but *empty* world-parented
    # ancestor with none of the actual geometry (see
    # discover_cabinet_support_bodies's own docstring).
    goal_body = resolve_body_name(attach_scene, "wooden_cabinet_1_base")

    result = skill.execute(attach_scene, attached, goal_body, start_q, place_surface="top")
    assert result.success, result.diagnostics
    assert result.planned_trajectory is not None
    assert result.planned_trajectory.success
    assert "hover_q" in result.planned_trajectory.metadata


@requires_libero_drawer
def test_transport_skill_reaches_hover_pose_above_measured_container_top(libero_drawer_scene):
    """The planned hover pose's height should sit `hover_clearance_m` above
    the cabinet's own measured top surface, not some other value -- a
    direct check that TransportSkill and PlaceSkill agree on where "over
    the goal" is (see this module's own docstring on why that matters)."""
    scene = libero_drawer_scene
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    obj_body = resolve_body_name(scene, "akita_black_bowl_1")
    attach_scene, start_q, attached = _attach_object_near_gripper(scene, ik, obj_body, JOINT_NAMES)

    checker = CollisionChecker(attach_scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES)
    lo, hi = attach_scene.joint_qpos_range(JOINT_NAMES)
    config = MotionConfig()
    config.transport.planning_attempts = 6  # RRTConnect is stochastic; the real cabinet geometry is a harder plan than an empty placeholder body
    planner = MotionPlanner(JOINT_NAMES, lo, hi, checker, PlannerConfig(timeout_sec=20.0, range=0.2))
    skill = TransportSkill(ik, planner, checker, config)
    goal_body = resolve_body_name(attach_scene, "wooden_cabinet_1_base")

    from step.system_one.motion.skills.place_skill import _support_floor_and_rim_z

    top_z, _ = _support_floor_and_rim_z(attach_scene, goal_body)
    result = skill.execute(attach_scene, attached, goal_body, start_q, place_surface="top")
    assert result.success, result.diagnostics
    expected_hover_z = top_z + config.place.hover_clearance_m
    assert abs(result.diagnostics["hover_xyz"][2] - expected_hover_z) < 1e-6
