import numpy as np

from conftest import EE_SITE, JOINT_NAMES, requires_libero
from step.system_one.motion.kinematics import IKSolver


@requires_libero
def test_ik_converges_to_nearby_target(libero_scene):
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    tcp_pos, tcp_quat = libero_scene.site_pose(EE_SITE)
    target_pos = tcp_pos + np.array([0.05, 0.02, -0.03])
    scratch = libero_scene.clone()
    result = ik.solve(scratch, target_pos, tcp_quat)
    assert result.success
    assert result.position_error < 0.002
    assert result.orientation_error < 0.02

    scratch.set_qpos(JOINT_NAMES, result.q)
    achieved_pos, _ = scratch.site_pose(EE_SITE)
    assert np.linalg.norm(achieved_pos - target_pos) < 0.002


@requires_libero
def test_ik_does_not_mutate_input_scene(libero_scene):
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    tcp_pos, tcp_quat = libero_scene.site_pose(EE_SITE)
    before = libero_scene.get_qpos(JOINT_NAMES).copy()
    scratch = libero_scene.clone()
    ik.solve(scratch, tcp_pos + np.array([0.1, 0.0, 0.0]), tcp_quat)
    after = libero_scene.get_qpos(JOINT_NAMES)
    assert np.allclose(before, after)


@requires_libero
def test_ik_rejects_unreachable_target(libero_scene):
    ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
    tcp_pos, tcp_quat = libero_scene.site_pose(EE_SITE)
    # 5 meters away -- far outside the Panda's ~0.85m reach.
    far_target = tcp_pos + np.array([5.0, 0.0, 0.0])
    scratch = libero_scene.clone()
    result = ik.solve(scratch, far_target, tcp_quat)
    assert not result.success
    assert result.q is None
