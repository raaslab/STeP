import numpy as np

from conftest import EE_SITE, JOINT_NAMES, requires_libero
from step.system_one.motion.mpc import JointSpaceMPC, MPCContext, ObstacleSphere, TimingWindow
from step.system_one.motion.types import MPCCostConfig


@requires_libero
def test_tracking_reference_advances_with_t0_step(libero_scene):
    """Regression test for a real bug found during development: the tracking
    cost indexed the reference by the *horizon-local* step instead of the
    *global* step, so the MPC always chased the beginning of the reference no
    matter how far execution had actually progressed, and tracking error grew
    without bound instead of converging."""
    q0 = libero_scene.get_qpos(JOINT_NAMES)
    n_joints = len(JOINT_NAMES)
    horizon = 6
    # A long reference the horizon can't span in one shot, so t0_step matters.
    ref = np.tile(q0, (50, 1))
    ref[10:] += 0.3  # a step change well beyond one horizon from the start

    mpc = JointSpaceMPC(JOINT_NAMES, EE_SITE, horizon=horizon, num_iters=10, cost_config=MPCCostConfig())
    ctx = MPCContext(reference=ref)
    scratch = libero_scene.clone()

    errors = []
    q = q0.copy()
    for t0 in range(0, 20):
        scratch.set_qpos(JOINT_NAMES, q)
        result = mpc.step(scratch, q, ctx, t0_step=t0)
        q = result.joint_command
        errors.append(result.tracking_error)

    # Once execution has caught up to and passed the step change in the
    # reference (t0 >= 10), tracking error must not be monotonically growing --
    # the bug produced strictly increasing error with no ceiling.
    tail = errors[12:]
    assert max(tail) < 0.5, f"tracking error should stay bounded, got {errors}"


@requires_libero
def test_clearance_cost_penalizes_proximity_to_obstacle(libero_scene):
    q0 = libero_scene.get_qpos(JOINT_NAMES)
    ee_pos, _ = libero_scene.site_pose(EE_SITE)

    mpc = JointSpaceMPC(JOINT_NAMES, EE_SITE, horizon=4, cost_config=MPCCostConfig(clearance_weight=5.0))
    ref = np.tile(q0, (10, 1))
    scratch = libero_scene.clone()

    far_obstacle = MPCContext(reference=ref, obstacles=[ObstacleSphere(center=ee_pos + 5.0, radius=0.05)])
    near_obstacle = MPCContext(reference=ref, obstacles=[ObstacleSphere(center=ee_pos.copy(), radius=0.2)])

    q_traj = mpc._rollout(q0, np.zeros((4, len(JOINT_NAMES))))
    cost_far, _ = mpc._clearance_cost_and_grad(scratch, q_traj, far_obstacle.obstacles)
    cost_near, _ = mpc._clearance_cost_and_grad(scratch, q_traj, near_obstacle.obstacles)
    assert cost_near > cost_far


@requires_libero
def test_timing_window_not_before_penalizes_early_arrival(libero_scene):
    q0 = libero_scene.get_qpos(JOINT_NAMES)
    ee_pos, _ = libero_scene.site_pose(EE_SITE)
    mpc = JointSpaceMPC(JOINT_NAMES, EE_SITE, horizon=4, cost_config=MPCCostConfig(timing_weight=5.0))
    scratch = libero_scene.clone()
    q_traj = mpc._rollout(q0, np.zeros((4, len(JOINT_NAMES))))

    # start_step far in the future -> being at the event target *now* is "too early".
    window_far_future = [TimingWindow(predicate="Grasped", event_target_xyz=ee_pos.copy(), start_step=1000, end_step=None, radius=0.5)]
    window_already_open = [TimingWindow(predicate="Grasped", event_target_xyz=ee_pos.copy(), start_step=0, end_step=None, radius=0.5)]

    cost_forbidden, _ = mpc._timing_cost_and_grad(scratch, q_traj, window_far_future, t0_step=0)
    cost_allowed, _ = mpc._timing_cost_and_grad(scratch, q_traj, window_already_open, t0_step=0)
    assert cost_forbidden > cost_allowed


def test_mpc_cost_config_fields_match_spec():
    cfg = MPCCostConfig()
    for field in (
        "tracking_weight",
        "terminal_weight",
        "control_weight",
        "smoothness_weight",
        "joint_limit_weight",
        "clearance_weight",
        "stl_weight",
        "timing_weight",
    ):
        assert hasattr(cfg, field)
