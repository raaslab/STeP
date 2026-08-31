import numpy as np

from step.system_one.motion.config import TimeParamConfig
from step.system_one.motion.time_parameterize import TimeParameterizer
from step.system_one.motion.types import MotionPlan, PlannerFailureReason


def _straight_line_plan(n_joints=7, n_wp=3, delta=0.2):
    path = np.zeros((n_wp, n_joints))
    for i in range(1, n_wp):
        path[i] = path[i - 1] + delta
    return MotionPlan(success=True, geometric_path=path, failure_reason=PlannerFailureReason.NONE)


def test_noop_on_failed_plan():
    plan = MotionPlan(success=False, failure_reason=PlannerFailureReason.NO_PATH_FOUND)
    out = TimeParameterizer().parameterize(plan)
    assert out.positions is None


def test_produces_monotonic_time_grid_and_matching_shapes():
    plan = _straight_line_plan()
    tp = TimeParameterizer(TimeParamConfig(control_hz=20.0, default_vel_limit=1.0, default_accel_limit=2.0))
    out = tp.parameterize(plan)
    assert out.times is not None
    assert np.all(np.diff(out.times) > 0)
    assert out.positions.shape[0] == len(out.times)
    assert out.positions.shape[1] == 7
    assert out.velocities.shape == out.positions.shape
    assert out.accelerations.shape == out.positions.shape


def test_endpoints_match_original_path():
    plan = _straight_line_plan()
    out = TimeParameterizer(TimeParamConfig(control_hz=20.0)).parameterize(plan)
    assert np.allclose(out.positions[0], plan.geometric_path[0], atol=1e-6)
    assert np.allclose(out.positions[-1], plan.geometric_path[-1], atol=1e-6)


def test_duration_scales_with_distance_not_number_of_waypoints():
    """Regression test for a real bug found during development: a naive
    per-segment 'v = a*t/2' floor made total duration scale with the NUMBER of
    (arbitrarily dense) interpolated waypoints instead of the actual path
    distance. Re-sampling the same straight-line path more densely must not
    change its time-optimal duration."""
    coarse = _straight_line_plan(n_wp=3, delta=0.2)  # total distance 0.4 per joint
    dense_path = np.linspace(coarse.geometric_path[0], coarse.geometric_path[-1], 30)
    dense = MotionPlan(success=True, geometric_path=dense_path, failure_reason=PlannerFailureReason.NONE)

    cfg = TimeParamConfig(control_hz=20.0, default_vel_limit=1.0, default_accel_limit=2.0)
    coarse_out = TimeParameterizer(cfg).parameterize(coarse)
    dense_out = TimeParameterizer(cfg).parameterize(dense)

    assert coarse_out.times[-1] < 2.0  # sane: ~0.4 rad at 1 rad/s plus accel ramp, not tens of seconds
    assert abs(coarse_out.times[-1] - dense_out.times[-1]) < 0.05


def test_respects_velocity_limit():
    plan = _straight_line_plan(n_wp=2, delta=1.0)
    cfg = TimeParamConfig(control_hz=50.0, default_vel_limit=0.5, default_accel_limit=100.0)
    out = TimeParameterizer(cfg).parameterize(plan)
    max_speed = np.max(np.abs(out.velocities))
    assert max_speed <= 0.5 + 1e-2
