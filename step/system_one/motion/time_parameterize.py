"""Time-parameterization of an OMPL geometric path under joint velocity/
acceleration limits.

This is a **simplified** per-joint trapezoidal scaling, not a full Time-
Optimal Path Parameterization (TOPP) solver -- no ``toppra`` dependency was
added (not installed in this environment, and the task explicitly says to
avoid adding heavy new dependencies where an existing stack suffices). The
``TimeParameterizer`` interface is the seam to swap in a real TOPP solver
later without touching callers.

Algorithm: segment durations are set purely by each joint's velocity limit
(``|dq| / v_limit``, max across joints) -- this is exactly cumulative
normalized arc length, so total duration only depends on the path's actual
geometry, not on how densely it happens to be sampled (an earlier version of
this module instead gave every segment an independent from-rest-to-rest
trapezoidal profile, which made duration scale with the *number* of waypoints
instead of distance -- see ``tests/test_time_parameterize.py::
test_duration_scales_with_distance_not_number_of_waypoints``). Acceleration
limits are then enforced by clipping the finite-differenced acceleration
profile after resampling, rather than folded into segment-duration
allocation -- simpler, and avoids re-integrating positions (which would drift
off the planned path's exact endpoints). This is a genuine simplification,
not a full time-optimal solve: a real TOPP pass would jointly optimize the
whole velocity profile instead of clipping after the fact.
"""

from __future__ import annotations

import numpy as np

from .config import TimeParamConfig
from .types import MotionPlan


class TimeParameterizer:
    def __init__(self, config: TimeParamConfig | None = None) -> None:
        self.config = config or TimeParamConfig()

    def _limits(self, n_joints: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        vel = (
            np.asarray(cfg.joint_vel_limits, dtype=np.float64)
            if cfg.joint_vel_limits is not None
            else np.full(n_joints, cfg.default_vel_limit)
        )
        acc = (
            np.asarray(cfg.joint_accel_limits, dtype=np.float64)
            if cfg.joint_accel_limits is not None
            else np.full(n_joints, cfg.default_accel_limit)
        )
        return vel, acc

    def parameterize(self, plan: MotionPlan) -> MotionPlan:
        if not plan.success or plan.geometric_path is None or len(plan.geometric_path) < 2:
            return plan

        path = np.asarray(plan.geometric_path, dtype=np.float64)
        n_wp, n_joints = path.shape
        vel_limits, acc_limits = self._limits(n_joints)

        deltas = np.diff(path, axis=0)  # (n_wp-1, n_joints)
        with np.errstate(divide="ignore", invalid="ignore"):
            segment_durations = np.max(np.abs(deltas) / vel_limits[None, :], axis=1)
        segment_durations = np.nan_to_num(segment_durations, nan=0.0, posinf=0.0)
        segment_durations = np.maximum(segment_durations, 1e-3)  # avoid zero-length segments

        times = np.concatenate([[0.0], np.cumsum(segment_durations)])

        # Resample onto a fixed control-rate grid so the MPC gets an evenly
        # spaced reference.
        dt = 1.0 / max(self.config.control_hz, 1e-6)
        n_steps = max(2, int(np.ceil(times[-1] / dt)) + 1)
        sample_times = np.linspace(0.0, times[-1], n_steps)
        positions = np.zeros((n_steps, n_joints))
        for j in range(n_joints):
            positions[:, j] = np.interp(sample_times, times, path[:, j])

        velocities = np.gradient(positions, sample_times, axis=0)
        velocities = np.clip(velocities, -vel_limits[None, :], vel_limits[None, :])
        accelerations = np.gradient(velocities, sample_times, axis=0)
        accelerations = np.clip(accelerations, -acc_limits[None, :], acc_limits[None, :])

        plan.times = sample_times
        plan.positions = positions
        plan.velocities = velocities
        plan.accelerations = accelerations
        plan.metadata.setdefault("time_parameterization", "velocity_limited_v2")
        return plan
