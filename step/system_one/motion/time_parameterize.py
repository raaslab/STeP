"""Time-parameterization of an OMPL geometric path under joint velocity/
acceleration limits.

A simplified per-joint trapezoidal scaling, not a full Time-Optimal Path
Parameterization (TOPP) solver -- the existing stack doesn't need a
toppra dependency. TimeParameterizer is the seam to swap one in later.

Segment durations are set by each joint's velocity limit (|dq| / v_limit,
max across joints), so total duration depends only on path geometry.
Acceleration limits are enforced by clipping the finite-differenced
acceleration profile after resampling, rather than folded into duration
allocation -- simpler, and avoids re-integrating positions off the path's exact endpoints.
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
