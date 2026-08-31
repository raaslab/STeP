"""Typed configuration for the motion pipeline. Loaded from a scene_context's
``platform.libero.motion`` block (see ``configs/libero_template.json``'s ``motion``
section) so every tunable below has a named, documented home instead of being a
hidden magic number in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraspSampleConfig:
    num_yaw_samples: int = 8
    yaw_range: tuple[float, float] = (0.0, 3.141592653589793)  # [0, pi)
    pregrasp_offset_m: float = 0.10
    lift_check_m: float = 0.05
    # Rim/edge grasp strategy (bowls, mugs, cups: objects whose full
    # cross-section doesn't fit the gripper at any yaw). See
    # grasp_generator.PrivilegedTopDownGraspGenerator's module docstring.
    rim_num_angle_samples: int = 8
    rim_offset_fraction: float = 0.6
    rim_grasp_width_m: float = 0.02


@dataclass
class PlannerConfig:
    backend: str = "rrtconnect"
    range: float = 0.05
    timeout_sec: float = 5.0
    seed: int = 0
    interpolation_density: int = 200
    validity_resolution: float = 0.01  # fraction of state-space extent per OMPL convention


@dataclass
class CollisionConfig:
    self_margin_m: float = 0.005
    env_margin_m: float = 0.01
    # Smaller than env_margin_m on purpose: the attached-object sphere fit
    # (see attached_object.fit_spheres_from_geoms) is already a conservative
    # over-approximation of the real geometry -- it's built to never
    # undercover the object's true collision shape -- so stacking a full
    # extra centimeter of margin on top of that already-conservative sphere
    # made every placement candidate in a modestly-sized container (sphere
    # radius and container clearance both a few centimeters) read as
    # colliding even when clear by several millimeters of real geometry.
    attached_margin_m: float = 0.003


@dataclass
class TimeParamConfig:
    joint_vel_limits: list[float] | None = None  # rad/s per joint; None -> read from model
    joint_accel_limits: list[float] | None = None  # rad/s^2 per joint; None -> conservative default
    default_vel_limit: float = 1.5
    default_accel_limit: float = 3.0
    control_hz: float = 20.0


@dataclass
class PlaceSampleConfig:
    # Candidate placement points are sampled within a shrunk copy of the goal
    # container's interior footprint (center + a ring), mirroring how the
    # grasp generator shrinks inward from an object's rim for the same
    # "don't aim at the wall" reason.
    interior_shrink_fraction: float = 0.35
    num_ring_samples: int = 4
    hover_clearance_m: float = 0.06  # above the container's rim before descending
    drop_clearance_m: float = 0.02  # above the container's floor (or the next object already in it)


@dataclass
class ReplanConfig:
    tracking_error_m: float = 0.03
    collision_margin_m: float = 0.01


@dataclass
class ExecutionConfig:
    """Grasp-closing timing. The MPC tracks a reference with nonzero steady-
    state lag (see mpc.py); closing the gripper as soon as the reference
    trajectory is exhausted -- instead of once the end effector has actually
    *arrived* -- can close on empty air or catch the object off-center."""

    settle_position_tol_m: float = 0.004
    settle_orientation_tol_rad: float = 0.05
    max_settle_steps: int = 150


@dataclass
class MotionConfig:
    """Top-level container. ``MPCCostConfig`` lives in ``types.py`` since it's part
    of the MPC's public result/contract surface, not just tuning; it's re-exported
    here via ``mpc``."""

    grasp: GraspSampleConfig = field(default_factory=GraspSampleConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    time_param: TimeParamConfig = field(default_factory=TimeParamConfig)
    place: PlaceSampleConfig = field(default_factory=PlaceSampleConfig)
    replan: ReplanConfig = field(default_factory=ReplanConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    mpc: dict[str, float] = field(default_factory=dict)  # raw dict -> MPCCostConfig(**mpc)

    @staticmethod
    def from_scene_context(scene_context: dict[str, Any] | None) -> "MotionConfig":
        raw = (
            ((scene_context or {}).get("platform", {}) or {})
            .get("libero", {}) or {}
        ).get("motion", {}) or {}
        return MotionConfig(
            grasp=GraspSampleConfig(**raw.get("grasp", {})),
            planner=PlannerConfig(**raw.get("planner", {})),
            collision=CollisionConfig(**raw.get("collision", {})),
            time_param=TimeParamConfig(**raw.get("time_param", {})),
            place=PlaceSampleConfig(**raw.get("place", {})),
            replan=ReplanConfig(**raw.get("replan", {})),
            execution=ExecutionConfig(**raw.get("execution", {})),
            mpc=dict(raw.get("mpc", {})),
        )
