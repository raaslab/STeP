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
    lift_check_m: float = 0.05
    # Standoff distance for the final approach segment (see PickSkill's
    # module docstring): the long-range approach plans under strict
    # FREE_SPACE to a point this far from the grasp pose along the approach
    # direction, then a short final descent from there to the grasp pose is
    # explicitly collision-validated (never blindly executed) under
    # TERMINAL_APPROACH. Not a separate caller-visible stage -- PickSkill
    # is still one atomic plan call -- just where the "no contact yet" ->
    # "contact with the target is now expected" transition happens.
    approach_standoff_m: float = 0.08
    # Rim/edge grasp strategy (bowls, mugs, cups: objects whose full
    # cross-section doesn't fit the gripper at any yaw). See
    # grasp_generator.PrivilegedTopDownGraspGenerator's module docstring.
    rim_num_angle_samples: int = 8
    rim_offset_fraction: float = 0.6
    rim_grasp_width_m: float = 0.02
    # Grasp-candidate strategy:
    #   "aabb"           -- the original PrivilegedTopDownGraspGenerator only
    #                       (AABB-center yaw samples + rim ring).
    #   "mesh_antipodal" -- also run MeshAntipodalGraspGenerator, which samples
    #                       the object's real collision-geom surface and keeps
    #                       finger-contact pairs that are antipodal within a
    #                       friction cone and fit the gripper opening. The two
    #                       generators' candidates are concatenated; PickSkill
    #                       still ranks everything by IK/collision/OMPL cost.
    strategy: str = "aabb"
    gripper_max_opening_m: float = 0.08
    # MeshAntipodalGraspGenerator knobs (only used when strategy="mesh_antipodal").
    mesh_samples_per_geom: int = 48
    mesh_num_height_levels: int = 7
    mesh_num_yaw_samples: int = 6
    mesh_num_perp_offsets: int = 5
    mesh_band_half_m: float = 0.012          # vertical half-thickness of a height slice
    mesh_line_half_m: float = 0.010          # lateral half-width of the "gripper line" corridor
    mesh_friction_cone_deg: float = 25.0     # max angle between a contact normal and the close axis
    mesh_min_width_m: float = 0.005
    mesh_width_margin_m: float = 0.004       # keep this much clearance inside the max opening
    mesh_max_candidates: int = 14            # keep only the best N (each costs IK+OMPL downstream)


@dataclass
class PlannerConfig:
    backend: str = "rrtconnect"
    range: float = 0.08
    # Generous enough to cover a full pick-to-container move in one plan
    # call (e.g. when `transport` is skipped and `place` has to connect
    # wherever `pick` left the arm all the way to the goal container,
    # rather than just a short final hover adjustment): RRTConnect only
    # takes this long when it actually needs to, since it returns as soon
    # as it finds an exact solution.
    timeout_sec: float = 20.0
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
