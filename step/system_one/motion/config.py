"""Typed configuration for the motion pipeline. Loaded from a scene_context's
``platform.libero.motion`` block (see ``configs/libero_template.json``'s ``motion``
section) so every tunable below has a named, documented home instead of being a
hidden magic number in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # Type-checking only -- see the real (lazy, runtime) imports just above
    # MotionConfig for why these aren't imported at module level.
    from ..skills.close.close_skill import CloseConfig
    from ..skills.open_drawer.open_drawer_skill import DrawerConfig
    from ..skills.pick.contact_graspnet_generator import ContactGraspNetConfig
    from ..skills.pick.grasp_generator import GraspSampleConfig
    from ..skills.place.place_skill import PlaceSampleConfig
    from ..skills.transport.transport_skill import TransportConfig
    from ..skills.turn_knob.turn_knob_skill import KnobConfig


@dataclass
class PlannerConfig:
    backend: str = "rrtconnect"
    # Max joint-space step per RRTConnect extension. The FREE_SPACE leg
    # (home -> standoff near the next target) can span several radians, and
    # too small a range makes the two trees need far more extensions to
    # link up, so OMPL gives up with an "approximate solution" (treated as
    # failure) well before timeout. Safety is unaffected -- validity
    # checking and the post-hoc edge re-validation both use a fixed
    # resolution independent of this value.
    range: float = 0.2
    # Not the lever that fixes long-horizon failures (see `range` above) --
    # kept generous so a merely-slow plan still has room to finish.
    timeout_sec: float = 20.0
    seed: int = 0
    interpolation_density: int = 200
    validity_resolution: float = 0.01  # fraction of state-space extent per OMPL convention
    # Budget for OMPL's post-solve path shortening. The unbounded default
    # routinely cost several seconds per plan call; since the simplified
    # path is re-validated edge-by-edge afterward regardless, a short
    # budget trades a slightly less-shortened path for faster planning.
    simplify_time_sec: float = 0.25


@dataclass
class CollisionConfig:
    self_margin_m: float = 0.005
    env_margin_m: float = 0.01
    # Smaller than env_margin_m on purpose: the attached-object sphere fit
    # (see attached_object.fit_spheres_from_geoms) is already a conservative
    # over-approximation, so stacking a full extra margin on top of it made
    # placement candidates in modestly-sized containers read as colliding
    # even when clear by several millimeters of real geometry.
    attached_margin_m: float = 0.003
    # Tolerated interpenetration between the gripper's FINGER geoms (not the
    # palm) and the grasp target, during TERMINAL_APPROACH only -- mirrors
    # real gripper compliance, since a rigid check would reject any object
    # slightly wider than the gripper's rated opening even though only the
    # final pinch was ever going to be snug. 0.0 disables this entirely.
    finger_overlap_tolerance_m: float = 0.006
    # LIBERO-Constrained keep-out zones: the STL task spec's G(Safe(obj))
    # obligations, made enforceable at plan/control time. Empty by default, so
    # an unconstrained run behaves exactly as before. See KeepoutZone.
    keepout_zones: list["KeepoutZone"] = field(default_factory=list)


@dataclass
class KeepoutZone:
    """An infinite vertical cylinder the robot must stay out of.

    Shares its geometry definition with the benchmark's scoring SDF
    (``CircleSDF(convention="outside_positive")``) so enforcement and the
    reported violation rate can't drift apart. Unbounded in z, so the
    constraint must be satisfied by routing around in xy, not by lifting
    over it.
    """

    name: str = ""
    center_xy: tuple[float, float] = (0.0, 0.0)
    radius_m: float = 0.06
    # Bodies exempt from this zone -- normally the obstacle's own body, so the
    # zone does not forbid the scene object that defines it from existing, and
    # the manipulated object when it is also the obstacle.
    exempt_bodies: tuple[str, ...] = ()

    def clearance(self, xy: Any) -> float:
        """Signed distance to the zone boundary: >0 outside, <0 inside."""
        p = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
        c = np.asarray(self.center_xy, dtype=np.float64).reshape(-1)[:2]
        return float(np.linalg.norm(p - c)) - float(self.radius_m)


@dataclass
class TimeParamConfig:
    joint_vel_limits: list[float] | None = None  # rad/s per joint; None -> read from model
    joint_accel_limits: list[float] | None = None  # rad/s^2 per joint; None -> conservative default
    default_vel_limit: float = 1.5
    default_accel_limit: float = 3.0
    control_hz: float = 20.0


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
    # cart_pos = clip(pos_err / settle_pos_scale_m, -1, 1). A smaller divisor
    # saturates to sustained max-speed motion for any larger residual gap --
    # fine for the few-mm steady-state MPC lag this loop targets, but
    # place's settle phase can hand it a much larger gap, and full-speed
    # motion there was enough to yank a friction-only, off-center grasp
    # loose. A larger divisor ramps down earlier instead; max_settle_steps
    # gives it room to cover a larger gap at a gentler pace.
    settle_pos_scale_m: float = 0.15
    # If position error stops improving by more than this over
    # `stall_window_steps` steps, the loop has stopped converging and is
    # just hunting in place -- almost always because the target sits right
    # at a real contact boundary. Breaking out early doesn't change where
    # the arm ends up, it just stops the pointless, visible struggling.
    stall_window_steps: int = 15
    stall_position_tol_m: float = 0.0015


# MotionConfig composes each skill's own config dataclass (grasp from
# grasp_generator.py, drawer from open_drawer_skill.py, etc.). These
# factories and from_scene_context below import the skill modules lazily
# rather than at module level: this module must not pull in the parent
# `system_one` package at import time, and skills/ already imports *from*
# motion/ (e.g. grasp_generator -> ik_utils -> collision -> config), so an
# eager import back into config.py here would be a real circular import.
def _default_grasp_config():
    from ..skills.pick.grasp_generator import GraspSampleConfig

    return GraspSampleConfig()


def _default_place_config():
    from ..skills.place.place_skill import PlaceSampleConfig

    return PlaceSampleConfig()


def _default_transport_config():
    from ..skills.transport.transport_skill import TransportConfig

    return TransportConfig()


def _default_drawer_config():
    from ..skills.open_drawer.open_drawer_skill import DrawerConfig

    return DrawerConfig()


def _default_knob_config():
    from ..skills.turn_knob.turn_knob_skill import KnobConfig

    return KnobConfig()


def _default_close_config():
    from ..skills.close.close_skill import CloseConfig

    return CloseConfig()


def _default_contact_graspnet_config():
    from ..skills.pick.contact_graspnet_generator import ContactGraspNetConfig

    return ContactGraspNetConfig()


@dataclass
class MotionConfig:
    """Top-level container. ``MPCCostConfig`` lives in ``types.py`` since it's part
    of the MPC's public result/contract surface, not just tuning; it's re-exported
    here via ``mpc``."""

    grasp: "GraspSampleConfig" = field(default_factory=_default_grasp_config)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    time_param: TimeParamConfig = field(default_factory=TimeParamConfig)
    place: "PlaceSampleConfig" = field(default_factory=_default_place_config)
    transport: "TransportConfig" = field(default_factory=_default_transport_config)
    replan: ReplanConfig = field(default_factory=ReplanConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    drawer: "DrawerConfig" = field(default_factory=_default_drawer_config)
    knob: "KnobConfig" = field(default_factory=_default_knob_config)
    close: "CloseConfig" = field(default_factory=_default_close_config)
    contact_graspnet: "ContactGraspNetConfig" = field(default_factory=_default_contact_graspnet_config)
    mpc: dict[str, float] = field(default_factory=dict)  # raw dict -> MPCCostConfig(**mpc)

    @staticmethod
    def from_scene_context(scene_context: dict[str, Any] | None) -> "MotionConfig":
        # See the module-level comment above these classes' own factories for
        # why these imports are lazy, not at module level.
        from ..skills.close.close_skill import CloseConfig
        from ..skills.open_drawer.open_drawer_skill import DrawerConfig
        from ..skills.pick.contact_graspnet_generator import ContactGraspNetConfig
        from ..skills.pick.grasp_generator import GraspSampleConfig
        from ..skills.place.place_skill import PlaceSampleConfig
        from ..skills.transport.transport_skill import TransportConfig
        from ..skills.turn_knob.turn_knob_skill import KnobConfig

        raw = (
            ((scene_context or {}).get("platform", {}) or {})
            .get("libero", {}) or {}
        ).get("motion", {}) or {}
        collision = CollisionConfig(**raw.get("collision", {}))
        # G(Safe(obj)) reaches the motion layer only here: the scene's SDF
        # keep-out zones become constraints the OMPL validity check and the
        # live MPC both enforce, so the formula actually shapes trajectories
        # instead of only being monitored after the fact.
        collision.keepout_zones = list(collision.keepout_zones) + _keepout_zones_from_scene(scene_context)
        return MotionConfig(
            grasp=GraspSampleConfig(**raw.get("grasp", {})),
            planner=PlannerConfig(**raw.get("planner", {})),
            collision=collision,
            time_param=TimeParamConfig(**raw.get("time_param", {})),
            place=PlaceSampleConfig(**raw.get("place", {})),
            transport=TransportConfig(**raw.get("transport", {})),
            replan=ReplanConfig(**raw.get("replan", {})),
            execution=ExecutionConfig(**raw.get("execution", {})),
            drawer=DrawerConfig(**raw.get("drawer", {})),
            knob=KnobConfig(**raw.get("knob", {})),
            close=CloseConfig(**raw.get("close", {})),
            contact_graspnet=ContactGraspNetConfig(**raw.get("contact_graspnet", {})),
            mpc=dict(raw.get("mpc", {})),
        )


def _keepout_zones_from_scene(scene_context: dict[str, Any] | None) -> list[KeepoutZone]:
    """Collect the scene's ``outside_positive`` circle SDFs as keep-out zones.

    Reads the same ``geometry[name].sdf_geometry`` blocks that
    ``geometry_sdf.parse_sdf_constraints`` turns into the benchmark's SDF
    constraints, so the planner and the metric are driven from one source.
    Parsed off the raw dict rather than by importing that parser, since this
    module must not pull in the parent ``system_one`` package.

    Only circles are read; a box/polygon keep-out has no cylinder equivalent
    here, so it's skipped rather than approximated.
    """
    if not scene_context:
        return []
    # RQ2 ablation switch: disabling this leaves the scene's SDF constraints
    # (and every scored signal) untouched, and only stops the motion layer
    # from planning around them -- isolating "STL reaches control" from "STL
    # is monitored" under an identical scene and metric.
    experiment = ((scene_context.get("platform") or {}).get("libero_experiment") or {})
    if not bool(experiment.get("enforce_keepout", True)):
        return []
    geometry = scene_context.get("geometry") or {}
    if not isinstance(geometry, dict):
        return []

    # The manipulated object is never its own keep-out -- some tasks name
    # the object being picked as the obstacle, which would make the task
    # unsolvable by construction.
    manipulated: set[str] = set()
    for ent in scene_context.get("entities") or []:
        if isinstance(ent, dict) and str(ent.get("role", "")) in {"manipulated_object", "target_object"}:
            name = ent.get("name")
            if name:
                manipulated.add(str(name))

    zones: list[KeepoutZone] = []
    for name, g in geometry.items():
        if not isinstance(g, dict):
            continue
        sdf = g.get("sdf_geometry")
        if not isinstance(sdf, dict):
            continue
        if str(sdf.get("type", "")).lower() != "circle":
            continue
        if str(sdf.get("convention", "")) != "outside_positive":
            continue
        center = sdf.get("center_xy") or g.get("xy")
        radius = sdf.get("radius", g.get("radius"))
        if center is None or radius is None:
            continue
        if str(name) in manipulated:
            continue
        try:
            cx, cy = float(center[0]), float(center[1])
            r = float(radius)
        except (TypeError, ValueError, IndexError):
            continue
        if not (r > 0.0):
            continue
        zones.append(
            KeepoutZone(
                name=str(name),
                center_xy=(cx, cy),
                radius_m=r,
                exempt_bodies=(str(name),),
            )
        )
    return zones
