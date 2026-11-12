"""Typed configuration for the motion pipeline. Loaded from a scene_context's
platform.libero.motion block (see configs/libero_template.json's motion
section) so every tunable below has a named, documented home instead of being a
hidden magic number in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # Type-checking only; the real imports are lazy (see MotionConfig factories below).
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
    # Max joint-space step per RRTConnect extension. Too small makes long
    # FREE_SPACE hops time out before the trees connect.
    range: float = 0.2
    # Generous on purpose: a merely-slow plan still has room to finish.
    timeout_sec: float = 20.0
    seed: int = 0
    interpolation_density: int = 200
    validity_resolution: float = 0.01  # fraction of state-space extent per OMPL convention
    # Capped short: the unbounded default cost several seconds per plan,
    # and the simplified path gets re-validated edge-by-edge anyway.
    simplify_time_sec: float = 0.25


@dataclass
class CollisionConfig:
    self_margin_m: float = 0.005 # This one is legacy 
    env_margin_m: float = 0.01
    # Clearance kept around the carried object's spheres in transit. 0.7cm was
    # enough when the spheres were ~2x too wide; with the tight fit it let a
    # carried moka pot skim (and topple) its neighbor. Goal poses that need to
    # pack tighter are checked on exact geometry instead.
    attached_margin_m: float = 0.02
    # Tolerated finger (not palm) interpenetration with the grasp target
    # during TERMINAL_APPROACH, mirroring real gripper compliance. 0.0 disables it.
    finger_overlap_tolerance_m: float = 0.006
    # The same compliance allowance, for the PALM, against the grasp target only.
    # Zero by default: on LIBERO the objects are small enough that the hand never
    # needs to touch them. RoboCasa's produce is not -- its own human demos
    # penetrate the object 1.1 to 6.9 mm at the palm and lift it successfully, so
    # rejecting that outright threw away real grasps.
    palm_overlap_tolerance_m: float = 0.0
    # During TERMINAL_APPROACH, let the fingers (never the palm or arm) touch
    # the grasp target. For perceived targets: a top-surface reconstruction can't
    # show overhangs, so a bowl's flared wall is solid underneath and blocks
    # the outer finger of every rim grasp (MoveIt's allowed-collision matrix
    # and cuRobo excuse the target the same way).
    permit_gripper_target_contact: bool = False
    # STL G(Safe(obj)) keep-out zones, enforced at plan/control time. Empty by default (no constraints).
    keepout_zones: list["KeepoutZone"] = field(default_factory=list)
    # STL G(Upright(obj, deg)): the most the carried object may tip from how it
    # stood before it was picked up. None = no limit (the default).
    max_tilt_deg: float | None = None
    # VLM-MPC-Cost baseline only: the active subtask's cost terms. Empty under
    # STeP, which drives the controller from the STL formula instead.
    mpc_cost_terms: list["MPCCostTerm"] = field(default_factory=list)
    # VLM-MPC-Cost baseline: how the VLM's weights are mapped onto the
    # controller's own scale.
    #
    # The VLM writes weights over Cartesian predicate distances (metres), and
    # they were summed raw into an objective whose native terms are joint-space
    # (radians, tracking 1.0 / terminal 4.0) and hand-tuned over many runs. The
    # VLM is picking numbers on a scale it has never seen, so how hard its
    # avoid-term fights its own reach-term was partly an accident of magnitude
    # rather than anything the baseline's formalism implies -- measured, when
    # max(avoid)/max(reach) >= 0.5 the baseline's task success is 27%, and below
    # 0.5 it is 50%.
    #
    # Normalising divides by the largest reach weight, so the VLM's RELATIVE
    # priorities -- the thing actually under test -- are preserved exactly, and
    # only the overall magnitude is set by the gain below. Sweeping that one
    # number gives the baseline the same tuning budget the method's own weights
    # had, which is what a fair comparison needs.
    mpc_cost_normalize: bool = True
    mpc_cost_weight_gain: float = 1.0


@dataclass
class KeepoutZone:
    """An infinite vertical cylinder the robot must stay out of.

    Shares its geometry definition with the benchmark's scoring SDF
    (CircleSDF(convention="outside_positive")) so enforcement and the
    reported violation rate can't drift apart. Unbounded in z, so the
    constraint must be satisfied by routing around in xy, not by lifting
    over it.
    """

    name: str = ""
    center_xy: tuple[float, float] = (0.0, 0.0)
    radius_m: float = 0.06
    # Bodies exempt from this zone -- normally the obstacle's own body (so it
    # doesn't forbid the scene object that defines it from existing).
    exempt_bodies: tuple[str, ...] = ()
    # What must stay out. None is the end effector (the original behaviour).
    # An object name means that object while it is carried, placed by the grasp
    # transform -- G(Safe(obj, other, d)) is about the object, not the hand.
    subject: str | None = None
    # Part of radius_m that is planning buffer rather than the rule itself. The
    # tracker lets the hand lag the plan by up to ~2 cm, so the plan keeps
    # clear by that much more; a goal is judged against the rule alone.
    buffer_m: float = 0.0

    def clearance(self, xy: Any) -> float:
        """Signed distance to the zone boundary: >0 outside, <0 inside."""
        p = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
        c = np.asarray(self.center_xy, dtype=np.float64).reshape(-1)[:2]
        return float(np.linalg.norm(p - c)) - float(self.radius_m)


@dataclass
class MPCCostTerm:
    """One weighted term of a VLM-authored MPC cost function.

    The VLM-MPC-Cost baseline replaces STeP's STL specification with a cost
    function System 2 writes directly over the predicate set. Each predicate it
    names becomes one of these, resolved to a world position by the executor
    before it reaches the controller.

    Deliberately has no interval, no ordering, and no temporal operator of any
    kind -- that absence *is* the baseline. A weighted sum of distance terms can
    say "be near this" and "stay off that", but it cannot say "after" or
    "within [a,b]", which is what the comparison against STL is meant to expose.

    ``kind`` is either ``reach`` (quadratic pull toward ``center``) or ``avoid``
    (softplus push out of a radius around it). ``xy_only`` mirrors the keep-out
    cylinder: unbounded in z, so it cannot be satisfied by lifting over.
    """

    predicate: str = ""
    kind: str = "reach"  # "reach" | "avoid"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_m: float = 0.0
    weight: float = 1.0
    xy_only: bool = False


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
    state lag, so closing the gripper as soon as the reference trajectory is
    exhausted -- instead of once the end effector has actually arrived --
    can close on empty air or catch the object off-center."""

    settle_position_tol_m: float = 0.004
    settle_orientation_tol_rad: float = 0.05
    max_settle_steps: int = 150
    # Transport waits for the hand to come within this distance of each
    # waypoint before advancing; ungated, the arm lagged the 0.5s reference
    # and cut straight through a neighboring moka pot the plan went around.
    transport_max_reference_lead_m: float = 0.02
    # cart_pos = clip(pos_err / settle_pos_scale_m, -1, 1). Large enough that
    # full-speed motion doesn't yank a friction-only, off-center grasp loose.
    settle_pos_scale_m: float = 0.15
    # If position error stops improving over this many steps, the loop is
    # just hunting at a real contact boundary; breaking out early is harmless.
    stall_window_steps: int = 15
    stall_position_tol_m: float = 0.0015


# Lazy: skills/ already imports from motion/, so importing a skill's config
# class here at module level would be circular.
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
    """Top-level container. MPCCostConfig lives in types.py since it's part
    of the MPC's public result/contract surface, not just tuning; it's re-exported
    here via mpc."""

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
        # Lazy for the same circular-import reason as the factories above.
        from ..skills.close.close_skill import CloseConfig
        from ..skills.open_drawer.open_drawer_skill import DrawerConfig
        from ..skills.pick.contact_graspnet_generator import ContactGraspNetConfig
        from ..skills.pick.grasp_generator import GraspSampleConfig
        from ..skills.place.place_skill import PlaceSampleConfig
        from ..skills.transport.transport_skill import TransportConfig
        from ..skills.turn_knob.turn_knob_skill import KnobConfig

        # Read the motion block from whichever platform the scene declares.
        # Hardcoding "libero" meant a RoboCasa scene silently received every
        # default -- including a validity_resolution tuned for a tabletop, which
        # rejects paths in a kitchen because a microwave door is thin enough to
        # slip between collision samples. Falls back to the libero block so a
        # scene that declares no backend behaves exactly as before.
        platform = (scene_context or {}).get("platform", {}) or {}
        backend = str(platform.get("backend") or "libero")
        raw = (
            ((platform.get(backend) or {}).get("motion", {}) or {})
            or ((platform.get("libero") or {}).get("motion", {}) or {})
        )
        collision = CollisionConfig(**raw.get("collision", {}))
        # G(Safe(obj)) reaches the motion layer only here: the scene's SDF
        # zones become constraints the planner and MPC both enforce.
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
    """Collect the scene's outside_positive circle SDFs as keep-out zones.

    Reads the same geometry[name].sdf_geometry blocks the benchmark's SDF
    constraints come from, so planner and metric agree. Only circles are
    read; a box/polygon keep-out has no cylinder equivalent and is skipped.
    """
    if not scene_context:
        return []
    # RQ2 ablation switch: disabling this stops only the motion layer from
    # planning around the zones, isolating "STL reaches control" from "STL is monitored".
    # stl_experiment is the backend-neutral spelling; libero_experiment is kept
    # so existing LIBERO scene contexts keep working untouched.
    _plat = scene_context.get("platform") or {}
    experiment = (_plat.get("libero_experiment") or _plat.get("stl_experiment") or {})
    if not bool(experiment.get("enforce_keepout", True)):
        return []
    geometry = scene_context.get("geometry") or {}
    if not isinstance(geometry, dict):
        return []

    # The manipulated object is never its own keep-out -- some tasks name it
    # as the obstacle too, which would make the task unsolvable otherwise.
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
