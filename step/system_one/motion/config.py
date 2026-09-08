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
    # Center-grasp yaws are sampled across [best-half_range, best+half_range],
    # where "best" is computed from the object's own real geometry (an
    # oriented-bounding-box fit -- see grasp_generator._object_xy_obb_world),
    # not blindly across the full [0, pi) world-frame range: the object's
    # true shape and current orientation already tell us where the good
    # candidates are, so samples are concentrated there instead of mostly
    # landing on angles we already know are worse.
    yaw_search_half_range_rad: float = 0.7853981633974483  # pi/4, i.e. +-45 degrees
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
    # Lateral offsets (in the plane perpendicular to the approach direction)
    # tried for the standoff position if the straight-line one (directly
    # above the object, for a top-down grasp) is unreachable or in
    # collision. A fixed vertical standoff assumes the airspace directly
    # above the object is always clear -- not true once an earlier subtask
    # has moved something else into it (measured directly: after opening a
    # drawer that ends up hovering above a bowl, every rim-edge candidate at
    # every angle failed identically, because they all share the same
    # straight-up standoff direction that's now blocked). Offsetting
    # sideways lets the same IK/collision validation find a standoff that
    # approaches from a shallow angle instead, still ending at the same
    # top-down grasp pose. `(0.0, 0.0)` (straight up) is tried implicitly
    # first by the caller before any of these.
    # Multiples of `approach_standoff_m` tried for the standoff distance
    # itself (1.0 = the nominal distance), before/alongside the lateral
    # search below. An obstacle hovering over the object can block every
    # lateral direction at the *nominal* standoff height, since reaching
    # that height at all can require an arm link other than the gripper
    # (elbow/forearm) to sweep through the obstacle's footprint regardless
    # of which horizontal direction the endpoint sits in -- a taller
    # standoff (further back along the approach direction) can clear the
    # obstacle's height entirely instead. See _standoff_candidates.
    standoff_distance_multipliers: tuple[float, ...] = (1.0, 2.5, 4.0)
    # Widened from an initial (0.05, 0.08)/4-direction search: measured
    # directly against a real drawer-overhang scene, that narrower search
    # still left some candidates blocked (the drawer's overhang extends
    # ~10cm past the bowl's position in the worst direction, and 4 evenly-
    # spaced directions at those radii doesn't reliably land a sample past
    # the edge of an obstacle that size). More radii and finer angular
    # resolution costs a few extra IK/collision checks per candidate (cheap
    # relative to a full OMPL plan) but meaningfully improves the odds of
    # finding a genuinely clear approach angle.
    standoff_lateral_offsets_m: tuple[float, ...] = (0.05, 0.08, 0.12)
    standoff_lateral_directions: int = 8  # tried around a full circle at each offset magnitude
    # Rim/edge grasp strategy (bowls, mugs, cups: objects whose full
    # cross-section doesn't fit the gripper at any yaw). See
    # grasp_generator.PrivilegedTopDownGraspGenerator's module docstring.
    rim_num_angle_samples: int = 8
    rim_offset_fraction: float = 0.6
    rim_grasp_width_m: float = 0.02
    # Safety margin (m) added on top of the gripper's real measured reach
    # (see SceneQuery.palm_clearance_above_tcp_m / finger_reach_below_tcp_m)
    # when raising a grasp height to keep the palm off a short object's top,
    # or the fingertips off the floor under a thin one.
    grasp_height_margin_m: float = 0.01


@dataclass
class PlannerConfig:
    backend: str = "rrtconnect"
    # RRTConnect's max joint-space step per tree extension. The long-range
    # FREE_SPACE leg this planner is used for (home/wherever-the-previous-
    # subtask-left-the-arm -> a standoff near the next target) is typically
    # a genuinely large joint-space distance -- measured directly against
    # the open-drawer scene, ~3.5-5.6 rad across the 7 joints combined, not
    # a short local adjustment. A smaller range doesn't make that safer, it
    # makes RRTConnect's two trees need far more extensions to link up
    # exactly; OMPL gives up and returns an "Approximate solution" (treated
    # as failure, see ompl_planner.py's own documented policy on that) long
    # before any timeout is reached. Measured directly at the previous
    # default (0.08): 8/12 real plans succeeded (2 of the 4 failures were
    # exactly this approximate-solution rejection, not a genuine
    # infeasibility -- the other 2 were an unrelated goal-in-collision
    # issue with that trial's IK solution). At 0.2, the same benchmark
    # (same scene, same random IK seeds) went 12/12, with solve times
    # unchanged (~0.85s either way) -- this was never a timeout problem.
    # Collision safety isn't traded away for this: OMPL's own internal edge
    # checking (`validity_resolution`, below) still validates at a fixed
    # *fraction* of the space's extent regardless of `range` -- a longer
    # edge gets proportionally more sub-checks, not fewer -- and every
    # accepted plan is independently re-validated edge-by-edge afterward
    # (see ``MotionPlanner.plan``'s post-hoc ``check_edge`` pass) at a fixed
    # absolute resolution that doesn't depend on this value either.
    range: float = 0.2
    # Not the lever that actually fixes long-horizon failures (see `range`
    # above) -- kept generous mainly so a plan that's merely slow, rather
    # than stuck on approximate-solution rejection, still has room to finish.
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
    # A small, tolerated amount of interpenetration between the gripper's
    # FINGER geoms (fingertip body + pad -- not the palm) and the grasp
    # target, during TERMINAL_APPROACH only. Mirrors real gripper compliance:
    # a rigid open-loop check requires the fully-open gripper to already be
    # clear of the object before ever closing, but a real (or MuJoCo's own
    # soft-contact) gripper can close through a few mm of an object slightly
    # wider than its rated opening. Without this, an object only ~5mm wider
    # than the gripper's max opening was rejected outright even though nothing
    # about the approach itself was unsafe -- only the final pinch was ever
    # going to be snug. 0.0 disables this (strict, zero-tolerance) entirely.
    finger_overlap_tolerance_m: float = 0.006


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
class DrawerConfig:
    """Tuning for ``OpenDrawerSkill``/``LiberoOpenDrawerAdapter`` (slide-joint
    drawers/cabinets only -- see ``motion/articulation.py``)."""

    approach_standoff_m: float = 0.08
    # IK is not unique -- different seeds can converge to different elbow/
    # wrist configurations that reach the identical grasp pose while sweeping
    # through different parts of the scene. This many (grasp, standoff)
    # candidate pairs are tried (fresh perturbed seed each time) before
    # giving up, taking the first whose connecting segment is actually
    # collision-clear -- see OpenDrawerSkill.execute()'s docstring comment.
    pose_selection_attempts: int = 6
    # RRTConnect is stochastic and can occasionally return only an
    # "approximate" (rejected) solution or hit its timeout on a genuinely
    # feasible start/goal pair; this many attempts are retried before
    # reporting NO_FEASIBLE_CANDIDATE.
    planning_attempts: int = 3
    # Tried largest-first; the first fraction of the joint's full measured
    # travel (see articulation.measure_pull_axis_and_travel) that's both
    # IK-reachable and collision-clear wins -- mirrors the same "try a few,
    # take the best feasible" shape PickSkill/PlaceSkill already use, rather
    # than failing outright when the full travel distance isn't reachable.
    pull_fraction_fallbacks: tuple[float, ...] = (0.85, 0.7, 0.5, 0.3)
    # Fraction of the joint's full qpos range travelled (0=closed, 1=open)
    # required to call the drawer "open".
    open_threshold_frac: float = 0.8
    retreat_m: float = 0.08
    retreat_steps: int = 40


@dataclass
class KnobConfig:
    """Tuning for ``TurnKnobSkill``/``LiberoStoveAdapter`` (hinge-jointed
    knobs/dials only -- see ``motion/articulation.py``'s ``discover_knobs``).
    """

    approach_standoff_m: float = 0.06
    # A knob's full diameter is typically wider than this gripper's max
    # opening (measured ~5.5cm radius on LIBERO's flat_stove asset, i.e.
    # ~11cm across -- wider than the gripper can span). Grasping instead
    # targets a narrow chord near the rim this wide -- see
    # turn_knob_skill.py's module docstring for the chord-geometry
    # derivation. Must be comfortably under the gripper's real max opening.
    grasp_chord_width_m: float = 0.035
    # Vertical clearance (m), added along the knob's own rotation axis, above
    # the knob's own measured surface before computing the grasp/standoff
    # pose. A flat stove knob sits recessed relative to the raised burner
    # grate/control-panel housing around it -- gripping exactly at the
    # knob's own measured height put the gripper's *open* fingers into that
    # surrounding housing at most tested angles (confirmed directly against
    # the real flat_stove_1 asset: 0/8 candidate angles were collision-clear
    # at 0 margin, climbing to 7/8 at 0.04). Same "raise the grasp height to
    # clear a short/recessed feature" pattern as GraspSampleConfig.
    # grasp_height_margin_m and ContactGraspNetConfig.grasp_height_margin_m,
    # just a larger value -- this fixture's surrounding geometry is taller
    # than either of those cases' short objects.
    grasp_height_margin_m: float = 0.04
    # Unlike a drawer handle (one measured position), a knob has a full
    # circle of equally-valid grasp points -- rotational symmetry means there
    # is no single "correct" angular position to grip it at, only ones that
    # happen to be reachable/collision-clear given whatever else (a burner
    # grate, an adjacent knob) sits around this particular knob. So the
    # search has two dimensions where a drawer only has one: which angle
    # around the rim to grip, and (per angle) which IK seed. This many
    # evenly-spaced angles around the full circle are tried; each one
    # delegates its own IK-seed retries to ``_solve_ik_robust``.
    radial_angle_samples: int = 8
    # Same "try a few, take the first feasible" shape as
    # DrawerConfig.pose_selection_attempts -- now spent per angle, not
    # instead of the angle search.
    pose_selection_attempts: int = 6
    planning_attempts: int = 3
    # Tried largest-first; the first fraction of the joint's full measured
    # range (from its current qpos toward its "on" endpoint) that's both
    # IK-reachable and collision-clear wins -- mirrors
    # DrawerConfig.pull_fraction_fallbacks.
    turn_fraction_fallbacks: tuple[float, ...] = (0.75, 0.6, 0.45, 0.3)
    # Absolute qpos (radians) at/above which the knob counts as "turned on".
    # Matches LIBERO's own flat_stove.turn_on() threshold (qpos >= 0.5) --
    # see turn_knob_skill.py's module docstring for the source. Overridable
    # per-scene for a different knob/threshold.
    turn_on_qpos_threshold: float = 0.5
    retreat_m: float = 0.06
    retreat_steps: int = 40


@dataclass
class ContactGraspNetConfig:
    """Tuning for ``ContactGraspNetGenerator`` (contact_graspnet_generator.py)
    -- a learned grasp-candidate source, called in-process. When enabled,
    ``LiberoPickAdapter`` uses it *instead of* ``PrivilegedTopDownGraspGenerator``
    (not composited with it -- see ``libero_pick_adapter.py``'s comment: a
    real run showed PickSkill's OR-set cost function always preferring the
    heuristic's cheap-but-possibly-wrong candidates over the network's).
    Disabled by default so nothing changes for a machine without
    ``contact_graspnet_pytorch`` + CUDA + a real checkpoint installed (that
    package is only ever imported lazily, inside
    ``contact_graspnet_generator._load_real_estimator``, never at module
    import time)."""

    enabled: bool = False
    # Path to the Contact-GraspNet checkpoint directory (expects model.pt
    # plus its config.yaml alongside it, matching Cap-X's own server loader).
    checkpoint_dir: str = ""
    device: str = "cuda"
    forward_passes: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    # Returned candidates are capped at this many (by score) before being
    # handed to PickSkill, so a large network output doesn't blow up the
    # downstream IK/collision feasibility search.
    max_candidates: int = 32
    # Cameras rendered and fused for the point cloud (SceneQuery.
    # render_depth_and_body_ids/deproject_camera_depth) -- standard LIBERO/
    # robosuite camera names; mirrors Cap-X's own agentview + wrist-camera
    # fusion for its LIBERO integration. A camera not present in a given
    # model is skipped, not an error.
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    render_height: int = 480
    render_width: int = 480
    # Depth far-clip (m) -- drops background/sky rays (MuJoCo reports a very
    # large depth, not a sentinel, where a ray hits nothing) before they
    # pollute pc_full.
    depth_max_m: float = 3.0
    # Mirrors Cap-X's own synthetic-viewpoint retry
    # (capx/serving/launch_contact_graspnet_server.py's _do_plan_point_clouds):
    # if filter_grasps's real, tight (~0.1mm) contact-to-segment-cloud
    # distance check leaves zero survivors -- which real testing showed
    # happening for this pipeline's first (world-frame) attempt -- re-express
    # the *same* captured points in a freshly sampled synthetic camera frame
    # (looking at the object's own centroid from a random nearby pose) and
    # retry, up to this many times, before giving up on this object. No new
    # rendering needed: Contact-GraspNet's predictions are sensitive to which
    # frame its input is expressed in, so a rigid re-expression alone is
    # enough to reliably produce a different result.
    max_viewpoint_retries: int = 10
    viewpoint_xy_extent_m: float = 0.25
    viewpoint_z_min_m: float = 0.4
    viewpoint_z_max_m: float = 0.9
    # None: a fresh (non-reproducible) RNG per generate() call, matching real
    # deployment. Tests pass a fixed seed for determinism.
    viewpoint_seed: int | None = None
    # Same value and purpose as GraspSampleConfig.grasp_height_margin_m --
    # see ContactGraspNetGenerator._generate_for_object's floor/palm-
    # clearance clamp for why this generator needs it too.
    grasp_height_margin_m: float = 0.01


@dataclass
class ExecutionConfig:
    """Grasp-closing timing. The MPC tracks a reference with nonzero steady-
    state lag (see mpc.py); closing the gripper as soon as the reference
    trajectory is exhausted -- instead of once the end effector has actually
    *arrived* -- can close on empty air or catch the object off-center."""

    settle_position_tol_m: float = 0.004
    settle_orientation_tol_rad: float = 0.05
    max_settle_steps: int = 150
    # If the settle loop's position error stops improving by more than this
    # much over `stall_window_steps` consecutive steps, it's stopped
    # converging further and is instead hunting in place -- almost always
    # because the target sits right at/through a real contact boundary
    # (e.g. a grasp pose that asks the fingers to close the last couple mm
    # against the object/handle itself). Continuing to command at full
    # reactive gain toward a target it physically can't reach any closer to
    # doesn't help -- it just repeatedly pushes into and eases back off the
    # obstacle for the rest of the budget, which is exactly what looks like
    # the arm "hitting something and retrying" on video. Breaking out early
    # once stalled doesn't change *where* the arm ends up (it wasn't getting
    # any closer anyway) -- it just stops the pointless, visible struggling.
    stall_window_steps: int = 15
    stall_position_tol_m: float = 0.0015


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
    drawer: DrawerConfig = field(default_factory=DrawerConfig)
    knob: KnobConfig = field(default_factory=KnobConfig)
    contact_graspnet: ContactGraspNetConfig = field(default_factory=ContactGraspNetConfig)
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
            drawer=DrawerConfig(**raw.get("drawer", {})),
            knob=KnobConfig(**raw.get("knob", {})),
            contact_graspnet=ContactGraspNetConfig(**raw.get("contact_graspnet", {})),
            mpc=dict(raw.get("mpc", {})),
        )
