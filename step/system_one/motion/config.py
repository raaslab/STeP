"""Typed configuration for the motion pipeline. Loaded from a scene_context's
``platform.libero.motion`` block (see ``configs/libero_template.json``'s ``motion``
section) so every tunable below has a named, documented home instead of being a
hidden magic number in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
    # Extra height the pick adapter raises the grasped object *after* the
    # test lift above has already confirmed the grip holds. Purely to leave
    # the next skill a plannable start pose: the test lift is deliberately
    # short (it only has to prove the grasp survives being loaded), which
    # leaves the arm low and table-adjacent, and planning a carry from there
    # with the object attached leaves RRTConnect almost no free corridor.
    # Measured on the real libero_object cream-cheese task: transport from
    # the post-test-lift pose returned NO_PATH_FOUND on all 8 candidate goals
    # after 85.8s; with this lift applied first the same transport found a
    # direct path in 1.4s. Set to 0 to disable.
    post_grasp_carry_lift_m: float = 0.10
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
    # obstacle's height entirely instead. See standoff_search.standoff_candidates.
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
    # Edge-pinch grasp strategy (plates, saucers, lids: flat/thin objects
    # whose *height*, not just cross-section, makes the rim-edge wall-
    # straddle strategy above physically meaningless -- there's no hollow
    # wall to straddle at any radius, just uniformly solid material all the
    # way to the edge. See grasp_generator._edge_pinch_candidates).
    edge_pinch_num_angle_samples: int = 8
    edge_pinch_offset_fraction: float = 0.92
    edge_pinch_width_m: float = 0.02
    # Randomized-seed IK retries for the standoff/grasp solves below (same
    # pattern as PlaceSampleConfig.pose_selection_attempts): a single fixed
    # seed (the arm's current resting pose) is a good seed for a top-down
    # grasp, since that's kinematically close to how the arm usually rests,
    # but not always for others -- confirmed directly via a real rollout
    # that this alone let a rim-edge candidate for a real plate object
    # succeed where it hadn't before.
    pose_selection_attempts: int = 6
    # Safety margin (m) added on top of the gripper's real measured reach
    # (see SceneQuery.palm_clearance_above_tcp_m / finger_reach_below_tcp_m)
    # when raising a grasp height to keep the palm off a short object's top,
    # or the fingertips off the floor under a thin one.
    grasp_height_margin_m: float = 0.01
    # RRTConnect is stochastic (see TransportConfig's own comment on this) --
    # PickSkill's long-range approach-to-standoff plan had no retry loop at
    # all until a real rollout showed several otherwise-valid candidates
    # (passed IK/collision) rejected outright on a single NO_PATH_FOUND,
    # for a scene no more cluttered than ones that succeed on a later
    # attempt of the same plan.
    planning_attempts: int = 6


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
    # Time budget for OMPL's post-solve path shortening. Called with no
    # argument (an earlier version's behavior), simplifySolution() uses a
    # generous internal default and routinely spent 2.5-6.7s per plan call --
    # measured from real rollout logs, often longer than the search that
    # produced the path. It's an optimization pass, not a correctness one:
    # the simplified path is independently re-validated edge-by-edge below
    # regardless, so a short budget trades a slightly less-shortened path
    # (the same call still collapsed e.g. 14 states to 5) for seconds back on
    # every single planning call in a rollout.
    simplify_time_sec: float = 0.25


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
    # LIBERO-Constrained keep-out zones: the STL task spec's G(Safe(obj))
    # obligations, made enforceable at plan/control time. Empty by default, so
    # an unconstrained run behaves exactly as before. See KeepoutZone.
    keepout_zones: list["KeepoutZone"] = field(default_factory=list)


@dataclass
class KeepoutZone:
    """An infinite vertical cylinder the robot must stay out of.

    This is the *same* geometry the benchmark scores against: a
    ``CircleSDF(convention="outside_positive")`` whose robustness is
    ``norm(p_xy - center_xy) - radius``, evaluated on the end-effector and on
    the carried object (see ``System1Executor._sample_constraint_margins``).
    Enforcement and measurement deliberately share one definition -- a keep-out
    the planner interprets differently from the metric would make the reported
    violation rate a property of the mismatch rather than of the policy.

    2D on purpose: the constraint is unbounded in z, so it cannot be satisfied
    by lifting over the obstacle, only by routing around it in xy. That is what
    makes it a non-trivial planning requirement rather than one the existing
    carry height already satisfies for free.
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
class PlaceSampleConfig:
    # Candidate placement points are sampled within a shrunk copy of the goal
    # container's interior footprint (center + a ring), mirroring how the
    # grasp generator shrinks inward from an object's rim for the same
    # "don't aim at the wall" reason.
    interior_shrink_fraction: float = 0.35
    num_ring_samples: int = 4
    # place_surface="top" candidate search (a large open support like the
    # table -- see [[move_bowl_via_table]]/libero_scene.infer_table_support_
    # entity) -- deliberately wider/denser than the container-interior
    # defaults above. A container's interior is small and empty by
    # construction (nothing else is normally *inside* a drawer/bowl yet), so
    # one shrunk ring is plenty; the table is large, shared across every
    # "move this out of the way" detour in a plan, and has a real fixed
    # obstacle sitting on it (the cabinet itself) -- confirmed directly via
    # a real multi-object rollout (bowl, then plate, then wine bottle all
    # needing their own clear spot on the table in the same run): the single
    # 4-point ring above put one candidate almost exactly on top of the
    # cabinet's own footprint, and left too few alternatives once an earlier
    # subtask's object already occupied the center. Multiple radii and more
    # angles give a real chance of landing somewhere clear of both the
    # cabinet and whatever an earlier subtask already parked there.
    top_surface_shrink_fractions: tuple[float, ...] = (0.2, 0.35, 0.5)
    top_surface_num_ring_samples: int = 10
    hover_clearance_m: float = 0.06  # above the container's rim before descending
    drop_clearance_m: float = 0.02  # above the container's floor (or the next object already in it)
    # How far the hand lifts away after opening, *before* the release is
    # verified. _is_grasped() is a contact test, so a genuinely released
    # object still reads as grasped while it rests against the open fingers --
    # the normal case when placing into an already-occupied container, where
    # the object settles higher than the empty-container floor the place pose
    # aimed at. Confirmed on the real two-object libero_10 basket task: the
    # second place failed with "gripper opened but object is still reported
    # grasped" despite placing correctly.
    release_withdraw_m: float = 0.06
    # IK is not unique -- same reasoning as every other skill in this
    # package (OpenDrawerSkill, TurnKnobSkill, TransportSkill): a single
    # DLS solve from one seed can land in a bad basin even when a feasible
    # solution exists nearby. PlaceSkill's own hover/place IK solves were
    # the one holdout still using a single, non-robust attempt -- confirmed
    # via a real rollout (placing into an opened drawer) that all 5
    # placement candidates failed hover IK with large (0.18-0.64m) position
    # errors from the single seed tried, despite the target itself being a
    # perfectly reasonable, reachable Cartesian pose.
    pose_selection_attempts: int = 6


@dataclass
class TransportConfig:
    """Tuning for ``TransportSkill``/``LiberoTransportAdapter`` -- moves the
    currently-attached object to a hover point above the goal container/
    support, the same target ``PlaceSkill``'s own internal hover leg plans
    to (see ``place_skill.py``'s module docstring), just exposed as its own
    subtask instead of only ever running as a step inside ``place``. Exists
    because the *legacy* stage-machine transport backend has no notion of
    the carried object's own footprint at all (a single fixed carry height,
    no attached-object collision checking) -- confirmed via a real rollout
    that it drove a grasped bowl straight into a cabinet's wall reaching
    for a goal on top of it, dropping the grasp.
    """

    # IK is not unique -- see OpenDrawerSkill/PlaceSkill's own comments on
    # the same point. This many attempts (a mix of near-start_q perturbation
    # and, for the back half, full-joint-range random seeds -- see
    # TransportSkill.execute's own comment on why both are needed) are
    # tried before giving up on one hover-pose candidate.
    pose_selection_attempts: int = 8
    # RRTConnect is stochastic; retried more times than most other
    # OMPL-backed skills in this package (their own default is 3) --
    # confirmed via real rollouts that transporting a carried object into
    # an already-open drawer (a tight, recessed target, harder than most
    # other transport goals) needed more than 3 attempts to reliably
    # succeed.
    planning_attempts: int = 8


@dataclass
class ReplanConfig:
    tracking_error_m: float = 0.03
    collision_margin_m: float = 0.01


@dataclass
class DrawerConfig:
    """Tuning for ``OpenDrawerSkill``/``LiberoOpenDrawerAdapter`` (slide-joint
    drawers/cabinets only -- see ``motion/articulation.py``)."""

    approach_standoff_m: float = 0.08
    # Downward tilts (radians) tried for the handle approach, in order. 0 is
    # the straight-along-the-pull-axis approach, which is the cleanest pull
    # when the space in front of the drawer is clear; the tilted ones bring
    # the hand in from above instead, which is the only way past clutter
    # sitting in that space (measured on the middle drawer: every one of 44
    # reachable straight-on standoffs had its descent blocked by the tabletop
    # or by objects parked in front of the cabinet). Opening a drawer does
    # not need a clean grasp -- an open gripper straddling the handle pulls
    # it fine -- so a tilted, hook-style approach is a perfectly good way to
    # do it, not a compromise.
    approach_tilts_rad: tuple[float, ...] = (0.0, 0.35, 0.61, 0.87)  # 0, 20, 35, 50 degrees
    # Wrist roll about the approach axis. The two values are two genuinely
    # fingers sit above/below a horizontal handle bar and close on it -- the
    # plain pinch, and the only orientation kept. Rolls of pi/2 and +/-45
    # were both tried against the middle drawer and both measured worse: at
    # pi/2 the fingers lie *along* the bar so the gripper cannot close at all
    # (a hook that slipped after ~7% of travel), and at +/-45 it closes
    # properly (0.0744m, inside the 8cm opening) but still cannot reach past
    # the tabletop, managing 1% while brushing all three drawers. The
    # blocker was never the wrist angle; it is the clutter in front of the
    # cabinet, so that is what gets cleared instead.
    # which keeps the palm and forearm out of the thin band in front of the
    # cabinet entirely. Measured on the middle drawer, and confirmed visually
    # by the user against rendered poses: at 35-50 degrees of tilt with a
    # pi/2 roll the arm reaches the handle with *nothing* else in contact --
    # no forearm in the bowl, no palm in the plate -- where every roll-0
    # orientation is blocked by one or the other.
    approach_rolls_rad: tuple[float, ...] = (0.0,)  # straight pinch only
    # Offsets applied to the handle target, as (along-pull-axis, vertical)
    # metres. Aiming dead centre drives the fingers into the bar itself
    # (measured: 1.8cm of interpenetration at roll pi/2, the *only* thing
    # left blocking an otherwise clear pose). Opening a drawer does not need
    # the bar centred between the pads -- it needs the fingers somewhere they
    # catch it when pulled -- so a few offsets are tried around it.
    grasp_offsets_m: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.02),
        (-0.02, 0.02),
        (-0.03, 0.02),
        (-0.04, 0.02),
        (-0.03, 0.0),
        (0.0, -0.02),
        (-0.015, 0.0),
    )
    # How close the hand has to end up to the handle for the pull to count as
    # "the handle is between the fingers". Deliberately looser than a grasp
    # test: the fingers are ~8cm apart when open, so the handle can sit well
    # off the exact target and still be hooked. Tight enough, though, to
    # still reject the failure this guards against -- an approach that ended
    # 0.10m away, resting on the cabinet face near the *next* drawer's
    # handle, which pulled that one open instead.
    handle_capture_radius_m: float = 0.045
    # Clearance left on each side of the handle when approaching with
    # pre-narrowed fingers (see OpenDrawerSkill's approach_finger_qpos).
    # Enough that the handle passes cleanly between the fingers without
    # needing the full open span, which is what does not fit the gap above a
    # lower drawer's handle.
    approach_finger_clearance_m: float = 0.006
    # IK is not unique -- different seeds can converge to different elbow/
    # wrist configurations that reach the identical grasp pose while sweeping
    # through different parts of the scene. This many (grasp, standoff)
    # candidate pairs are tried (fresh perturbed seed each time) before
    # giving up, taking the first whose connecting segment is actually
    # collision-clear -- see OpenDrawerSkill.execute()'s docstring comment.
    pose_selection_attempts: int = 6
    # Same standoff-candidate search PickSkill uses (see
    # standoff_search.standoff_candidates) -- the nominal straight-back
    # standoff (multiplier 1.0, no lateral offset) is always tried first, so
    # an uncluttered scene still gets the simple, direct approach; these
    # only matter once that's blocked. Confirmed necessary directly: a real
    # cluttered scene (objects left on the table between the robot and a
    # cabinet) blocked the single fixed standoff position at *every* IK seed
    # tried, for every drawer level in that cabinet, since a fixed-direction
    # standoff has no way to route the approach past something sitting
    # directly in that one path.
    standoff_distance_multipliers: tuple[float, ...] = (1.0, 2.5, 4.0)
    standoff_lateral_offsets_m: tuple[float, ...] = (0.05, 0.08, 0.12)
    standoff_lateral_directions: int = 8
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
    # After retreating, servo back to the robot's own defined home/reset
    # joint configuration (``robot.init_qpos``) before ending the subtask.
    # The handle-approach IK solution that got the drawer open is often a
    # genuinely awkward configuration (elbow twisted, wrist near a joint
    # limit) -- confirmed directly: left as-is, it measurably reduced the
    # arm's reachable workspace for whatever subtask ran next (a
    # following pick's own IK solve had far less room to maneuver, and
    # visibly moved stiffly). A dense joint-space interpolation (not a
    # single Cartesian jump) is tracked here, same reasoning as the
    # pull/turn legs elsewhere in this package: OSC's own IK is not
    # unique, so commanding the home pose's Cartesian target directly can
    # still converge to a *different* elbow config than true home: only
    # incremental, densely-sampled joint waypoints reliably guide the
    # actual joint configuration back to the target, not just the
    # end-effector's Cartesian position.
    return_home_steps: int = 60


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
    # Vertical offsets (m), added along the knob's own rotation axis above
    # the knob body's own origin, tried (together with the angle search
    # below) when picking the grasp/standoff pose. A flat stove knob's own
    # collision geometry is a real, non-cylindrical mesh (a raised
    # pointer/dial shape on a wider base, confirmed directly against the
    # real flat_stove_1 asset via its geoms) -- so unlike a drawer handle's
    # simple measured axis, there is no single "right" height to grip it at
    # a fixed radius; the true solid cross-section at that radius varies
    # with height, and only a real contact check (not just "not colliding",
    # see turn_knob_skill.py's ``_verify_grip_contact``) can tell which
    # height actually has material there. An earlier version picked a single
    # large fixed margin (0.04) reasoned from the body's overall AABB alone,
    # assuming raising the grip height would simply clear the surrounding
    # burner housing -- it did (collision-clear at 7/8 angles), but a direct
    # contact check showed *zero* of those candidates actually touched real
    # knob material: the mesh narrows sharply above its base, so the
    # gripper closed on empty air every time. Real closed-fingertip contact
    # was only ever found at small margins close to the knob's own base
    # (0.0-0.02); this list is tried smallest-real-contact-distance-first
    # (largest measured overlap = most robust grip), largest margin last.
    grasp_height_margins_m: tuple[float, ...] = (0.01, 0.0, 0.02, 0.015, 0.03, 0.005)
    # Unlike a drawer handle (one measured position), a knob has a full
    # circle of equally-valid grasp points -- rotational symmetry means there
    # is no single "correct" angular position to grip it at, only ones that
    # happen to be reachable/collision-clear given whatever else (a burner
    # grate, an adjacent knob) sits around this particular knob. So the
    # search has two dimensions where a drawer only has one: which angle
    # around the rim to grip, and (per angle) which IK seed. This many
    # evenly-spaced angles around the full circle are tried; each one
    # delegates its own IK-seed retries to ``_solve_ik_robust``.
    radial_angle_samples: int = 16
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
    # How far past `turn_on_qpos_threshold` to target turning the knob to
    # (not the joint's full range end). Confirmed directly against the real
    # flat_stove_1 asset (full range [-0.005, 2.1] rad, threshold 0.5):
    # targeting the far endpoint (2.1) asks the wrist to sweep ~120 degrees
    # while rigidly gripping a point near table height, dragging the arm
    # through the surrounding burner grate/table at every tested turn
    # fraction. A real stove only needs turning to "click on", not cranking
    # to the mechanical stop -- targeting just past the threshold keeps the
    # turn to the smallest rotation that reliably registers as on.
    turn_on_safety_margin_rad: float = 0.6
    retreat_m: float = 0.06
    retreat_steps: int = 40


@dataclass
class CloseConfig:
    """Tuning for ``CloseSkill``/``LiberoCloseAdapter`` -- closes a drawer
    (slide joint) or a microwave door (hinge joint) by pushing it shut with
    a closed gripper, never grasping it. Pushing has none of
    ``DrawerConfig``'s grasp-precision machinery (offsets, tilts, per-angle
    finger width) because a closed fist just needs to *contact* the fixture
    somewhere and keep moving through it -- there is no handle to miss."""

    approach_standoff_m: float = 0.08
    # Higher than every other skill's own pose_selection_attempts (6) --
    # measured necessary specifically for the microwave door: its hinge axis
    # is vertical, so every approach reaches sideways across the robot's own
    # base/pedestal instead of straight down onto a tabletop object, and a
    # DLS solve from a perturbed seed converges to a self-colliding elbow
    # configuration (forearm swept behind the pedestal) far more often than
    # not. Measured directly: 6 restarts found a real descent-clear
    # candidate ~1/75 tries; 40 found 5/75, with every candidate's IK
    # converging at all (vs. some failing outright at 6). Drawer closing
    # doesn't need this itself (reaches forward across the tabletop, not
    # sideways past the base) but shares this config field for simplicity.
    pose_selection_attempts: int = 24
    standoff_distance_multipliers: tuple[float, ...] = (1.0, 2.5, 4.0)
    standoff_lateral_offsets_m: tuple[float, ...] = (0.05, 0.08, 0.12)
    standoff_lateral_directions: int = 8
    # Drawer branch only (see DrawerConfig.approach_tilts_rad, same
    # reasoning): a purely horizontal approach sweeps the forearm through
    # whatever sits on the table in front of the cabinet at handle height --
    # confirmed necessary, not just ported for parity, by a real close
    # attempt on libero_10's own bottom-drawer task, whose scene has a wine
    # rack sitting right next to the cabinet: every one of 31 reachable
    # straight-on standoffs had its descent blocked (table or the wine rack)
    # until this tilt search was added. Not used by the door branch, whose
    # approach direction already comes from the arc's own tangent.
    approach_tilts_rad: tuple[float, ...] = (0.0, 0.35, 0.61, 0.87)
    # Drawer branch only -- mirrors DrawerConfig.pull_fraction_fallbacks
    # (tried largest/most-closed first): pushing all the way to the joint's
    # exact closed end can bring the palm into contact with whatever sits
    # flush around the drawer opening once it's fully shut -- confirmed
    # directly on a real stacked-cabinet scene, where fully closing the
    # bottom drawer swept the palm into the middle drawer's own body sitting
    # just above it. A push that only gets most of the way still counts as
    # closed (see close_threshold_frac) and avoids the tightest final
    # clearance entirely instead of trying to thread it.
    close_fraction_fallbacks: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.75, 0.6)
    # ``_execute_drawer_linear`` only: a fixed orientation (built from three
    # independent rotations applied to the arm's home pose -- yaw about
    # world +Z, then roll about the resulting local +Z/approach axis, then
    # tilt about the resulting local +X, perpendicular to both; these three,
    # in this order, span the full orientation space) plus a fixed height
    # offset relative to the handle's own z. All four pinned rather than
    # swept, tuned by hand against real physics rollouts on a real scene
    # where an object placed inside the drawer sits at almost exactly handle
    # height -- not the static collision check, which is measurably too
    # conservative for this leg (see close_fraction_fallbacks's own
    # comment). Progression (open_frac after the push; lower = more closed):
    # yaw=0/roll=0/tilt=0/height=+2cm -> 0.456; yaw=90 -> 0.420; height=0cm
    # (right at the handle, pushing lower than that plateaus/doesn't help
    # further) -> 0.341; roll=270 on top of that -> 0.147 (first time under
    # this struct's own close_threshold_frac); tilt=-30 with roll back at 0
    # -> **0.049** -- the first value to cross LIBERO's *own* success bar
    # (WhiteCabinet.is_close needs qpos > 0.0, i.e. open_frac < 0.0588 for
    # this cabinet) rather than just this codebase's own more lenient
    # threshold. scripts/tune_close_drawer.py exposes all four as CLI flags
    # (--yaw/--roll/--tilt/--height) for exploring further without editing
    # this file at all.
    linear_yaws_deg: tuple[float, ...] = (90.0,)
    linear_roll_deg: float = 0.0
    linear_tilt_deg: float = -30.0
    linear_height_offsets_m: tuple[float, ...] = (0.0,)
    planning_attempts: int = 3
    # Number of Cartesian waypoints FK-sampled along a microwave door's
    # swept closing arc (see articulation.measure_door_close_arc) -- each is
    # IK-solved in sequence, warm-started from the previous one, to build the
    # joint-space reference the MPC tracks. Irrelevant for a drawer, whose
    # closing path is a straight line (2 endpoints, IK-solved directly).
    door_arc_waypoints: int = 10
    # Fraction of the joint's full qpos range still open, required to call
    # the fixture "closed" -- the mirror image of
    # DrawerConfig.open_threshold_frac, just checked in the other direction
    # (<= this counts as closed, vs. >= open_threshold_frac counts as open).
    # Looser than 0.0 because a real push rarely drives the joint to its
    # exact mechanical zero.
    close_threshold_frac: float = 0.15
    retreat_m: float = 0.06
    retreat_steps: int = 40
    return_home_steps: int = 60


@dataclass
class ContactGraspNetConfig:
    """Tuning for ``ContactGraspNetGenerator`` (contact_graspnet_generator.py)
    -- a learned grasp-candidate source, called in-process. When enabled,
    ``LiberoPickAdapter`` uses it *instead of* ``PrivilegedTopDownGraspGenerator``
    (not composited with it -- see ``libero_pick_adapter.py``'s comment: a
    real run showed PickSkill's OR-set cost function always preferring the
    heuristic's cheap-but-possibly-wrong candidates over the network's).
    Disabled by default so nothing changes for a machine without
    ``contact_graspnet_pytorch`` or a real checkpoint installed (that
    package is only ever imported lazily, inside
    ``contact_graspnet_generator._load_real_estimator``, never at module
    import time).

    CUDA is *not* required: ``contact_graspnet_pytorch`` ships no compiled CUDA
    extensions (its PointNet++ ops are pure PyTorch), so CPU inference works and
    measures ~1.6s/call -- verified end-to-end on an Apple-Silicon Mac, where a
    full LIBERO pick+transport+place rollout using it took ~70s."""

    enabled: bool = False
    # Path to the Contact-GraspNet checkpoint directory (expects model.pt
    # plus its config.yaml alongside it, matching Cap-X's own server loader).
    # Machine-specific, so it must never be hardcoded to one person's checkout:
    # the CONTACT_GRASPNET_CHECKPOINT_DIR environment variable overrides
    # whatever is configured here (see _load_real_estimator).
    checkpoint_dir: str = ""
    # "auto" resolves to cuda when available, else cpu. An explicit "cuda"/"cpu"/
    # "mps" is honored as-is.
    device: str = "auto"
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
    # cart_pos = clip(pos_err / settle_pos_scale_m, -1, 1): with the old
    # hardcoded 0.03, any residual gap over 3cm saturates to a sustained
    # max-speed command for every step until it drops below 3cm -- fine for
    # the few-mm steady-state MPC lag this loop was designed for, but the
    # actual residual gap handed to place's settle phase (whatever's left
    # after the reference-tracked descent runs out) isn't reliably that
    # small -- measured up to ~15cm in a real libero_10 moka-pot run.
    # Sustained full-speed motion over that large a gap, while still
    # carrying a friction-only-held, off-center (handle) grasp, was enough
    # to physically separate the object from the gripper mid-air with zero
    # other contact recorded -- not a weak grasp, an abrupt, uncushioned
    # yank. A larger divisor here only widens the proportional band (it
    # still saturates for genuinely large gaps, just fewer of them, and
    # ramps down earlier/more gradually approaching the target); it doesn't
    # change the converged endpoint, and max_settle_steps=150 leaves ample
    # room to cover a larger gap at a gentler pace instead.
    settle_pos_scale_m: float = 0.15
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
    transport: TransportConfig = field(default_factory=TransportConfig)
    replan: ReplanConfig = field(default_factory=ReplanConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    drawer: DrawerConfig = field(default_factory=DrawerConfig)
    knob: KnobConfig = field(default_factory=KnobConfig)
    close: CloseConfig = field(default_factory=CloseConfig)
    contact_graspnet: ContactGraspNetConfig = field(default_factory=ContactGraspNetConfig)
    mpc: dict[str, float] = field(default_factory=dict)  # raw dict -> MPCCostConfig(**mpc)

    @staticmethod
    def from_scene_context(scene_context: dict[str, Any] | None) -> "MotionConfig":
        raw = (
            ((scene_context or {}).get("platform", {}) or {})
            .get("libero", {}) or {}
        ).get("motion", {}) or {}
        collision = CollisionConfig(**raw.get("collision", {}))
        # G(Safe(obj)) reaches the motion layer here, and nowhere else. The
        # scene's outside_positive circle SDFs (written by
        # libero_scene.apply_libero_experiment_geometry from the task's
        # avoid_region) become keep-out zones the OMPL state-validity check and
        # the live MPC both enforce. Without this the formula is monitored but
        # never planned for, which makes a 6cm constraint and a 0cm constraint
        # produce identical trajectories.
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
    ``geometry_sdf.parse_sdf_constraints`` turns into the SDF constraints the
    benchmark scores against, so the planner and the metric are driven from one
    source. Parsed off the raw dict rather than by importing that parser: this
    module is imported by every motion skill and must not pull in the parent
    ``system_one`` package.

    Only circles are read. A box/polygon keep-out is expressible in the scene
    format but has no cylinder equivalent here, so it is skipped rather than
    silently approximated by a circle of the wrong size.
    """
    if not scene_context:
        return []
    # The RQ2 ablation switch. Setting this false leaves the scene's SDF
    # constraints (and therefore every scored signal -- safe_ee, safe_object,
    # the violation rate) completely untouched, and only stops the motion layer
    # from planning around them. That is what isolates "STL reaches control"
    # from "STL is monitored": both arms of the comparison see an identical
    # scene and are measured by an identical metric.
    experiment = ((scene_context.get("platform") or {}).get("libero_experiment") or {})
    if not bool(experiment.get("enforce_keepout", True)):
        return []
    geometry = scene_context.get("geometry") or {}
    if not isinstance(geometry, dict):
        return []

    # The manipulated object is never its own keep-out: two libero_10 tasks
    # name the object being picked as the obstacle, which would make the task
    # unsolvable by construction rather than hard.
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
