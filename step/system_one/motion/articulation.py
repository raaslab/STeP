"""Discovery and measurement of slide-jointed (prismatic) articulated fixtures
-- drawers/cabinet levels in LIBERO's kitchen scenes. Pure MuJoCo
introspection, so it works for any slide-joint fixture, not just
WoodenCabinet by name.

A drawer's handle has no dedicated geom or site in LIBERO's assets, and the
joint's own "region" site sits behind the drawer's front panel (it's
LIBERO's In-predicate placement region, not a grasp point). The handle
is instead located geometrically in locate_handle_position_world:
whichever geom protrudes furthest toward the measured "open" direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene_query import SceneQuery, quat_to_rotmat_wxyz

_LEVEL_KEYWORDS = ("top", "upper", "middle", "bottom", "lower")
_EXCLUDE_NAME_PARTS = ("robot", "gripper", "panda")


@dataclass
class DrawerHandle:
    drawer_body: str
    joint_name: str
    level_label: str


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def _level_label_for(joint_name: str) -> str:
    lower = _normalize(joint_name)
    for kw in _LEVEL_KEYWORDS:
        if kw in lower:
            return "top" if kw == "upper" else ("bottom" if kw == "lower" else kw)
    return "drawer"


def discover_drawers(scene: SceneQuery) -> list[DrawerHandle]:
    """Every slide-joint drawer/cabinet level in the loaded model."""
    model = scene.model
    out: list[DrawerHandle] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_SLIDE:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not joint_name or any(part in _normalize(joint_name) for part in _EXCLUDE_NAME_PARTS):
            continue
        body_id = int(model.jnt_bodyid[jid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        out.append(
            DrawerHandle(
                drawer_body=body_name,
                joint_name=joint_name,
                level_label=_level_label_for(joint_name),
            )
        )
    return out


def discover_cabinet_support_bodies(scene: SceneQuery) -> list[str]:
    """The static (non-sliding) frame body of every drawer/cabinet fixture
    discovered by discover_drawers -- the fixed frame the sliding levels
    hang off of, distinct from any DrawerHandle.drawer_body. Needed so
    "place X on top of the cabinet" has a resolvable target body.

    Found by walking each drawer's parent chain up to the first ancestor
    that actually owns collision geoms, not simply the topmost ancestor --
    a world-parented root can be an empty pass-through frame, and targeting
    it would leave its own top slab unexcluded from the attached-object
    check, making a placement on the cabinet top read as colliding with itself.
    """
    return list(dict.fromkeys(discover_drawer_support_frames(scene).values()))


def discover_drawer_support_frames(scene: SceneQuery) -> dict[str, str]:
    """Like discover_cabinet_support_bodies, but keyed by each drawer's
    drawer_body name instead of deduplicated into a flat list -- lets
    callers say which drawers share a given cabinet frame body."""
    model = scene.model
    out: dict[str, str] = {}
    for drawer in discover_drawers(scene):
        body_id = int(model.body_parentid[scene.body_id(drawer.drawer_body)])
        while body_id != 0:
            if any(model.geom_bodyid[g] == body_id for g in range(model.ngeom)):
                break
            parent_id = int(model.body_parentid[body_id])
            if parent_id == body_id:
                break
            body_id = parent_id
        frame_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if frame_name:
            out[drawer.drawer_body] = frame_name
    return out


def _handle_cluster_geom_ids(
    scene: SceneQuery, drawer: DrawerHandle, pull_axis_world: np.ndarray, cluster_tolerance_m: float
) -> list[int]:
    """Geoms making up the handle assembly: whichever cluster of the drawer
    body's own geoms protrudes furthest toward the measured "open" direction
    -- the part that physically sticks out toward whoever pulls it. Includes
    every geom within cluster_tolerance_m of the outermost projection,
    since a real handle is usually a multi-geom assembly at similar depth."""
    gids = scene.body_geom_ids(drawer.drawer_body, include_children=False)
    if not gids:
        raise KeyError(f"drawer body {drawer.drawer_body!r} has no geoms to locate a handle from")
    projections = {g: float(np.dot(scene.data.geom_xpos[g], pull_axis_world)) for g in gids}
    max_proj = max(projections.values())
    return [g for g, p in projections.items() if p >= max_proj - cluster_tolerance_m]


def locate_handle_position_world(
    scene: SceneQuery, drawer: DrawerHandle, pull_axis_world: np.ndarray, cluster_tolerance_m: float = 0.03
) -> np.ndarray:
    """World position of the drawer's handle: laterally the centre of the
    handle assembly, but at the depth of its *most protruding* geom rather
    than the assembly's mean depth.

    Averaging in the mounting studs behind the bar would put the target
    behind the bar, so the fingers close past it and the palm hits the bar
    instead -- missing the handle. Aiming at the outermost depth is also
    just what "grasp the handle" means physically."""
    outer = _handle_cluster_geom_ids(scene, drawer, pull_axis_world, cluster_tolerance_m)
    positions = np.asarray([scene.data.geom_xpos[g] for g in outer], dtype=np.float64)
    center = positions.mean(axis=0)
    axis = np.asarray(pull_axis_world, dtype=np.float64)
    axis = axis / float(np.linalg.norm(axis))
    # Shift only the along-pull-axis component out to the most protruding
    # geom, leaving the lateral/vertical centre of the assembly alone.
    outermost_depth = float(np.max(positions @ axis))
    return center + (outermost_depth - float(center @ axis)) * axis


def estimate_handle_closing_width_m(
    scene: SceneQuery,
    drawer: DrawerHandle,
    pull_axis_world: np.ndarray,
    closing_axis_world: np.ndarray,
    cluster_tolerance_m: float = 0.03,
) -> float:
    """How wide the handle assembly is along closing_axis_world -- what
    the gripper's fingers actually close around, so a candidate grasp's
    "closed" position is validated against the handle's real width, not the
    joint's own (physically impossible) zero-gap closing limit. Conservative:
    takes the largest single handle-geom span projected onto the closing axis.
    """
    handle_gids = _handle_cluster_geom_ids(scene, drawer, pull_axis_world, cluster_tolerance_m)
    axis = np.asarray(closing_axis_world, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    # Measure only the most protruding geom (the bar), not the whole cluster
    # -- mounting studs/slabs would inflate the width, opening the gripper too wide and letting the pull slip.
    axis_pull = np.asarray(pull_axis_world, dtype=np.float64)
    axis_pull = axis_pull / np.linalg.norm(axis_pull)
    if handle_gids:
        outermost = max(handle_gids, key=lambda g: float(scene.data.geom_xpos[g] @ axis_pull))
        handle_gids = [outermost]
    max_half_span = 0.0
    for gid in handle_gids:
        gmat = scene.data.geom_xmat[gid].reshape(3, 3)
        local_half = scene._geom_local_half_extent(scene.model.geom_type[gid], scene.model.geom_size[gid])
        world_half = np.abs(gmat) @ local_half
        half_span_along_axis = float(np.abs(world_half) @ np.abs(axis))
        max_half_span = max(max_half_span, half_span_along_axis)
    return 2.0 * max_half_span


def resolve_drawer(scene: SceneQuery, canonical_name: str) -> DrawerHandle:
    """Keyword-match a canonical name (e.g. "top_drawer_handle",
    "top_drawer", or a raw joint/body name) against the discovered
    drawers' level labels and real names."""
    drawers = discover_drawers(scene)
    query = _normalize(canonical_name)
    for d in drawers:
        if d.level_label != "drawer" and d.level_label in query:
            return d
    for d in drawers:
        if _normalize(d.joint_name) in query or _normalize(d.drawer_body) in query:
            return d
    if len(drawers) == 1:
        return drawers[0]
    available = [f"{d.level_label}_drawer (joint={d.joint_name})" for d in drawers]
    raise KeyError(
        f"Could not resolve {canonical_name!r} to a known drawer/cabinet level. "
        f"Available: {available}"
    )


def measure_pull_axis_and_travel(
    scene: SceneQuery, drawer: DrawerHandle
) -> tuple[np.ndarray, float, float, float]:
    """Measure (via FK on two scratch clones) the unit direction the drawer
    moves when opening, and its travel distance.

    Returns (pull_axis_world, travel_m, qpos_closed, qpos_open); the raw
    joint-range endpoints aren't assumed ordered low-to-high -> closed-to-open
    -- whichever moves the drawer body further in world space is "open".
    """
    lo, hi = scene.joint_qpos_range([drawer.joint_name])
    lo, hi = float(lo[0]), float(hi[0])

    scene_lo = scene.clone()
    scene_lo.set_qpos([drawer.joint_name], np.array([lo]))
    pos_lo, _ = scene_lo.body_pose(drawer.drawer_body)

    scene_hi = scene.clone()
    scene_hi.set_qpos([drawer.joint_name], np.array([hi]))
    pos_hi, _ = scene_hi.body_pose(drawer.drawer_body)

    # Whichever endpoint sits at/near qpos=0 is "closed" -- a closed drawer's
    # slide joint sits near zero displacement from its resting pose.
    if abs(lo) <= abs(hi):
        qpos_closed, qpos_open = lo, hi
        pos_closed, pos_open = pos_lo, pos_hi
    else:
        qpos_closed, qpos_open = hi, lo
        pos_closed, pos_open = pos_hi, pos_lo

    delta = pos_open - pos_closed
    travel_m = float(np.linalg.norm(delta))
    pull_axis_world = delta / travel_m if travel_m > 1e-9 else np.array([0.0, 1.0, 0.0])
    return pull_axis_world, travel_m, qpos_closed, qpos_open


# --------------------------------------------------------------------- knobs/dials


@dataclass
class KnobHandle:
    knob_body: str
    joint_name: str
    label: str


def discover_knobs(scene: SceneQuery) -> list[KnobHandle]:
    """Every hinge-jointed knob/dial in the loaded model, discovered by
    joint type (mjJNT_HINGE) rather than name -- LIBERO's flat_stove
    knob is itself named "button" in the source asset despite being a rotary dial."""
    model = scene.model
    out: list[KnobHandle] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not joint_name or any(part in _normalize(joint_name) for part in _EXCLUDE_NAME_PARTS):
            continue
        body_id = int(model.jnt_bodyid[jid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        out.append(KnobHandle(knob_body=body_name, joint_name=joint_name, label=_normalize(body_name)))
    return out


def resolve_knob(scene: SceneQuery, canonical_name: str) -> KnobHandle:
    """Keyword-match a canonical name (e.g. "flat_stove_1", a raw joint
    name, or the knob body's own name) against the discovered knobs."""
    knobs = discover_knobs(scene)
    query = _normalize(canonical_name)
    for k in knobs:
        if _normalize(k.joint_name) in query or _normalize(k.knob_body) in query or query in _normalize(k.knob_body):
            return k
    if len(knobs) == 1:
        return knobs[0]
    available = [f"{k.label} (joint={k.joint_name})" for k in knobs]
    raise KeyError(f"Could not resolve {canonical_name!r} to a known knob/dial. Available: {available}")


def measure_knob_rotation(scene: SceneQuery, knob: KnobHandle) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Measure (via FK on two scratch clones at the joint's range endpoints)
    the fixed axis a knob's hinge rotates about, and its pivot center.

    Extracted from the measured delta rotation rather than trusted from the
    joint's local jnt_axis, since a body's mounting quat can make the
    XML-defined hinge axis not the actual world axis it spins about.

    Returns (axis_world, center_world, qpos_lo, qpos_hi); the endpoints
    aren't assumed ordered "off -> on".
    """
    lo, hi = scene.joint_qpos_range([knob.joint_name])
    lo, hi = float(lo[0]), float(hi[0])

    scene_lo = scene.clone()
    scene_lo.set_qpos([knob.joint_name], np.array([lo]))
    pos_lo, quat_lo = scene_lo.body_pose(knob.knob_body)

    scene_hi = scene.clone()
    scene_hi.set_qpos([knob.joint_name], np.array([hi]))
    pos_hi, quat_hi = scene_hi.body_pose(knob.knob_body)

    r_lo = quat_to_rotmat_wxyz(quat_lo)
    r_hi = quat_to_rotmat_wxyz(quat_hi)
    r_delta = r_hi @ r_lo.T
    axis = np.array(
        [
            r_delta[2, 1] - r_delta[1, 2],
            r_delta[0, 2] - r_delta[2, 0],
            r_delta[1, 0] - r_delta[0, 1],
        ]
    )
    norm = float(np.linalg.norm(axis))
    axis_world = axis / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    # A knob's body origin only rotates about the joint pivot (flush-mounted,
    # so pos_lo/pos_hi should already match) -- averaging is just a safety net against float drift.
    center_world = 0.5 * (pos_lo + pos_hi)
    return axis_world, center_world, lo, hi


def estimate_knob_radius_m(scene: SceneQuery, knob: KnobHandle, axis_world: np.ndarray) -> float:
    """Real radius of the knob body, measured perpendicular to its rotation
    axis from its own collision geoms' world AABB."""
    extent = scene.body_extent_world(knob.knob_body)
    if extent is None:
        raise KeyError(f"knob body {knob.knob_body!r} has no collision geoms to measure a radius from")
    axis = np.abs(axis_world / np.linalg.norm(axis_world))
    perp_weight = 1.0 - axis  # ~0 along the rotation axis itself, ~1 perpendicular to it
    return float(np.max(extent.half_extents * perp_weight))


# --------------------------------------------------------------------- doors (hinged)

# Hinge-type alone can't tell a door from a knob (both mjJNT_HINGE), so a
# door additionally needs a name hint ("microjoint") to avoid double-discovery.
_DOOR_NAME_HINTS = ("micro",)


@dataclass
class MicrowaveDoor:
    door_body: str
    joint_name: str


def discover_microwave_doors(scene: SceneQuery) -> list[MicrowaveDoor]:
    """Every hinge-jointed microwave door in the loaded model."""
    model = scene.model
    out: list[MicrowaveDoor] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not joint_name or not any(hint in _normalize(joint_name) for hint in _DOOR_NAME_HINTS):
            continue
        body_id = int(model.jnt_bodyid[jid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        out.append(MicrowaveDoor(door_body=body_name, joint_name=joint_name))
    return out


def resolve_microwave_door(scene: SceneQuery, canonical_name: str) -> MicrowaveDoor:
    """Keyword-match a canonical name (e.g. "microwave_1", a raw joint
    name, or the door body's own name) against the discovered doors."""
    doors = discover_microwave_doors(scene)
    query = _normalize(canonical_name)
    for d in doors:
        if _normalize(d.joint_name) in query or _normalize(d.door_body) in query or query in _normalize(d.door_body):
            return d
    if len(doors) == 1:
        return doors[0]
    available = [f"{d.door_body} (joint={d.joint_name})" for d in doors]
    raise KeyError(f"Could not resolve {canonical_name!r} to a known microwave door. Available: {available}")


def _door_hinge_frame(scene: SceneQuery, door: MicrowaveDoor) -> tuple[np.ndarray, np.ndarray]:
    """World-frame (pivot, axis) of a door's hinge, read directly from
    MuJoCo's FK (xanchor/xaxis), which makes no assumption that the
    pivot coincides with the body origin -- true for a flush-mounted knob,
    not necessarily for a door hinged at its edge."""
    jid = scene.joint_id(door.joint_name)
    pivot_world = np.array(scene.data.xanchor[jid], dtype=np.float64)
    axis_world = np.array(scene.data.xaxis[jid], dtype=np.float64)
    axis_world = axis_world / np.linalg.norm(axis_world)
    return pivot_world, axis_world


def _door_push_point_local(
    scene: SceneQuery, door: MicrowaveDoor, pivot_world: np.ndarray, axis_world: np.ndarray
) -> np.ndarray:
    """Body-local offset of the door's outermost geom (measured
    perpendicular to the hinge axis) -- the point a push naturally contacts
    first. Returned body-local, invariant to qpos, so it can be re-projected
    at any qpos via FK."""
    gids = scene.body_geom_ids(door.door_body, include_children=False)
    if not gids:
        raise KeyError(f"door body {door.door_body!r} has no geoms to locate a push point from")

    def _perp_dist(g: int) -> float:
        rel = scene.data.geom_xpos[g] - pivot_world
        return float(np.linalg.norm(rel - np.dot(rel, axis_world) * axis_world))

    outer_gid = max(gids, key=_perp_dist)
    body_pos, body_quat = scene.body_pose(door.door_body)
    R = quat_to_rotmat_wxyz(body_quat)
    return R.T @ (np.array(scene.data.geom_xpos[outer_gid], dtype=np.float64) - body_pos)


def measure_door_close_arc(
    scene: SceneQuery, door: MicrowaveDoor, num_waypoints: int = 10
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Measure the real swept arc a push-contact point travels from the
    current qpos to the joint's closed end, via FK on num_waypoints
    scratch clones -- not just the two range endpoints, since a hinge's
    travel is a genuine arc, unlike a drawer's straight-line slide.

    Returns (waypoints_world, qpos_current, qpos_closed, axis_world, pivot_world).
    """
    lo, hi = scene.joint_qpos_range([door.joint_name])
    lo, hi = float(lo[0]), float(hi[0])
    # Whichever endpoint sits at/near zero is "closed" -- LIBERO's
    # microwave.xml happens to define 0 as closed, but this doesn't hardcode that.
    qpos_closed = lo if abs(lo) <= abs(hi) else hi
    qpos_current = float(scene.get_qpos([door.joint_name])[0])

    pivot_world, axis_world = _door_hinge_frame(scene, door)
    contact_local = _door_push_point_local(scene, door, pivot_world, axis_world)

    waypoints = []
    for q in np.linspace(qpos_current, qpos_closed, max(2, int(num_waypoints))):
        sample = scene.clone()
        sample.set_qpos([door.joint_name], np.array([q]))
        body_pos, body_quat = sample.body_pose(door.door_body)
        R = quat_to_rotmat_wxyz(body_quat)
        waypoints.append(body_pos + R @ contact_local)
    return np.asarray(waypoints), qpos_current, qpos_closed, axis_world, pivot_world
