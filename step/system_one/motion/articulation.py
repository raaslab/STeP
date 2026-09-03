"""Discovery and measurement of slide-jointed (prismatic) articulated fixtures --
drawers/cabinet levels in LIBERO's kitchen scenes. Pure MuJoCo introspection, no
LIBERO Python API dependency, so it works for any slide-joint fixture the loaded
model happens to contain, not just ``WoodenCabinet`` by name.

A drawer's handle has no dedicated geom or site in LIBERO's assets (see
``wooden_cabinet.xml``). The joint's own named "region" site (e.g.
``wooden_cabinet_1_top_region``) is **not** a handle anchor either -- measured
directly against the real model, it sits *behind* the drawer's own front-face
panel (it's the interior placement-goal region LIBERO's ``In`` BDDL predicate
uses for "put the bowl inside", not a grasp point). The handle is instead
*located* geometrically in ``locate_handle_position_world``: it's whichever of
the drawer body's own geoms protrudes furthest toward the measured "open"
direction, which is what a handle physically is by construction. Everything
here (which direction "open" moves, how far, where the handle is) is measured
from real forward kinematics/geometry rather than inferred from naming
conventions or the joint's local axis sign convention -- mirroring
``scene_query.py``'s general preference for measured geometry over inferred
constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene_query import SceneQuery

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


def _handle_cluster_geom_ids(
    scene: SceneQuery, drawer: DrawerHandle, pull_axis_world: np.ndarray, cluster_tolerance_m: float
) -> list[int]:
    """Geoms making up the handle assembly: among everything owned directly
    by the drawer body (front panel, side walls, handle), the handle is
    whichever cluster protrudes furthest toward the measured "open"
    direction -- a handle is, by physical necessity, the part of a drawer
    that sticks out toward whoever pulls it. Includes every geom within
    ``cluster_tolerance_m`` of the outermost projection (the real handle is
    usually a small multi-geom assembly: a bar/knob plus one or two
    mounting studs, all at roughly the same depth)."""
    gids = scene.body_geom_ids(drawer.drawer_body, include_children=False)
    if not gids:
        raise KeyError(f"drawer body {drawer.drawer_body!r} has no geoms to locate a handle from")
    projections = {g: float(np.dot(scene.data.geom_xpos[g], pull_axis_world)) for g in gids}
    max_proj = max(projections.values())
    return [g for g, p in projections.items() if p >= max_proj - cluster_tolerance_m]


def locate_handle_position_world(
    scene: SceneQuery, drawer: DrawerHandle, pull_axis_world: np.ndarray, cluster_tolerance_m: float = 0.03
) -> np.ndarray:
    """World position of the drawer's handle -- the mean position of the
    handle-assembly geoms found by ``_handle_cluster_geom_ids``."""
    outer = _handle_cluster_geom_ids(scene, drawer, pull_axis_world, cluster_tolerance_m)
    return np.mean([scene.data.geom_xpos[g] for g in outer], axis=0)


def estimate_handle_closing_width_m(
    scene: SceneQuery,
    drawer: DrawerHandle,
    pull_axis_world: np.ndarray,
    closing_axis_world: np.ndarray,
    cluster_tolerance_m: float = 0.03,
) -> float:
    """How wide the handle assembly is along ``closing_axis_world`` -- the
    real quantity the gripper's fingers close around, needed so a candidate
    grasp's "closed" finger position can be validated (and later commanded)
    at the width the handle *actually* has there, not the joint's own
    mechanical closing limit (which corresponds to a zero gap -- fine for
    "nothing between the fingers", wrong and physically impossible for any
    real object of nonzero width; see ``CollisionChecker.check_gripper_
    closure``'s docstring for why that distinction matters). Conservative
    (never under-covers): for each handle-assembly geom, projects its own
    real half-extent onto the closing axis and takes the largest single-geom
    span, mirroring this package's general "the fitted measure must not be
    smaller than the true geometry" rule (e.g. ``attached_object.
    fit_spheres_from_geoms``).
    """
    handle_gids = _handle_cluster_geom_ids(scene, drawer, pull_axis_world, cluster_tolerance_m)
    axis = np.asarray(closing_axis_world, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    max_half_span = 0.0
    for gid in handle_gids:
        gmat = scene.data.geom_xmat[gid].reshape(3, 3)
        local_half = scene._geom_local_half_extent(scene.model.geom_type[gid], scene.model.geom_size[gid])
        world_half = np.abs(gmat) @ local_half
        half_span_along_axis = float(np.abs(world_half) @ np.abs(axis))
        max_half_span = max(max_half_span, half_span_along_axis)
    return 2.0 * max_half_span


def resolve_drawer(scene: SceneQuery, canonical_name: str) -> DrawerHandle:
    """Keyword-match a canonical name (e.g. ``"top_drawer_handle"``,
    ``"top_drawer"``, or a raw joint/body name) against the discovered
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
    """Measure (in world space, via real FK on two scratch clones) the unit
    direction the drawer moves when opening, and the physical travel distance.

    Returns ``(pull_axis_world, travel_m, qpos_closed, qpos_open)`` where
    ``qpos_closed``/``qpos_open`` are the raw joint-range endpoints (not
    assumed to be ordered low-to-high -> low-to-open; the endpoint that moves
    the drawer body *further* in world space is treated as "open").
    """
    lo, hi = scene.joint_qpos_range([drawer.joint_name])
    lo, hi = float(lo[0]), float(hi[0])

    scene_lo = scene.clone()
    scene_lo.set_qpos([drawer.joint_name], np.array([lo]))
    pos_lo, _ = scene_lo.body_pose(drawer.drawer_body)

    scene_hi = scene.clone()
    scene_hi.set_qpos([drawer.joint_name], np.array([hi]))
    pos_hi, _ = scene_hi.body_pose(drawer.drawer_body)

    # Whichever endpoint sits at qpos=0 (or closer to it) is "closed" -- a
    # closed drawer's slide joint sits at/near zero displacement from its
    # resting pose; the other endpoint is where it travels to when opened.
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
