"""LIBERO scene-context utilities.

This module keeps all LIBERO-specific scene introspection and experiment
augmentation out of the generic STL/MPC executor.  The goal is to let the same
System-2 -> STL compiler -> System-1 pipeline run on robosuite, real hardware,
and LIBERO by only changing the backend adapter and scene_context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import copy
import json
import os
import re

import sys

import numpy as np


def _import_libero_benchmark():
    """Return ``(benchmark_module, get_libero_path)``.

    Handles both a properly pip/uv-installed ``libero`` (single-level --
    ``from libero import benchmark``, e.g. Cap-X's own ``.venv-libero``,
    which is how a "normal" LIBERO checkout is actually packaged) and a
    double-nested vendored checkout reached via ``_ensure_libero_on_path``
    (``from libero.libero import benchmark`` -- the historical vlm-stl
    layout, where the repo root itself is added to ``sys.path`` instead of
    the package root).
    """
    try:
        from libero import benchmark, get_libero_path
    except ImportError:
        from libero.libero import benchmark, get_libero_path  # type: ignore
    return benchmark, get_libero_path


def _import_libero_env():
    """Return the ``OffScreenRenderEnv`` class, single- or double-nested (see
    ``_import_libero_benchmark``)."""
    try:
        from libero.envs import OffScreenRenderEnv
    except ImportError:
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    return OffScreenRenderEnv


def _ensure_libero_on_path() -> None:
    """Add the LIBERO source tree to sys.path if libero is not already importable.

    The LIBERO repo is checked in at <workspace>/LIBERO/libero, so the package
    root (the directory that contains the top-level ``libero/`` package) is the
    ``LIBERO/libero`` subdirectory relative to this file's repository root.
    """
    try:
        import libero  # noqa: F401
        return
    except ImportError:
        pass
    import importlib.util
    here = Path(__file__).resolve()
    # Walk up to find the repo root (contains LIBERO/ sibling directory).
    # The pip package root is LIBERO/ itself; add that so that
    # `import libero` resolves to LIBERO/libero/ and
    # `from libero.libero import …` resolves to LIBERO/libero/libero/.
    for parent in here.parents:
        candidate = parent / "LIBERO"
        if (candidate / "libero" / "libero").is_dir():
            path_str = str(candidate)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return


_MOVABLE_HINTS = (
    "block",
    "cube",
    "mug",
    "cup",
    "bowl",
    "plate",
    "alphabet_soup",
    "cream_cheese",
    "tomato_sauce",
    "butter",
    "basket",
    "can",
    "box",
    "object",
)
_GOAL_OR_SUPPORT_HINTS = (
    "tray",
    "basket",
    "bowl",
    "plate",
    "stove",
    "cabinet",
    "drawer",
    "container",
    "region",
    "target",
    "goal",
)
_ROBOT_KEY_PARTS = (
    "robot",
    "eef",
    "gripper",
    "joint",
    "qpos",
    "qvel",
    "image",
    "camera",
)


@dataclass(frozen=True)
class LiberoTaskInfo:
    suite: str | None
    task_id: int | None
    task_name: str | None
    language: str | None
    bddl_file: str | None
    init_state_id: int


def _as_float_list(value: Any, n: int | None = None) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if n is not None and arr.size < n:
        return None
    if n is not None:
        arr = arr[:n]
    if not np.all(np.isfinite(arr)):
        return None
    return [float(x) for x in arr.tolist()]


def normalize_entity_name(name: str) -> str:
    """Convert LIBERO/obs names into stable scene-context entity names."""
    s = str(name).strip()
    s = s.replace("/", "_").replace(".", "_").replace("-", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "entity"


def _strip_pose_suffix(key: str) -> str | None:
    s = normalize_entity_name(key)
    for suffix in ("_pos", "_xyz", "_position"):
        if s.endswith(suffix):
            base = s[: -len(suffix)]
            if base and not any(part in base for part in _ROBOT_KEY_PARTS):
                return base
    return None


def role_for_object_name(name: str) -> str:
    """Best-effort role assignment for LIBERO object names.

    Roles are used only as hints for System 2 and target resolution.  The
    runtime remains object-agnostic: any entity can still be referred to by
    name if its pose is available in live_scene_state/geometry.
    """
    lower = normalize_entity_name(name)
    if "obstacle" in lower or "keepout" in lower:
        return "obstacle"
    if any(h in lower for h in _GOAL_OR_SUPPORT_HINTS) and not any(
        h in lower for h in ("block", "cube")
    ):
        # Containers / fixtures can be placement references.  Some can also be
        # movable in LIBERO, but marking them as support_object prevents the
        # planner from moving them unless explicitly instructed.
        return "support_object"
    if any(h in lower for h in _MOVABLE_HINTS):
        return "manipulated_object"
    return "support_object"


def infer_objects_from_obs(obs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract object poses from a LIBERO observation dict.

    LIBERO/robosuite observations commonly include keys like
    ``<object>_pos`` and ``<object>_quat``.  This parser deliberately ignores
    robot/eef/camera keys and returns a normalized object map.
    """
    objects: dict[str, dict[str, Any]] = {}
    for key, value in (obs or {}).items():
        base = _strip_pose_suffix(key)
        if base is None:
            continue
        xyz = _as_float_list(value, 3)
        if xyz is None:
            continue
        objects.setdefault(base, {})
        objects[base].update({"xyz": xyz, "xy": xyz[:2], "z": xyz[2], "source": "libero_obs"})

    for key, value in (obs or {}).items():
        skey = normalize_entity_name(key)
        if skey.endswith("_quat"):
            base = skey[: -len("_quat")]
            quat = _as_float_list(value, 4)
            if quat is not None and base in objects:
                objects[base]["quat"] = quat
    return objects


def infer_objects_from_mujoco(env: Any) -> dict[str, dict[str, Any]]:
    """Best-effort fallback when object poses are absent from obs.

    Uses MuJoCo body names while filtering out robot/world/table bodies.  This
    is intentionally permissive because LIBERO tasks use many object names.
    """
    out: dict[str, dict[str, Any]] = {}
    sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return out
    try:
        model = sim.model
        data = sim.data
        nbody = int(model.nbody)
        for i in range(nbody):
            try:
                raw_name = model.body_id2name(i)
            except Exception:
                raw_name = None
            if not raw_name:
                continue
            name = normalize_entity_name(raw_name)
            if not name or any(part in name for part in ("world", "table", "robot", "panda", "gripper", "mount", "base")):
                continue
            pos = _as_float_list(data.body_xpos[i], 3)
            if pos is None:
                continue
            out.setdefault(name, {"xyz": pos, "xy": pos[:2], "z": pos[2], "source": "libero_mujoco_body"})
    except Exception:
        return out
    return out


def infer_drawer_fixtures(env: Any) -> list[dict[str, Any]]:
    """Discover slide-jointed drawer/cabinet fixtures in the live LIBERO env
    (e.g. ``wooden_cabinet_1``'s top/middle/bottom drawers) and return them as
    scene-context entities. Unlike movable objects, these never appear in
    ``infer_objects_from_obs`` -- LIBERO only emits ``<name>_pos``/``_quat``
    obs keys for ``env.objects_dict``, never for ``env.fixtures_dict`` -- so
    without this, System 2 has no real entity to reference and can only
    hallucinate a name like ``top_drawer_handle`` for an "open the drawer"
    instruction. Lazily imports the motion package's MuJoCo introspection
    (pure model discovery, no LIBERO API needed) to avoid a hard dependency
    for callers that never touch LIBERO/mujoco.
    """
    sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return []
    try:
        from .motion.articulation import discover_drawer_support_frames, discover_drawers
        from .motion.scene_query import SceneQuery

        scene = SceneQuery.from_robosuite_sim(sim)
        drawers = discover_drawers(scene)
        support_frames = discover_drawer_support_frames(scene)
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for d in drawers:
        name = normalize_entity_name(f"{d.level_label}_drawer")
        aliases = sorted(
            {
                name,
                name.replace("_", " "),
                normalize_entity_name(d.joint_name),
                normalize_entity_name(f"{d.level_label}_drawer_handle"),
                f"{d.level_label} drawer handle",
            }
        )
        support_frame = support_frames.get(d.drawer_body)
        support_note = ""
        if support_frame:
            support_note = (
                f" This drawer is physically part of the {normalize_entity_name(support_frame)} "
                "cabinet frame entity -- when checking rule 15/15a clearance for this drawer's "
                f"opening path, use {normalize_entity_name(support_frame)}'s own xy/radius in "
                "scene_context.geometry as the cabinet position to measure distance from."
            )
        out.append(
            {
                "name": name,
                "role": "articulated_fixture",
                "aliases": aliases,
                "notes": (
                    f"Auto-discovered articulated fixture (slide joint {d.joint_name!r}); "
                    "use skill=\"open\" to grasp its handle and pull it open, or skill=\"close\" "
                    "to push it shut." + support_note
                ),
            }
        )
    return out


def infer_cabinet_support_entities(env: Any) -> list[dict[str, Any]]:
    """Discover each drawer cabinet's own static frame body (see
    ``discover_cabinet_support_bodies`` for why this is a *different* body
    from any drawer level) and return it as a ``support_object`` scene
    entity, so "place X on top of the cabinet" has a real name to target --
    without this, System 2 has nothing to reference but a drawer level
    itself (already seen hallucinating a drawer's name as a stand-in for
    "on top of the cabinet", which is wrong: a drawer is the sliding part,
    not the cabinet's static top surface)."""
    sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return []
    try:
        from .motion.articulation import discover_cabinet_support_bodies, discover_drawer_support_frames, discover_drawers
        from .motion.scene_query import SceneQuery

        scene = SceneQuery.from_robosuite_sim(sim)
        root_bodies = discover_cabinet_support_bodies(scene)
        support_frames = discover_drawer_support_frames(scene)
        drawer_entity_names_by_frame: dict[str, list[str]] = {}
        for d in discover_drawers(scene):
            frame = support_frames.get(d.drawer_body)
            if frame:
                drawer_entity_names_by_frame.setdefault(frame, []).append(
                    normalize_entity_name(f"{d.level_label}_drawer")
                )
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for body_name in root_bodies:
        # Register the body's own exact name verbatim -- NOT stripped down
        # to a shorter/prettier form the way free-body objects normally are
        # (see resolve_body_name's own "{name}_main" convention). That
        # stripping is wrong here: this fixture's world-parented root (e.g.
        # "wooden_cabinet_1_main") is a real body in its own right, just an
        # empty pass-through frame with none of the actual geometry -- so a
        # shortened "wooden_cabinet_1" would round-trip straight back to
        # that *wrong*, empty body via resolve_body_name's own "_main"
        # convention, not to this (correctly-chosen, geometry-owning) one.
        # Confirmed directly: this exact collision was the reason "place on
        # top of the cabinet" kept failing before this was fixed.
        name = normalize_entity_name(body_name)
        aliases = sorted({name, name.replace("_", " ")})
        # Real xy/z geometry, not just a name -- without this, the legacy
        # transport backend's own SceneResolver.resolve_goal() has nothing
        # to look up for a fixture-only entity (which normally has none,
        # see merge_drawer_fixtures_into_scene_context's own docstring) and
        # silently falls back to a workspace-center placeholder instead of
        # the cabinet's real position. Confirmed directly: transporting to
        # this entity before this fix aimed at goal_xy=(0.05, 0.2) -- an
        # unrelated fallback point, not the cabinet at all. "open" doesn't
        # need this (it re-measures the live model directly via
        # resolve_drawer), but transport/place's own resolvers do.
        pos, _ = scene.body_pose(body_name)
        extent = scene.body_extent_world(body_name)
        top_z = float(pos[2] + extent.center_offset[2] + extent.half_extents[2]) if extent is not None else float(pos[2])
        radius = float(max(extent.half_extents[0], extent.half_extents[1])) if extent is not None else 0.1
        owned_drawers = drawer_entity_names_by_frame.get(body_name, [])
        drawer_note = ""
        if owned_drawers:
            drawer_note = (
                f" This is the shared cabinet frame for {', '.join(owned_drawers)} -- when "
                "checking rule 15/15a clearance for opening any of those drawers, use this "
                "entity's own xy/radius here as the cabinet position to measure distance from."
            )
        out.append(
            {
                "name": name,
                "role": "support_object",
                "aliases": aliases,
                "notes": (
                    "Auto-discovered cabinet/fixture frame body -- use as params.goal_region "
                    'with params.place_surface="top" to place something on top of it (not '
                    "inside any of its drawers)." + drawer_note
                ),
                "_geometry": {
                    "xy": [float(pos[0]), float(pos[1])],
                    "xyz": [float(pos[0]), float(pos[1]), top_z],
                    "z": top_z,
                    "radius": radius,
                },
                "_owned_drawers": owned_drawers,
            }
        )
    return out


def infer_table_support_entity(env: Any) -> list[dict[str, Any]]:
    """Register the scene's own tabletop as a resolvable ``support_object``
    entity, so System 2 has a simple, always-available "somewhere clear"
    destination for temporarily relocating an object -- e.g. moving a bowl
    out of a drawer's way before opening it (see prompts.py's rule on this).

    Deliberately *not* discovered via ``infer_objects_from_mujoco`` (which
    explicitly excludes any body name containing "table", since it's scene
    furniture, not a manipulable/placeable object in the usual sense) --
    registered here on purpose instead, since placing something *on* the
    table is exactly the simple, well-tested operation this fixture-move
    heuristic actually wants, as opposed to placing on top of a specific
    fixture's own frame (more error-prone, and leaves the object sitting
    somewhere elevated and awkward to re-grasp from -- confirmed via a real
    rollout that re-picking a bowl off a cabinet's own top surface is
    measurably harder than picking it up off the table it started on).
    """
    sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return []
    try:
        from .motion.scene_query import SceneQuery

        scene = SceneQuery.from_robosuite_sim(sim)
        scene.body_id("table")
    except Exception:
        return []

    pos, _ = scene.body_pose("table")
    extent = scene.body_extent_world("table")
    top_z = float(pos[2] + extent.center_offset[2] + extent.half_extents[2]) if extent is not None else float(pos[2])
    radius = float(max(extent.half_extents[0], extent.half_extents[1])) if extent is not None else 0.3
    return [
        {
            "name": "table",
            "role": "support_object",
            "aliases": ["table", "tabletop", "the table"],
            "notes": (
                "The scene's own tabletop -- use as params.goal_region with "
                'params.place_surface="top" as a simple, always-available spot '
                "to temporarily set an object down on (e.g. moving something out "
                "of a drawer/cabinet's way before opening it)."
            ),
            "_geometry": {
                "xy": [float(pos[0]), float(pos[1])],
                "xyz": [float(pos[0]), float(pos[1]), top_z],
                "z": top_z,
                "radius": radius,
            },
        }
    ]


def merge_drawer_fixtures_into_scene_context(scene_context: dict[str, Any], env: Any) -> dict[str, Any]:
    """Append discovered drawer/cabinet entities (see ``infer_drawer_fixtures``),
    their cabinets' own static frame bodies (see
    ``infer_cabinet_support_entities``), and the scene's own tabletop (see
    ``infer_table_support_entity``) into ``scene_context["entities"]``,
    skipping any name already present. A plain drawer/cabinet-level entity
    has no meaningful single xyz/geometry entry to add -- its entity record
    (name/role/aliases) is enough for System 2 to reference it by name and
    for ``resolve_drawer``/``resolve_body_name`` to look it up again from the
    live model at execution time. A cabinet/table support entity is
    different: it also carries a ``_geometry`` entry (popped into
    ``scene_context["geometry"]`` here, matching
    ``update_scene_context_from_objects``'s own convention) since the
    *legacy* transport backend's ``SceneResolver.resolve_goal()`` -- unlike
    ``open``'s own live re-measurement -- has no other way to find a
    fixture's position and silently falls back to a workspace-center
    placeholder without it."""
    updated = copy.deepcopy(scene_context)
    entities = updated.setdefault("entities", [])
    geometry = updated.setdefault("geometry", {})
    existing_names = {e.get("name") for e in entities if e.get("name")}
    cabinets_with_drawers: list[tuple[str, list[str]]] = []
    for entity in infer_drawer_fixtures(env) + infer_cabinet_support_entities(env) + infer_table_support_entity(env):
        entity = dict(entity)
        entity_geometry = entity.pop("_geometry", None)
        owned_drawers = entity.pop("_owned_drawers", None)
        if entity["name"] not in existing_names:
            entities.append(entity)
            existing_names.add(entity["name"])
        if entity_geometry is not None:
            geometry.setdefault(entity["name"], entity_geometry)
        if owned_drawers:
            cabinets_with_drawers.append((entity["name"], owned_drawers))

    # The robot's own base position, as plain geometry rather than an entity
    # (it is not a thing to manipulate or place onto). System 2 otherwise has
    # no way to tell which parts of a large table the arm can comfortably
    # work in, and "put it somewhere clear of the cabinet" alone pushes
    # objects to whichever table edge is furthest from the fixture --
    # confirmed via a real rollout that left a bowl at the far edge where
    # re-grasping it failed, despite the gripper reaching its target to
    # within 0.9mm. See prompts.py rule 15's reach band.
    try:
        from .motion.scene_query import SceneQuery

        sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
        if sim is not None:
            base_scene = SceneQuery.from_robosuite_sim(sim)
            base_pos, _ = base_scene.body_pose("robot0_base")
            geometry.setdefault(
                "robot_base",
                {
                    "xy": [float(base_pos[0]), float(base_pos[1])],
                    "xyz": [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])],
                    "z": float(base_pos[2]),
                },
            )
    except Exception:
        pass

    _annotate_drawer_clearance_conflicts(entities, geometry, cabinets_with_drawers)
    _annotate_drawer_approach_corridor(entities, geometry, env)
    return updated


DRAWER_BLOCKING_CLEARANCE_M = 0.15


def _annotate_drawer_clearance_conflicts(
    entities: list[dict[str, Any]],
    geometry: dict[str, Any],
    cabinets_with_drawers: list[tuple[str, list[str]]],
) -> None:
    """Precompute, in code rather than leaving it to System 2's own arithmetic,
    which other objects sit too close to each drawer cabinet's own footprint
    to leave room for that drawer to open -- and stamp the verdict directly
    onto that object's entity ``notes`` (mutated in place).

    This exists because asking the LLM to do this distance/threshold
    computation itself (rule 15a in prompts.py) turned out unreliable in
    practice: given the same scene four separate real rollouts wrongly opted
    to move an object with plenty of clearance (0.22m) while leaving the one
    actually crowding the cabinet's footprint (0.05m clearance) untouched.
    Handing System 2 the already-computed verdict per object removes that
    failure mode -- it only has to read a stamped conclusion, not compute
    one across several candidates itself."""
    for cabinet_name, owned_drawers in cabinets_with_drawers:
        cab_geom = geometry.get(cabinet_name)
        if cab_geom is None:
            continue
        drawers_text = "/".join(owned_drawers)
        for entity in entities:
            name = entity.get("name")
            if not name or name == cabinet_name or name == "table" or name in owned_drawers:
                continue
            obj_geom = geometry.get(name)
            if obj_geom is None:
                continue
            dx = obj_geom["xy"][0] - cab_geom["xy"][0]
            dy = obj_geom["xy"][1] - cab_geom["xy"][1]
            dist = (dx * dx + dy * dy) ** 0.5
            clearance = dist - cab_geom.get("radius", 0.0) - obj_geom.get("radius", 0.0)
            if clearance < DRAWER_BLOCKING_CLEARANCE_M:
                flag = (
                    f" [Precomputed clearance to {cabinet_name}: {clearance:.2f}m, below the "
                    f"{DRAWER_BLOCKING_CLEARANCE_M:.2f}m rule-15 threshold -- this object BLOCKS "
                    f"{drawers_text} from opening. If it is the object the task places into that "
                    "fixture, it must be moved aside before the open subtask, per rule 15.]"
                )
                entity["notes"] = entity.get("notes", "") + flag



# The band around the line the hand travels to reach a handle. Half-width is
# about the hand's own half-width, and the depth covers only the final
# stretch where the gripper is down at handle height -- further out the arm
# is still up high and passes over things. Deliberately narrow: an object
# merely *near* the approach does not have to be relocated, because the arm
# can nudge a small light one aside at no cost, and every object moved is
# three more subtasks that can fail. Measured on the middle drawer, this
# flags the plate (0.043m off-axis, 0.125m out -- a wide flat obstacle right
# at the handle) while leaving the cream cheese (0.270m out), the bowl
# (0.098m off-axis) and the wine bottle (0.232m off-axis) alone.
APPROACH_CORRIDOR_HALF_WIDTH_M = 0.07
APPROACH_CORRIDOR_DEPTH_M = 0.20


def _annotate_drawer_approach_corridor(
    entities: list[dict[str, Any]],
    geometry: dict[str, Any],
    env: Any,
) -> None:
    """Flag objects sitting in the straight line the gripper must travel to
    reach a drawer handle.

    Distinct from ``_annotate_drawer_clearance_conflicts``, which asks whether
    an object fouls the *drawer's own swing*. That misses the other way a
    drawer becomes unopenable: the handle stays perfectly clear while the
    space the *arm* needs to get to it does not. Measured on the middle
    drawer, whose handle is only ~11.6cm above the table: every reachable
    approach was blocked, by the tabletop and by a plate and cream cheese
    parked in front -- and the cream cheese was never flagged by the
    footprint test at all, being 0.245m clear of the cabinet.
    """
    try:
        from .motion.articulation import discover_drawers, locate_handle_position_world, measure_pull_axis_and_travel
        from .motion.scene_query import SceneQuery

        sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
        if sim is None:
            return
        scene = SceneQuery.from_robosuite_sim(sim)
        drawers = discover_drawers(scene)
    except Exception:
        return

    corridors = []
    for d in drawers:
        try:
            axis, _travel, _lo, _hi = measure_pull_axis_and_travel(scene, d)
            handle = locate_handle_position_world(scene, d, axis)
        except Exception:
            continue
        corridors.append((normalize_entity_name(f"{d.level_label}_drawer"), np.asarray(axis[:2], dtype=float), np.asarray(handle[:2], dtype=float)))
    if not corridors:
        return

    for entity in entities:
        name = entity.get("name")
        geom = geometry.get(name) if name else None
        if not geom or "xy" not in geom or name == "table" or "drawer" in (name or ""):
            continue
        xy = np.asarray(geom["xy"], dtype=float)
        blocked_for = []
        for drawer_name, axis_xy, handle_xy in corridors:
            delta = xy - handle_xy
            along = float(delta @ axis_xy)
            lateral = float(np.linalg.norm(delta - along * axis_xy))
            if 0.0 < along < APPROACH_CORRIDOR_DEPTH_M and lateral < APPROACH_CORRIDOR_HALF_WIDTH_M:
                blocked_for.append(drawer_name)
        if blocked_for:
            entity["notes"] = entity.get("notes", "") + (
                f" [Precomputed: this object stands in the arm's approach path to "
                f"{'/'.join(blocked_for)} (within {APPROACH_CORRIDOR_HALF_WIDTH_M:.2f}m of the line "
                "the gripper travels to reach the handle). The handle cannot be reached until it is "
                "moved, even though the drawer's own swing may be clear.]"
            )

def _entity_aliases(name: str) -> list[str]:
    base = normalize_entity_name(name)
    aliases = {base, base.replace("_", " ")}
    # Make phrases such as "blue block" resolve when the raw LIBERO entity is
    # "blue_wooden_block_1".
    no_idx = re.sub(r"_\d+$", "", base)
    aliases.add(no_idx.replace("_", " "))
    aliases.add(no_idx)
    return sorted(a for a in aliases if a)


def update_scene_context_from_objects(
    scene_context: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    *,
    source: str = "libero",
) -> dict[str, Any]:
    """Return a scene_context copy populated with object-centric geometry."""
    updated = copy.deepcopy(scene_context)
    existing = {str(e.get("name")): e for e in updated.get("entities", []) if e.get("name")}
    entities = list(updated.get("entities", []))
    geometry = updated.setdefault("geometry", {})
    live = updated.setdefault("live_scene_state", {})
    live_objects = live.setdefault("objects", {})

    for name, obj in sorted(objects.items()):
        canon = normalize_entity_name(name)
        xyz = obj.get("xyz") or _as_float_list(obj.get("pos"), 3)
        if xyz is None:
            continue
        xy = obj.get("xy") or xyz[:2]
        role = existing.get(canon, {}).get("role") or role_for_object_name(canon)
        if canon not in existing:
            ent = {
                "name": canon,
                "role": role,
                "object_class": re.sub(r"_\d+$", "", canon),
                "aliases": _entity_aliases(canon),
                "notes": f"Auto-discovered from LIBERO {source} state.",
            }
            entities.append(ent)
            existing[canon] = ent
        geometry.setdefault(canon, {})
        geometry[canon].setdefault("xy", [float(xy[0]), float(xy[1])])
        geometry[canon].setdefault("xyz", [float(xyz[0]), float(xyz[1]), float(xyz[2])])
        geometry[canon].setdefault("z", float(xyz[2]))
        geometry[canon].setdefault("radius", float(obj.get("radius", 0.045)))
        geometry[canon].setdefault("descend_radius", float(obj.get("descend_radius", 0.075)))
        if role == "obstacle" and "sdf_geometry" not in geometry[canon]:
            geometry[canon]["sdf_geometry"] = {
                "type": "circle",
                "center_xy": [float(xy[0]), float(xy[1])],
                "radius": float(geometry[canon].get("radius", 0.055)),
                "convention": "outside_positive",
            }
        live_objects[canon] = {
            **{k: v for k, v in obj.items() if k not in {"pos"}},
            "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "xy": [float(xy[0]), float(xy[1])],
            "z": float(xyz[2]),
            "source": obj.get("source", source),
        }

    updated["entities"] = entities
    live["notes"] = (
        "LIBERO live_scene_state contains object poses in the simulator/world frame. "
        "Use these poses for spatial offsets such as 5cm left of another object; "
        "do not invent absolute coordinates when a reference entity is available."
    )
    updated.setdefault("coordinate_convention", {})
    updated["coordinate_convention"].setdefault("frame", "LIBERO/robosuite world xy")
    updated["coordinate_convention"].setdefault("left", "-x")
    updated["coordinate_convention"].setdefault("right", "+x")
    updated["coordinate_convention"].setdefault("front", "+y")
    updated["coordinate_convention"].setdefault("back", "-y")
    updated["coordinate_convention"].setdefault("offset_units", "meters")
    return updated


def load_libero_task_from_scene_context(scene_context: dict[str, Any]) -> LiberoTaskInfo:
    cfg = scene_context.get("platform", {}).get("libero", {}) or {}
    suite = cfg.get("suite") or cfg.get("task_suite")
    task_id = cfg.get("task_id")
    init_state_id = int(cfg.get("init_state_id", cfg.get("initial_state_id", 0)))
    bddl_file = cfg.get("bddl_file") or cfg.get("bddl_file_name")
    task_name = cfg.get("task_name")
    language = cfg.get("language") or cfg.get("instruction")

    if suite is not None and task_id is not None and not bddl_file:
        _ensure_libero_on_path()
        benchmark, get_libero_path = _import_libero_benchmark()

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[str(suite)]()
        task_id_i = int(task_id)
        task = task_suite.get_task(task_id_i)
        task_name = task_name or getattr(task, "name", None)
        language = language or getattr(task, "language", None)
        bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    elif bddl_file:
        bddl_file = str(Path(bddl_file).expanduser())

    return LiberoTaskInfo(
        suite=None if suite is None else str(suite),
        task_id=None if task_id is None else int(task_id),
        task_name=task_name,
        language=language,
        bddl_file=bddl_file,
        init_state_id=init_state_id,
    )


def get_libero_init_state(scene_context: dict[str, Any]) -> Any | None:
    cfg = scene_context.get("platform", {}).get("libero", {}) or {}
    if cfg.get("init_state") is not None:
        return cfg.get("init_state")
    suite = cfg.get("suite") or cfg.get("task_suite")
    task_id = cfg.get("task_id")
    if suite is None or task_id is None:
        return None
    try:
        _ensure_libero_on_path()
        benchmark, _ = _import_libero_benchmark()

        task_suite = benchmark.get_benchmark_dict()[str(suite)]()
        init_states = task_suite.get_task_init_states(int(task_id))
        init_state_id = int(cfg.get("init_state_id", cfg.get("initial_state_id", 0)))
        if init_states is None or len(init_states) == 0:
            return None
        init_state_id = max(0, min(init_state_id, len(init_states) - 1))
        return init_states[init_state_id]
    except Exception:
        return None


def append_experiment_prompt_modifiers(prompt: str, scene_context: dict[str, Any]) -> str:
    """Append STL-stress instructions declared in scene_context.platform.libero_experiment.

    This lets you take a normal LIBERO task and systematically add timing,
    avoidance, or spatial-precision requirements while keeping the instruction
    language-conditioned for all compared methods.
    """
    exp = scene_context.get("platform", {}).get("libero_experiment", {}) or {}
    additions: list[str] = []
    timing = exp.get("timing_window") or {}
    if timing:
        target = timing.get("target") or timing.get("object") or "the target object"
        start = timing.get("start_sec")
        end = timing.get("end_sec")
        predicate = timing.get("predicate", "pick")
        if start is not None and end is not None:
            additions.append(f"Ensure the {predicate} of {target} happens between {float(start):g} and {float(end):g} seconds, not before and not after.")
    avoid = exp.get("avoid_region") or exp.get("avoid") or {}
    if avoid:
        region = avoid.get("region") or avoid.get("object") or avoid.get("obstacle")
        if region:
            additions.append(f"Avoid the {str(region).replace('_', ' ')} during the motion.")
    spatial = exp.get("spatial_offset") or {}
    if spatial:
        obj = spatial.get("object", "the object")
        ref = spatial.get("reference") or spatial.get("reference_object") or "the reference object"
        amount = float(spatial.get("offset_m", spatial.get("meters", 0.05)))
        direction = spatial.get("direction", "left")
        additions.append(f"Place {obj} {amount * 100:g} cm to the {direction} of {ref} with high spatial precision.")
    if not additions:
        return prompt
    return prompt.rstrip() + " " + " ".join(additions)


def apply_libero_experiment_geometry(scene_context: dict[str, Any]) -> dict[str, Any]:
    """Materialize experiment modifiers as scene-context geometry when possible."""
    updated = copy.deepcopy(scene_context)
    exp = updated.get("platform", {}).get("libero_experiment", {}) or {}
    geom = updated.setdefault("geometry", {})
    entities = updated.setdefault("entities", [])
    entity_names = {e.get("name") for e in entities if e.get("name")}

    # Avoidance: make sure the named obstacle has an SDF geometry.  If the
    # object pose was discovered, a circle SDF is enough as a generic fallback;
    # users can override with a box/polygon in the scene_context.
    avoid = exp.get("avoid_region") or exp.get("avoid") or {}
    if avoid:
        region = normalize_entity_name(avoid.get("region") or avoid.get("object") or avoid.get("obstacle") or "")
        margin = float(avoid.get("margin", avoid.get("radius", 0.06)))
        if region:
            if region not in entity_names:
                entities.append({"name": region, "role": "obstacle", "aliases": _entity_aliases(region)})
                entity_names.add(region)
            g = geom.setdefault(region, {})
            xy = g.get("xy") or updated.get("live_scene_state", {}).get("objects", {}).get(region, {}).get("xy")
            if xy is not None and "sdf_geometry" not in g:
                g["sdf_geometry"] = {
                    "type": "circle",
                    "center_xy": [float(xy[0]), float(xy[1])],
                    "radius": margin,
                    "convention": "outside_positive",
                }
            g.setdefault("radius", margin)

    # Spatial precision: create a virtual goal_region at an offset from a live
    # reference.  System 2 can either use this named region or compute the same
    # explicit goal_x/goal_y from coordinate_convention.
    spatial = exp.get("spatial_offset") or {}
    if spatial:
        ref = normalize_entity_name(spatial.get("reference") or spatial.get("reference_object") or "")
        obj = normalize_entity_name(spatial.get("object") or "object")
        direction = str(spatial.get("direction", "left")).lower()
        amount = float(spatial.get("offset_m", spatial.get("meters", 0.05)))
        ref_xy = None
        if ref:
            ref_xy = geom.get(ref, {}).get("xy") or updated.get("live_scene_state", {}).get("objects", {}).get(ref, {}).get("xy")
        if ref_xy is not None:
            dx, dy = 0.0, 0.0
            if direction in {"left", "west", "minus_x", "-x"}:
                dx = -amount
            elif direction in {"right", "east", "plus_x", "+x"}:
                dx = amount
            elif direction in {"front", "forward", "plus_y", "+y"}:
                dy = amount
            elif direction in {"back", "behind", "minus_y", "-y"}:
                dy = -amount
            name = normalize_entity_name(spatial.get("name") or f"{obj}_{direction}_of_{ref}_{int(round(amount*100))}cm")
            xy = [float(ref_xy[0]) + dx, float(ref_xy[1]) + dy]
            geom[name] = {
                "xy": xy,
                "xyz": [xy[0], xy[1], float(geom.get(ref, {}).get("z", updated.get("live_scene_state", {}).get("objects", {}).get(ref, {}).get("z", 0.8)))],
                "radius": float(spatial.get("goal_radius", 0.015)),
                "descend_radius": float(spatial.get("descend_radius", 0.06)),
                "derived_from": {"reference": ref, "direction": direction, "offset_m": amount},
            }
            if name not in entity_names:
                entities.append({
                    "name": name,
                    "role": "goal_region",
                    "aliases": _entity_aliases(name) + [f"{amount*100:g} cm {direction} of {ref.replace('_', ' ')}"],
                    "notes": "Virtual goal region generated for a LIBERO STL spatial-precision experiment.",
                })
                entity_names.add(name)
    return updated


def _all_known_entity_names(scene_context: dict[str, Any]) -> set[str]:
    """Return the union of entity names from entities list, geometry, and live_scene_state."""
    names: set[str] = set()
    for ent in scene_context.get("entities", []):
        if ent.get("name"):
            names.add(str(ent["name"]))
            for alias in ent.get("aliases", []):
                names.add(str(alias))
    names.update(scene_context.get("geometry", {}).keys())
    names.update(scene_context.get("live_scene_state", {}).get("objects", {}).keys())
    return names


def format_available_entities(scene_context: dict[str, Any]) -> str:
    """Return a human-readable summary of all known entities in the scene context."""
    entities = scene_context.get("entities", [])
    geometry = scene_context.get("geometry", {})
    live_objects = scene_context.get("live_scene_state", {}).get("objects", {})

    # Collect all names, preferring the structured entities list first.
    seen: set[str] = set()
    lines: list[str] = []

    for ent in entities:
        name = ent.get("name")
        if not name:
            continue
        seen.add(name)
        role = ent.get("role", "unknown")
        g = geometry.get(name, {})
        xyz = g.get("xyz") or live_objects.get(name, {}).get("xyz")
        pos_str = f"  xyz={[round(v, 3) for v in xyz]}" if xyz else ""
        aliases = ent.get("aliases", [])
        alias_str = f"  aliases={aliases}" if aliases else ""
        lines.append(f"  {name!r:40s} role={role}{pos_str}{alias_str}")

    # Include objects that appear only in geometry or live_scene_state.
    extra_names = (set(geometry.keys()) | set(live_objects.keys())) - seen
    for name in sorted(extra_names):
        g = geometry.get(name, {})
        xyz = g.get("xyz") or live_objects.get(name, {}).get("xyz")
        pos_str = f"  xyz={[round(v, 3) for v in xyz]}" if xyz else ""
        lines.append(f"  {name!r:40s} (geometry only){pos_str}")

    if not lines:
        return "  (no entities discovered yet)"
    return "\n".join(lines)


def validate_libero_experiment_references(scene_context: dict[str, Any]) -> list[str]:
    """Check that experiment modifier entity references exist in the scene context.

    Returns a list of error strings (empty if everything is valid).  The caller
    should treat a non-empty list as a fatal configuration error.
    """
    exp = scene_context.get("platform", {}).get("libero_experiment", {}) or {}
    known = _all_known_entity_names(scene_context)
    errors: list[str] = []

    def _check(name: str | None, field: str) -> None:
        if not name:
            return
        canon = normalize_entity_name(name)
        if canon not in known and name not in known:
            errors.append(
                f"'{name}' (field: {field}) not found in the scene. "
                f"Known entities: {sorted(known)}"
            )

    avoid = exp.get("avoid_region") or exp.get("avoid") or {}
    if avoid:
        _check(avoid.get("region") or avoid.get("object") or avoid.get("obstacle"), "avoid_region.region")

    spatial = exp.get("spatial_offset") or {}
    if spatial:
        _check(spatial.get("reference") or spatial.get("reference_object"), "spatial_offset.reference")

    timing = exp.get("timing_window") or {}
    if timing:
        _check(timing.get("target") or timing.get("object"), "timing_window.target")

    return errors


def write_scene_context(path: str | Path, scene_context: dict[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scene_context, indent=2))
    return str(p)
