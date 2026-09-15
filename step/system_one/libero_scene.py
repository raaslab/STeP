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
import math
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


def _patch_torch_load_for_libero() -> None:
    """LIBERO's ``benchmark.get_task_init_states`` calls ``torch.load(path)`` with
    no ``weights_only`` argument. PyTorch >= 2.6 flipped that default to ``True``,
    which cannot unpickle LIBERO's numpy-backed init-state files.

    Without this, ``get_libero_init_state``'s own bare ``except`` swallows the
    resulting ``UnpicklingError`` and returns ``None``, so ``--init-state-id`` is
    *silently ignored* and every rollout runs the BDDL default layout instead --
    i.e. an N-init-state benchmark secretly evaluates one state N times. The
    init-state files ship inside the pinned LIBERO checkout and are trusted, so
    restore the pre-2.6 behavior for them.
    """
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.load, "_libero_weights_only_patched", False):
        return
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._libero_weights_only_patched = True
    torch.load = _load


def _ensure_libero_on_path() -> None:
    """Add the LIBERO source tree to sys.path if libero is not already importable.

    The LIBERO repo is checked in at <workspace>/LIBERO/libero, so the package
    root (the directory that contains the top-level ``libero/`` package) is the
    ``LIBERO/libero`` subdirectory relative to this file's repository root.
    """
    try:
        import libero  # noqa: F401
        _patch_torch_load_for_libero()
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
            _patch_torch_load_for_libero()
            return
    _patch_torch_load_for_libero()


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
        from .motion.articulation import discover_drawer_support_frames, discover_drawers, measure_pull_axis_and_travel
        from .motion.config import MotionConfig
        from .motion.scene_query import SceneQuery

        scene = SceneQuery.from_robosuite_sim(sim)
        drawers = discover_drawers(scene)
        support_frames = discover_drawer_support_frames(scene)
    except Exception:
        return []

    motion_cfg = MotionConfig()
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
        # Live open/closed state, computed the same way OpenDrawerSkill/
        # CloseSkill themselves judge success (DrawerConfig.
        # open_threshold_frac / CloseConfig.close_threshold_frac) -- not
        # just captured once at scene-authoring time, since this reads the
        # *current* sim state, whatever task/reset left it at. Plans were
        # measured emitting a spurious "open" subtask against a drawer that
        # started ~91% open already (a real libero_10 task's own initial
        # condition, not an authoring quirk) -- with nothing in the prompt
        # ever saying so, System 2 had no way to know not to. Mirrors rule
        # 17's existing "[Precomputed: already satisfies In(...)]" pattern,
        # just for Opened/Closed on a fixture instead of In(...) on an
        # object.
        state_note = ""
        try:
            _, _, qpos_closed, qpos_open = measure_pull_axis_and_travel(scene, d)
            qpos_now = float(scene.get_qpos([d.joint_name])[0])
            denom = qpos_open - qpos_closed
            frac_open = (qpos_now - qpos_closed) / denom if abs(denom) > 1e-9 else 0.0
            if frac_open >= motion_cfg.drawer.open_threshold_frac:
                state_note = (
                    f" [Precomputed: this drawer is already {frac_open * 100:.0f}% open, which "
                    "already satisfies Opened(...) -- do NOT emit an \"open\" subtask for it; if "
                    "the task needs it opened, treat that as already done.]"
                )
            elif frac_open <= motion_cfg.close.close_threshold_frac:
                state_note = (
                    f" [Precomputed: this drawer is already {frac_open * 100:.0f}% open (i.e. "
                    "closed), which already satisfies Closed(...) -- do NOT emit a \"close\" "
                    "subtask for it; if the task needs it closed, treat that as already done.]"
                )
        except Exception:
            state_note = ""
        out.append(
            {
                "name": name,
                "role": "articulated_fixture",
                "aliases": aliases,
                "notes": (
                    f"Auto-discovered articulated fixture (slide joint {d.joint_name!r}); "
                    "use skill=\"open\" to grasp its handle and pull it open, or skill=\"close\" "
                    "to push it shut." + support_note + state_note
                ),
            }
        )
    return out


# Suffix convention several LIBERO turbosquid assets use for named,
# functionally-distinct sub-regions of a container -- confirmed against
# desk_caddy's own MJCF (assets/turbosquid_objects/desk_caddy/
# desk_caddy_site.xml): real box-type sites named "back_contain_region",
# "front_contain_region", "left_contain_region", "right_contain_region",
# each with its own local pos/quat/size -- and the BDDL goal for a real
# task (STUDY_SCENE1's "back compartment of the caddy") targets
# "desk_caddy_1_back_contain_region" directly by that exact name.
_NAMED_REGION_SITE_SUFFIX = "_contain_region"


def infer_named_region_entities(env: Any) -> list[dict[str, Any]]:
    """Discover every "_contain_region" site on every free-body object in
    the scene and register each as its own scene entity, carrying its live,
    exactly-measured world position and a usable placement radius in its
    notes -- as a point-target hint (rule 6), not a resolvable body (no
    plumbing anywhere resolves a goal_region to a site, only to a body).

    Exists because these regions were previously invisible to System 2
    entirely: nothing scanned for them, so a task like "put the book in the
    back compartment of the caddy" left System 2 to either eyeball a point
    target from the image (imprecise) or, worse, invent a plausible-sounding
    but nonexistent body name for it -- confirmed via a real rollout, a
    recalled plan referencing goal_region="back_compartment_of_caddy" (no
    such body exists under any name) crashed the pipeline outright wherever
    that string later reached resolve_body_name uncaught. Supplying the
    real, precomputed coordinates removes the reason to guess or invent one.
    """
    sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return []
    try:
        import mujoco

        from .motion.scene_query import SceneQuery, quat_to_rotmat_wxyz

        scene = SceneQuery.from_robosuite_sim(sim)
    except Exception:
        return []

    model = scene.model
    out: list[dict[str, Any]] = []
    for sid in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if not site_name.endswith(_NAMED_REGION_SITE_SUFFIX):
            continue
        bid = model.site_bodyid[sid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        # The body this site sits on is an internal child frame (e.g.
        # desk_caddy_1's own "object" sub-body, per its MJCF), not the
        # canonical free-joint body scene_context/params refer to
        # ("desk_caddy_1_main") -- walk up to the nearest ancestor whose own
        # name looks like a real registered LIBERO object (ends in "_main"
        # or has a free joint of its own) so the entity name this produces
        # matches what the rest of the scene context already calls it.
        root_bid = bid
        seen = set()
        while root_bid != 0 and root_bid not in seen:
            seen.add(root_bid)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_bid) or ""
            if name.endswith("_main"):
                body_name = name
                break
            parent = int(model.body_parentid[root_bid])
            if parent == root_bid:
                break
            root_bid = parent
        canonical_object = body_name[: -len("_main")] if body_name.endswith("_main") else body_name
        if not canonical_object:
            continue

        pos = scene.data.site_xpos[sid].copy()
        R = scene.data.site_xmat[sid].reshape(3, 3)
        local_half = np.asarray(model.site_size[sid], dtype=np.float64)
        # World-frame AABB half-extent under an arbitrary rotation: standard
        # re-projection (world_half[i] = sum_j |R[i,j]| * local_half[j]) --
        # needed because these sites are not all axis-aligned with their
        # parent (front_contain_region's own quat swaps its local X/Y into
        # world Y/Z), so just reading local_half directly would silently
        # report the wrong two axes' extents for that one.
        world_half = np.abs(R) @ local_half
        radius = float(min(world_half[0], world_half[1])) * 0.8

        # `site_name` is already fully namespaced by robosuite's own
        # instance-prefixing when it merges the asset into the scene (the
        # source XML just declares name="back_contain_region"; the compiled
        # model's real name is "desk_caddy_1_back_contain_region") -- use it
        # directly as the entity name rather than prefixing canonical_object
        # a second time. This also happens to exactly match the BDDL goal's
        # own predicate naming for this region, a useful (if not load-
        # bearing) coincidence.
        entity_name = site_name
        region_label = entity_name
        if canonical_object and region_label.startswith(canonical_object + "_"):
            region_label = region_label[len(canonical_object) + 1 :]
        # Two cases after the prefix strip above: a genuinely distinct
        # compartment still has its own connecting underscore before the
        # suffix ("back_contain_region"), but a site with *no* distinguishing
        # label at all is left as the bare suffix alone ("contain_region") --
        # its own underscore was already consumed stripping the object
        # prefix, so `.endswith(_NAMED_REGION_SITE_SUFFIX)` (which requires
        # a leading "_") silently fails to match and never reduces it to ""
        # the way it should. Strip the bare suffix too, then any leftover
        # trailing "_", so both cases correctly converge on "" for a
        # single/generic region.
        if region_label.endswith(_NAMED_REGION_SITE_SUFFIX):
            region_label = region_label[: -len(_NAMED_REGION_SITE_SUFFIX)]
        elif region_label.endswith(_NAMED_REGION_SITE_SUFFIX.lstrip("_")):
            region_label = region_label[: -len(_NAMED_REGION_SITE_SUFFIX.lstrip("_"))]
        region_label = region_label.rstrip("_")
        if not region_label:
            # A container with one single, unnamed "contain_region" site --
            # its own generic default interior, not one of several distinct
            # named compartments the way desk_caddy has "front"/"back"/
            # "left"/"right" -- isn't what this point-target hint is for.
            # Routing every object placed into it through the exact same
            # fixed point is fine for exactly one object, but wrong the
            # moment a second needs to go in alongside the first: confirmed
            # via a real libero_10 rollout (alphabet_soup_1 then
            # tomato_sauce_1, both told to target the identical
            # basket_1_contain_region point) -- the second transport failed
            # outright, every hover candidate near that point colliding with
            # the first object already resting there. The plain container
            # body (already registered as its own "basket_1"-style entity)
            # plus System 1's own interior-ring-search candidate generation
            # already finds a different, unoccupied spot for each object
            # without this -- only skip straight to the fixed point when the
            # site name actually distinguishes a specific sub-compartment.
            continue
        out.append(
            {
                "name": entity_name,
                "role": "support_object",
                "aliases": sorted({entity_name, f"{canonical_object} {region_label} compartment"}),
                "notes": (
                    f"Auto-discovered named sub-region ({region_label!r} compartment) of "
                    f"{canonical_object}, from its own asset's named site -- not a body, so it "
                    "cannot be used as params.goal_region directly. To target it precisely, use "
                    "params.goal_region={\"type\":\"point\",\"xy\":["
                    f"{pos[0]:.4f}, {pos[1]:.4f}],\"z\":{pos[2]:.4f},\"radius\":{radius:.4f}}} "
                    "on both transport and place -- these are the region's own real, measured "
                    "coordinates; do not estimate your own or invent a body name for this region. "
                    f"ALSO set params.goal_surface=\"{canonical_object}\" on both subtasks -- a point "
                    "target on its own defaults to resting on the bare table (System 1's own "
                    "fallback when goal_surface is unset), which is wrong here: this point sits "
                    "well above the table, inside a compartment on top of it, so place needs "
                    f"{canonical_object}'s own geometry, not the table's, to know how far down to "
                    "actually lower the object -- confirmed by a real rollout that omitting this "
                    "released the object ~20cm above where it should have settled, catching on the "
                    "compartment's own divider wall instead of descending into the slot."
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
        from .motion.scene_query import SceneQuery, resolve_body_name

        scene = SceneQuery.from_robosuite_sim(sim)
        # Not always literally named "table" -- a kitchen scene's arena
        # does use that exact body name, but a living-room scene's is
        # "living_room_table", a study scene's "study_table", etc. (see
        # resolve_body_name's own "table" fallback for why/how). Resolving
        # through it here, instead of a bare scene.body_id("table"), is
        # what makes this entity -- and every point-target place that
        # defaults to it as a supporting surface -- actually exist outside
        # kitchen scenes at all.
        table_body = resolve_body_name(scene, "table")
    except Exception:
        return []

    pos, _ = scene.body_pose(table_body)
    extent = scene.body_extent_world(table_body)
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


def infer_stove_fixtures(env: Any) -> list[dict[str, Any]]:
    """Register each stove/burner fixture (discovered the same way
    ``discover_knobs`` finds its turn-on target -- by hinge joint, not by
    name) as a resolvable ``support_object`` entity, so "turn on the stove
    and put the moka pot on it" has one real name to reference for *both*
    the turn_on target and the place/transport goal.

    Without this, a stove is invisible to System 2 -- it never appears in
    ``infer_objects_from_obs`` (LIBERO only emits `<name>_pos`/`_quat` obs
    keys for movable ``env.objects_dict`` members, never fixtures) and
    ``infer_objects_from_mujoco``'s own body-name filter excludes it too
    (``flat_stove_1_base`` matches the "base" exclusion). Confirmed directly
    on a real libero_10 rollout: with no stove entity to reference, System 2
    picked the only other object in the scene wide of the mark instead
    (`chefmate_8_frypan_1`, a frypan sharing the table) for *both*
    params.object on the turn_on subtask and params.goal_region on the
    place -- turn_on still worked by luck (resolve_knob falls back to "the
    only knob available" when a name doesn't match anything), but the moka
    pot ended up placed on the frypan instead of the stove, which has no
    equivalent silent-fallback rescue.
    """
    sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return []
    try:
        from .motion.articulation import discover_knobs
        from .motion.scene_query import SceneQuery, resolve_body_name

        scene = SceneQuery.from_robosuite_sim(sim)
        knobs = discover_knobs(scene)
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for knob in knobs:
        # The knob body itself (LIBERO names it "button") is a small part
        # deep in the fixture, not representative of the stove's own
        # footprint/top surface -- strip the known knob-part suffix to get
        # back to the fixture's own canonical name (e.g.
        # "flat_stove_1_button" -> "flat_stove_1"), then resolve *that*
        # to its real geometry-owning body the same way any other object
        # would be (the "_main" convention), rather than trusting the
        # knob's own position as a stand-in for the whole fixture.
        base = normalize_entity_name(knob.knob_body)
        for suffix in ("_button", "_knob", "_dial", "_switch"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base in seen_names:
            continue
        # resolve_body_name already prefers "_burner" over "_main" for
        # exactly this reason: the burner's own collision top (0.9305) sits
        # *below* the outer frame's (0.955) -- a burner recessed slightly
        # into the stovetop, not level with it. Using the frame's height (or
        # its xy -- confirmed 0.15m away from the burner's, a completely
        # different local position) as "where a pot rests" aimed the
        # hover/place pose at the wrong spot entirely, and every candidate
        # near it collided with the real burner/frame geometry (confirmed:
        # hover_pose_in_collision at nearly every one of 124 tried offsets).
        try:
            surface_body = resolve_body_name(scene, base)
        except KeyError:
            surface_body = knob.knob_body
        pos, _ = scene.body_pose(surface_body)
        # Collision-only, not the merged visual+collision extent: the two
        # can disagree (confirmed elsewhere this session for both
        # ContactGraspNetGenerator's height clamp and this same asset's own
        # main/frame body, whose visual mesh top sits ~1cm above its real
        # collision top) -- and it's the collision geometry a placed object
        # actually rests against.
        extent = scene.body_collision_extent_world(surface_body)
        top_z = float(pos[2] + extent.center_offset[2] + extent.half_extents[2]) if extent is not None else float(pos[2])
        radius = float(max(extent.half_extents[0], extent.half_extents[1])) if extent is not None else 0.15
        seen_names.add(base)
        out.append(
            {
                "name": base,
                "role": "support_object",
                "aliases": sorted({base, base.replace("_", " ")}),
                "notes": (
                    "Auto-discovered stove/burner fixture. Use this same entity for both: "
                    "(1) skill=\"turn_on\" with params.object naming it, to rotate its knob; "
                    'and (2) params.goal_region on transport/place with params.place_surface='
                    '"top", to set something down on its cooking surface. Do not substitute '
                    "any other object in the scene for either use."
                ),
                "_geometry": {
                    "xy": [float(pos[0]), float(pos[1])],
                    "xyz": [float(pos[0]), float(pos[1]), top_z],
                    "z": top_z,
                    "radius": radius,
                },
            }
        )
    return out


def merge_drawer_fixtures_into_scene_context(scene_context: dict[str, Any], env: Any) -> dict[str, Any]:
    """Append discovered drawer/cabinet entities (see ``infer_drawer_fixtures``),
    their cabinets' own static frame bodies (see
    ``infer_cabinet_support_entities``), the scene's own tabletop (see
    ``infer_table_support_entity``), and any named sub-region sites like a
    caddy's own compartments (see ``infer_named_region_entities``) into
    ``scene_context["entities"]``, skipping any name already present. A
    plain drawer/cabinet-level entity
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
    for entity in (
        infer_drawer_fixtures(env)
        + infer_cabinet_support_entities(env)
        + infer_table_support_entity(env)
        + infer_stove_fixtures(env)
        + infer_named_region_entities(env)
    ):
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
    # Derived from the "agentview" camera's own real extrinsics (the camera
    # System 2's scene image is rendered from), not assumed -- its image-
    # "right" direction is exactly world +y, and its image-"up"/away-from-
    # camera direction is mostly world -x (cam_xmat columns: local x = image
    # right, local y = image up, -local z = viewing direction). Previously
    # had x and y's roles swapped (claimed left/right = x, front/back = y),
    # which silently sent every "place it to the right of X" a quarter-turn
    # off -- confirmed on a real rollout: told to put the chocolate pudding
    # "to the right of the plate", the LLM correctly followed the (wrong)
    # convention and moved +x relative to the plate, which the real camera
    # geometry above shows lands *in front of* the plate, not to its side.
    updated["coordinate_convention"].setdefault("left", "-y")
    updated["coordinate_convention"].setdefault("right", "+y")
    updated["coordinate_convention"].setdefault("front", "+x")
    updated["coordinate_convention"].setdefault("back", "-x")
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
        explicit_xy = avoid.get("center_xy")
        if region and explicit_xy is not None:
            # A VIRTUAL zone: a marked region on the table at a fixed position,
            # with no object behind it. This is what LIBERO-Constrained uses.
            # Deriving the zone from a real object instead forces a choice
            # between a keep-out that never binds (the object sits off the
            # route) and one that blocks the task outright (the object is
            # physically in the way) -- both measured, neither usable.
            g = geom.setdefault(region, {})
            g["xy"] = [float(explicit_xy[0]), float(explicit_xy[1])]
            g["sdf_geometry"] = {
                "type": "circle",
                "center_xy": [float(explicit_xy[0]), float(explicit_xy[1])],
                "radius": margin,
                "convention": "outside_positive",
                "always_active": True,
                # Marks it as having no body to follow, so the per-reset
                # re-centring leaves it exactly where the benchmark put it.
                "virtual": True,
            }
            g.setdefault("radius", margin)
            if region not in entity_names:
                entities.append({
                    "name": region,
                    "role": "obstacle",
                    "aliases": _entity_aliases(region) + [
                        "the red zone", "the marked zone", "the danger zone",
                        "red zone marked on the table", "the zone",
                    ],
                    "notes": "Virtual keep-out region marked on the table; no physical object.",
                })
                entity_names.add(region)
        elif region:
            if region not in entity_names:
                entities.append({"name": region, "role": "obstacle", "aliases": _entity_aliases(region)})
                entity_names.add(region)
            else:
                # The obstacle is usually an already-discovered scene object, so
                # it arrives with whatever role discovery guessed (typically
                # support_object). Declaring it as a keep-out must override that:
                # parse_sdf_constraints derives the sign convention and whether
                # the constraint applies to the end effector from this role, so
                # leaving it as support_object silently produces an
                # inside_positive, object-only constraint -- the inverse of a
                # keep-out.
                for ent in entities:
                    if ent.get("name") == region:
                        ent["role"] = "obstacle"
            g = geom.setdefault(region, {})
            xy = g.get("xy") or updated.get("live_scene_state", {}).get("objects", {}).get(region, {}).get("xy")
            if xy is not None and "sdf_geometry" not in g:
                g["sdf_geometry"] = {
                    "type": "circle",
                    "center_xy": [float(xy[0]), float(xy[1])],
                    "radius": margin,
                    "convention": "outside_positive",
                    # A keep-out zone is a G(Safe(obstacle)) obligation: it holds
                    # for every step of the episode, not only while some subtask
                    # names it in applied_constraints (nothing populates that).
                    "always_active": True,
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


def bddl_goal_object_names(bddl_file: str | Path | None) -> list[str]:
    """Object instances named in a BDDL ``:goal`` block, in the order they
    appear -- conventionally the manipulated object first, then what it must end
    up on/in.

    Read from the BDDL rather than from the language string or from discovered
    entity roles: the goal is the task's own machine-readable statement of which
    objects matter, and it stays correct across init states and across the role
    guesses in ``role_for_object_name``.
    """
    if not bddl_file:
        return []
    try:
        text = Path(bddl_file).read_text()
    except OSError:
        return []
    match = re.search(r"\(:goal(.*?)\n\s*\)\s*\n", text, re.S) or re.search(r"\(:goal(.*)", text, re.S)
    if not match:
        return []
    seen: list[str] = []
    # Scene object instances are named "<object>_<index>" in LIBERO's BDDL.
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*_\d+", match.group(1)):
        if token not in seen:
            seen.append(token)
    return seen


def relocate_obstacle_into_path(env: Any, scene_context: dict[str, Any], bddl_file: str | None) -> dict[str, Any] | None:
    """Move the task's keep-out obstacle onto the straight line between the
    manipulated object and its goal, so the keep-out is something the robot must
    actually plan around.

    This is what makes LIBERO-Constrained a benchmark rather than a prompt
    overlay. Measured across the 30 configured tasks, the designated obstacle
    sits a median ~13cm off the transport path and up to 26cm away, so a 6cm
    keep-out is satisfied by the naive trajectory on most tasks and the
    constrained and unconstrained arms produce the same motion. Three tasks are
    worse than uninformative: they name the picked object or the goal itself as
    the obstacle, which no trajectory can satisfy.

    Placement is recomputed at reset from live positions rather than baked into
    the task config as a fixed coordinate, because LIBERO's init states move the
    objects: a coordinate that sits on the path in one init state sits well off
    it in another.

    The obstacle is NOT placed on the segment itself. It goes on the
    perpendicular bisector, offset by ``h``, which is what makes the placement
    solvable on short spans: at offset ``h`` the distance to the path is ``h``
    (so the constraint binds while ``h < margin``) but the distance to each
    endpoint is ``hypot(span/2, h)``, which grows with ``h`` and can exceed the
    margin even when ``span/2`` alone does not. Placing the obstacle on the line
    instead forces ``span >= 2*margin``, which fails on the many LIBERO tasks
    whose pick and goal are barely 12cm apart.

    So ``h`` is chosen from the window

        h >= sqrt((margin + buffer)^2 - (span/2)^2)   # endpoints stay reachable
        h <= margin - buffer                          # the path still violates

    and the task is skipped only when that window is genuinely empty.
    """
    experiment = (scene_context.get("platform", {}) or {}).get("libero_experiment", {}) or {}
    avoid = experiment.get("avoid_region") or {}
    if not avoid or not avoid.get("place_in_path"):
        return None
    obstacle = normalize_entity_name(avoid.get("region") or avoid.get("object") or avoid.get("obstacle") or "")
    if not obstacle:
        return None

    goal_objects = [normalize_entity_name(n) for n in bddl_goal_object_names(bddl_file)]
    endpoints = [n for n in goal_objects if n != obstacle]
    if len(endpoints) < 2:
        return {"obstacle": obstacle, "skipped": f"goal names {len(endpoints)} usable object(s), need 2"}

    sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return None
    model, data = sim.model, sim.data

    def _body_id(name: str) -> int | None:
        for candidate in (name, f"{name}_main"):
            try:
                bid = model.body_name2id(candidate)
            except Exception:
                continue
            if bid is not None:
                return int(bid)
        return None

    pick_id, goal_id, obstacle_id = (_body_id(endpoints[0]), _body_id(endpoints[1]), _body_id(obstacle))
    if pick_id is None or goal_id is None or obstacle_id is None:
        return {"obstacle": obstacle, "skipped": "could not resolve one of the bodies in MuJoCo"}

    pick_xy = np.asarray(data.body_xpos[pick_id][:2], dtype=np.float64)
    goal_xy = np.asarray(data.body_xpos[goal_id][:2], dtype=np.float64)
    span = float(np.linalg.norm(goal_xy - pick_xy))
    margin = float(avoid.get("margin", avoid.get("radius", 0.06)))
    # How far inside/outside the margin the placement aims to land. Too large
    # and the window closes on short spans; too small and rounding or a few
    # millimetres of settling decides whether the benchmark is satisfiable.
    buffer = float(avoid.get("placement_buffer_m", 0.01))
    if span < 1e-3:
        return {"obstacle": obstacle, "skipped": "pick and goal are at the same position"}

    half = 0.5 * span
    h_min = math.sqrt(max(0.0, (margin + buffer) ** 2 - half**2))
    h_max = margin - buffer
    if h_min > h_max:
        return {
            "obstacle": obstacle,
            "skipped": (
                f"no offset satisfies both ends: span {span:.3f}m needs h>={h_min:.3f}m "
                f"to keep the endpoints reachable but h<={h_max:.3f}m to stay on the path"
            ),
        }
    # The smallest valid offset: closest to the path, so the constraint binds as
    # hard as it can while the endpoints stay reachable.
    h = h_min if h_min > 0.0 else min(0.5 * margin, h_max)

    mid = 0.5 * (pick_xy + goal_xy)
    direction = (goal_xy - pick_xy) / span
    perp = np.array([-direction[1], direction[0]], dtype=np.float64)
    # Two perpendicular directions are equally good for the constraint; pick the
    # one whose landing spot is further from every other object, so the
    # relocated obstacle does not end up stacked on top of unrelated clutter.
    others = []
    for bid in range(int(model.nbody)):
        if bid in (pick_id, goal_id, obstacle_id):
            continue
        try:
            raw = model.body_id2name(bid)
        except Exception:
            raw = None
        if not raw or any(p in raw for p in ("world", "table", "robot", "panda", "gripper", "mount", "base")):
            continue
        others.append(np.asarray(data.body_xpos[bid][:2], dtype=np.float64))

    def _min_other_dist(xy: np.ndarray) -> float:
        return min((float(np.linalg.norm(xy - o)) for o in others), default=float("inf"))

    candidates = [mid + h * perp, mid - h * perp]
    target_xy = max(candidates, key=_min_other_dist)

    # The obstacle is moved by writing its free joint's qpos. A body without a
    # free joint (a fixture bolted into the scene) cannot be moved this way.
    jnt_adr = int(model.body_jntadr[obstacle_id])
    jnt_num = int(model.body_jntnum[obstacle_id])
    if jnt_num < 1 or jnt_adr < 0:
        return {"obstacle": obstacle, "skipped": "body has no free joint (fixed fixture)"}
    qpos_adr = int(model.jnt_qposadr[jnt_adr])
    before = [float(v) for v in data.body_xpos[obstacle_id][:3]]
    data.qpos[qpos_adr + 0] = float(target_xy[0])
    data.qpos[qpos_adr + 1] = float(target_xy[1])
    # z and the quaternion are left alone: the object keeps its resting height
    # and orientation, and the settling steps after this drop it onto whatever
    # is beneath its new position.
    try:
        sim.forward()
    except Exception:
        pass
    return {
        "obstacle": obstacle,
        "between": [endpoints[0], endpoints[1]],
        "from_xy": [round(before[0], 4), round(before[1], 4)],
        "to_xy": [round(float(target_xy[0]), 4), round(float(target_xy[1]), 4)],
        "pick_goal_span_m": round(span, 4),
        # What the placement guarantees, and the two numbers to check when a
        # task's results look wrong: path_clearance < margin means the naive
        # trajectory violates, endpoint_clearance > margin means the task is
        # still satisfiable.
        "path_clearance_m": round(float(h), 4),
        "endpoint_clearance_m": round(math.hypot(half, float(h)), 4),
        "margin_m": round(margin, 4),
        "clutter_clearance_m": round(_min_other_dist(target_xy), 4),
    }


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
