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


def _patch_torch_load_for_libero() -> None:
    """LIBERO's ``benchmark.get_task_init_states`` calls ``torch.load(path)`` with
    no ``weights_only`` argument. PyTorch >= 2.6 flipped that default to ``True``,
    which cannot unpickle LIBERO's numpy-backed init-state files -- so every
    ``set_init_state`` silently falls back to the BDDL default layout (in this
    repo the failure is swallowed by ``get_libero_init_state``'s bare except).
    The init-state files ship inside the pinned LIBERO checkout and are trusted,
    so restore the pre-2.6 behavior for them.
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
        from libero.libero import benchmark, get_libero_path  # type: ignore

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
        from libero.libero import benchmark  # type: ignore

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
