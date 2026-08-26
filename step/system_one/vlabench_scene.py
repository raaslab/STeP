"""Utilities for adapting VLABench episodes to the VLM-STL scene format.

The functions here intentionally use only public, best-effort VLABench hooks:
``load_env(...)``, ``env.get_observation(...)``, ``env.task.entities``, and
``env.task.get_instruction()``.  They do not assume a particular task category,
object name, or object class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import re

import numpy as np


DEFAULT_GOAL_RADIUS = 0.035
DEFAULT_DESCEND_RADIUS = 0.075
DEFAULT_OBJECT_RADIUS = 0.035


@dataclass
class VLABenchSnapshot:
    task: str
    instruction: str
    scene_context: dict[str, Any]
    image_path: str | None = None


def _safe_float_list(x: Any, n: int | None = None) -> list[float] | None:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except Exception:
        return None
    if n is not None and arr.size < n:
        return None
    if n is not None:
        arr = arr[:n]
    return [float(v) for v in arr.tolist()]


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(s).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "entity"


def _direction_to_xy(direction: str, offset_m: float) -> list[float]:
    """Map human spatial words to a planar offset in the VLABench world frame.

    This is intentionally simple and configurable from the CLI.  For a paper
    benchmark, we should report the convention and keep it fixed.  In VLABench,
    the exact camera view may vary, but the world-frame relation is what the STL
    monitor evaluates.
    """
    d = str(direction).lower().strip().replace("-", "_")
    val = float(offset_m)
    if d in {"left", "minus_x", "negative_x"}:
        return [-val, 0.0]
    if d in {"right", "plus_x", "positive_x"}:
        return [val, 0.0]
    if d in {"front", "forward", "plus_y", "positive_y"}:
        return [0.0, val]
    if d in {"back", "behind", "backward", "minus_y", "negative_y"}:
        return [0.0, -val]
    raise ValueError(f"Unsupported spatial direction {direction!r}")


def _entity_names(env: Any) -> list[str]:
    names: list[str] = []
    entities = getattr(getattr(env, "task", None), "entities", None)
    if isinstance(entities, dict):
        names.extend(str(k) for k in entities.keys())
    # Some VLABench task observables include object-like keys; keep only true
    # task entities by default, but make this robust if the task has no entities.
    if not names:
        obs = env.get_observation(require_pcd=False)
        for key in obs.keys():
            if key.endswith("_pos") or key.endswith("_position"):
                names.append(key.rsplit("_", 1)[0])
    return sorted(dict.fromkeys(names))


def _entity_handle(env: Any, name: str) -> Any | None:
    entities = getattr(getattr(env, "task", None), "entities", None)
    if isinstance(entities, dict):
        return entities.get(name)
    return None


def _extract_entity_position(env: Any, name: str) -> np.ndarray | None:
    """Best-effort world position for a VLABench entity."""
    # Public helper on LM4ManipDMEnv; for many entities name/type="geom" works.
    for typ in ("geom", "body", "site"):
        try:
            pos = env.get_xpos_by_name(name, type=typ)
            if pos is not None:
                return np.asarray(pos, dtype=np.float32).reshape(3).copy()
        except Exception:
            pass

    entity = _entity_handle(env, name)
    physics = getattr(env, "physics", None)
    if entity is not None and physics is not None:
        # Most VLABench entities expose geoms.  Bind the first one and read xpos.
        for attr in ("geoms", "sites", "bodies"):
            items = getattr(entity, attr, None)
            if items is None:
                continue
            try:
                seq = list(items)
            except Exception:
                seq = [items]
            for item in seq:
                try:
                    bound = physics.bind(item)
                    pos = getattr(bound, "xpos", None)
                    if pos is not None:
                        return np.asarray(pos, dtype=np.float32).reshape(3).copy()
                except Exception:
                    continue

    # Fallback through task observables if present.
    try:
        obs = env.get_observation(require_pcd=False)
        for key in (f"{name}_pos", f"{name}_position", name):
            val = obs.get(key)
            xyz = _safe_float_list(val, 3)
            if xyz is not None:
                return np.asarray(xyz, dtype=np.float32)
    except Exception:
        pass
    return None


def _infer_object_class(name: str, entity: Any | None = None) -> str:
    for attr in ("object_class", "class_name", "category", "name"):
        val = getattr(entity, attr, None) if entity is not None else None
        if isinstance(val, str) and val.strip():
            return _slug(val).lower()
    parts = str(name).lower().split("_")
    # Drop common numeric suffixes.
    parts = [p for p in parts if not p.isdigit()]
    return "_".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "object")


def _entity_aliases(name: str) -> list[str]:
    raw = str(name)
    aliases = {raw, raw.replace("_", " ")}
    # VLABench names often look like akita_black_bowl_1; the human instruction
    # may simply say "black bowl" or "bowl".
    toks = [t for t in raw.split("_") if not t.isdigit()]
    if len(toks) >= 2:
        aliases.add(" ".join(toks[-2:]))
    if toks:
        aliases.add(toks[-1])
    return sorted(a for a in aliases if a)


def build_scene_context_from_env(
    env: Any,
    *,
    task_name: str,
    episode_config: dict[str, Any] | None = None,
    avoid_entities: list[str] | None = None,
    spatial_goals: list[dict[str, Any]] | None = None,
    timing_modifiers: list[dict[str, Any]] | None = None,
    workspace_bounds: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Create a scene_context dictionary from a live VLABench env."""
    avoid_set = {str(x) for x in (avoid_entities or [])}
    spatial_goals = list(spatial_goals or [])
    timing_modifiers = list(timing_modifiers or [])

    entities: list[dict[str, Any]] = []
    geometry: dict[str, Any] = {}
    live_objects: dict[str, Any] = {}

    for name in _entity_names(env):
        entity = _entity_handle(env, name)
        pos = _extract_entity_position(env, name)
        role = "obstacle" if name in avoid_set else "manipulated_object"
        obj_class = _infer_object_class(name, entity)
        item = {
            "name": name,
            "role": role,
            "object_class": obj_class,
            "aliases": _entity_aliases(name),
        }
        entities.append(item)
        if pos is not None:
            xyz = [float(pos[0]), float(pos[1]), float(pos[2])]
            live_objects[name] = {
                "xyz": xyz,
                "xy": xyz[:2],
                "z": xyz[2],
                "source": "vlabench_snapshot",
            }
            geometry[name] = {
                "xyz": xyz,
                "xy": xyz[:2],
                "z": xyz[2],
                "radius": DEFAULT_OBJECT_RADIUS,
                "descend_radius": DEFAULT_DESCEND_RADIUS,
            }
        if name in avoid_set:
            center = geometry.get(name, {}).get("xy")
            if center is not None:
                geometry[name].setdefault("sdf_geometry", {
                    "type": "circle",
                    "center_xy": center,
                    "radius": float(DEFAULT_OBJECT_RADIUS + 0.04),
                    "sdf_convention": "outside_positive",
                    "applies_to": ["ee", "object"],
                    "margin": 0.025,
                    "weight": 34.0,
                })

    # Add virtual goal regions for spatial precision modifiers.
    for spec in spatial_goals:
        goal_name = str(spec["name"])
        ref = str(spec["reference"])
        offset_xy = np.asarray(spec["offset_xy"], dtype=np.float32).reshape(2)
        ref_xy_raw = geometry.get(ref, {}).get("xy") if ref in geometry else None
        try:
            ref_xy = np.asarray(ref_xy_raw, dtype=np.float32).reshape(-1)
        except Exception:
            ref_xy = np.asarray([], dtype=np.float32)
        if ref_xy.size < 2:
            raise ValueError(f"Cannot create spatial goal {goal_name!r}; reference {ref!r} has no xy pose")
        center_xy = (ref_xy[:2] + offset_xy).astype(float).tolist()
        entities.append({
            "name": goal_name,
            "role": "goal_region",
            "object_class": "relative_offset_region",
            "aliases": [goal_name.replace("_", " "), spec.get("language", goal_name)],
            "notes": f"Virtual goal: {spec.get('language', goal_name)}.",
        })
        geometry[goal_name] = {
            "xy": center_xy,
            "z": float(geometry.get(ref, {}).get("z", 0.0)),
            "radius": float(spec.get("radius", 0.015)),
            "descend_radius": float(spec.get("descend_radius", 0.035)),
            "reference": ref,
            "offset_xy": offset_xy.astype(float).tolist(),
        }

    try:
        instruction = env.task.get_instruction()
    except Exception:
        instruction = task_name

    ctx: dict[str, Any] = {
        "scene_name": f"vlabench_{task_name}",
        "description": "VLABench episode adapted to the VLM-STL scene_context format.",
        "platform": {
            "backend": "vlabench",
            "vlabench": {
                "task": task_name,
                "robot": "franka",
                "random_init": episode_config is None,
                "run_mode": "eval",
            },
            "workspace_bounds": workspace_bounds or {
                "x": [-0.75, 0.75],
                "y": [-0.75, 0.75],
                "z": [0.55, 1.55],
            },
            "perception": {
                "carry_height_above_table": 0.17,
                "model": "gpt-4o",
            },
        },
        "entities": entities,
        "geometry": geometry,
        "live_scene_state": {
            "objects": live_objects,
            "notes": "VLABench simulator ground-truth poses used as relocalized object states.",
        },
        "execution_proxy": {
            "system2_recall_enabled": False,
            "recall_on_subtask_success": False,
            "recall_on_final_success": False,
            "recall_failure_threshold": -0.02,
            "max_subtask_steps": 450,
            "mpc_timing_enabled": True,
            "mpc_timing_early_weight": 220.0,
            "mpc_timing_deadline_weight": 55.0,
            "mpc_timing_window_weight": 10.0,
            "mpc_timing_early_hold_weight": 260.0,
        },
        "vlabench_task_instruction": instruction,
        "vlabench_modifiers": {
            "avoid_entities": sorted(avoid_set),
            "spatial_goals": spatial_goals,
            "timing_modifiers": timing_modifiers,
        },
    }
    if episode_config is not None:
        ctx["platform"]["vlabench"]["episode_config"] = episode_config
    return ctx


def load_vlabench_env(
    task: str,
    *,
    robot: str = "franka",
    episode_config: dict[str, Any] | None = None,
    random_init: bool = True,
    run_mode: str = "eval",
    time_limit: float = math.inf,
    reset_wait_step: int = 10,
) -> Any:
    """Import and construct a VLABench environment with a helpful error."""
    try:
        import importlib
        importlib.import_module("VLABench.tasks")  # registers tasks
        importlib.import_module("VLABench.robots")  # registers robots
        from VLABench.envs import load_env
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "VLABench is not installed/importable. Install it in this environment: "
            "git clone https://github.com/OpenMOSS/VLABench.git && cd VLABench && "
            "pip install -r requirements.txt && pip install -e . && python scripts/download_assets.py"
        ) from exc
    return load_env(
        task,
        robot=robot,
        time_limit=time_limit,
        reset_wait_step=reset_wait_step,
        episode_config=episode_config,
        random_init=bool(random_init),
        run_mode=run_mode,
    )


def make_spatial_goal_spec(
    *,
    obj: str,
    reference: str,
    direction: str,
    offset_m: float,
    radius: float = 0.015,
) -> dict[str, Any]:
    offset_xy = _direction_to_xy(direction, offset_m)
    cm = int(round(float(offset_m) * 100.0))
    goal_name = _slug(f"{obj}_{cm}cm_{direction}_of_{reference}_goal")
    return {
        "name": goal_name,
        "object": obj,
        "reference": reference,
        "direction": direction,
        "offset_m": float(offset_m),
        "offset_xy": offset_xy,
        "radius": float(radius),
        "descend_radius": max(float(radius) * 2.5, 0.035),
        "language": f"{cm} cm to the {direction} of {reference}",
    }


def write_json(path: str | Path, data: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return str(path)
