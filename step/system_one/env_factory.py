"""Environment construction from scene_context + runtime config.

LIBERO-only: the legacy ``sim``/``real``/``vlabench`` backends (and their
supporting modules -- ``sim_env.py``, ``real_env.py``, ``vlabench_env.py``,
``robosuite_utils.py``, ``hardware/``, ``perception/``) were removed as part
of the System 1 cleanup once the project's scope narrowed to LIBERO. See git
history (pre-cleanup commits) if any of that needs to be resurrected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_env import BaseManipulationEnv


def _make_libero(**kw):
    raise RuntimeError("unreachable: LIBERO is constructed inline below")


# Backends this build can construct. A backend registers a callable taking the
# same keyword arguments as make_env_from_scene_context; LIBERO stays inline
# because it is the default path and predates the hook.
_BACKENDS: dict[str, Any] = {"libero": _make_libero}


def register_backend(name: str, factory) -> None:
    """Register an additional System 1 backend (e.g. RoboCasa).

    Deliberately a registry rather than an import: the RoboCasa backend needs
    robosuite 1.5 while LIBERO pins 1.4, so the two cannot be imported into one
    process. The caller that has the right environment registers its own.
    """
    _BACKENDS[str(name)] = factory


def make_env_from_scene_context(
    scene_context: dict[str, Any],
    runtime: dict[str, Any],
    *,
    render: bool = False,
    env_horizon: int | None = None,
    debug_dir: str | Path | None = None,
) -> BaseManipulationEnv:
    """Build the LIBERO backend env for the platform declared in scene_context."""
    platform = scene_context.get("platform", {})
    backend = platform.get("backend", "libero")
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unsupported platform.backend {backend!r}. Known backends: "
            f"{sorted(_BACKENDS)}. (The 'sim'/'real'/'vlabench' backends were "
            "removed in the System 1 cleanup -- see git history.)"
        )
    if backend != "libero":
        # Non-LIBERO backends build their own env and then run the *same*
        # System 1: the motion pipeline addresses MuJoCo through SceneQuery and
        # robosuite's standard Panda names, so nothing below the scene-context
        # layer is backend-specific. Keeping the construction behind this hook
        # is what makes that claim checkable rather than assumed.
        return _BACKENDS[backend](
            scene_context=scene_context, runtime=runtime, render=render,
            env_horizon=env_horizon, debug_dir=debug_dir,
        )

    perception_cfg = platform.get("perception", {}) or {}
    placement_defaults = scene_context.get("placement_defaults", {}) or {}
    exec_proxy = scene_context.get("execution_proxy", {}) or {}
    # Retreat height is a System-1 placement/execution parameter, not something
    # System 2 should have to rediscover during replanning. Prefer
    # placement_defaults, with backward-compatible fallbacks for older scenes.
    retreat_height_above_table = float(
        placement_defaults.get(
            "retreat_height_above_table",
            perception_cfg.get(
                "retreat_height_above_table",
                exec_proxy.get("retreat_height_above_table", 0.28),
            ),
        )
    )
    from .geometry_sdf import parse_sdf_constraints
    from .libero_env import LiberoManipulationEnv

    return LiberoManipulationEnv(
        scene_context=scene_context,
        mode=runtime["mode"],
        horizon=800 if env_horizon is None else int(env_horizon),
        has_renderer=render,
        active_object_name=runtime.get("active_object_name") or "object",
        goal_reference_name=runtime.get("goal_reference_name"),
        goal_xy=tuple(runtime["goal_xy"]),
        goal_radius=runtime["goal_radius"],
        goal_z=runtime.get("goal_z"),
        descend_radius=runtime["descend_radius"],
        keepout_xy=tuple(runtime["keepout_xy"]),
        keepout_radius=runtime["keepout_radius"],
        carry_height_above_table=runtime["carry_height"]
        if runtime["carry_height"] is not None
        else float(perception_cfg.get("carry_height_above_table", 0.17)),
        retreat_height_above_table=retreat_height_above_table,
        mpc_timing_enabled=bool(exec_proxy.get("mpc_timing_enabled", True)),
        mpc_timing_early_weight=float(exec_proxy.get("mpc_timing_early_weight", 220.0)),
        mpc_timing_deadline_weight=float(exec_proxy.get("mpc_timing_deadline_weight", 55.0)),
        mpc_timing_window_weight=float(exec_proxy.get("mpc_timing_window_weight", 10.0)),
        mpc_timing_early_hold_weight=float(exec_proxy.get("mpc_timing_early_hold_weight", 260.0)),
    )
