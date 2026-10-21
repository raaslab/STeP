"""Environment construction from scene_context + runtime config. LIBERO-only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_env import BaseManipulationEnv


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
    if backend != "libero":
        raise ValueError(f"Unsupported platform.backend {backend!r}: only 'libero' is supported.")

    perception_cfg = platform.get("perception", {}) or {}
    placement_defaults = scene_context.get("placement_defaults", {}) or {}
    exec_proxy = scene_context.get("execution_proxy", {}) or {}
    # Prefers placement_defaults, with fallbacks for scenes that predate it.
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
