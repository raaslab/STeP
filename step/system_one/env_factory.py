"""Environment construction from scene_context + runtime config."""

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
    mock_robot: bool = False,
    debug_dir: str | Path | None = None,
) -> BaseManipulationEnv:
    """Build the right backend env for the platform declared in scene_context."""
    platform = scene_context.get("platform", {})
    backend = platform.get("backend", "sim")
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
    if backend == "vlabench":
        from .vlabench_env import VLABenchManipulationEnv

        return VLABenchManipulationEnv(
            scene_context=scene_context,
            mode=runtime["mode"],
            horizon=220 if env_horizon is None else int(env_horizon),
            render=render,
            active_object_name=runtime.get("active_object_name"),
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
            workspace_bounds=platform.get("workspace_bounds"),
            sdf_constraints=parse_sdf_constraints(scene_context),
            mpc_timing_enabled=bool(exec_proxy.get("mpc_timing_enabled", True)),
            mpc_timing_early_weight=float(exec_proxy.get("mpc_timing_early_weight", 220.0)),
            mpc_timing_deadline_weight=float(exec_proxy.get("mpc_timing_deadline_weight", 55.0)),
            mpc_timing_window_weight=float(exec_proxy.get("mpc_timing_window_weight", 10.0)),
            mpc_timing_early_hold_weight=float(exec_proxy.get("mpc_timing_early_hold_weight", 260.0)),
        )

    if backend == "libero":
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

    if backend == "real":
        from .real_env import RealManipulationEnv

        return RealManipulationEnv(
            scene_context=scene_context,
            mode=runtime["mode"],
            mock=mock_robot,
            debug_dir=debug_dir,
            # If run_full_system already ran perception before System 2, do not
            # run SAM/VLM again inside env.reset(). The geometry/live_scene_state
            # in scene_context is now the source of truth for execution.
            skip_perception=bool(
                scene_context.get("_runtime", {}).get("skip_perception_on_reset", False)
            ),
            active_object_name=runtime.get("active_object_name"),
            goal_reference_name=runtime.get("goal_reference_name"),
            goal_xy=runtime["goal_xy"],
            goal_radius=runtime["goal_radius"],
            goal_z=runtime.get("goal_z"),
            descend_radius=runtime["descend_radius"],
            keepout_xy=runtime["keepout_xy"],
            keepout_radius=runtime["keepout_radius"],
            carry_height_above_table=runtime["carry_height"]
            if runtime["carry_height"] is not None
            else None,
            retreat_height_above_table=retreat_height_above_table,
            sdf_constraints=parse_sdf_constraints(scene_context),
            mpc_timing_enabled=bool(exec_proxy.get("mpc_timing_enabled", True)),
            mpc_timing_early_weight=float(exec_proxy.get("mpc_timing_early_weight", 220.0)),
            mpc_timing_deadline_weight=float(exec_proxy.get("mpc_timing_deadline_weight", 55.0)),
            mpc_timing_window_weight=float(exec_proxy.get("mpc_timing_window_weight", 10.0)),
            mpc_timing_early_hold_weight=float(exec_proxy.get("mpc_timing_early_hold_weight", 260.0)),
        )
    # Default: sim.
    from .sim_env import SimManipulationEnv

    return SimManipulationEnv(
        mode=runtime["mode"],
        horizon=220 if env_horizon is None else int(env_horizon),
        has_renderer=render,
        goal_xy=runtime["goal_xy"],
        goal_radius=runtime["goal_radius"],
        goal_z=runtime.get("goal_z"),
        descend_radius=runtime["descend_radius"],
        keepout_xy=runtime["keepout_xy"],
        keepout_radius=runtime["keepout_radius"],
        carry_height_above_table=runtime["carry_height"]
        if runtime["carry_height"] is not None
        else float(perception_cfg.get("carry_height_above_table", 0.17)),
        retreat_height_above_table=retreat_height_above_table,
        num_blocks=runtime["num_blocks"],
        active_object_name=runtime.get("active_object_name") or "green_cube",
        goal_reference_name=runtime.get("goal_reference_name"),
        workspace_bounds=platform.get("workspace_bounds"),
        sdf_constraints=parse_sdf_constraints(scene_context),
        mpc_timing_enabled=bool(exec_proxy.get("mpc_timing_enabled", True)),
        mpc_timing_early_weight=float(exec_proxy.get("mpc_timing_early_weight", 220.0)),
        mpc_timing_deadline_weight=float(exec_proxy.get("mpc_timing_deadline_weight", 55.0)),
        mpc_timing_window_weight=float(exec_proxy.get("mpc_timing_window_weight", 10.0)),
        mpc_timing_early_hold_weight=float(exec_proxy.get("mpc_timing_early_hold_weight", 260.0)),
        tray_xy=tuple(platform.get("robosuite", {}).get("tray", {}).get("xy"))
        if platform.get("robosuite", {}).get("tray", {}).get("enabled", False)
        else None,
        tray_half_extents_xy=tuple(platform.get("robosuite", {}).get("tray", {}).get("half_extents_xy", [0.085, 0.065])),
        tray_wall_height=float(platform.get("robosuite", {}).get("tray", {}).get("wall_height", 0.018)),
        tray_wall_thickness=float(platform.get("robosuite", {}).get("tray", {}).get("wall_thickness", 0.006)),
        large_block_xy=tuple(platform.get("robosuite", {}).get("large_block_obstacle", {}).get("xy"))
        if platform.get("robosuite", {}).get("large_block_obstacle", {}).get("enabled", False)
        else None,
        large_block_size=tuple(platform.get("robosuite", {}).get("large_block_obstacle", {}).get("size", [0.060, 0.060, 0.060])),
        large_block_collision=bool(platform.get("robosuite", {}).get("large_block_obstacle", {}).get("collision", False)),
    )
