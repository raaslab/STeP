from __future__ import annotations

from collections import OrderedDict
import os
import sys
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np


def _try_import_robosuite():
    try:
        import robosuite as suite  # type: ignore
        from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config  # type: ignore
        return suite, refactor_composite_controller_config
    except Exception as exc:
        raise ImportError("robosuite is required for this project. Please install robosuite, mujoco, and gymnasium first.") from exc


def _format_vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _patch_mjviewer_visual_options() -> None:
    try:
        from robosuite.renderers.viewer import mjviewer_renderer as mvr
    except Exception:
        return
    if getattr(mvr.MjviewerRenderer.update, "_system12_mjvopt_patched", False):
        return
    _orig = mvr.MjviewerRenderer.update

    def update(self):  # noqa: ANN001
        _orig(self)
        if self.viewer is None:
            return
        try:
            import mujoco
            mujoco.mjv_defaultOption(self.viewer.opt)
            self.viewer.opt.geomgroup[0] = 0
        except Exception:
            pass

    update._system12_mjvopt_patched = True  # type: ignore[attr-defined]
    mvr.MjviewerRenderer.update = update  # type: ignore[method-assign]


def _table_surface_z_world(env) -> float:
    import mujoco
    env.sim.forward()
    m = env.sim.model._model
    d = env.sim.data._data
    for sid in range(env.sim.model.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name and name.endswith("table_top"):
            return float(d.site_xpos[sid, 2])
    try:
        return float(env.model.mujoco_arena.table_top_abs[2])
    except Exception:
        pass
    try:
        return float(env.model.mujoco_arena.table_offset[2])
    except Exception:
        return 0.8


def _inject_static_scene_geoms(
    xml_string: str,
    *,
    keepout_xy: tuple[float, float],
    table_surface_z: float,
    keepout_radius: float,
    keepout_half_extents: tuple[float, float, float] = (0.065, 0.010, 0.004),
    goal_xy: tuple[float, float] | None = None,
    goal_radius: float | None = None,
    descend_radius: float | None = None,
    tray_xy: tuple[float, float] | None = None,
    tray_half_extents_xy: tuple[float, float] = (0.085, 0.065),
    tray_wall_height: float = 0.018,
    tray_wall_thickness: float = 0.006,
    large_block_xy: tuple[float, float] | None = None,
    large_block_size: tuple[float, float, float] = (0.060, 0.060, 0.060),
    large_block_collision: bool = False,
) -> str:
    root = ET.fromstring(xml_string)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Could not find <worldbody> in robosuite MJCF.")
    for elem in list(worldbody.findall("body")):
        if (elem.get("name") or "") in {
            "system12_keepout_body",
            "system12_tray_body",
            "system12_large_block_body",
        }:
            worldbody.remove(elem)
    for elem in list(worldbody.findall("geom")):
        name = elem.get("name") or ""
        if (
            name.startswith("system12_keepout_")
            or name.startswith("system12_goal_")
            or name.startswith("system12_descend_")
            or name.startswith("system12_tray_")
            or name.startswith("system12_large_block_")
        ):
            worldbody.remove(elem)

    hx, hy, hz = [float(v) for v in keepout_half_extents]
    rx, ry = [float(v) for v in keepout_xy]
    keepout_body = ET.SubElement(worldbody, "body", name="system12_keepout_body", pos=_format_vec([rx, ry, table_surface_z + hz + 0.002]))
    for grp in ("0", "1"):
        ET.SubElement(keepout_body, "geom", name=f"system12_keepout_knife_blade_g{grp}", type="box", size=_format_vec([hx, hy, hz]), rgba=_format_vec([0.78, 0.78, 0.80, 1.0]), contype="0", conaffinity="0", group=grp)
        ET.SubElement(keepout_body, "geom", name=f"system12_keepout_knife_handle_g{grp}", type="box", pos=_format_vec([-0.050, 0.0, 0.0]), size=_format_vec([0.018, 0.012, hz * 1.15]), rgba=_format_vec([0.18, 0.18, 0.20, 1.0]), contype="0", conaffinity="0", group=grp)
        if float(keepout_radius) > 0.0:
            ET.SubElement(worldbody, "geom", name=f"system12_keepout_red_region_g{grp}", type="cylinder", pos=_format_vec([rx, ry, table_surface_z + 0.004]), size=_format_vec([keepout_radius, 0.003]), rgba=_format_vec([0.95, 0.15, 0.15, 0.20]), contype="0", conaffinity="0", group=grp)
    if goal_xy is not None and goal_radius is not None:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        gr = float(goal_radius)
        for grp in ("0", "1"):
            ET.SubElement(worldbody, "geom", name=f"system12_goal_blue_region_g{grp}", type="cylinder", pos=_format_vec([gx, gy, table_surface_z + 0.0055]), size=_format_vec([gr, 0.002]), rgba=_format_vec([0.12, 0.45, 0.92, 0.30]), contype="0", conaffinity="0", group=grp)
            if descend_radius is not None and float(descend_radius) > gr:
                ET.SubElement(worldbody, "geom", name=f"system12_descend_blue_region_g{grp}", type="cylinder", pos=_format_vec([gx, gy, table_surface_z + 0.0045]), size=_format_vec([float(descend_radius), 0.0015]), rgba=_format_vec([0.12, 0.45, 0.92, 0.12]), contype="0", conaffinity="0", group=grp)
    if tray_xy is not None:
        tx, ty = float(tray_xy[0]), float(tray_xy[1])
        hx_tray, hy_tray = float(tray_half_extents_xy[0]), float(tray_half_extents_xy[1])
        wall_h = float(tray_wall_height)
        wall_t = float(tray_wall_thickness)
        tray_body = ET.SubElement(worldbody, "body", name="system12_tray_body", pos=_format_vec([tx, ty, table_surface_z + 0.006]))
        for grp in ("0", "1"):
            ET.SubElement(tray_body, "geom", name=f"system12_tray_bottom_g{grp}", type="box", size=_format_vec([hx_tray, hy_tray, 0.003]), rgba=_format_vec([0.10, 0.35, 0.90, 0.22]), contype="0", conaffinity="0", group=grp)
            ET.SubElement(tray_body, "geom", name=f"system12_tray_front_g{grp}", type="box", pos=_format_vec([0.0, hy_tray + wall_t, wall_h]), size=_format_vec([hx_tray + wall_t, wall_t, wall_h]), rgba=_format_vec([0.05, 0.18, 0.55, 0.55]), contype="0", conaffinity="0", group=grp)
            ET.SubElement(tray_body, "geom", name=f"system12_tray_back_g{grp}", type="box", pos=_format_vec([0.0, -hy_tray - wall_t, wall_h]), size=_format_vec([hx_tray + wall_t, wall_t, wall_h]), rgba=_format_vec([0.05, 0.18, 0.55, 0.55]), contype="0", conaffinity="0", group=grp)
            ET.SubElement(tray_body, "geom", name=f"system12_tray_left_g{grp}", type="box", pos=_format_vec([-hx_tray - wall_t, 0.0, wall_h]), size=_format_vec([wall_t, hy_tray + wall_t, wall_h]), rgba=_format_vec([0.05, 0.18, 0.55, 0.55]), contype="0", conaffinity="0", group=grp)
            ET.SubElement(tray_body, "geom", name=f"system12_tray_right_g{grp}", type="box", pos=_format_vec([hx_tray + wall_t, 0.0, wall_h]), size=_format_vec([wall_t, hy_tray + wall_t, wall_h]), rgba=_format_vec([0.05, 0.18, 0.55, 0.55]), contype="0", conaffinity="0", group=grp)

    if large_block_xy is not None:
        bx, by = float(large_block_xy[0]), float(large_block_xy[1])
        sx, sy, sz = [float(v) for v in large_block_size]
        contype = "1" if bool(large_block_collision) else "0"
        conaffinity = "1" if bool(large_block_collision) else "0"
        block_body = ET.SubElement(worldbody, "body", name="system12_large_block_body", pos=_format_vec([bx, by, table_surface_z + sz]))
        for grp in ("0", "1"):
            ET.SubElement(block_body, "geom", name=f"system12_large_block_obstacle_g{grp}", type="box", size=_format_vec([sx, sy, sz]), rgba=_format_vec([0.55, 0.35, 0.12, 0.85]), contype=contype, conaffinity=conaffinity, group=grp)
    return ET.tostring(root, encoding="unicode")


def _controller_configs(suite, refactor_composite_controller_config):
    arm_cfg = suite.load_part_controller_config(default_controller="OSC_POSE")
    arm_cfg["control_delta"] = True
    arm_cfg["uncouple_pos_ori"] = True
    arm_cfg["input_max"] = 1.0
    arm_cfg["input_min"] = -1.0
    arm_cfg["output_max"] = [0.05, 0.05, 0.05, 0.20, 0.20, 0.20]
    arm_cfg["output_min"] = [-0.05, -0.05, -0.05, -0.20, -0.20, -0.20]
    return refactor_composite_controller_config(arm_cfg, "Panda", ["right"])


def make_lift_env(seed: int = 0, horizon: int = 220, control_freq: int = 20, has_renderer: bool = False, render_camera: str = "agentview", *, keepout_xy: tuple[float, float] = (0.12, 0.11), keepout_radius: float = 0.065, keepout_half_extents: tuple[float, float, float] = (0.065, 0.010, 0.004), goal_xy: tuple[float, float] | None = (0.22, 0.22), goal_radius: float | None = 0.055, descend_radius: float | None = 0.075, tray_xy: tuple[float, float] | None = None, tray_half_extents_xy: tuple[float, float] = (0.085, 0.065), tray_wall_height: float = 0.018, tray_wall_thickness: float = 0.006, large_block_xy: tuple[float, float] | None = None, large_block_size: tuple[float, float, float] = (0.060, 0.060, 0.060), large_block_collision: bool = False, has_offscreen_renderer: bool = False, use_camera_obs: bool = False, camera_names: str | list[str] | tuple[str, ...] | None = None, camera_heights: int | list[int] = 512, camera_widths: int | list[int] = 512):
    if has_offscreen_renderer and sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "cgl")
    suite, refactor_composite_controller_config = _try_import_robosuite()
    if has_renderer:
        _patch_mjviewer_visual_options()
    if camera_names is None:
        camera_names = render_camera
    env = suite.make(env_name="Lift", robots="Panda", controller_configs=_controller_configs(suite, refactor_composite_controller_config), has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer, render_camera=render_camera, use_camera_obs=use_camera_obs, use_object_obs=True, reward_shaping=True, control_freq=control_freq, horizon=horizon, ignore_done=False, hard_reset=False, lite_physics=True, seed=seed, camera_names=camera_names, camera_heights=camera_heights, camera_widths=camera_widths)
    env.reset()
    init_state = np.array(env.sim.get_state().flatten())
    table_surface_z = _table_surface_z_world(env)
    setattr(env, "_system12_table_surface_z", float(table_surface_z))
    xml_string = _inject_static_scene_geoms(
        env.sim.model.get_xml(),
        keepout_xy=keepout_xy,
        table_surface_z=table_surface_z,
        keepout_radius=keepout_radius,
        keepout_half_extents=keepout_half_extents,
        goal_xy=goal_xy,
        goal_radius=goal_radius,
        descend_radius=descend_radius,
        tray_xy=tray_xy,
        tray_half_extents_xy=tray_half_extents_xy,
        tray_wall_height=tray_wall_height,
        tray_wall_thickness=tray_wall_thickness,
        large_block_xy=large_block_xy,
        large_block_size=large_block_size,
        large_block_collision=large_block_collision,
    )
    env.reset_from_xml_string(xml_string)
    env.sim.reset(); env.sim.set_state_from_flattened(init_state); env.sim.forward()
    return env


def make_triple_place_lift_env(seed: int = 0, horizon: int = 220, control_freq: int = 20, has_renderer: bool = False, render_camera: str = "agentview", *, keepout_xy: tuple[float, float] = (0.12, 0.11), keepout_radius: float = 0.065, keepout_half_extents: tuple[float, float, float] = (0.065, 0.010, 0.004), goal_xy: tuple[float, float] | None = (0.22, 0.22), goal_radius: float | None = 0.055, descend_radius: float | None = 0.075, tray_xy: tuple[float, float] | None = None, tray_half_extents_xy: tuple[float, float] = (0.085, 0.065), tray_wall_height: float = 0.018, tray_wall_thickness: float = 0.006, large_block_xy: tuple[float, float] | None = None, large_block_size: tuple[float, float, float] = (0.060, 0.060, 0.060), large_block_collision: bool = False, has_offscreen_renderer: bool = False, use_camera_obs: bool = False, camera_names: str | list[str] | tuple[str, ...] | None = None, camera_heights: int | list[int] = 512, camera_widths: int | list[int] = 512):
    import step.system_one.triple_lift_env  # noqa: F401
    if has_offscreen_renderer and sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "cgl")
    suite, refactor_composite_controller_config = _try_import_robosuite()
    if has_renderer:
        _patch_mjviewer_visual_options()
    if camera_names is None:
        camera_names = render_camera
    env = suite.make(env_name="TriplePlaceLift", robots="Panda", controller_configs=_controller_configs(suite, refactor_composite_controller_config), has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer, render_camera=render_camera, use_camera_obs=use_camera_obs, use_object_obs=True, reward_shaping=False, control_freq=control_freq, horizon=horizon, ignore_done=False, hard_reset=False, lite_physics=True, seed=seed, camera_names=camera_names, camera_heights=camera_heights, camera_widths=camera_widths)
    env.reset()
    init_state = np.array(env.sim.get_state().flatten())
    table_surface_z = _table_surface_z_world(env)
    setattr(env, "_system12_table_surface_z", float(table_surface_z))
    setattr(env, "_system12_active_block_idx", 1)
    xml_string = _inject_static_scene_geoms(
        env.sim.model.get_xml(),
        keepout_xy=keepout_xy,
        table_surface_z=table_surface_z,
        keepout_radius=keepout_radius,
        keepout_half_extents=keepout_half_extents,
        goal_xy=goal_xy,
        goal_radius=goal_radius,
        descend_radius=descend_radius,
        tray_xy=tray_xy,
        tray_half_extents_xy=tray_half_extents_xy,
        tray_wall_height=tray_wall_height,
        tray_wall_thickness=tray_wall_thickness,
        large_block_xy=large_block_xy,
        large_block_size=large_block_size,
        large_block_collision=large_block_collision,
    )
    env.reset_from_xml_string(xml_string)
    env.sim.reset(); env.sim.set_state_from_flattened(init_state); env.sim.forward()
    setattr(env, "_system12_active_block_idx", 1)
    return env


def action_spec(env) -> tuple[np.ndarray, np.ndarray]:
    low, high = env.action_spec
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def get_gripper_dim(env) -> int:
    robot = env.robots[0]
    try:
        return int(robot.gripper["right"].dof)
    except Exception:
        try:
            return int(robot.gripper.dof)
        except Exception:
            return max(int(robot.action_dim) - 6, 1)


def find_obs_key(obs: OrderedDict[str, Any] | dict[str, Any], candidates: list[str], contains: list[str] | None = None):
    for key in candidates:
        if key in obs:
            return key
    if contains is not None:
        for key in obs.keys():
            lower = key.lower()
            if all(token in lower for token in contains):
                return key
    raise KeyError(f"Could not find observation key from candidates={candidates} contains={contains}; got {list(obs.keys())}")


def obs_array(obs: OrderedDict[str, Any] | dict[str, Any], candidates: list[str], contains: list[str] | None = None) -> np.ndarray:
    key = find_obs_key(obs, candidates=candidates, contains=contains)
    return np.asarray(obs[key], dtype=np.float32).copy()
