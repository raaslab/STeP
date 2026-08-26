"""Robosuite backend for the System 1 manipulation environment.

This is the concrete subclass of :class:`BaseManipulationEnv` that drives a
robosuite Lift / TriplePlaceLift scene. All shared stage-machine / MPC logic
lives in ``base_env.py``; this file only deals with robosuite-specific I/O:
making the environment, injecting goal/keepout MJCF, reading EE/object pose,
checking grasps, and rendering frames with MuJoCo's native renderer.

The class is exported as both ``SimManipulationEnv`` and the legacy name
``CubeGoalLiftEnv`` so existing imports (``from step.system_one.env import
CubeGoalLiftEnv``) keep working.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .base_env import BaseManipulationEnv
from .robosuite_utils import (
    action_spec,
    get_gripper_dim,
    make_lift_env,
    make_triple_place_lift_env,
    obs_array,
)


class SimManipulationEnv(BaseManipulationEnv):
    """Robosuite Lift / TriplePlaceLift backend with a configurable goal."""

    def __init__(
        self,
        mode: str = "top",
        seed: int = 0,
        horizon: int = 1200,
        control_freq: int = 20,
        has_renderer: bool = False,
        render_camera: str = "agentview",
        open_gripper_cmd: float = -1.0,
        close_gripper_cmd: float = 1.0,
        nominal_ori_gain: float = 3.6,
        keepout_xy: Tuple[float, float] = (0.12, 0.11),
        keepout_radius: float = 0.065,
        object_keepout_radius: float = 0.05,
        goal_xy: Tuple[float, float] = (0.22, 0.22),
        goal_radius: float = 0.055,
        goal_z: float | None = None,
        descend_radius: float = 0.075,
        carry_height_above_table: float = 0.17,
        retreat_height_above_table: float = 0.28,
        close_steps: int = 14,
        num_blocks: int = 3,
        active_object_name: str = "green_cube",
        goal_reference_name: str | None = None,
        workspace_bounds: dict[str, Any] | None = None,
        sdf_constraints: list[Any] | None = None,
        tray_xy: tuple[float, float] | None = None,
        tray_half_extents_xy: tuple[float, float] = (0.085, 0.065),
        tray_wall_height: float = 0.018,
        tray_wall_thickness: float = 0.006,
        large_block_xy: tuple[float, float] | None = None,
        large_block_size: tuple[float, float, float] = (0.060, 0.060, 0.060),
        large_block_collision: bool = False,
        mpc_timing_enabled: bool = True,
        mpc_timing_early_weight: float = 220.0,
        mpc_timing_deadline_weight: float = 55.0,
        mpc_timing_window_weight: float = 10.0,
        mpc_timing_early_hold_weight: float = 260.0,
    ) -> None:
        if num_blocks not in (1, 3):
            raise ValueError("num_blocks must be 1 or 3")
        self.has_renderer = bool(has_renderer)
        self.num_blocks = int(num_blocks)
        _mk_env = make_triple_place_lift_env if self.num_blocks == 3 else make_lift_env
        self.env = _mk_env(
            seed=seed,
            horizon=horizon,
            control_freq=control_freq,
            has_renderer=has_renderer,
            render_camera=render_camera,
            keepout_xy=(float(keepout_xy[0]), float(keepout_xy[1])),
            keepout_radius=float(keepout_radius),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            goal_radius=float(goal_radius),
            descend_radius=float(descend_radius),
            tray_xy=tray_xy,
            tray_half_extents_xy=tray_half_extents_xy,
            tray_wall_height=float(tray_wall_height),
            tray_wall_thickness=float(tray_wall_thickness),
            large_block_xy=large_block_xy,
            large_block_size=large_block_size,
            large_block_collision=large_block_collision,
        )
        low, high = action_spec(self.env)
        action_dim_underlying = int(low.shape[0])
        gripper_dim = int(get_gripper_dim(self.env))

        table_z = 0.8
        try:
            table_z = float(self.env.model.mujoco_arena.table_offset[2])
        except Exception:
            pass

        super().__init__(
            mode=mode,
            seed=seed,
            nominal_ori_gain=nominal_ori_gain,
            keepout_xy=keepout_xy,
            keepout_radius=keepout_radius,
            object_keepout_radius=object_keepout_radius,
            goal_xy=goal_xy,
            goal_radius=goal_radius,
            goal_z=goal_z,
            descend_radius=descend_radius,
            carry_height_above_table=carry_height_above_table,
            retreat_height_above_table=retreat_height_above_table,
            close_steps=close_steps,
            active_object_name=active_object_name,
            goal_reference_name=goal_reference_name,
            workspace_bounds=workspace_bounds,
            sdf_constraints=sdf_constraints,
            mpc_timing_enabled=mpc_timing_enabled,
            mpc_timing_early_weight=mpc_timing_early_weight,
            mpc_timing_deadline_weight=mpc_timing_deadline_weight,
            mpc_timing_window_weight=mpc_timing_window_weight,
            mpc_timing_early_hold_weight=mpc_timing_early_hold_weight,
            open_gripper_cmd=open_gripper_cmd,
            close_gripper_cmd=close_gripper_cmd,
            action_dim=action_dim_underlying,
            gripper_dim=gripper_dim,
            action_low=low.astype(np.float32),
            action_high=high.astype(np.float32),
            table_z=table_z,
        )

    # ------------------------------------------------------------ backends

    def _reset_backend(self) -> Dict[str, Any]:
        return self.env.reset()

    def _step_backend(self, full_action: np.ndarray):
        return self.env.step(full_action)

    # ----------------------------------------------------- observation api

    def _cube_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        return obs_array(obs, ["cube_pos", "can_pos"], contains=["cube", "pos"])

    def _eef_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        try:
            return obs_array(
                obs,
                ["robot0_eef_pos", "robot0_right_eef_pos"],
                contains=["eef", "pos"],
            )
        except Exception:
            sim = self.env.sim
            robot = self.env.robots[0]
            for attr in ["eef_site_id", "eef_site_ids"]:
                if hasattr(robot, attr):
                    site = getattr(robot, attr)
                    if isinstance(site, dict):
                        site = next(iter(site.values()))
                    elif isinstance(site, (list, tuple)):
                        site = site[0]
                    return np.asarray(
                        sim.data.site_xpos[site], dtype=np.float32
                    ).copy()
            raise

    def _eef_rotmat(self, obs: Dict[str, Any]) -> np.ndarray:
        from .orientation_utils import orthonormalize

        sim = self.env.sim
        robot = self.env.robots[0]
        for attr in ["eef_site_id", "eef_site_ids"]:
            if hasattr(robot, attr):
                site = getattr(robot, attr)
                if isinstance(site, dict):
                    site = next(iter(site.values()))
                elif isinstance(site, (list, tuple)):
                    site = site[0]
                xmat = np.asarray(
                    sim.data.site_xmat[site], dtype=np.float32
                ).reshape(3, 3)
                return orthonormalize(xmat)
        raise RuntimeError("Could not recover end-effector orientation.")

    def _is_grasped(self) -> bool:
        try:
            return bool(
                self.env._check_grasp(
                    gripper=self.env.robots[0].gripper,
                    object_geoms=self._active_pick_object(),
                )
            )
        except Exception:
            return False

    # ---------------------------------------------------- active object api

    def set_active_object(self, object_name: str) -> None:
        super().set_active_object(object_name)
        if self.num_blocks == 1:
            return
        mapping = {"red_cube": 0, "green_cube": 1, "blue_cube": 2}
        idx = mapping.get(self.active_object_name, 1)
        setattr(self.env, "_system12_active_block_idx", idx)
        setattr(self.env, "_stl_active_block_idx", idx)
        cubes = getattr(self.env, "cubes", None)
        if cubes is not None:
            try:
                self.env.cube = cubes[idx]
                self.env.cube_body_id = self.env.cube_body_ids[idx]
            except Exception:
                pass

    def _active_pick_object(self):
        cubes = getattr(self.env, "cubes", None)
        if cubes is None:
            return getattr(self.env, "cube", None)
        idx = int(
            getattr(
                self.env,
                "_stl_active_block_idx",
                getattr(self.env, "_system12_active_block_idx", 1),
            )
        )
        idx = max(0, min(idx, len(cubes) - 1))
        return cubes[idx]

    def _object_index_by_name(self, object_name: str | None) -> int | None:
        if object_name is None:
            return None
        mapping = {"red_cube": 0, "green_cube": 1, "blue_cube": 2}
        return mapping.get(str(object_name))

    def _object_pos_by_name(self, object_name: str | None) -> np.ndarray | None:
        # First try the in-sim cube ids (works for the legacy three-cube scene),
        # then fall back to the generic scene_objects map.
        idx = self._object_index_by_name(object_name)
        body_ids = getattr(self.env, "cube_body_ids", None)
        if idx is not None and body_ids is not None:
            idx = max(0, min(idx, len(body_ids) - 1))
            return np.asarray(
                self.env.sim.data.body_xpos[body_ids[idx]], dtype=np.float32
            ).copy()
        return super()._object_pos_by_name(object_name)

    # ------------------------------------------------------------- markers

    def _add_marker(
        self,
        *,
        pos: np.ndarray,
        type_id: int,
        size: np.ndarray,
        rgba: np.ndarray,
        label: str,
    ) -> None:
        if not self.has_renderer:
            return
        try:
            viewer = getattr(self.env, "viewer", None)
            if viewer is None:
                return
            marker_api = getattr(viewer, "add_marker", None)
            if marker_api is None and hasattr(viewer, "viewer"):
                marker_api = getattr(viewer.viewer, "add_marker", None)
            if marker_api is None:
                return
            marker_api(
                pos=pos,
                type=type_id,
                size=size,
                rgba=rgba,
                mat=np.eye(3).flatten(),
                label=label,
            )
        except Exception:
            pass

    def _draw_debug_markers(self) -> None:
        if not self.has_renderer:
            return
        z0 = self.table_z + 0.003
        self._add_marker(
            pos=np.array([self.goal_xy[0], self.goal_xy[1], z0]),
            type_id=5,
            size=np.array([self.goal_radius, self.goal_radius, 0.003]),
            rgba=np.array([0.1, 0.3, 0.95, 0.22]),
            label="goal region",
        )
        self._add_marker(
            pos=np.array([self.goal_xy[0], self.goal_xy[1], z0 + 0.003]),
            type_id=5,
            size=np.array([self.descend_radius, self.descend_radius, 0.002]),
            rgba=np.array([0.1, 0.3, 0.95, 0.10]),
            label="descend region",
        )
        z = self.table_z + float(self.knife_visual_half_extents[2]) + 0.002
        self._add_marker(
            pos=np.array([self.keepout_xy[0], self.keepout_xy[1], z]),
            type_id=6,
            size=self.knife_visual_half_extents,
            rgba=np.array([0.75, 0.75, 0.78, 0.95]),
            label="support object",
        )
        self._add_marker(
            pos=np.array([self.keepout_xy[0], self.keepout_xy[1], z0]),
            type_id=5,
            size=np.array([self.keepout_radius, self.keepout_radius, 0.003]),
            rgba=np.array([0.95, 0.15, 0.15, 0.22]),
            label="obstacle region",
        )

    def _on_after_reset(self) -> None:
        self._draw_debug_markers()
        if self.has_renderer:
            try:
                self.env.render()
            except Exception:
                pass

    def _on_after_step(self) -> None:
        self._draw_debug_markers()

    # --------------------------------------------------------- rendering

    @staticmethod
    def capture_initial_scene_image(
        save_path: str,
        seed: int = 0,
        camera_name: str = "agentview",
        width: int = 512,
        height: int = 512,
        goal_xy: tuple[float, float] = (0.22, 0.22),
        goal_radius: float = 0.055,
        descend_radius: float = 0.075,
        keepout_xy: tuple[float, float] = (0.12, 0.11),
        keepout_radius: float = 0.065,
        num_blocks: int = 3,
    ) -> str:
        """Capture an initial scene frame for System 2 without using GLFW offscreen."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.image as mpimg
        import mujoco

        _mk_env = make_triple_place_lift_env if int(num_blocks) == 3 else make_lift_env
        raw_env = _mk_env(
            seed=seed,
            has_renderer=False,
            render_camera=camera_name,
            keepout_xy=(float(keepout_xy[0]), float(keepout_xy[1])),
            keepout_radius=float(keepout_radius),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            goal_radius=float(goal_radius),
            descend_radius=float(descend_radius),
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        renderer = None
        try:
            raw_env.reset()
            model = raw_env.sim.model._model
            max_height = getattr(
                getattr(getattr(model, "vis", None), "global_", None), "offheight", None
            )
            max_width = getattr(
                getattr(getattr(model, "vis", None), "global_", None), "offwidth", None
            )
            if max_height is None:
                max_height = 480
            if max_width is None:
                max_width = 640
            render_height = int(min(height, max_height))
            render_width = int(min(width, max_width))
            renderer = mujoco.Renderer(
                model, height=render_height, width=render_width
            )
            renderer.update_scene(raw_env.sim.data._data, camera=camera_name)
            image = renderer.render()
            save_path = str(save_path)
            mpimg.imsave(save_path, image)
            return save_path
        finally:
            try:
                if renderer is not None:
                    renderer.close()
            except Exception:
                pass
            raw_env.close()

    def render_rgb_frame(
        self,
        camera_name: str = "agentview",
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        import mujoco

        model = self.env.sim.model._model
        data = self.env.sim.data._data
        max_height = getattr(
            getattr(getattr(model, "vis", None), "global_", None), "offheight", None
        )
        max_width = getattr(
            getattr(getattr(model, "vis", None), "global_", None), "offwidth", None
        )
        if max_height is None:
            max_height = 480
        if max_width is None:
            max_width = 640
        render_height = int(min(int(height), int(max_height)))
        render_width = int(min(int(width), int(max_width)))
        shape = (render_height, render_width)
        if self._video_renderer is None or self._video_renderer_shape != shape:
            if self._video_renderer is not None:
                try:
                    self._video_renderer.close()
                except Exception:
                    pass
            self._video_renderer = mujoco.Renderer(
                model, height=render_height, width=render_width
            )
            self._video_renderer_shape = shape
        self._video_renderer.update_scene(data, camera=camera_name)
        frame = self._video_renderer.render()
        return np.asarray(frame, dtype=np.uint8).copy()

    def close(self):
        try:
            super().close()
        finally:
            return self.env.close()


# ----------------------------------------------------------------------------
# Backward-compat alias. Existing code does ``from step.system_one.env import
# CubeGoalLiftEnv``; we re-export from this module for clarity, and ``env.py``
# continues to forward this name.
CubeGoalLiftEnv = SimManipulationEnv
