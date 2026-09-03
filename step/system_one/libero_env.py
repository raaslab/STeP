"""LIBERO backend for the STL-MPC manipulation executor.

The adapter intentionally implements the same BaseManipulationEnv hooks used by
robosuite and real hardware.  LIBERO-specific details stay here: task loading,
initial-state selection, observation parsing, object pose lookup, rendering, and
best-effort grasp checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os

import numpy as np

from .base_env import BaseManipulationEnv
from .geometry_sdf import parse_sdf_constraints
from .libero_scene import (
    apply_libero_experiment_geometry,
    get_libero_init_state,
    infer_objects_from_mujoco,
    infer_objects_from_obs,
    load_libero_task_from_scene_context,
    merge_drawer_fixtures_into_scene_context,
    normalize_entity_name,
    update_scene_context_from_objects,
)


def _libero_obs_array(obs: dict[str, Any], candidates: list[str], contains: list[str] | None = None) -> np.ndarray:
    for key in candidates:
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32).reshape(-1).copy()
    contains = contains or []
    for key, value in obs.items():
        low = str(key).lower()
        if all(c.lower() in low for c in contains):
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 3:
                return arr.copy()
    raise KeyError(f"Could not find observation array; candidates={candidates}, contains={contains}")


class LiberoManipulationEnv(BaseManipulationEnv):
    """Run the shared STL-MPC skill executor inside a LIBERO task."""

    def __init__(
        self,
        *,
        scene_context: dict[str, Any],
        mode: str = "top",
        horizon: int = 600,
        has_renderer: bool = False,
        camera_name: str = "agentview",
        camera_height: int = 128,
        camera_width: int = 128,
        active_object_name: str = "object",
        goal_reference_name: str | None = None,
        goal_xy: tuple[float, float] = (0.0, 0.0),
        goal_radius: float = 0.055,
        goal_z: float | None = None,
        descend_radius: float = 0.075,
        keepout_xy: tuple[float, float] = (0.0, 0.0),
        keepout_radius: float = 0.0,
        carry_height_above_table: float = 0.17,
        retreat_height_above_table: float = 0.28,
        mpc_timing_enabled: bool = True,
        mpc_timing_early_weight: float = 220.0,
        mpc_timing_deadline_weight: float = 55.0,
        mpc_timing_window_weight: float = 10.0,
        mpc_timing_early_hold_weight: float = 260.0,
    ) -> None:
        self.scene_context = apply_libero_experiment_geometry(scene_context)
        self.platform_cfg = self.scene_context.get("platform", {})
        self.libero_cfg = self.platform_cfg.get("libero", {}) or {}
        self.has_renderer = bool(has_renderer)
        self.camera_name = str(self.libero_cfg.get("camera_name", camera_name))
        self.camera_height = int(self.libero_cfg.get("camera_height", self.libero_cfg.get("camera_heights", camera_height)))
        self.camera_width = int(self.libero_cfg.get("camera_width", self.libero_cfg.get("camera_widths", camera_width)))
        self.horizon = int(self.libero_cfg.get("horizon", horizon))
        self._task_info = load_libero_task_from_scene_context(self.scene_context)
        if not self._task_info.bddl_file:
            raise ValueError(
                "LIBERO backend needs platform.libero.bddl_file or platform.libero.suite + task_id."
            )

        from step.system_one.libero_scene import _ensure_libero_on_path
        _ensure_libero_on_path()
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore

        env_args = {
            "bddl_file_name": self._task_info.bddl_file,
            "camera_heights": self.camera_height,
            "camera_widths": self.camera_width,
        }
        camera_names = self.libero_cfg.get("camera_names")
        if camera_names is not None:
            env_args["camera_names"] = camera_names
        self.env = OffScreenRenderEnv(**env_args)
        try:
            self.env.seed(int(self.libero_cfg.get("seed", 0)))
        except Exception:
            pass

        self._init_state = get_libero_init_state(self.scene_context)
        self._last_obs_objects: dict[str, dict[str, Any]] = {}
        self._last_gripper_cmd: float = -1.0
        self._grasped_latch: bool = False
        self._active_obj_at_grasp: np.ndarray | None = None

        low = -np.ones(7, dtype=np.float32)
        high = np.ones(7, dtype=np.float32)
        try:
            spec_env = getattr(self.env, "env", self.env)
            spec = spec_env.action_spec
            if callable(spec):
                low, high = spec()
            elif isinstance(spec, (tuple, list)) and len(spec) == 2:
                low, high = spec
            low = np.asarray(low, dtype=np.float32).reshape(-1)
            high = np.asarray(high, dtype=np.float32).reshape(-1)
        except Exception:
            pass
        if low.size < 7:
            low = -np.ones(7, dtype=np.float32)
            high = np.ones(7, dtype=np.float32)

        # Tabletop domains (e.g. libero_spatial) expose table_offset on the arena;
        # floor domains (e.g. libero_object's EmptyArena) don't have a table at
        # all, so fall back to floor level (z=0) rather than the tabletop height.
        table_z = float(self.platform_cfg.get("table_z", 0.0))
        try:
            table_z = float(self.env.env.model.mujoco_arena.table_offset[2])
        except Exception:
            pass

        exec_proxy = self.scene_context.get("execution_proxy", {}) or {}

        # configs/libero_template.json's workspace_bounds.z is tuned for the
        # kitchen-tabletop domain (table_z ~0.8). Floor domains (libero_object,
        # etc.) have table_z ~0.0, so that absolute z-band would clip every
        # pick/place target up to ~0.78 and the arm would never descend to the
        # object. Re-anchor the z band to the detected table height whenever
        # the configured band doesn't actually bracket it.
        workspace_bounds = self.platform_cfg.get("workspace_bounds")
        if workspace_bounds:
            z_range = workspace_bounds.get("z")
            if z_range and not (float(z_range[0]) <= table_z <= float(z_range[1])):
                workspace_bounds = dict(workspace_bounds)
                workspace_bounds["z"] = [table_z - 0.05, table_z + 0.55]

        super().__init__(
            mode=mode,
            active_object_name=active_object_name,
            goal_reference_name=goal_reference_name,
            goal_xy=goal_xy,
            goal_radius=goal_radius,
            goal_z=goal_z,
            descend_radius=descend_radius,
            keepout_xy=keepout_xy,
            keepout_radius=keepout_radius,
            carry_height_above_table=carry_height_above_table,
            retreat_height_above_table=retreat_height_above_table,
            workspace_bounds=workspace_bounds,
            sdf_constraints=parse_sdf_constraints(self.scene_context),
            mpc_timing_enabled=mpc_timing_enabled,
            mpc_timing_early_weight=mpc_timing_early_weight,
            mpc_timing_deadline_weight=mpc_timing_deadline_weight,
            mpc_timing_window_weight=mpc_timing_window_weight,
            mpc_timing_early_hold_weight=mpc_timing_early_hold_weight,
            action_dim=int(low.shape[0]),
            gripper_dim=1,
            action_low=low.astype(np.float32),
            action_high=high.astype(np.float32),
            table_z=table_z,
        )
        # Allow the scene template to override top-grasp approach height.
        # LIBERO bowls/cups need a higher hover than the default 6 cm so the
        # open gripper clears the rim during lateral approach.
        if "top_grasp_hover_z" in exec_proxy:
            self.top_grasp_hover_z = float(exec_proxy["top_grasp_hover_z"])
        if "top_grasp_close_xy_tol" in exec_proxy:
            self.top_grasp_close_xy_tol = float(exec_proxy["top_grasp_close_xy_tol"])
        if "top_grasp_close_z_tol" in exec_proxy:
            self.top_grasp_close_z_tol = float(exec_proxy["top_grasp_close_z_tol"])

    # ------------------------------------------------------------ backends

    # Number of zero-action steps taken after reset to let the LIBERO physics
    # settle before the first video frame is captured.  LIBERO's own evaluation
    # harness typically uses 10 steps; 15 gives a clean settled scene without
    # meaningfully affecting the episode horizon.
    SETTLING_STEPS: int = 15

    def _reset_backend(self) -> Dict[str, Any]:
        obs = self.env.reset()
        if self._init_state is not None:
            try:
                obs = self.env.set_init_state(self._init_state)
            except Exception:
                try:
                    self.env.set_init_state(self._init_state)
                    obs = self.env._get_observations()
                except Exception:
                    pass
        # Run a handful of zero-action steps so the objects fully settle onto
        # the table before the first video frame is captured.
        zero_action = np.zeros(7, dtype=np.float32)
        for _ in range(self.SETTLING_STEPS):
            try:
                result = self.env.step(zero_action)
                if isinstance(result, tuple):
                    obs = result[0]
                else:
                    obs = result
            except Exception:
                break
        self._grasped_latch = False
        self._active_obj_at_grasp = None
        self._refresh_scene_objects(obs)
        if self.scene_objects:
            self._grow_workspace_bounds_to_include(np.stack(list(self.scene_objects.values())))
        return obs

    def _step_backend(self, full_action: np.ndarray):
        action = np.asarray(full_action, dtype=np.float32).reshape(-1)
        if action.size > 0:
            self._last_gripper_cmd = float(action[-1])
        obs, reward, done, info = self.env.step(action)
        self._refresh_scene_objects(obs)
        return obs, reward, done, info

    # ----------------------------------------------------- observation api

    def _refresh_scene_objects(self, obs: Dict[str, Any]) -> None:
        objects = infer_objects_from_obs(obs)
        if not objects:
            objects = infer_objects_from_mujoco(self.env)
        self._last_obs_objects = objects
        self.scene_objects.clear()
        for name, obj in objects.items():
            xyz = obj.get("xyz")
            if xyz is not None:
                self.scene_objects[normalize_entity_name(name)] = np.asarray(xyz, dtype=np.float32)

    def _eef_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        return _libero_obs_array(
            obs,
            ["robot0_eef_pos", "robot0_right_eef_pos", "eef_pos"],
            contains=["eef", "pos"],
        )[:3].astype(np.float32)

    def _eef_rotmat(self, obs: Dict[str, Any]) -> np.ndarray:
        try:
            mat = _libero_obs_array(
                obs,
                ["robot0_eef_rotmat", "robot0_right_eef_rotmat", "eef_rotmat"],
                contains=["eef", "rot"],
            )
            if mat.size >= 9:
                return mat[:9].reshape(3, 3).astype(np.float32)
        except Exception:
            pass
        # Fallback: read MuJoCo site orientation.
        try:
            sim = self.env.env.sim
            robot = self.env.env.robots[0]
            site = getattr(robot, "eef_site_id", None) or getattr(robot, "eef_site_ids", None)
            if isinstance(site, dict):
                site = next(iter(site.values()))
            elif isinstance(site, (list, tuple)):
                site = site[0]
            if site is not None:
                return np.asarray(sim.data.site_xmat[site], dtype=np.float32).reshape(3, 3).copy()
        except Exception:
            pass
        return np.eye(3, dtype=np.float32)

    def _cube_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        pos = self._object_pos_by_name(self.active_object_name)
        if pos is not None:
            return pos.astype(np.float32)
        # Best-effort fallback to first discovered object.
        if self.scene_objects:
            return next(iter(self.scene_objects.values())).astype(np.float32).copy()
        objects = infer_objects_from_obs(obs)
        if objects:
            xyz = next(iter(objects.values())).get("xyz")
            if xyz is not None:
                return np.asarray(xyz, dtype=np.float32)
        raise KeyError(f"Could not find active LIBERO object {self.active_object_name!r} in observation.")

    def _object_pos_by_name(self, object_name: str | None) -> np.ndarray | None:
        if object_name is None:
            return None
        name = normalize_entity_name(object_name)
        if name in self.scene_objects:
            return np.asarray(self.scene_objects[name], dtype=np.float32).copy()
        # Try fuzzy alias-less matching against discovered object names.
        for k, v in self.scene_objects.items():
            if name in k or k in name:
                return np.asarray(v, dtype=np.float32).copy()
        return super()._object_pos_by_name(name)

    def _active_pick_object(self):
        # Try to return LIBERO/robosuite object handles when available.  The
        # heuristic grasp checker below does not require this, but robosuite's
        # _check_grasp can use it if found.
        name = normalize_entity_name(self.active_object_name)
        candidates = []
        for attr in ("objects", "mujoco_objects", "obj_body_id", "obj_body_ids"):
            try:
                candidates.append(getattr(self.env.env, attr))
            except Exception:
                pass
        for c in candidates:
            if isinstance(c, dict):
                for k, v in c.items():
                    if name in normalize_entity_name(k) or normalize_entity_name(k) in name:
                        return v
            elif isinstance(c, (list, tuple)):
                for v in c:
                    vname = normalize_entity_name(getattr(v, "name", str(v)))
                    if name in vname or vname in name:
                        return v
        return None

    def _is_grasped(self) -> bool:
        # Prefer backend grasp check if available.
        try:
            obj = self._active_pick_object()
            if obj is not None:
                return bool(self.env.env._check_grasp(gripper=self.env.env.robots[0].gripper, object_geoms=obj))
        except Exception:
            pass
        # Generic fallback: gripper recently commanded closed and EE is close to
        # active object.  Latch while the object is carried near the gripper.
        if self.last_raw_obs is None:
            return False
        try:
            ee = self._eef_pos(self.last_raw_obs)
            obj = self._cube_pos(self.last_raw_obs)
            close = float(np.linalg.norm(ee - obj)) < 0.075
            if self._last_gripper_cmd > 0.2 and close:
                self._grasped_latch = True
                self._active_obj_at_grasp = obj.copy()
            if self._last_gripper_cmd < -0.2:
                self._grasped_latch = False
            if self._grasped_latch and float(np.linalg.norm(ee - obj)) < 0.12:
                return True
        except Exception:
            return False
        return False

    def _is_success(self) -> bool:
        # Trust the benchmark's own ground-truth goal check over the generic
        # base-class heuristic. LIBERO's bddl domains force
        # done = self._check_success() on every step (see
        # Libero_Floor_Manipulation.step()), so an episode can end the instant
        # the task's real goal is met -- e.g. the object lands in its target
        # container -- even while it's still grasped, before the base class's
        # own (stricter, "not grasped") _is_success() would agree. Without
        # this, a genuinely successful place gets reported as a failure
        # because System 1's release/retreat state machine never gets to run
        # another step.
        try:
            return bool(self.env.env._check_success())
        except Exception:
            return super()._is_success()

    # --------------------------------------------------------- rendering

    def render_rgb_frame(
        self,
        camera_name: str = "agentview",
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        cam = camera_name or self.camera_name
        # Most LIBERO obs dicts expose '<camera>_image'.  If possible, use a
        # fresh observation image; otherwise call the underlying renderer.
        # All three sources ultimately come from MuJoCo's raw sim.render(),
        # whose OpenGL row order is bottom-to-top; without robosuite's private
        # macro file (IMAGE_CONVENTION="opencv") flipping it, images come out
        # upside down when saved/displayed top-to-bottom, so flip explicitly
        # here rather than depending on that optional macro setup.
        if self.last_raw_obs is not None:
            for key in (f"{cam}_image", f"{self.camera_name}_image", "agentview_image", "robot0_eye_in_hand_image"):
                if key in self.last_raw_obs:
                    frame = np.asarray(self.last_raw_obs[key], dtype=np.uint8)
                    if frame.ndim == 3:
                        return frame[::-1].copy()
        try:
            frame = self.env.render(height=int(height), width=int(width), camera_name=cam)
            return np.asarray(frame, dtype=np.uint8)[::-1].copy()
        except Exception:
            try:
                frame = self.env.env.sim.render(camera_name=cam, height=int(height), width=int(width))
                return np.asarray(frame, dtype=np.uint8)[::-1].copy()
            except Exception:
                return np.zeros((int(height), int(width), 3), dtype=np.uint8)

    def relocalize_scene(self) -> Dict[str, Any]:
        if self.last_raw_obs is not None:
            self._refresh_scene_objects(self.last_raw_obs)
        return dict(self.scene_objects)

    def close(self) -> None:
        try:
            super().close()
        finally:
            try:
                self.env.close()
            except Exception:
                pass

    @staticmethod
    def capture_initial_scene_image(
        save_path: str,
        scene_context: dict[str, Any],
        camera_name: str = "agentview",
        width: int = 512,
        height: int = 512,
    ) -> str:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.image as mpimg

        env = LiberoManipulationEnv(
            scene_context=scene_context,
            has_renderer=False,
            camera_name=camera_name,
            camera_height=height,
            camera_width=width,
        )
        try:
            obs = env.reset()
            frame = env.render_rgb_frame(camera_name=camera_name, width=width, height=height)
            save_path = str(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            mpimg.imsave(save_path, frame)
            return save_path
        finally:
            env.close()

    @staticmethod
    def snapshot_scene_context(scene_context: dict[str, Any]) -> dict[str, Any]:
        """Reset LIBERO once and return scene_context populated with live poses."""
        env = LiberoManipulationEnv(scene_context=scene_context, has_renderer=False)
        try:
            obs = env.reset()
            objects = infer_objects_from_obs(obs) or infer_objects_from_mujoco(env.env)
            ctx = update_scene_context_from_objects(scene_context, objects, source="libero_snapshot")
            ctx = merge_drawer_fixtures_into_scene_context(ctx, env.env)
            return apply_libero_experiment_geometry(ctx)
        finally:
            env.close()
