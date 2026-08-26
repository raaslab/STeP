"""VLABench backend for the VLM-STL System-1 executor.

This adapter lets the existing STL-MPC stage machine run inside VLABench by
mapping normalized Cartesian delta commands to VLABench's absolute joint-action
API.  It is intentionally a thin backend: task semantics remain in System 2,
STL compilation, and the common BaseManipulationEnv logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import math

import numpy as np

from .base_env import BaseManipulationEnv
from .orientation_utils import orthonormalize, rotvec_to_matrix
from .vlabench_scene import load_vlabench_env, _extract_entity_position


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size < 4:
        return np.eye(3, dtype=np.float32)
    # Try wxyz, which is what dm_control/MuJoCo commonly use.
    w, x, y, z = [float(v) for v in q[:4]]
    n = math.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return orthonormalize(R)


def _matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    R = orthonormalize(R).astype(np.float64)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.asarray([x, y, z], dtype=np.float32)


class VLABenchManipulationEnv(BaseManipulationEnv):
    """VLABench backend with the same public API as sim/real backends."""

    def __init__(
        self,
        *,
        scene_context: dict[str, Any],
        mode: str = "top",
        horizon: int = 600,
        render: bool = False,
        active_object_name: str | None = None,
        goal_reference_name: str | None = None,
        goal_xy: tuple[float, float] = (0.0, 0.0),
        goal_radius: float = 0.035,
        goal_z: float | None = None,
        descend_radius: float = 0.075,
        keepout_xy: tuple[float, float] = (0.0, 0.0),
        keepout_radius: float = 0.0,
        carry_height_above_table: float = 0.17,
        retreat_height_above_table: float = 0.28,
        workspace_bounds: dict[str, Any] | None = None,
        sdf_constraints: list[Any] | None = None,
        mpc_timing_enabled: bool = True,
        mpc_timing_early_weight: float = 220.0,
        mpc_timing_deadline_weight: float = 55.0,
        mpc_timing_window_weight: float = 10.0,
        mpc_timing_early_hold_weight: float = 260.0,
    ) -> None:
        self.scene_context = scene_context
        self.has_renderer = bool(render)
        vlb = (scene_context.get("platform", {}) or {}).get("vlabench", {}) or {}
        self.task_name = str(vlb.get("task") or vlb.get("task_name") or "select_toy")
        self.robot_name = str(vlb.get("robot", "franka"))
        self.episode_config = vlb.get("episode_config")
        self.random_init = bool(vlb.get("random_init", self.episode_config is None))
        self.run_mode = str(vlb.get("run_mode", "eval"))
        self.reset_wait_step = int(vlb.get("reset_wait_step", 10))
        self.control_hz = float(vlb.get("control_hz", 20.0))
        self.ee_delta_scale_m = float(vlb.get("ee_delta_scale_m", 0.032))
        self.ee_rot_scale_rad = float(vlb.get("ee_rot_scale_rad", 0.08))
        self.open_width = float(vlb.get("open_width", 0.04))
        self.close_width = float(vlb.get("close_width", 0.0))
        self.default_camera_id = int(vlb.get("camera_id", 0))
        self._last_observation: Dict[str, Any] | None = None
        self._last_timestep = None

        self.env = load_vlabench_env(
            self.task_name,
            robot=self.robot_name,
            episode_config=self.episode_config,
            random_init=self.random_init,
            run_mode=self.run_mode,
            time_limit=int(horizon),
            reset_wait_step=self.reset_wait_step,
        )

        table_z = float(vlb.get("table_z", scene_context.get("table_z", 0.75)))
        # The common controller emits [dx,dy,dz,drot_x,drot_y,drot_z,gripper].
        low = -np.ones(7, dtype=np.float32)
        high = np.ones(7, dtype=np.float32)
        super().__init__(
            mode=mode,
            keepout_xy=keepout_xy,
            keepout_radius=keepout_radius,
            object_keepout_radius=float(vlb.get("object_keepout_radius", 0.05)),
            goal_xy=goal_xy,
            goal_radius=goal_radius,
            goal_z=goal_z,
            descend_radius=descend_radius,
            carry_height_above_table=carry_height_above_table,
            retreat_height_above_table=retreat_height_above_table,
            active_object_name=active_object_name or self._first_entity_name() or "object",
            goal_reference_name=goal_reference_name,
            workspace_bounds=workspace_bounds,
            sdf_constraints=sdf_constraints,
            mpc_timing_enabled=mpc_timing_enabled,
            mpc_timing_early_weight=mpc_timing_early_weight,
            mpc_timing_deadline_weight=mpc_timing_deadline_weight,
            mpc_timing_window_weight=mpc_timing_window_weight,
            mpc_timing_early_hold_weight=mpc_timing_early_hold_weight,
            open_gripper_cmd=-1.0,
            close_gripper_cmd=1.0,
            action_dim=7,
            gripper_dim=1,
            action_low=low,
            action_high=high,
            table_z=table_z,
            mpc_action_scale_m=self.ee_delta_scale_m,
        )
        self._sync_scene_objects()

    def _first_entity_name(self) -> str | None:
        entities = self.scene_context.get("entities", []) or []
        for ent in entities:
            if ent.get("role") in {"manipulated_object", "support_object"}:
                return str(ent.get("name"))
        return None

    # ------------------------------------------------------------ backend API

    def _obs(self) -> Dict[str, Any]:
        return self.env.get_observation(require_pcd=False)

    def _reset_backend(self) -> Dict[str, Any]:
        self._last_timestep = self.env.reset()
        obs = self._obs()
        self._last_observation = obs
        self._sync_scene_objects()
        return obs

    def _step_backend(self, full_action: np.ndarray):
        action = self._to_vlabench_action(full_action)
        timestep = self.env.step(action)
        self._last_timestep = timestep
        obs = self._obs()
        self._last_observation = obs
        self._sync_scene_objects()
        reward = 0.0
        try:
            reward = float(timestep.reward or 0.0)
        except Exception:
            pass
        done = False
        try:
            done = bool(timestep.last())
        except Exception:
            done = False
        return obs, reward, done, {}

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
        super().close()

    # ----------------------------------------------------- observation hooks

    def _eef_state(self, obs: Dict[str, Any]) -> np.ndarray:
        st = obs.get("ee_state")
        if st is None:
            st = self.env.robot.get_ee_state(self.env.physics)
        return np.asarray(st, dtype=np.float32).reshape(-1)

    def _eef_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        st = self._eef_state(obs)
        return st[:3].astype(np.float32).copy()

    def _eef_rotmat(self, obs: Dict[str, Any]) -> np.ndarray:
        st = self._eef_state(obs)
        if st.size >= 7:
            # If the last four values look like a quaternion, use them.
            q = st[3:7]
            qnorm = float(np.linalg.norm(q))
            if 0.5 <= qnorm <= 1.5:
                return _quat_to_matrix(q)
            # Otherwise interpret 3:6 as Euler.
            euler = st[3:6]
            return self._euler_to_rotmat(euler)
        return np.eye(3, dtype=np.float32)

    def _cube_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        pos = self._object_pos_by_name(self.active_object_name)
        if pos is not None:
            return pos.astype(np.float32)
        # Last resort: keep the EE as object proxy so failures are visible but
        # the code does not crash while debugging a new VLABench task.
        return self._eef_pos(obs)

    def _is_grasped(self) -> bool:
        try:
            names, _entities = self.env.get_grasped_entity()
            return str(self.active_object_name) in {str(n) for n in names}
        except Exception:
            return False

    def _object_pos_by_name(self, object_name: str | None) -> np.ndarray | None:
        if object_name is None:
            return None
        pos = _extract_entity_position(self.env, str(object_name))
        if pos is not None:
            return pos.astype(np.float32)
        return super()._object_pos_by_name(object_name)

    # --------------------------------------------------------- VLABench action

    @staticmethod
    def _euler_to_rotmat(euler: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = [float(v) for v in np.asarray(euler).reshape(3)]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float32)
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float32)
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float32)
        return orthonormalize(Rz @ Ry @ Rx)

    def _to_vlabench_action(self, full_action: np.ndarray) -> np.ndarray:
        obs = self.last_raw_obs if self.last_raw_obs is not None else self._obs()
        ee = self._eef_pos(obs)
        R = self._eef_rotmat(obs)
        cmd = np.asarray(full_action, dtype=np.float32).reshape(-1)
        delta_xyz = np.clip(cmd[:3], -1.0, 1.0) * float(self.ee_delta_scale_m)
        target_pos = ee + delta_xyz
        drot = np.zeros(3, dtype=np.float32)
        if cmd.size >= 6:
            drot = np.clip(cmd[3:6], -1.0, 1.0) * float(self.ee_rot_scale_rad)
        target_R = rotvec_to_matrix(drot) @ R
        target_euler = _matrix_to_euler_xyz(target_R)
        try:
            from VLABench.utils.utils import euler_to_quaternion
            quat = euler_to_quaternion(*[float(v) for v in target_euler])
        except Exception:
            quat = self._euler_to_quat_wxyz(target_euler)
        try:
            _success, qpos = self.env.robot.get_qpos_from_ee_pos(
                physics=self.env.physics,
                pos=np.asarray(target_pos, dtype=np.float32),
                quat=np.asarray(quat, dtype=np.float32),
            )
        except Exception:
            qpos = self.env.robot.get_qpos(self.env.physics)
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        grip_scalar = float(cmd[-1]) if cmd.size else -1.0
        width = self.close_width if grip_scalar > 0.0 else self.open_width
        gripper_state = np.ones(2, dtype=np.float32) * float(width)
        return np.concatenate([qpos, gripper_state]).astype(np.float32)

    @staticmethod
    def _euler_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = [float(v) for v in np.asarray(euler).reshape(3)]
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        return np.asarray([w, x, y, z], dtype=np.float32)

    # -------------------------------------------------------------- scene I/O

    def _sync_scene_objects(self) -> None:
        for ent in self.scene_context.get("entities", []) or []:
            name = ent.get("name")
            if not name:
                continue
            pos = _extract_entity_position(self.env, str(name))
            if pos is not None:
                self.update_scene_object(str(name), pos)

    def relocalize_scene(self) -> Dict[str, Any]:
        self._sync_scene_objects()
        return {k: np.asarray(v, dtype=np.float32).copy() for k, v in self.scene_objects.items()}

    # -------------------------------------------------------------- rendering

    def render_rgb_frame(self, camera_name: str = "agentview", width: int = 640, height: int = 480) -> np.ndarray:
        # VLABench exposes multiple cameras as ids.  Use a numeric camera name if
        # provided; otherwise fall back to configured/default camera 0.
        cam_id = self.default_camera_id
        try:
            cam_id = int(camera_name)
        except Exception:
            pass
        try:
            frame = self.env.render(camera_id=cam_id, height=int(height), width=int(width))
        except Exception:
            obs = self._last_observation or self._obs()
            rgb = np.asarray(obs.get("rgb"))
            if rgb.ndim == 4:
                frame = rgb[min(cam_id, rgb.shape[0] - 1)]
            else:
                frame = rgb
        return np.asarray(frame, dtype=np.uint8)

    @staticmethod
    def capture_initial_scene_image(
        save_path: str,
        *,
        scene_context: dict[str, Any],
        camera_name: str = "0",
        width: int = 512,
        height: int = 512,
    ) -> str:
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:  # pragma: no cover
            imageio = None
        env = VLABenchManipulationEnv(
            scene_context=scene_context,
            horizon=int((scene_context.get("platform", {}).get("vlabench", {}) or {}).get("horizon", 600)),
            render=False,
        )
        try:
            env.reset()
            frame = env.render_rgb_frame(camera_name=camera_name, width=width, height=height)
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if imageio is not None:
                imageio.imwrite(str(path), frame)
            else:
                from PIL import Image
                Image.fromarray(frame).save(str(path))
            return str(path)
        finally:
            env.close()
