"""Thin wrapper around ``ur_rtde`` for the UR3e arm.

The wrapper exposes a small, focused API tailored to the System 1 stage
machine:

* ``connect`` / ``disconnect``
* ``get_eef_pose`` / ``get_eef_xyz`` / ``get_eef_rotmat``
* ``get_joints``
* ``move_home``                     blocking ``moveJ`` to a known home
* ``move_linear``                   blocking ``moveL`` (one waypoint)
* ``move_linear_path``              blocking ``moveL`` over a list of waypoints
* ``servo_to``                      non-blocking ``servoL`` for tight MPC loops
* ``stop_servo``                    flushes the active servo motion
* ``validate_ik``                   reachability + singularity check

The controller can also operate in mock mode, where it returns a fixed pose
and accepts any command. This lets us run the entire real-world pipeline
end-to-end (perception, planning, gripper sequencing) without a robot.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _try_import_rtde():
    try:
        import rtde_control  # type: ignore
        import rtde_receive  # type: ignore

        return rtde_control, rtde_receive
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "ur_rtde is required for the real-robot backend. Install with `pip install ur_rtde`."
        ) from exc


def _rotvec_to_matrix(rotvec: Sequence[float]) -> np.ndarray:
    from ..orientation_utils import rotvec_to_matrix  # local to avoid cycles

    return rotvec_to_matrix(np.asarray(rotvec, dtype=np.float64))


def _matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    from ..orientation_utils import matrix_to_rotvec

    return matrix_to_rotvec(R)


class UR3eController:
    """Real UR3e controller driven by ``ur_rtde``.

    All motions clamp to the workspace bounds passed at construction. The
    workspace bounds match the convention in ``temp_real_robot_files``:
    ``{"x": [-0.34, 0.32], "y": [-0.453, -0.11], "z": [-0.016, 0.45]}``.
    """

    def __init__(
        self,
        ip: str,
        *,
        workspace_bounds: dict[str, list[float]] | None = None,
        servo_dt: float = 0.05,
        servo_lookahead: float = 0.10,
        servo_gain: int = 300,
        force_limit: int = 255,
    ) -> None:
        self.ip = str(ip)
        self.workspace_bounds = workspace_bounds or {
            "x": [-0.34, 0.32],
            "y": [-0.453, -0.11],
            "z": [-0.016, 0.45],
        }
        self.servo_dt = float(servo_dt)
        self.servo_lookahead = float(servo_lookahead)
        self.servo_gain = int(servo_gain)
        self.force_limit = int(force_limit)
        self._rtde_c = None
        self._rtde_r = None
        self._is_servoing = False

    # ----------------------------------------------------- lifecycle

    def connect(self) -> None:
        rtde_control, rtde_receive = _try_import_rtde()
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
        self._rtde_c = rtde_control.RTDEControlInterface(self.ip)

    def disconnect(self) -> None:
        try:
            if self._is_servoing and self._rtde_c is not None:
                self._rtde_c.servoStop()
                self._is_servoing = False
        except Exception:
            pass
        try:
            if self._rtde_c is not None:
                self._rtde_c.disconnect()
        except Exception:
            pass
        try:
            if self._rtde_r is not None:
                self._rtde_r.disconnect()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return self._rtde_c is not None and self._rtde_r is not None

    # ----------------------------------------------------- state api

    def get_joints(self) -> np.ndarray:
        if self._rtde_r is None:
            raise RuntimeError("UR3eController is not connected.")
        return np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)

    def get_eef_pose(self) -> np.ndarray:
        """Return the 6D TCP pose in robot base frame: ``[x, y, z, rx, ry, rz]``."""
        if self._rtde_r is None:
            raise RuntimeError("UR3eController is not connected.")
        return np.asarray(self._rtde_r.getActualTCPPose(), dtype=np.float64)

    def get_eef_xyz(self) -> np.ndarray:
        return self.get_eef_pose()[:3].astype(np.float32)

    def get_eef_rotvec(self) -> np.ndarray:
        return self.get_eef_pose()[3:6].astype(np.float32)

    def get_eef_rotmat(self) -> np.ndarray:
        return _rotvec_to_matrix(self.get_eef_pose()[3:6])

    # ----------------------------------------------------- motion api

    def _clamp_to_workspace(self, xyz: np.ndarray) -> np.ndarray:
        x = np.clip(
            xyz[0],
            float(self.workspace_bounds["x"][0]),
            float(self.workspace_bounds["x"][1]),
        )
        y = np.clip(
            xyz[1],
            float(self.workspace_bounds["y"][0]),
            float(self.workspace_bounds["y"][1]),
        )
        z = np.clip(
            xyz[2],
            float(self.workspace_bounds["z"][0]),
            float(self.workspace_bounds["z"][1]),
        )
        return np.array([x, y, z], dtype=np.float64)

    def move_home(self, home_joints_rad: Sequence[float], speed: float = 0.5, accel: float = 0.5) -> None:
        if self._rtde_c is None:
            raise RuntimeError("UR3eController is not connected.")
        self.stop_servo()
        self._rtde_c.moveJ(list(home_joints_rad), float(speed), float(accel))

    def move_linear(
        self,
        target_xyz: Sequence[float],
        target_rotvec: Sequence[float],
        *,
        speed: float = 0.05,
        accel: float = 0.10,
        clamp: bool = True,
    ) -> None:
        if self._rtde_c is None:
            raise RuntimeError("UR3eController is not connected.")
        self.stop_servo()
        xyz = np.asarray(target_xyz, dtype=np.float64)
        if clamp:
            xyz = self._clamp_to_workspace(xyz)
        rot = np.asarray(target_rotvec, dtype=np.float64)
        pose = np.concatenate([xyz, rot]).tolist()
        self._rtde_c.moveL(pose, float(speed), float(accel))

    def move_linear_path(
        self,
        waypoints: Iterable[Sequence[float]],
        *,
        speed: float = 0.05,
        accel: float = 0.10,
        blend: float = 0.02,
    ) -> None:
        """Execute a list of Cartesian waypoints with ``moveL(path)``.

        Each waypoint must be ``[x, y, z, rx, ry, rz]`` (or include trailing
        per-waypoint ``[speed, accel, blend]`` fields, in which case those
        override the defaults). XYZ are clamped to the workspace.
        """
        if self._rtde_c is None:
            raise RuntimeError("UR3eController is not connected.")
        self.stop_servo()
        path = []
        for wp in waypoints:
            wp = list(wp)
            xyz = self._clamp_to_workspace(np.asarray(wp[:3], dtype=np.float64))
            rot = wp[3:6]
            tail = wp[6:9] if len(wp) >= 9 else [float(speed), float(accel), float(blend)]
            path.append([float(xyz[0]), float(xyz[1]), float(xyz[2]), *rot, *tail])
        self._rtde_c.moveL(path)

    def servo_to(
        self,
        target_xyz: Sequence[float],
        target_rotvec: Sequence[float],
        *,
        dt: float | None = None,
        lookahead: float | None = None,
        gain: int | None = None,
        speed: float = 0.5,
        accel: float = 0.5,
        clamp: bool = True,
    ) -> None:
        """Stream a single Cartesian target via ``servoL`` (non-blocking)."""
        if self._rtde_c is None:
            raise RuntimeError("UR3eController is not connected.")
        xyz = np.asarray(target_xyz, dtype=np.float64)
        if clamp:
            xyz = self._clamp_to_workspace(xyz)
        rot = list(np.asarray(target_rotvec, dtype=np.float64))
        pose = [float(xyz[0]), float(xyz[1]), float(xyz[2]), *rot]
        self._rtde_c.servoL(
            pose,
            float(speed),
            float(accel),
            float(self.servo_dt if dt is None else dt),
            float(self.servo_lookahead if lookahead is None else lookahead),
            int(self.servo_gain if gain is None else gain),
        )
        self._is_servoing = True

    def stop_servo(self) -> None:
        if self._rtde_c is None or not self._is_servoing:
            return
        try:
            self._rtde_c.servoStop()
        except Exception:
            pass
        self._is_servoing = False

    # ---------------------------------------------------- safety helpers

    def validate_path(
        self,
        waypoints: Iterable[Sequence[float]],
        *,
        max_joint_jump_rad: float = 0.8,
        seed_joints: Sequence[float] | None = None,
    ) -> bool:
        """Validate IK reachability + singularity for a Cartesian path."""
        if self._rtde_c is None:
            return True
        current = list(seed_joints) if seed_joints is not None else list(self.get_joints())
        for wp in waypoints:
            pose = list(wp[:6])
            ik = self._rtde_c.getInverseKinematics(pose, current)
            if not ik or len(ik) == 0:
                return False
            max_jump = max(abs(a - b) for a, b in zip(ik, current))
            if max_jump > float(max_joint_jump_rad):
                return False
            current = list(ik)
        return True


class UR3eMockController(UR3eController):
    """Drop-in replacement that pretends to be a real UR3e for dry-run testing.

    Holds an internal pose, accepts any command, and updates the pose to track
    the most recent target. Useful when iterating on perception / planning code
    without a connected robot.
    """

    def __init__(
        self,
        *,
        initial_pose: Sequence[float] = (0.0, -0.30, 0.30, 3.14, 0.0, 0.0),
        workspace_bounds: dict[str, list[float]] | None = None,
    ) -> None:
        super().__init__("mock", workspace_bounds=workspace_bounds)
        self._pose = np.asarray(initial_pose, dtype=np.float64)
        self._joints = np.zeros(6, dtype=np.float64)

    def connect(self) -> None:  # type: ignore[override]
        # Nothing to do.
        self._rtde_c = object()  # truthy sentinel
        self._rtde_r = object()

    def disconnect(self) -> None:  # type: ignore[override]
        self._rtde_c = None
        self._rtde_r = None

    def get_joints(self) -> np.ndarray:
        return self._joints.copy()

    def get_eef_pose(self) -> np.ndarray:
        return self._pose.copy()

    def move_home(self, home_joints_rad, speed: float = 0.5, accel: float = 0.5) -> None:
        self._joints = np.asarray(home_joints_rad, dtype=np.float64)

    def move_linear(self, target_xyz, target_rotvec, *, speed: float = 0.05, accel: float = 0.10, clamp: bool = True) -> None:
        xyz = np.asarray(target_xyz, dtype=np.float64)
        if clamp:
            xyz = self._clamp_to_workspace(xyz)
        rot = np.asarray(target_rotvec, dtype=np.float64)
        self._pose = np.concatenate([xyz, rot])

    def move_linear_path(self, waypoints, *, speed=0.05, accel=0.10, blend=0.02) -> None:
        last = None
        for wp in waypoints:
            last = list(wp[:6])
        if last is not None:
            xyz = self._clamp_to_workspace(np.asarray(last[:3], dtype=np.float64))
            self._pose = np.concatenate([xyz, last[3:6]])

    def servo_to(self, target_xyz, target_rotvec, *, dt=None, lookahead=None, gain=None, speed=0.5, accel=0.5, clamp=True) -> None:
        self.move_linear(target_xyz, target_rotvec, clamp=clamp)
        self._is_servoing = True

    def stop_servo(self) -> None:
        self._is_servoing = False

    def validate_path(self, waypoints, *, max_joint_jump_rad=0.8, seed_joints=None) -> bool:
        return True
