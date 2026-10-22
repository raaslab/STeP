"""Backend-agnostic manipulation environment.

This module factors out everything that does not depend on whether we are
running in robosuite or on a real UR3e. Concrete backends (``sim_env.py``,
``real_env.py``) inherit from :class:`BaseManipulationEnv` and only need to
implement a small set of low-level hooks (read EE pose, read object pose,
apply a Cartesian command, render a frame, etc).

What lives here:

* the stage machine (pregrasp -> grasp -> close -> lift -> ... -> retreat)
* STL-style nominal MPC stepping with stage-adaptive smoothing
* the goal / pass-through / keep-out configuration surface
* status + reward + rollout logging
* video capture / mp4 writing helpers

The shared logic is identical between sim and real, so improvements to the
controller transfer for free in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .modes import MODE_CONFIGS, GraspMode
from .nominal_mpc import STLCartesianMPC, STLStageContext
from .geometry_sdf import SDFConstraint
from .orientation_utils import orientation_error_rotvec, orthonormalize, rotvec_to_matrix


@dataclass
class StageTargets:
    pregrasp: np.ndarray
    grasp: np.ndarray
    carry: np.ndarray
    over_goal: np.ndarray
    place: np.ndarray
    retreat: np.ndarray


class BaseManipulationEnv:
    """Common stage-machine + STL-MPC shell shared by sim and real backends.

    Subclasses must implement the low-level hooks listed under
    :py:meth:`BaseManipulationEnv` (search for ``raise NotImplementedError``)
    and call :py:meth:`_post_init_finalize` once their backend is ready.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        *,
        mode: str = "top",
        seed: int = 0,
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
        active_object_name: str = "green_cube",
        goal_reference_name: str | None = None,
        open_gripper_cmd: float = -1.0,
        close_gripper_cmd: float = 1.0,
        action_dim: int = 7,
        gripper_dim: int = 1,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        table_z: float = 0.8,
        # Where a task object's reported pose sits on the object itself.
        # LIBERO reports the mesh BASE (table level), so reaching mid-height
        # means lifting the grasp target by half the object's height. RoboCasa
        # reports the CENTROID -- measured on its eggplant, the body origin sits
        # 0.003 m from the midpoint of the mesh and 0.086 m above its bottom --
        # so applying that same lift raises the target from the middle of the
        # object to above its top, and the gripper closes on air.
        object_pose_is_base: bool = True,
        # MPC tuning (sane defaults shared by sim + real)
        mpc_action_scale_m: float = 0.032,
        mpc_horizon: int = 16,
        mpc_num_iters: int = 30,
        mpc_step_size: float = 0.12,
        mpc_stage_weight: float = 0.24,
        mpc_terminal_weight: float = 28.0,
        mpc_control_weight: float = 0.06,
        mpc_smooth_weight: float = 1.85,
        mpc_table_clearance_z: float = 0.020,
        mpc_knife_weight: float = 24.0,
        mpc_carry_weight: float = 18.0,
        mpc_goal_over_weight: float = 7.5,
        mpc_barrier_beta: float = 30.0,
        # Optional axis-aligned workspace bounds in the same frame as EE poses.
        # Format: {"x": [xmin, xmax], "y": [ymin, ymax], "z": [zmin, zmax]}.
        workspace_bounds: dict[str, Any] | None = None,
        mpc_workspace_weight: float = 0.0,
        mpc_timing_enabled: bool = True,
        mpc_timing_early_weight: float = 220.0,
        mpc_timing_deadline_weight: float = 55.0,
        mpc_timing_window_weight: float = 10.0,
        mpc_timing_early_hold_weight: float = 260.0,
        sdf_constraints: list[SDFConstraint] | None = None,
    ) -> None:
        if mode not in MODE_CONFIGS:
            raise ValueError(
                f"Unknown mode {mode!r}; expected one of {list(MODE_CONFIGS)}"
            )
        self.mode_name = mode
        self.seed = int(seed)
        self.active_object_name = str(active_object_name)
        self.goal_reference_name = (
            None if goal_reference_name is None else str(goal_reference_name)
        )

        self.open_gripper_cmd = float(open_gripper_cmd)
        self.close_gripper_cmd = float(close_gripper_cmd)
        self.nominal_ori_gain = float(nominal_ori_gain)
        self.close_steps = int(close_steps)

        # Action layout (subclasses can override action_dim/gripper_dim).
        self.action_dim_underlying = int(action_dim)
        self.gripper_dim = int(gripper_dim)
        self.arm_dim = self.action_dim_underlying - self.gripper_dim
        if action_low is None:
            action_low = -np.ones(self.action_dim_underlying, dtype=np.float32)
        if action_high is None:
            action_high = np.ones(self.action_dim_underlying, dtype=np.float32)
        self.low = np.asarray(action_low, dtype=np.float32)
        self.high = np.asarray(action_high, dtype=np.float32)

        # State.
        self.stage = 0
        self.close_counter = 0
        self.grasp_retry_counter = 0
        self.grasp_offset_override: np.ndarray | None = None
        self.active_object_height: float | None = None
        self.step_idx = 0
        self.last_raw_obs: Dict[str, Any] | None = None
        self.last_target_xyz = np.zeros(3, dtype=np.float32)
        self.last_nominal_pos = np.zeros(3, dtype=np.float32)
        self.last_nominal_ori = np.zeros(3, dtype=np.float32)
        self.last_full_action = np.zeros(self.action_dim_underlying, dtype=np.float32)
        self.top_orientation_R = np.eye(3, dtype=np.float32)
        self.prev_target_dist = 0.0
        self.prev_cube_height = 0.0
        self.prev_ee_pos = np.zeros(3, dtype=np.float32)
        self.prev_ori_err = 0.0
        self.rollout_log: List[Dict[str, Any]] = []
        self._video_renderer = None
        self._video_renderer_shape: tuple[int, int] | None = None
        self.release_hold_pos: np.ndarray | None = None
        self.retreat_anchor_xy: np.ndarray | None = None

        # Clean top-grasp settings. For top grasp, the target is the cube
        # center; the gripper closes only when the end-effector is actually
        # centered on the cube in XY and Z.
        self.top_grasp_hover_z = 0.06
        self.top_grasp_close_xy_tol = 0.025
        self.top_grasp_close_z_tol = 0.018
        self.top_pick_servo_scale_m = 0.025
        self.top_pick_servo_alpha = 0.70
        self.top_pick_servo_max_delta = 0.24

        # Side-grasp acquisition should also be a literal Cartesian servo.
        # The older path sent side grasps through the soft STL-MPC during
        # stages 0/1/2, which often hovered near the cube and never satisfied
        # the close predicate.
        self.side_pick_servo_scale_m = 0.030
        self.side_pick_servo_alpha = 0.68
        self.side_pick_servo_max_delta = 0.24
        self.side_grasp_close_pos_tol = 0.055
        self.side_grasp_close_ori_tol = 1.25

        # Table / goal geometry.
        self.table_z = float(table_z)
        self.object_pose_is_base = bool(object_pose_is_base)
        self.carry_z = self.table_z + float(carry_height_above_table)
        self.retreat_z = self.table_z + float(retreat_height_above_table)
        self.keepout_xy = np.asarray(keepout_xy, dtype=np.float32)
        self.keepout_radius = float(keepout_radius)
        self.object_keepout_radius = float(object_keepout_radius)
        self.goal_xy = np.asarray(goal_xy, dtype=np.float32)
        self.goal_radius = float(goal_radius)
        self.goal_z_override = None if goal_z is None else float(goal_z)
        self.descend_radius = float(descend_radius)

        # The visual half-extents of the legacy "knife" support object. Kept
        # for sim debug-marker rendering; real backends ignore it.
        self.knife_visual_half_extents = np.asarray(
            [0.065, 0.010, 0.004], dtype=np.float32
        )

        # Pass-through region.
        self.pass_through_xy: np.ndarray | None = None
        self.pass_through_radius = 0.10
        self.pass_through_z: float | None = None
        self.pass_through_label: str | None = None
        self.pass_through_visited = False

        # Per-instance scene state for entities that are NOT hard-coded into
        # the simulator (used by real backends, optional in sim).
        self.scene_objects: Dict[str, np.ndarray] = {}

        # Axis-aligned workspace limits for EE-space planning.
        self.workspace_min_xyz, self.workspace_max_xyz = self._parse_workspace_bounds(
            workspace_bounds
        )
        self.mpc_workspace_weight = float(mpc_workspace_weight)
        self.sdf_constraints: list[SDFConstraint] = list(sdf_constraints or [])
        self.active_sdf_constraint_names: set[str] = set()

        # Local timing-window constraints for the current subtask. These are
        # configured by the executor from System-2 STL constraints. They are
        # represented generically and fed into the MPC as temporal robustness
        # costs, while the existing online monitor remains the final checker.
        self.active_timing_windows: list[dict[str, Any]] = []
        self.timing_control_hz: float = 20.0
        self.timing_subtask_step_idx: int = 0
        self.timing_task_step_offset: int = 0
        self.timing_mpc_enabled: bool = bool(mpc_timing_enabled)
        self.timing_early_weight: float = float(mpc_timing_early_weight)
        self.timing_deadline_weight: float = float(mpc_timing_deadline_weight)
        self.timing_window_weight: float = float(mpc_timing_window_weight)
        self.timing_early_hold_weight: float = float(mpc_timing_early_hold_weight)

        # MPC controller (the same one used by sim + real).
        self.nominal_mpc = STLCartesianMPC(
            action_scale_m=float(mpc_action_scale_m),
            horizon=int(mpc_horizon),
            num_iters=int(mpc_num_iters),
            step_size=float(mpc_step_size),
            stage_weight=float(mpc_stage_weight),
            terminal_weight=float(mpc_terminal_weight),
            control_weight=float(mpc_control_weight),
            smooth_weight=float(mpc_smooth_weight),
            table_clearance_z=float(mpc_table_clearance_z),
            knife_weight=float(mpc_knife_weight),
            carry_weight=float(mpc_carry_weight),
            goal_over_weight=float(mpc_goal_over_weight),
            barrier_beta=float(mpc_barrier_beta),
        )

    # ------------------------------------------------------------ properties

    @property
    def mode(self) -> GraspMode:
        return MODE_CONFIGS[self.mode_name]

    def switch_mode(self, mode: str) -> None:
        if mode not in MODE_CONFIGS:
            raise ValueError(f"Unknown mode {mode!r}")
        self.mode_name = mode

    def configure_from_constraints(
        self,
        carry_height: float | None = None,
        descend_radius: float | None = None,
        keepout_radius: float | None = None,
    ) -> None:
        if carry_height is not None:
            self.carry_z = self.table_z + float(carry_height)
        if descend_radius is not None:
            self.descend_radius = float(descend_radius)
        if keepout_radius is not None:
            self.keepout_radius = float(keepout_radius)

    def configure_timing_windows(
        self,
        windows: list[dict[str, Any]] | None,
        *,
        control_hz: float | None = None,
        task_elapsed_steps: int = 0,
    ) -> None:
        """Attach local STL timing windows to the low-level MPC context.

        The executor calls this at the start of each subtask. The MPC then
        uses these windows as differentiable temporal robustness costs. This
        keeps timing in the STL/MPC backbone instead of treating it only as a
        post-hoc Python gate.
        """
        self.active_timing_windows = [dict(w) for w in (windows or [])]
        try:
            hz = float(control_hz) if control_hz is not None else self.timing_control_hz
            self.timing_control_hz = hz if hz > 0.0 else 20.0
        except Exception:
            self.timing_control_hz = 20.0
        self.timing_subtask_step_idx = 0
        self.timing_task_step_offset = int(max(0, task_elapsed_steps))
        try:
            self.nominal_mpc.reset()
        except Exception:
            pass

    def clear_timing_windows(self) -> None:
        self.configure_timing_windows([], control_hz=self.timing_control_hz, task_elapsed_steps=0)

    def _timing_elapsed_sec(self, window: dict[str, Any]) -> float:
        reference = str(window.get("reference") or "subtask_start").strip().lower()
        if reference == "task_start":
            steps = self.timing_task_step_offset + self.timing_subtask_step_idx
        else:
            steps = self.timing_subtask_step_idx
        return float(steps) / max(float(self.timing_control_hz), 1e-6)

    @staticmethod
    def _timing_predicate_name(window: dict[str, Any]) -> str:
        pred = str(window.get("predicate") or "").strip().lower()
        return pred.replace(" ", "_").replace("-", "_")

    def _timing_event_target_for_predicate(
        self,
        predicate: str,
        *,
        ee_pos: np.ndarray | None,
        cube_pos: np.ndarray,
    ) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray | None] | None:
        """Map a symbolic event predicate to a continuous EE-space proxy.

        This is deliberately object-agnostic: the active object pose and the
        live goal/reference geometry define the proxy. For grasped, the proxy
        is the grasp pose; for goal/placement predicates, it is the EE pose
        that would put the active object at the current live target.
        """
        pred = str(predicate).strip().lower().replace(" ", "_").replace("-", "_")
        targets = self._targets(cube_pos)
        if pred in {"grasp", "grasped", "still_grasped"}:
            # Use a radius slightly larger than the hard close tolerance so the
            # early-window cost keeps the system outside the close predicate.
            radius = max(0.035, float(self.top_grasp_close_xy_tol) + 0.010)
            # Tuple form: (event target, event radius, precursor hold target).
            return targets.grasp.astype(np.float32), radius, targets.pregrasp.astype(np.float32)
        if pred in {"pregrasp", "pre_grasp"}:
            return targets.pregrasp.astype(np.float32), 0.045, None

        offset = np.zeros(3, dtype=np.float32)
        if ee_pos is not None and (self._is_grasped() or self.stage in (3, 4, 5, 6)):
            offset = np.asarray(ee_pos, dtype=np.float32) - np.asarray(cube_pos, dtype=np.float32)

        if pred in {"over_goal", "overgoal"}:
            return (targets.over_goal + offset).astype(np.float32), max(float(self.goal_radius), 0.035), None
        if pred in {"placed", "placed_xy", "placed_released", "placed_and_released", "released"}:
            return (targets.place + offset).astype(np.float32), max(float(self.goal_radius), 0.035), None
        return None

    def _mpc_timing_windows_for_context(
        self,
        *,
        ee_pos: np.ndarray | None,
        cube_pos: np.ndarray,
    ) -> list[dict[str, Any]]:
        if not self.timing_mpc_enabled or not self.active_timing_windows:
            return []
        out: list[dict[str, Any]] = []
        for window in self.active_timing_windows:
            pred = self._timing_predicate_name(window)
            proxy = self._timing_event_target_for_predicate(pred, ee_pos=ee_pos, cube_pos=cube_pos)
            if proxy is None:
                continue
            if len(proxy) == 3:
                event_target, event_radius, early_hold = proxy
            else:
                event_target, event_radius = proxy
                early_hold = None
            item = dict(window)
            payload = {
                "predicate": pred,
                "elapsed_sec": self._timing_elapsed_sec(window),
                "event_target_xyz": event_target.astype(np.float32),
                "event_radius": float(event_radius),
                "early_weight": float(self.timing_early_weight),
                "deadline_weight": float(self.timing_deadline_weight),
                "window_weight": float(self.timing_window_weight),
                "early_hold_weight": float(self.timing_early_hold_weight),
            }
            if early_hold is not None:
                payload["early_hold_xyz"] = np.asarray(early_hold, dtype=np.float32)
            item.update(payload)
            out.append(item)
        return out

    def _pre_window_timing_window_for_predicate(self, predicate_names: set[str]) -> dict[str, Any] | None:
        """Return an active timing window whose lower bound has not opened yet.

        This is used to translate STL not-before semantics, e.g.
        G[0,a)(!Grasped(o)), into a time-varying MPC target.  The helper is
        predicate-generic: callers provide the event predicate names they care
        about, and the actual object/goal geometry comes from the live scene.
        """
        if not self.timing_mpc_enabled or not self.active_timing_windows:
            return None
        names = {str(n).strip().lower().replace(" ", "_").replace("-", "_") for n in predicate_names}
        for window in self.active_timing_windows:
            pred = self._timing_predicate_name(window)
            if pred not in names:
                continue
            start_sec = window.get("start_sec")
            if start_sec is None:
                continue
            try:
                start = float(start_sec)
            except Exception:
                continue
            if self._timing_elapsed_sec(window) < start:
                return window
        return None

    def _pre_window_hold_target_for_current_pick(self, cube_pos: np.ndarray) -> np.ndarray | None:
        """MPC hold target for a pick event whose timing window has not opened.

        For a timed Grasped/StillGrasped predicate, the event region is the
        grasp-success zone.  Before the lower bound, that region should be
        treated as forbidden by the MPC, while the nominal target should remain
        the predecessor set (pregrasp).  Once the window opens this returns
        None and the existing pick FSM/targets take over unchanged.
        """
        window = self._pre_window_timing_window_for_predicate({"grasp", "grasped", "still_grasped"})
        if window is None:
            return None
        return self._targets(cube_pos).pregrasp.astype(np.float32)

    def _pre_window_hold_active_for_current_pick(self, cube_pos: np.ndarray) -> bool:
        return self._pre_window_hold_target_for_current_pick(cube_pos) is not None

    def _timing_mpc_active_for_current_stage(
        self,
        *,
        ee_pos: np.ndarray | None,
        cube_pos: np.ndarray,
    ) -> bool:
        return bool(self._mpc_timing_windows_for_context(ee_pos=ee_pos, cube_pos=cube_pos))

    @staticmethod
    def _parse_workspace_bounds(
        workspace_bounds: dict[str, Any] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not workspace_bounds:
            return None, None
        try:
            lo = np.array(
                [
                    float(workspace_bounds["x"][0]),
                    float(workspace_bounds["y"][0]),
                    float(workspace_bounds["z"][0]),
                ],
                dtype=np.float32,
            )
            hi = np.array(
                [
                    float(workspace_bounds["x"][1]),
                    float(workspace_bounds["y"][1]),
                    float(workspace_bounds["z"][1]),
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
                return None, None
            if np.any(hi <= lo):
                return None, None
            return lo, hi
        except Exception:
            return None, None

    def _grow_workspace_bounds_to_include(
        self, points: np.ndarray, margin: float = 0.05
    ) -> None:
        """Widen (never shrink) the workspace box so it covers known object
        positions, plus a margin.

        `configs/libero_template.json`'s workspace_bounds is a single static
        box tuned for one scene layout (the kitchen-tabletop demo). Other
        scenes/suites place objects outside that box on some axis, which
        silently clips the pick/place target to the box edge and the arm can
        never actually reach the object (see the table_z fix in
        LiberoManipulationEnv for the same class of bug on the z axis).
        """
        if self.workspace_min_xyz is None or self.workspace_max_xyz is None:
            return
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if pts.size == 0:
            return
        lo = pts.min(axis=0) - float(margin)
        hi = pts.max(axis=0) + float(margin)
        self.workspace_min_xyz = np.minimum(self.workspace_min_xyz, lo).astype(np.float32)
        self.workspace_max_xyz = np.maximum(self.workspace_max_xyz, hi).astype(np.float32)

    def _clip_xyz_to_workspace(self, xyz: np.ndarray) -> np.ndarray:
        arr = np.asarray(xyz, dtype=np.float32).copy()
        if self.workspace_min_xyz is None or self.workspace_max_xyz is None:
            return arr
        return np.clip(arr, self.workspace_min_xyz, self.workspace_max_xyz).astype(np.float32)

    def _clip_delta_action_to_workspace(
        self,
        ee_pos: np.ndarray,
        action: np.ndarray,
        *,
        scale_m: float,
    ) -> np.ndarray:
        """Hard-clip the first commanded EE step into the workspace box.

        The MPC also receives workspace soft barriers over its full horizon, but
        this one-step clip is an extra safety layer for direct-servo pick stages
        and for optimizer transients near the workspace boundary.
        """
        raw = np.asarray(action, dtype=np.float32)
        if self.workspace_min_xyz is None or self.workspace_max_xyz is None:
            return np.clip(raw, -1.0, 1.0).astype(np.float32)
        scale = max(float(scale_m), 1e-6)
        ee = np.asarray(ee_pos, dtype=np.float32)
        next_xyz = ee + scale * np.clip(raw, -1.0, 1.0)
        clipped_next = np.clip(next_xyz, self.workspace_min_xyz, self.workspace_max_xyz)
        return np.clip((clipped_next - ee) / scale, -1.0, 1.0).astype(np.float32)

    def workspace_bounds_status(self) -> dict[str, Any]:
        return {
            "workspace_min_xyz": None
            if self.workspace_min_xyz is None
            else self.workspace_min_xyz.astype(float).tolist(),
            "workspace_max_xyz": None
            if self.workspace_max_xyz is None
            else self.workspace_max_xyz.astype(float).tolist(),
            "mpc_workspace_weight": float(self.mpc_workspace_weight),
        }

    # ----------------------------------------------------- low-level hooks
    #
    # Subclasses MUST override the following.  Default implementations raise
    # NotImplementedError on purpose so any sub-class missing a hook fails
    # loudly during a smoke test rather than silently producing garbage.

    def _reset_backend(self) -> Dict[str, Any]:
        """Reset the underlying simulator / real-robot connection.

        Returns a "raw observation" dict that subclasses can later look up via
        ``_eef_pos`` / ``_cube_pos``. The dict's contents are backend specific.
        """
        raise NotImplementedError

    def _step_backend(self, full_action: np.ndarray) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply a normalized Cartesian action and step the backend by one tick.

        ``full_action`` is shaped ``(action_dim_underlying,)`` and is already
        clipped into ``[low, high]``. Returns ``(obs, base_reward, terminated, info)``
        following the robosuite convention.
        """
        raise NotImplementedError

    def _eef_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def _eef_rotmat(self, obs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def _cube_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        """Position of the currently active manipulated object."""
        raise NotImplementedError

    def _is_grasped(self) -> bool:
        raise NotImplementedError

    def _object_pos_by_name(self, object_name: str | None) -> np.ndarray | None:
        """Return the world position of an arbitrary scene entity, or None.

        Used for stack-on-object placement. The default implementation looks
        up self.scene_objects, which real backends populate from perception.
        Sim subclasses override this to read live MuJoCo positions.
        """
        if object_name is None:
            return None
        if object_name in self.scene_objects:
            pos = self.scene_objects[object_name]
            return np.asarray(pos, dtype=np.float32).copy()
        return None

    def _active_pick_object(self) -> Any:
        """Backend-specific handle of the active object (used for grasp checks)."""
        return None

    def render_rgb_frame(
        self,
        camera_name: str = "agentview",
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        """Render a single RGB frame for video recording."""
        raise NotImplementedError

    def snapshot_scene_image(
        self,
        save_path: str,
        *,
        camera_name: str = "agentview",
        width: int = 640,
        height: int = 480,
    ) -> str:
        """Capture and save a fresh RGB frame of the current scene.

        Default implementation reuses :py:meth:`render_rgb_frame`. Real
        backends override this to also flush the RealSense buffer.
        """
        from pathlib import Path as _Path

        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:
            imageio = None
        frame = self.render_rgb_frame(
            camera_name=camera_name, width=int(width), height=int(height)
        )
        save_path = str(save_path)
        _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if imageio is not None:
            imageio.imwrite(save_path, np.asarray(frame, dtype=np.uint8))
        else:
            try:
                from PIL import Image  # type: ignore

                Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(save_path)
            except Exception:
                import matplotlib.image as mpimg

                mpimg.imsave(save_path, np.asarray(frame, dtype=np.uint8))
        return save_path

    def relocalize_scene(self) -> Dict[str, Any]:
        """Re-run perception against a fresh capture (real backend hook).

        Default is a no-op for sim. Real backends override to refresh
        ``scene_objects`` from a fresh RealSense capture between prompts.
        Returns the (possibly unchanged) ``scene_objects`` dict.
        """
        return dict(self.scene_objects)

    def close(self) -> None:
        if self._video_renderer is not None:
            try:
                self._video_renderer.close()
            except Exception:
                pass
            self._video_renderer = None

    # ----------------------------------------------------- active object api

    def set_active_object(self, object_name: str) -> None:
        """Default: just record the name. Sim overrides to swap the cube index."""
        self.active_object_name = str(object_name)

    def set_grasp_offset_override(self, offset_xyz: Any | None) -> None:
        """Set an explicit world-frame grasp offset from the active object pose."""
        if offset_xyz is None:
            self.grasp_offset_override = None
            return
        arr = np.asarray(offset_xyz, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            raise ValueError(f"grasp offset must have 3 values, got {offset_xyz!r}")
        self.grasp_offset_override = arr[:3].astype(np.float32)

    def set_active_object_height(self, height: float | None) -> None:
        """Set the active object's known height (m), e.g. from the Grasped
        predicate's height arg. Used by :meth:`_effective_grasp_rim_offset` to
        grasp at roughly mid-height instead of the object's base pose, for
        non-rim objects (cans, bottles, boxes) whose base is at table level.
        """
        self.active_object_height = None if height is None else float(height)

    def update_scene_object(self, name: str, xyz: np.ndarray) -> None:
        """Record the world XYZ of a named scene entity."""
        self.scene_objects[str(name)] = np.asarray(xyz, dtype=np.float32).copy()

    # ---------------------------------------------------- pass-through api

    def clear_pass_through(self) -> None:
        self.pass_through_xy = None
        self.pass_through_radius = 0.10
        self.pass_through_z = None
        self.pass_through_label = None
        self.pass_through_visited = False

    def configure_pass_through(
        self,
        xy: Tuple[float, float] | np.ndarray,
        radius: float = 0.10,
        z: float | None = None,
        label: str | None = None,
    ) -> None:
        self.pass_through_xy = np.asarray(xy, dtype=np.float32)
        self.pass_through_radius = float(radius)
        self.pass_through_z = None if z is None else float(z)
        self.pass_through_label = label
        self.pass_through_visited = False

    def _has_pending_pass_through(self) -> bool:
        return self.pass_through_xy is not None and not self.pass_through_visited

    def _pass_through_target(
        self,
        cube_pos: np.ndarray | None = None,
        ee_pos: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if self.pass_through_xy is None:
            return None
        desired_obj_z = float(
            self.pass_through_z if self.pass_through_z is not None else self.carry_z
        )
        target_xy = self.pass_through_xy.astype(np.float32).copy()
        target_z = desired_obj_z
        if cube_pos is not None and ee_pos is not None and self._is_grasped():
            offset = np.asarray(ee_pos, dtype=np.float32) - np.asarray(
                cube_pos, dtype=np.float32
            )
            target_xy = target_xy + offset[:2]
            target_z = desired_obj_z + float(offset[2])
        return np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)

    # ---------------------------------------------------- targets / context

    def _is_object_stack_target(self) -> bool:
        # Treat as "stack on top of another object" whenever the goal reference
        # is any scene entity whose position we can localize.
        if self.goal_reference_name is None:
            return False
        return self._object_pos_by_name(self.goal_reference_name) is not None

    def _release_dwell_steps(self) -> int:
        if self._is_object_stack_target():
            return max(20, self.close_steps)
        return max(8, self.close_steps // 2)

    def _goal_anchor(self) -> tuple[np.ndarray, float]:
        anchor_xy = self.goal_xy.astype(np.float32).copy()
        place_z = float(
            self.goal_z_override
            if self.goal_z_override is not None
            else (self.table_z + 0.03)
        )
        ref_name = self.goal_reference_name
        if ref_name is None:
            return anchor_xy, place_z
        ref_pos = self._object_pos_by_name(ref_name)
        if ref_pos is not None:
            anchor_xy = ref_pos[:2].astype(np.float32).copy()
            if self.goal_z_override is not None:
                place_z = float(self.goal_z_override)
            else:
                place_z = float(ref_pos[2] + 0.044)
            return anchor_xy, place_z
        # Legacy: a "knife" reference falls back to the keep-out XY.
        if str(ref_name) == "knife":
            anchor_xy = self.keepout_xy.astype(np.float32).copy()
            place_z = float(self.table_z + 0.032)
        return anchor_xy, place_z

    def _is_rim_object(self) -> bool:
        """Return True if the active object requires a rim grasp (bowl, mug, etc.)."""
        from .modes import RIM_GRASP_KEYWORDS
        name = str(getattr(self, "active_object_name", "") or "").lower()
        return any(kw in name for kw in RIM_GRASP_KEYWORDS)

    def _effective_grasp_rim_offset(self) -> np.ndarray:
        """Return the grasp rim offset only for rim-class objects; zero otherwise."""
        if self.grasp_offset_override is not None:
            return self.grasp_offset_override.astype(np.float32)
        if self._is_rim_object():
            return self.mode.grasp_rim_offset
        if self.active_object_height is not None and self.object_pose_is_base:
            # cube_pos is the object's base pose (table level), not its
            # centroid, so a zero offset drives the gripper down into tall
            # objects (cans, bottles, boxes) before it ever closes. Grasp at
            # roughly mid-height instead, using the known object height.
            #
            # Only where the platform reports the base. On one that reports the
            # centroid the pose already IS mid-height, and lifting again puts
            # the fingers at the top of the object -- see object_pose_is_base.
            half_height = float(np.clip(self.active_object_height * 0.5, 0.0, 0.15))
            return np.array([0.0, 0.0, half_height], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _effective_top_close_xy_tol(self) -> float:
        """Wider XY close tolerance for rim objects (gripper lands on the rim,
        not at the bowl centroid); tight tolerance for flat/block objects."""
        if self._is_rim_object():
            return self.top_grasp_close_xy_tol
        return 0.025

    def _targets(self, cube_pos: np.ndarray) -> StageTargets:
        cfg = self.mode
        goal_xy, place_z = self._goal_anchor()

        if self.mode_name == "top":
            grasp_offset = self._effective_grasp_rim_offset()
            # Hover above the object, then descend to the effective grasp point.
            # For rim objects (bowls, mugs, …) the offset lifts the target to the
            # rim so the gripper doesn't plunge into the interior.  For flat or
            # block-like objects the offset is zero and the gripper targets the
            # centroid directly.
            pregrasp = np.array(
                [
                    cube_pos[0] + grasp_offset[0],
                    cube_pos[1] + grasp_offset[1],
                    cube_pos[2] + self.top_grasp_hover_z,
                ],
                dtype=np.float32,
            )
            grasp = (cube_pos + grasp_offset).astype(np.float32)
        else:
            pregrasp = (cube_pos + cfg.pregrasp_offset).astype(np.float32)
            grasp = (cube_pos + cfg.grasp_offset).astype(np.float32)

        return StageTargets(
            pregrasp=pregrasp.astype(np.float32),
            grasp=grasp.astype(np.float32),
            carry=np.array([cube_pos[0], cube_pos[1], self.carry_z], dtype=np.float32),
            over_goal=np.array([goal_xy[0], goal_xy[1], self.carry_z], dtype=np.float32),
            place=np.array([goal_xy[0], goal_xy[1], place_z], dtype=np.float32),
            retreat=np.array([goal_xy[0], goal_xy[1], self.retreat_z], dtype=np.float32),
        )

    def _target_orientation(self) -> np.ndarray:
        base = rotvec_to_matrix(self.mode.world_rotvec_offset) @ self.top_orientation_R
        if abs(float(self.mode.local_wrist_roll)) > 1e-8:
            wrist_roll = rotvec_to_matrix(
                np.array([0.0, 0.0, self.mode.local_wrist_roll], dtype=np.float32)
            )
            base = base @ wrist_roll
        return orthonormalize(base).astype(np.float32)

    def _constraint_matches_current_goal(self, c: SDFConstraint) -> bool:
        if self.goal_reference_name is None:
            return False
        goal = str(self.goal_reference_name)
        return bool(c.name == goal or c.entity == goal)

    def set_active_sdf_constraints(self, names: set[str] | list[str] | tuple[str, ...] | None) -> None:
        self.active_sdf_constraint_names = {str(n) for n in (names or []) if n is not None}

    def _sdf_constraint_selected(self, c: SDFConstraint) -> bool:
        names = self.active_sdf_constraint_names
        return bool(c.always_active or c.name in names or (c.entity is not None and c.entity in names))

    def _active_sdf_constraints_for_stage(self) -> list[SDFConstraint]:
        active: list[SDFConstraint] = []
        for c in self.sdf_constraints:
            if not c.active_for_stage(self.stage):
                continue
            role = str(c.role)
            if role in {"obstacle", "keepout", "safety"}:
                if self._sdf_constraint_selected(c):
                    active.append(c)
            elif role == "goal_region":
                if self.stage >= 4 and self._constraint_matches_current_goal(c):
                    active.append(c)
            elif role in {"pass_through", "route"}:
                if self.stage == 4 and self.pass_through_label is not None:
                    if c.name == self.pass_through_label or c.entity == self.pass_through_label:
                        active.append(c)
        return active


    def _current_goal_sdf_constraint(self) -> SDFConstraint | None:
        for c in self.sdf_constraints:
            if str(c.role) == "goal_region" and self._constraint_matches_current_goal(c):
                return c
        return None

    def _goal_robustness(self, cube_pos: np.ndarray, *, inflation: float = 0.0) -> float:
        goal_c = self._current_goal_sdf_constraint()
        if goal_c is not None:
            return float(goal_c.robustness_xy(cube_pos[:2], inflation=inflation))
        goal_xy, _ = self._goal_anchor()
        return float(self.goal_radius + float(inflation) - np.linalg.norm(cube_pos[:2] - goal_xy))

    def _sdf_status_margins(self, ee_pos: np.ndarray, cube_pos: np.ndarray) -> dict[str, float | None]:
        margins: dict[str, float | None] = {}
        vals: list[float] = []
        for c in self._active_sdf_constraints_for_stage():
            if c.applies_to_kind("ee"):
                rho = float(c.robustness_xy(ee_pos[:2]))
                margins[f"sdf_{c.name}_ee"] = rho
                vals.append(rho)
            if c.applies_to_kind("object"):
                rho = float(c.robustness_xy(cube_pos[:2]))
                margins[f"sdf_{c.name}_object"] = rho
                vals.append(rho)
        margins["sdf_min"] = float(min(vals)) if vals else None
        return margins

    def _compute_robustness(self, ee_pos: np.ndarray, cube_pos: np.ndarray) -> Dict[str, float]:
        # Safety robustness is SDF-only.  The legacy analytic circular keepout
        # margin is deliberately not mixed into safe_ee/safe_object here; when
        # no SDF safety constraint is active, safety is treated as unconstrained
        # for this step instead of falling back to a hidden task-specific disk.
        sdf_margins = self._sdf_status_margins(ee_pos, cube_pos)
        ee_vals = [float(v) for k, v in sdf_margins.items() if v is not None and k.endswith("_ee")]
        object_vals = [float(v) for k, v in sdf_margins.items() if v is not None and k.endswith("_object")]
        safe_ee = float(min(ee_vals)) if ee_vals else 1.0
        safe_cube = float(min(object_vals)) if object_vals else 1.0

        over_goal = self._goal_robustness(cube_pos)
        carry = float(cube_pos[2] - self.carry_z)
        pass_through = None
        if self.pass_through_xy is not None:
            pass_through = float(
                self.pass_through_radius - np.linalg.norm(cube_pos[:2] - self.pass_through_xy)
            )

        return {
            "safe_ee": safe_ee,
            "safe_object": safe_cube,
            "over_goal": over_goal,
            "carry": carry,
            "pass_through": pass_through,
            **sdf_margins,
        }


    # ----------------------------------------------------------- stage fsm

    def _ready_to_close_gripper(
        self,
        ee_pos: np.ndarray,
        cube_pos: np.ndarray,
        ori_err: float,
        grasp_dist: float,
    ) -> bool:
        """Return True when the end-effector is at a valid grasp pose.

        For top grasp, this literally checks whether the end-effector is in the
        middle of the cube: small XY error and small Z error relative to the
        active cube center. No hidden offsets or large distance thresholds.
        """
        if self.mode_name == "top":
            rim_offset = self._effective_grasp_rim_offset()
            grasp_target = cube_pos + rim_offset
            xy_err = float(np.linalg.norm(ee_pos[:2] - grasp_target[:2]))
            z_err = abs(float(ee_pos[2]) - float(grasp_target[2]))
            return bool(
                xy_err <= self._effective_top_close_xy_tol()
                and z_err <= self.top_grasp_close_z_tol
                and ori_err < 0.70
            )

        if self.mode_name in {"left", "right"}:
            # Side grasps are more sensitive to wrist-orientation details in
            # robosuite. Use the planned side-grasp point as the primary close
            # predicate and keep orientation as a loose sanity check.
            return bool(
                grasp_dist <= self.side_grasp_close_pos_tol
                and ori_err <= self.side_grasp_close_ori_tol
            )

        return bool(grasp_dist < 0.045 and ori_err < 0.65)

    def _reset_grasp_attempt(self) -> None:
        """Reattempt the pick without resetting the simulator scene."""
        self.stage = 0
        self.close_counter = 0
        self.grasp_retry_counter += 1
        try:
            self.nominal_mpc.reset()
        except Exception:
            pass
        self.last_nominal_pos.fill(0.0)
        self.last_nominal_ori.fill(0.0)

    def _update_stage(self, ee_pos: np.ndarray, cube_pos: np.ndarray, ee_R: np.ndarray) -> None:
        targets = self._targets(cube_pos)
        pregrasp_dist = float(np.linalg.norm(ee_pos - targets.pregrasp))
        grasp_dist = float(np.linalg.norm(ee_pos - targets.grasp))
        goal_xy, goal_place_z = self._goal_anchor()
        over_goal_xy = float(np.linalg.norm(cube_pos[:2] - goal_xy))
        goal_margin = self._goal_robustness(cube_pos)
        descend_inflation = max(0.0, float(self.descend_radius) - float(self.goal_radius))
        descend_goal_margin = self._goal_robustness(cube_pos, inflation=descend_inflation)
        if self.pass_through_xy is not None:
            pass_through_xy_dist = float(
                np.linalg.norm(cube_pos[:2] - self.pass_through_xy)
            )
            visit_tol = self.pass_through_radius + 0.005
            if (
                self.stage == 4
                and (not self.pass_through_visited)
                and pass_through_xy_dist <= visit_tol
            ):
                self.pass_through_visited = True
        place_target_dist = float(np.linalg.norm(ee_pos - targets.place))
        ori_err = float(
            np.linalg.norm(orientation_error_rotvec(ee_R, self._target_orientation()))
        )

        z_ready = abs(float(cube_pos[2]) - float(goal_place_z)) <= 0.04
        ee_ready = place_target_dist <= 0.035
        xy_ready = goal_margin >= 0.0

        if self.stage == 0 and pregrasp_dist < 0.038:
            self.stage = 1
        elif self.stage == 1 and self._ready_to_close_gripper(ee_pos, cube_pos, ori_err, grasp_dist):
            self.stage = 2
            self.close_counter = 0
        elif self.stage == 2:
            self.close_counter += 1
            # Never lift just because the close timer elapsed. Stage 3 means
            # the backend reports a real grasp.
            if self._is_grasped():
                self.stage = 3
                self.close_counter = 0
                self.grasp_retry_counter = 0
            elif self.close_counter >= max(24, 2 * self.close_steps):
                self._reset_grasp_attempt()
        elif self.stage == 3:
            # If the object is lost during early lift, restart the pick.
            if not self._is_grasped():
                self._reset_grasp_attempt()
            elif cube_pos[2] >= self.carry_z - 0.02:
                self.stage = 4
        elif (
            self.stage == 4
            and (not self._has_pending_pass_through())
            and descend_goal_margin >= 0.0
        ):
            self.stage = 5
        elif self.stage == 5 and xy_ready and (ee_ready or z_ready):
            self.stage = 6
            self.close_counter = 0
            self.release_hold_pos = ee_pos.astype(np.float32).copy()
            self.retreat_anchor_xy = ee_pos[:2].astype(np.float32).copy()
        elif self.stage == 6:
            self.close_counter += 1
            if self.close_counter >= self._release_dwell_steps():
                self.stage = 7

    def _current_target(
        self,
        cube_pos: np.ndarray,
        ee_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        targets = self._targets(cube_pos)
        target_R = self._target_orientation()

        # Lower-bounded timing windows for pick are realized as an MPC target
        # switch, not as a post-hoc abort or task-specific gripper gate.  Before
        # the window opens, the grasp-success region is forbidden by the timing
        # robustness cost and the nominal target remains the predecessor set.
        # Once the lower bound opens, this returns None and the original stage
        # logic below runs unchanged.
        pre_window_hold = self._pre_window_hold_target_for_current_pick(cube_pos)
        if pre_window_hold is not None:
            return pre_window_hold, target_R, self.open_gripper_cmd

        if self.stage == 0:
            return targets.pregrasp, target_R, self.open_gripper_cmd
        if self.stage == 1:
            return targets.grasp, target_R, self.open_gripper_cmd
        if self.stage == 2:
            return targets.grasp, target_R, self.close_gripper_cmd
        if self.stage == 3:
            return targets.carry, target_R, self.close_gripper_cmd
        if self.stage == 4:
            pass_target = self._pass_through_target(cube_pos=cube_pos, ee_pos=ee_pos)
            if pass_target is not None and not self.pass_through_visited:
                return pass_target, target_R, self.close_gripper_cmd
            return targets.over_goal, target_R, self.close_gripper_cmd
        if self.stage == 5:
            return targets.place, target_R, self.close_gripper_cmd
        if self.stage == 6:
            if self.release_hold_pos is not None:
                return self.release_hold_pos.astype(np.float32), target_R, self.open_gripper_cmd
            return targets.place, target_R, self.open_gripper_cmd
        if self.stage == 7:
            if self.retreat_anchor_xy is not None:
                retreat = np.array(
                    [self.retreat_anchor_xy[0], self.retreat_anchor_xy[1], self.retreat_z],
                    dtype=np.float32,
                )
                return retreat, target_R, self.open_gripper_cmd
            return targets.retreat, target_R, self.open_gripper_cmd
        return targets.retreat, target_R, self.open_gripper_cmd

    def _stl_context(
        self,
        target_xyz: np.ndarray,
        cube_pos: np.ndarray,
        ee_pos: np.ndarray | None = None,
    ) -> STLStageContext:
        carry_active = self.stage in (3, 4)
        tracked_pos = (
            cube_pos
            if self.stage in (3, 4, 5) or self._is_grasped()
            else None
        )
        goal_xy, _ = self._goal_anchor()
        if (
            self.stage == 4
            and self._has_pending_pass_through()
            and self.pass_through_xy is not None
        ):
            goal_xy = self.pass_through_xy.astype(np.float32)
        timing_windows = self._mpc_timing_windows_for_context(ee_pos=ee_pos, cube_pos=cube_pos)
        return STLStageContext(
            target_xyz=target_xyz.astype(np.float32),
            # Legacy keepout fields are passed only for API compatibility.  The
            # MPC ignores them and uses sdf_constraints for geometric safety.
            knife_xy=self.keepout_xy.astype(np.float32),
            knife_radius=0.0,
            table_z=self.table_z,
            always_keepout=False,
            carry_active=carry_active,
            carry_min_z=self.carry_z,
            goal_xy=goal_xy.astype(np.float32),
            goal_radius=self.descend_radius,
            mug_xy_radius=0.0,
            mug_pos=None if tracked_pos is None else tracked_pos.astype(np.float32),
            workspace_min_xyz=self.workspace_min_xyz,
            workspace_max_xyz=self.workspace_max_xyz,
            workspace_weight=self.mpc_workspace_weight,
            sdf_constraints=self._active_sdf_constraints_for_stage(),
            timing_windows=timing_windows,
            dt_sec=1.0 / max(float(self.timing_control_hz), 1e-6),
        )

    def _configure_mpc_for_stage(self) -> tuple[float, float, float, float]:
        action_scale = 0.012
        pos_alpha = 0.28
        ori_alpha = 0.42
        max_pos_delta = 0.06
        max_ori_delta = 0.06
        if self.stage in (3, 4):
            action_scale = 0.026
            pos_alpha = 0.38
            ori_alpha = 0.34
            max_pos_delta = 0.13
            max_ori_delta = 0.13
        if self.stage in (5, 6, 7):
            action_scale = 0.012
            pos_alpha = 0.28
            ori_alpha = 0.26
            max_pos_delta = 0.075
            max_ori_delta = 0.08
        if self.goal_radius <= 0.015 and self.stage in (4, 5, 6):
            action_scale = min(action_scale, 0.010)
            pos_alpha = min(pos_alpha, 0.24)
            max_pos_delta = min(max_pos_delta, 0.055)
        self.nominal_mpc.action_scale_m = float(action_scale)
        return (
            float(pos_alpha),
            float(ori_alpha),
            float(max_pos_delta),
            float(max_ori_delta),
        )

    @staticmethod
    def _smooth_action(
        raw: np.ndarray,
        prev: np.ndarray,
        alpha: float,
        max_delta: float,
    ) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        prev = np.asarray(prev, dtype=np.float32)
        blended = alpha * raw + (1.0 - alpha) * prev
        delta = np.clip(blended - prev, -float(max_delta), float(max_delta))
        return np.clip(prev + delta, -1.0, 1.0).astype(np.float32)

    def _direct_cartesian_servo_action(
        self,
        ee_pos: np.ndarray,
        target_xyz: np.ndarray,
        *,
        scale_m: float,
    ) -> np.ndarray:
        """Convert Cartesian position error into normalized delta action."""
        target = self._clip_xyz_to_workspace(np.asarray(target_xyz, dtype=np.float32))
        err = target - np.asarray(ee_pos, dtype=np.float32)
        raw = np.clip(err / max(float(scale_m), 1e-6), -1.0, 1.0).astype(np.float32)
        return self._clip_delta_action_to_workspace(ee_pos, raw, scale_m=float(scale_m))

    def _nominal_action(
        self,
        ee_pos: np.ndarray,
        cube_pos: np.ndarray,
        ee_R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        self._update_stage(ee_pos, cube_pos, ee_R)
        target_xyz, target_R, grip = self._current_target(cube_pos, ee_pos=ee_pos)
        target_xyz = self._clip_xyz_to_workspace(target_xyz)
        pos_alpha, ori_alpha, max_pos_delta, max_ori_delta = self._configure_mpc_for_stage()
        ctx = self._stl_context(target_xyz, cube_pos, ee_pos=ee_pos)
        timing_mpc_active = bool(ctx.timing_windows)
        if (
            self.stage in (0, 1, 2)
            and self.mode_name in {"top", "left", "right"}
            and not timing_mpc_active
        ):
            # For grasp acquisition, use a literal Cartesian servo to the
            # requested grasp target. Keep MPC for transport/place where STL
            # constraints matter more. This is important for side grasps: the
            # old soft-MPC path often hovered near the side target but did not
            # descend/advance enough to trigger gripper closure.
            if self.mode_name == "top":
                scale_m = self.top_pick_servo_scale_m
                alpha = self.top_pick_servo_alpha
                max_delta = self.top_pick_servo_max_delta
            else:
                scale_m = self.side_pick_servo_scale_m
                alpha = self.side_pick_servo_alpha
                max_delta = self.side_pick_servo_max_delta
            raw_pos = self._direct_cartesian_servo_action(
                ee_pos=ee_pos,
                target_xyz=target_xyz,
                scale_m=scale_m,
            )
            nominal_pos = self._smooth_action(
                raw=raw_pos,
                prev=self.last_nominal_pos,
                alpha=alpha,
                max_delta=max_delta,
            )
        else:
            raw_pos = self.nominal_mpc.act(ee_pos, ctx).astype(np.float32)
            nominal_pos = self._smooth_action(
                raw=raw_pos,
                prev=self.last_nominal_pos,
                alpha=pos_alpha,
                max_delta=max_pos_delta,
            )
        nominal_pos = self._clip_delta_action_to_workspace(
            ee_pos,
            nominal_pos,
            scale_m=float(self.nominal_mpc.action_scale_m),
        )
        ori_err = orientation_error_rotvec(ee_R, target_R)
        raw_ori = np.clip(self.nominal_ori_gain * ori_err, -1.0, 1.0).astype(np.float32)
        nominal_ori = self._smooth_action(
            raw=raw_ori,
            prev=self.last_nominal_ori,
            alpha=ori_alpha,
            max_delta=max_ori_delta,
        )
        return (
            nominal_pos.astype(np.float32),
            nominal_ori.astype(np.float32),
            float(grip),
            target_xyz.astype(np.float32),
            target_R.astype(np.float32),
        )

    def _compose_full_action(
        self,
        cart_pos: np.ndarray,
        cart_ori: np.ndarray,
        gripper_cmd: float,
    ) -> np.ndarray:
        action = np.zeros((self.action_dim_underlying,), dtype=np.float32)
        action[:3] = cart_pos.astype(np.float32)
        action[3:6] = cart_ori.astype(np.float32)
        if self.gripper_dim > 0:
            action[self.arm_dim:] = gripper_cmd
        return np.clip(action, self.low, self.high)

    # ------------------------------------------------------------ orient util
    def orientation_error_rotvec(self, current_R: np.ndarray, target_R: np.ndarray) -> np.ndarray:
        return orientation_error_rotvec(current_R, target_R)

    # ------------------------------------------------------ simple real api

    def simple_pregrasp_target(self, object_xyz: np.ndarray) -> np.ndarray:
        """Object-agnostic top-approach waypoint used by the real simple executor."""
        obj = np.asarray(object_xyz, dtype=np.float32)
        offset = self._effective_grasp_rim_offset()
        return np.array(
            [obj[0] + offset[0], obj[1] + offset[1], obj[2] + float(self.top_grasp_hover_z)],
            dtype=np.float32,
        )

    def simple_grasp_target(self, object_xyz: np.ndarray) -> np.ndarray:
        """Object-agnostic grasp waypoint.

        For the real simple controller, perception decides what object_xyz means.
        In the current RealSense + SAM path this is usually a top/surface point,
        so the runtime should not add object-specific cube offsets here.
        """
        return (np.asarray(object_xyz, dtype=np.float32) + self._effective_grasp_rim_offset()).astype(np.float32)

    def simple_carry_target(self, object_xyz: np.ndarray) -> np.ndarray:
        """Lift the active object vertically to carry height."""
        obj = np.asarray(object_xyz, dtype=np.float32)
        return np.array([obj[0], obj[1], float(self.carry_z)], dtype=np.float32)

    def simple_over_goal_target(self, goal_xyz: np.ndarray | None = None) -> np.ndarray:
        """Waypoint above the current goal at carry height."""
        if goal_xyz is None:
            goal_xy, _ = self._goal_anchor()
            return np.array([goal_xy[0], goal_xy[1], float(self.carry_z)], dtype=np.float32)
        goal = np.asarray(goal_xyz, dtype=np.float32)
        return np.array([goal[0], goal[1], float(self.carry_z)], dtype=np.float32)

    # ------------------------------------------------------------ public api

    def reset(self) -> Dict[str, Any]:
        raw_obs = self._reset_backend()
        self.stage = 0
        self.close_counter = 0
        self.grasp_retry_counter = 0
        self.step_idx = 0
        self.nominal_mpc.reset()
        self.last_raw_obs = raw_obs
        self.last_nominal_pos.fill(0.0)
        self.last_nominal_ori.fill(0.0)
        self.last_full_action.fill(0.0)
        self.release_hold_pos = None
        self.retreat_anchor_xy = None
        self.top_orientation_R = self._eef_rotmat(raw_obs)
        self.set_active_object(self.active_object_name)
        self.prev_ee_pos = self._eef_pos(raw_obs)
        cube_pos = self._cube_pos(raw_obs)
        self.prev_cube_height = float(cube_pos[2])
        target_xyz, _, _ = self._current_target(cube_pos)
        self.prev_target_dist = float(np.linalg.norm(self.prev_ee_pos - target_xyz))
        self.prev_ori_err = float(
            np.linalg.norm(
                orientation_error_rotvec(
                    self.top_orientation_R, self._target_orientation()
                )
            )
        )
        self.rollout_log = []
        self._on_after_reset()
        return raw_obs

    def _on_after_reset(self) -> None:
        """Hook for backends to draw debug markers, etc."""

    def step_nominal(self) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.last_raw_obs is None:
            raise RuntimeError("Call reset() before step_nominal().")
        ee_pos = self._eef_pos(self.last_raw_obs)
        ee_R = self._eef_rotmat(self.last_raw_obs)
        cube_pos = self._cube_pos(self.last_raw_obs)
        nominal_pos, nominal_ori, grip, target_xyz, target_R = self._nominal_action(
            ee_pos, cube_pos, ee_R
        )
        full_action = self._compose_full_action(nominal_pos, nominal_ori, grip)
        raw_obs, base_reward, terminated, info = self._step_backend(full_action)
        self.last_raw_obs = raw_obs
        self.step_idx += 1
        self.timing_subtask_step_idx += 1
        self._on_after_step()

        new_ee = self._eef_pos(raw_obs)
        new_R = self._eef_rotmat(raw_obs)
        new_cube = self._cube_pos(raw_obs)
        target_dist = float(np.linalg.norm(new_ee - target_xyz))
        progress_reward = 3.2 * (self.prev_target_dist - target_dist)
        ori_err_now = float(
            np.linalg.norm(orientation_error_rotvec(new_R, target_R))
        )
        ori_progress_reward = 1.4 * (self.prev_ori_err - ori_err_now)
        robustness = self._compute_robustness(new_ee, new_cube)
        safety_violation_penalty = (
            -8.0 * max(0.0, -robustness["safe_ee"])
            - 10.0 * max(0.0, -robustness["safe_object"])
        )
        carry_reward = 0.0
        if self.stage in (3, 4):
            carry_reward += 1.5 * min(robustness["carry"], 0.05)
        if self.stage == 4:
            carry_reward += 1.0 * min(robustness["over_goal"], 0.05)
        goal_xy, goal_place_z = self._goal_anchor()
        place_reward = 0.0
        if self.stage >= 5:
            dist_to_goal_xy = float(np.linalg.norm(new_cube[:2] - goal_xy))
            place_reward += (
                1.2
                * (self.descend_radius - min(dist_to_goal_xy, self.descend_radius))
                / max(self.descend_radius, 1e-6)
            )
        lift_progress = np.clip(
            (float(new_cube[2]) - self.prev_cube_height) / 0.05, -1.0, 1.0
        )
        lift_reward = 1.2 * lift_progress
        grasp_bonus = 2.5 if self._is_grasped() else 0.0
        success = self._is_success()
        success_bonus = 20.0 if success else 0.0
        time_penalty = -0.01
        reward = (
            float(base_reward)
            + progress_reward
            + ori_progress_reward
            + safety_violation_penalty
            + carry_reward
            + place_reward
            + lift_reward
            + grasp_bonus
            + success_bonus
            + time_penalty
        )

        self.prev_target_dist = target_dist
        self.prev_cube_height = float(new_cube[2])
        self.prev_ee_pos = new_ee.copy()
        self.prev_ori_err = ori_err_now
        self.last_target_xyz = target_xyz.copy()
        self.last_nominal_pos = nominal_pos.copy()
        self.last_nominal_ori = nominal_ori.copy()
        self.last_full_action = full_action.copy()

        out_info: Dict[str, Any] = dict(info)
        out_info.update(
            {
                "mode": self.mode_name,
                "stage": int(self.stage),
                "grasp_retry_counter": int(self.grasp_retry_counter),
                "success": success,
                "base_reward": float(base_reward),
                "reward": float(reward),
                "nominal_pos": nominal_pos.copy(),
                "nominal_ori": nominal_ori.copy(),
                "mpc_action_scale_m": float(self.nominal_mpc.action_scale_m),
                "timing_mpc_active": bool(getattr(self, "active_timing_windows", [])),
                "timing_subtask_elapsed_sec": float(self.timing_subtask_step_idx) / max(float(self.timing_control_hz), 1e-6),
                "timing_pre_window_hold_active": self._pre_window_hold_active_for_current_pick(new_cube),
                "target_xyz": target_xyz.copy(),
                "target_pos_err": float(np.linalg.norm(new_ee - target_xyz)),
                "target_ori_err": ori_err_now,
                "cube_pos": new_cube.copy(),
                "eef_pos": new_ee.copy(),
                "top_grasp_xy_err": float(np.linalg.norm(new_ee[:2] - (new_cube + self._effective_grasp_rim_offset())[:2])) if self.mode_name == "top" else None,
                "top_grasp_z_err": abs(float(new_ee[2]) - float((new_cube + self._effective_grasp_rim_offset())[2])) if self.mode_name == "top" else None,
                "eef_minus_cube_z": float(new_ee[2] - new_cube[2]) if self.mode_name == "top" else None,
                "goal_xy": goal_xy.copy(),
                "goal_place_z": goal_place_z,
                "keepout_xy": self.keepout_xy.copy(),
                **robustness,
            }
        )
        self.rollout_log.append(out_info)
        terminated = bool(terminated)
        return raw_obs, float(reward), terminated, out_info

    def _on_after_step(self) -> None:
        """Hook called after each backend step (used for debug markers in sim)."""

    def _is_success(self) -> bool:
        if self.last_raw_obs is None:
            return False
        cube = self._cube_pos(self.last_raw_obs)
        grasped = self._is_grasped()
        goal_xy, goal_place_z = self._goal_anchor()
        over_goal = float(np.linalg.norm(cube[:2] - goal_xy)) <= self.goal_radius
        if self.goal_reference_name is None:
            target_z = float(
                self.goal_z_override
                if self.goal_z_override is not None
                else (self.table_z + 0.03)
            )
            z_ok = abs(float(cube[2]) - target_z) <= 0.05
        else:
            z_ok = abs(float(cube[2]) - float(goal_place_z)) <= 0.05
        return bool(over_goal and z_ok and not grasped)

    def status(self) -> Dict[str, Any]:
        if self.last_raw_obs is None:
            raise RuntimeError("Call reset() before status().")
        cube = self._cube_pos(self.last_raw_obs)
        ee = self._eef_pos(self.last_raw_obs)
        robustness = self._compute_robustness(ee, cube)
        goal_xy, goal_place_z = self._goal_anchor()
        return {
            "stage": self.stage,
            "grasp_retry_counter": int(self.grasp_retry_counter),
            "grasped": self._is_grasped(),
            "success": self._is_success(),
            "cube_pos": cube,
            "eef_pos": ee,
            "top_grasp_xy_err": float(np.linalg.norm(ee[:2] - (cube + self._effective_grasp_rim_offset())[:2])) if self.mode_name == "top" else None,
            "top_grasp_z_err": abs(float(ee[2]) - float((cube + self._effective_grasp_rim_offset())[2])) if self.mode_name == "top" else None,
            "eef_minus_cube_z": float(ee[2] - cube[2]) if self.mode_name == "top" else None,
            "goal_anchor_xy": goal_xy,
            "goal_place_z": goal_place_z,
            "goal_reference_name": self.goal_reference_name,
            "timing_mpc_active": bool(self.active_timing_windows),
            "timing_subtask_elapsed_sec": float(self.timing_subtask_step_idx) / max(float(self.timing_control_hz), 1e-6),
            "timing_pre_window_hold_active": self._pre_window_hold_active_for_current_pick(cube),
            "pass_through_xy": None
            if self.pass_through_xy is None
            else self.pass_through_xy.copy(),
            "pass_through_radius": self.pass_through_radius,
            "pass_through_visited": self.pass_through_visited,
            "pass_through_label": self.pass_through_label,
            **robustness,
        }

    # --------------------------------------------------------------- plot

    def plot_xy(self, save_path: str | None = None):
        if not self.rollout_log:
            raise RuntimeError("No rollout data. Run execution first.")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        ee_xy = np.array(
            [step["eef_pos"][:2] for step in self.rollout_log], dtype=np.float32
        )
        cube_xy = np.array(
            [step["cube_pos"][:2] for step in self.rollout_log], dtype=np.float32
        )
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        ax.plot(ee_xy[:, 0], ee_xy[:, 1], label="ee path", linewidth=2.0)
        ax.plot(cube_xy[:, 0], cube_xy[:, 1], label="object path", linewidth=2.0)
        ax.scatter([ee_xy[0, 0]], [ee_xy[0, 1]], marker="o", s=80, label="ee start")
        ax.scatter(
            [cube_xy[0, 0]], [cube_xy[0, 1]], marker="s", s=80, label="object start"
        )
        ax.scatter(
            [self.goal_xy[0]],
            [self.goal_xy[1]],
            marker="*",
            s=180,
            label="goal center",
        )
        if self.keepout_radius > 1e-6:
            keepout = patches.Circle(
                tuple(self.keepout_xy.tolist()),
                radius=self.keepout_radius,
                alpha=0.22,
                color="tab:red",
                label="obstacle keep-out",
            )
            ax.add_patch(keepout)
        goal = patches.Circle(
            tuple(self.goal_xy.tolist()),
            radius=self.goal_radius,
            fill=False,
            linestyle="--",
            linewidth=2.0,
            edgecolor="tab:blue",
            label="goal region",
        )
        ax.add_patch(goal)
        descend = patches.Circle(
            tuple(self.goal_xy.tolist()),
            radius=self.descend_radius,
            fill=False,
            linestyle=":",
            linewidth=2.0,
            edgecolor="tab:blue",
            label="descend radius",
        )
        ax.add_patch(descend)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Manipulation rollout ({self.mode_name})")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=160)
        return fig, ax
