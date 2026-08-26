"""Real-world manipulation backend (UR3e + 2x RealSense + Robotiq).

This is the concrete subclass of :class:`BaseManipulationEnv` that drives a
physical UR3e arm. Like ``sim_env.py``, it only contains backend-specific
I/O; the stage machine and STL/MPC code are inherited from
:mod:`system1.base_env`, so improvements to the controller transfer to both
sim and real automatically.

Key design choices:

* **Scene state lives in :py:attr:`scene_objects`.** At reset time we capture
  RGBD frames from the configured RealSense cameras, ask a VLM (via
  :class:`ScenePerception`) to point at every entity declared in
  ``scene_context.json``, deproject those pixels into the robot base frame,
  and store the result in :py:attr:`scene_objects`. The stage machine then
  uses ``scene_objects[active_object_name]`` as the cube/object position.
* **Cube position while grasped is approximated by the EE.** Once the gripper
  reports ``STOPPED_OUTER_OBJECT`` we record the EE-to-object offset and
  thereafter compute ``cube_pos = ee_pos + offset``. After release we clear
  the offset and re-localize.
* **Cartesian commands go through ``servoL``.** Each MPC step produces a
  normalized ``[-1, 1]`` delta which we convert into a per-step EE
  displacement of ``mpc.action_scale_m`` meters and stream to the robot.
* **Mock mode.** If ``--mock-robot`` is set on the CLI (or ``mock=True`` is
  passed here) the controller is replaced by :class:`UR3eMockController` and
  RealSense capture is skipped — useful for dry-running the whole pipeline
  on a laptop without hardware.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base_env import BaseManipulationEnv
from .hardware.robotiq_gripper import RobotiqGripper
from .hardware.ur3e_controller import UR3eController, UR3eMockController
from .orientation_utils import matrix_to_rotvec, rotvec_to_matrix
from .perception.projector import Projector
from .perception.realsense import RealSenseCamera, RealSenseStream
from .perception.sam_segmenter import (
    SAMSegmenter,
    create_point_cloud_from_rgbd,
    find_sam_checkpoint,
)
from .perception.scene_discovery import DiscoveredEntity, SceneDiscovery
from .perception.scene_perception import ScenePerception, save_color_image


class RealManipulationEnv(BaseManipulationEnv):
    """Cross-platform System 1 backend for the UR3e + RealSense rig."""

    def __init__(
        self,
        scene_context: Dict[str, Any],
        *,
        mode: str = "top",
        seed: int = 0,
        mock: bool = False,
        skip_perception: bool = False,
        active_object_name: Optional[str] = None,
        goal_reference_name: Optional[str] = None,
        carry_height_above_table: Optional[float] = None,
        retreat_height_above_table: Optional[float] = None,
        descend_radius: float = 0.075,
        keepout_xy: tuple[float, float] = (0.0, -0.30),
        keepout_radius: float = 0.0,
        goal_xy: Optional[tuple[float, float]] = None,
        goal_radius: float = 0.06,
        goal_z: Optional[float] = None,
        close_steps: int = 14,
        cwd: Optional[Path] = None,
        per_step_rot_scale: float = 0.0,
        sdf_constraints: list[Any] | None = None,
        debug_dir: str | Path | None = None,
        mpc_timing_enabled: bool = True,
        mpc_timing_early_weight: float = 220.0,
        mpc_timing_deadline_weight: float = 55.0,
        mpc_timing_window_weight: float = 10.0,
        mpc_timing_early_hold_weight: float = 260.0,
    ) -> None:
        self.scene_context = scene_context or {}
        self.mock = bool(mock)
        self.skip_perception = bool(skip_perception)
        self.cwd = Path(cwd or Path.cwd())
        self.per_step_rot_scale = float(per_step_rot_scale)
        
        if debug_dir is None:
            import datetime as _dt
            _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._debug_dir = self.cwd / "outputs" / "debug" / _ts
        else:
            self._debug_dir = Path(debug_dir)
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._gripper_closed_flag = False
        self._grasped_flag = False
        self._ee_to_object_offset: Optional[np.ndarray] = None
        self._latest_color: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        # Extra perception metadata used for debugging real-world grasp points.
        # Keys are entity names; values may include source pixel, SAM mark, etc.
        self.scene_object_debug: Dict[str, Any] = {}

        platform = self.scene_context.get("platform", {})
        self.platform_cfg = platform
        self.workspace_bounds = platform.get(
            "workspace_bounds",
            {"x": [-0.34, 0.32], "y": [-0.453, -0.11], "z": [-0.016, 0.45]},
        )
        perception_cfg = platform.get("perception", {})
        table_z = float(perception_cfg.get("table_surface_z", 0.0))
        if carry_height_above_table is None:
            carry_height_above_table = float(
                perception_cfg.get("carry_height_above_table", 0.20)
            )
        if retreat_height_above_table is None:
            retreat_height_above_table = float(
                perception_cfg.get("retreat_height_above_table", 0.28)
            )

        # Pick a default active_object_name if the caller did not.
        if active_object_name is None:
            for ent in self.scene_context.get("entities", []):
                if ent.get("role") == "manipulated_object":
                    active_object_name = ent["name"]
                    break
            if active_object_name is None:
                active_object_name = "object"

        # Pick a default goal_xy if the caller did not (use first goal_region's geometry).
        if goal_xy is None:
            for ent in self.scene_context.get("entities", []):
                if ent.get("role") == "goal_region":
                    geom = self.scene_context.get("geometry", {}).get(ent["name"], {})
                    if "xy" in geom:
                        goal_xy = (float(geom["xy"][0]), float(geom["xy"][1]))
                        if "radius" in geom:
                            goal_radius = float(geom["radius"])
                        break
        if goal_xy is None:
            goal_xy = (0.10, -0.30)

        super().__init__(
            mode=mode,
            seed=seed,
            keepout_xy=keepout_xy,
            keepout_radius=keepout_radius,
            goal_xy=goal_xy,
            goal_radius=goal_radius,
            goal_z=goal_z,
            descend_radius=descend_radius,
            carry_height_above_table=carry_height_above_table,
            retreat_height_above_table=retreat_height_above_table,
            close_steps=close_steps,
            active_object_name=active_object_name,
            goal_reference_name=goal_reference_name,
            open_gripper_cmd=-1.0,
            close_gripper_cmd=1.0,
            action_dim=7,
            gripper_dim=1,
            action_low=np.array([-1.0] * 7, dtype=np.float32),
            action_high=np.array([1.0] * 7, dtype=np.float32),
            table_z=table_z,
            workspace_bounds=self.workspace_bounds,
            sdf_constraints=sdf_constraints,
            mpc_workspace_weight=float(
                perception_cfg.get(
                    "mpc_workspace_weight",
                    platform.get("mpc_workspace_weight", 0.0),
                )
            ),
            mpc_timing_enabled=bool(mpc_timing_enabled),
            mpc_timing_early_weight=float(mpc_timing_early_weight),
            mpc_timing_deadline_weight=float(mpc_timing_deadline_weight),
            mpc_timing_window_weight=float(mpc_timing_window_weight),
            mpc_timing_early_hold_weight=float(mpc_timing_early_hold_weight),
        )

        # ---- Build hardware ----
        robot_cfg = platform.get("robot", {})
        self.home_joints_rad = list(
            robot_cfg.get(
                "home_joints_rad",
                [-1.5668, -1.3176, -1.3577, -2.0099, 1.5782, 0.0152],
            )
        )
        self.tcp_speed = float(robot_cfg.get("tcp_speed", 0.05))
        self.tcp_accel = float(robot_cfg.get("tcp_accel", 0.10))
        if self.mock:
            self.controller = UR3eMockController(
                workspace_bounds=self.workspace_bounds
            )
        else:
            self.controller = UR3eController(
                ip=str(robot_cfg.get("ip", "192.168.77.23")),
                workspace_bounds=self.workspace_bounds,
                servo_dt=float(robot_cfg.get("servo_dt", 0.05)),
                servo_lookahead=float(robot_cfg.get("servo_lookahead", 0.10)),
                servo_gain=int(robot_cfg.get("servo_gain", 300)),
            )

        # ---- Build gripper ----
        self.gripper_cfg = platform.get("gripper", {})
        # Real Robotiq grasp feedback is often unreliable for small cubes. By
        # default, the simple real executor treats successful pick motion
        # completion as enough to continue, while still logging the gripper
        # feedback separately. Set platform.gripper.require_grasp_feedback=true
        # if you want a failed feedback signal to stop the plan.
        self.require_grasp_feedback: bool = bool(
            self.gripper_cfg.get("require_grasp_feedback", False)
        )
        self.gripper: Optional[RobotiqGripper] = None
        if not self.mock:
            try:
                self.gripper = RobotiqGripper()
            except Exception:
                self.gripper = None

        # ---- Build cameras + projectors ----
        self.cameras = RealSenseStream()
        self.projectors: Dict[str, Projector] = {}
        cameras_cfg = platform.get("cameras", {})
        primary_cam_name: Optional[str] = None
        for cam_name, cam_cfg in cameras_cfg.items():
            calib_path = cam_cfg.get("calibration")
            if calib_path:
                calib_path = self.cwd / calib_path
                if Path(calib_path).is_file():
                    self.projectors[cam_name] = Projector(calib_path)
                else:
                    print(f"[real_env] WARNING: calibration not found at {calib_path}; camera '{cam_name}' will only be used for VLM image input.")
            self.cameras.add(
                RealSenseCamera(
                    serial=str(cam_cfg.get("serial", "")),
                    name=cam_name,
                    width=int(cam_cfg.get("resolution", [640, 480])[0]),
                    height=int(cam_cfg.get("resolution", [640, 480])[1]),
                    fps=int(cam_cfg.get("fps", 30)),
                )
            )
            if cam_cfg.get("is_primary_for_vlm", False) or primary_cam_name is None:
                primary_cam_name = cam_name
        self.primary_camera_name = primary_cam_name
        self.primary_projector: Optional[Projector] = (
            self.projectors.get(primary_cam_name) if primary_cam_name else None
        )

        # ---- Scene perception (lazy) ----
        self.scene_perception: Optional[ScenePerception] = None
        self.scene_discovery: Optional[SceneDiscovery] = None
        self._sam: Optional[SAMSegmenter] = None
        self.use_sam: bool = bool(perception_cfg.get("use_sam", True))
        self.use_pcd_fusion: bool = bool(perception_cfg.get("use_pcd_fusion", False))
        # Use the same VLM-grid + SAM grounding path that worked in the legacy
        # interactive planner. Set platform.perception.pipeline='legacy' to use
        # the older ScenePerception / SceneDiscovery classes instead.
        self.perception_pipeline: str = str(perception_cfg.get("pipeline", "interactive"))
        self.legacy_perception_root: str = str(
            perception_cfg.get("legacy_perception_root", "temp_real_robot_files")
        )
        self.use_som: bool = bool(perception_cfg.get("use_som", False))

        # Real top-grasp calibration.
        #
        # The SoM/SAM geometry reports a surface/top point for graspable
        # objects. If we command the TCP to exactly that z, the gripper closes
        # above the object. For real top grasps, descend below the perceived
        # surface point by a calibrated amount. This is gripper/scene geometry,
        # not object-specific logic.
        self.real_top_grasp_depth_below_surface = float(
            perception_cfg.get("top_grasp_depth_below_surface", 0.045)
        )
        self.real_top_grasp_table_clearance = float(
            perception_cfg.get("top_grasp_table_clearance", 0.005)
        )

        for _name, _meta in (
            self.scene_context.get("live_scene_state", {}).get("objects", {}) or {}
        ).items():
            if isinstance(_meta, dict):
                self.scene_object_debug[str(_name)] = dict(_meta)

    # ------------------------------------------------------------ helpers

    def _connect_hardware(self) -> None:
        if self.controller is None:
            raise RuntimeError("UR3e controller not configured.")
        self.controller.connect()
        if self.gripper is not None and not self.mock:
            try:
                self.gripper.connect(
                    hostname=str(self.platform_cfg.get("robot", {}).get("ip", "")),
                    port=int(self.platform_cfg.get("robot", {}).get("gripper_port", 63352)),
                )
                self.gripper.activate()
            except Exception as exc:
                print(f"[real_env] WARNING: gripper connect failed ({exc}); proceeding without gripper feedback.")
                self.gripper = None

        if not self.mock:
            try:
                self.cameras.start_all(warmup_frames=30)
            except Exception as exc:
                print(f"[real_env] WARNING: camera start failed ({exc}); perception disabled.")

    def _disconnect_hardware(self) -> None:
        try:
            self.controller.stop_servo()
        except Exception:
            pass
        try:
            self.controller.disconnect()
        except Exception:
            pass
        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            except Exception:
                pass
        try:
            self.cameras.stop_all()
        except Exception:
            pass

    def _ensure_sam(self) -> Optional[SAMSegmenter]:
        if not self.use_sam:
            return None
        if self._sam is not None:
            return self._sam
        cfg = self.platform_cfg.get("perception", {}).get("sam", {})
        self._sam = SAMSegmenter(
            model_type=str(cfg.get("model_type", "vit_b")),
            checkpoint_path=cfg.get("checkpoint_path"),
            device=cfg.get("device"),
        )
        return self._sam

    def _ensure_scene_perception(self) -> Optional[ScenePerception]:
        if self.scene_perception is not None or self.skip_perception:
            return self.scene_perception
        if self.primary_projector is None:
            return None
        self.scene_perception = ScenePerception(
            projector=self.primary_projector,
            workspace_bounds=self.workspace_bounds,
            model=self.platform_cfg.get("perception", {}).get("model", "gpt-4o"),
            sam=self._ensure_sam(),
            table_z=float(self.table_z),
        )
        return self.scene_perception

    def _ensure_scene_discovery(self) -> Optional[SceneDiscovery]:
        if self.scene_discovery is not None:
            return self.scene_discovery
        if self.primary_projector is None:
            return None
        self.scene_discovery = SceneDiscovery(
            projector=self.primary_projector,
            workspace_bounds=self.workspace_bounds,
            model=self.platform_cfg.get("perception", {}).get("model", "gpt-4o"),
            sam=self._ensure_sam(),
            table_z=float(self.table_z),
        )
        return self.scene_discovery

    # ----------------------------------------------- public perception api

    def capture_rgbd(self, *, save_dir: Optional[Path] = None):
        """Grab a fresh aligned (color, depth) frame from the primary camera."""
        if self.primary_camera_name is None:
            raise RuntimeError("No primary RealSense camera configured.")
        cam = self.cameras.cameras.get(self.primary_camera_name)
        if cam is None or cam.pipeline is None:
            raise RuntimeError(
                f"Primary camera {self.primary_camera_name!r} not started."
            )
        color, depth = cam.capture()
        self._latest_color = color
        self._latest_depth = depth
        if save_dir is not None:
            save_color_image(color, Path(save_dir) / "real_scene_capture.png")
        return color, depth

    def relocalize_scene(self) -> Dict[str, np.ndarray]:
        """Re-run perception against a fresh capture and refresh scene_objects.

        In the default interactive pipeline, this uses the same grid-VLM +
        cloud_grasp SAM refinement path as the working legacy interactive
        planner, then updates ``self.scene_context`` and ``self.scene_objects``.
        """
        if self.skip_perception or self.mock or self.primary_camera_name is None:
            return dict(self.scene_objects)
        try:
            color, depth = self.capture_rgbd(save_dir=self._debug_dir)
        except Exception as exc:
            print(f"[real_env] relocalize: capture failed: {exc}")
            return dict(self.scene_objects)

        if self.perception_pipeline == "interactive":
            try:
                from .perception.interactive_grounding import ground_scene_context_interactive

                pcd = self._maybe_build_pcd(color, depth) if self.use_pcd_fusion else None
                outdir = self.cwd / "outputs" / "interactive_relocalize"
                new_ctx, rows = ground_scene_context_interactive(
                    scene_context=self.scene_context,
                    color_image=color,
                    depth_image=depth,
                    projector=self.primary_projector,
                    mode="localize-known",
                    model=self.platform_cfg.get("perception", {}).get("model", "gpt-4o"),
                    legacy_perception_root=self.legacy_perception_root,
                    use_som=self.use_som,
                    use_sam=self.use_sam,
                    pcd=pcd,
                    outdir=outdir,
                )
                self.scene_context = new_ctx
                for row in rows:
                    if row.get("visible") and row.get("xyz_robot") is not None:
                        xyz = np.asarray(row["xyz_robot"], dtype=np.float32)
                        self.update_scene_object(row["name"], xyz)
                        self._remember_perception_row(row)
                        print(
                            f"[real_env] interactive re-localized {row['name']!r} ({row.get('role')}) at "
                            f"{tuple(round(float(c), 3) for c in xyz.tolist())} ({row.get('notes','')})"
                        )
                return dict(self.scene_objects)
            except Exception as exc:
                print(f"[real_env] interactive relocalize failed: {exc}; falling back to legacy localizer.")

        sp = self._ensure_scene_perception()
        if sp is None:
            return dict(self.scene_objects)
        pcd = self._maybe_build_pcd(color, depth)
        try:
            localized = sp.localize_from_scene_context(
                color, depth, self.scene_context, manual_only=False, pcd=pcd
            )
        except Exception as exc:
            print(f"[real_env] relocalize: VLM/SAM call failed: {exc}")
            return dict(self.scene_objects)
        for name, info in localized.items():
            if info.xyz_robot is not None:
                self.update_scene_object(name, info.xyz_robot)
                print(
                    f"[real_env] re-localized {name!r} ({info.role}) at "
                    f"{tuple(round(float(c), 3) for c in info.xyz_robot.tolist())} "
                    f"({info.notes})"
                )
        return dict(self.scene_objects)


    def discover_scene_context(self, base_scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build a fresh scene_context from a live capture.

        Default behavior uses the interactive-planner-style perception pipeline:
        VLM coordinate grid / optional SoM + cloud_grasp SAM geometry.
        """
        if not self.controller.connected:
            self._connect_hardware()
        if not self.mock and self.primary_camera_name is not None:
            color, depth = self.capture_rgbd(save_dir=self._debug_dir)
        else:
            color = np.zeros((480, 640, 3), dtype=np.uint8)
            depth = np.zeros((480, 640), dtype=np.uint16)

        if self.perception_pipeline == "interactive":
            try:
                from .perception.interactive_grounding import ground_scene_context_interactive

                pcd = self._maybe_build_pcd(color, depth) if self.use_pcd_fusion else None
                outdir = self._debug_dir / "interactive_discovery"
                new_ctx, rows = ground_scene_context_interactive(
                    scene_context=base_scene_context,
                    color_image=color,
                    depth_image=depth,
                    projector=self.primary_projector,
                    mode="discover",
                    model=self.platform_cfg.get("perception", {}).get("model", "gpt-4o"),
                    legacy_perception_root=self.legacy_perception_root,
                    use_som=self.use_som,
                    use_sam=self.use_sam,
                    pcd=pcd,
                    outdir=outdir,
                )
                self.scene_context = new_ctx
                self.scene_objects.clear()
                for row in rows:
                    if row.get("visible") and row.get("xyz_robot") is not None:
                        self.update_scene_object(row["name"], np.asarray(row["xyz_robot"], dtype=np.float32))
                        self._remember_perception_row(row)
                return new_ctx
            except Exception as exc:
                print(f"[real_env] interactive discover failed: {exc}; falling back to legacy discovery.")

        sd = self._ensure_scene_discovery()
        if sd is None:
            print("[real_env] discover_scene_context: no projector / discovery available; returning base context.")
            return dict(base_scene_context)
        pcd = self._maybe_build_pcd(color, depth)
        discovered: list[DiscoveredEntity] = []
        try:
            discovered = sd.discover(color, depth, pcd=pcd)
        except Exception as exc:
            print(f"[real_env] discover_scene_context: VLM call failed: {exc}")
        if not discovered:
            print("[real_env] discover_scene_context: VLM returned no entities; keeping base context.")
            return dict(base_scene_context)
        new_ctx = sd.build_scene_context(base_scene_context, discovered)
        # Refresh local state so subsequent step calls see the new entities.
        self.scene_context = new_ctx
        self.scene_objects.clear()
        for ent in discovered:
            if ent.xyz_robot is not None:
                self.update_scene_object(ent.name, ent.xyz_robot)
        return new_ctx


    def _maybe_build_pcd(self, color: np.ndarray, depth: np.ndarray):
        """Build a single-camera Open3D point cloud if fusion is enabled."""
        if not self.use_pcd_fusion or self.primary_projector is None:
            return None
        cam = self.cameras.cameras.get(self.primary_camera_name)
        if cam is None or cam.intrinsics is None:
            return None
        return create_point_cloud_from_rgbd(
            color,
            depth,
            cam.intrinsics,
            self.primary_projector.T_base_cam,
        )

    # ----------------------------------------------- BaseManipulationEnv hooks

    def _reset_backend(self) -> Dict[str, Any]:
        # Lazy-connect on first reset so simply constructing a RealManipulationEnv
        # (e.g. for static scene introspection) doesn't require live hardware.
        if not self.controller.connected:
            self._connect_hardware()

        # Move to home and open gripper.
        self.controller.move_home(self.home_joints_rad, speed=0.5, accel=0.5)
        if self.gripper is not None:
            try:
                self.gripper.move(
                    int(self.gripper_cfg.get("open_position", 3)),
                    int(self.gripper_cfg.get("speed", 100)),
                    int(self.gripper_cfg.get("force", 100)),
                )
            except Exception:
                pass
        elif self.mock:
            pass
        time.sleep(0.5)

        self._gripper_closed_flag = False
        self._grasped_flag = False
        self._ee_to_object_offset = None

        # Capture a fresh image and localize all scene entities.
        if not self.skip_perception and self.primary_camera_name is not None and not self.mock:
            try:
                color, depth = self.capture_rgbd(save_dir=self._debug_dir)
                if self.perception_pipeline == "interactive":
                    try:
                        from .perception.interactive_grounding import ground_scene_context_interactive

                        pcd = self._maybe_build_pcd(color, depth) if self.use_pcd_fusion else None
                        outdir = self._debug_dir / "interactive_reset_localize"
                        new_ctx, rows = ground_scene_context_interactive(
                            scene_context=self.scene_context,
                            color_image=color,
                            depth_image=depth,
                            projector=self.primary_projector,
                            mode="localize-known",
                            model=self.platform_cfg.get("perception", {}).get("model", "gpt-4o"),
                            legacy_perception_root=self.legacy_perception_root,
                            use_som=self.use_som,
                            use_sam=self.use_sam,
                            pcd=pcd,
                            outdir=outdir,
                        )
                        self.scene_context = new_ctx
                        for row in rows:
                            if row.get("visible") and row.get("xyz_robot") is not None:
                                xyz = np.asarray(row["xyz_robot"], dtype=np.float32)
                                self.update_scene_object(row["name"], xyz)
                                self._remember_perception_row(row)
                                print(
                                    f"[real_env] SoM/localized {row['name']!r} ({row.get('role')}) at "
                                    f"{tuple(round(float(c), 3) for c in xyz.tolist())} ({row.get('notes','')})"
                                )
                            else:
                                print(f"[real_env] could not localize {row.get('name')!r}: {row.get('notes')}")
                    except Exception as exc:
                        print(f"[real_env] interactive reset localization failed: {exc}; falling back to legacy localizer.")
                        raise

                else:
                    raise RuntimeError("legacy requested")

            except Exception as exc:
                # Legacy fallback path.
                try:
                    color = locals().get("color", None)
                    depth = locals().get("depth", None)
                    if color is None or depth is None:
                        color, depth = self.capture_rgbd(save_dir=self.cwd / "outputs")
                    sp = self._ensure_scene_perception()
                    if sp is not None:
                        auto = bool(self.platform_cfg.get("perception", {}).get("auto_localize_on_reset", True))
                        pcd = self._maybe_build_pcd(color, depth)
                        localized = sp.localize_from_scene_context(
                            color,
                            depth,
                            self.scene_context,
                            manual_only=not auto,
                            pcd=pcd,
                        )
                        for name, info in localized.items():
                            if info.xyz_robot is not None:
                                self.update_scene_object(name, info.xyz_robot)
                                print(
                                    f"[real_env] localized {name!r} ({info.role}) at "
                                    f"{tuple(round(float(c), 3) for c in info.xyz_robot.tolist())} ({info.notes})"
                                )
                            else:
                                print(f"[real_env] could not localize {name!r}: {info.notes}")
                except Exception as fallback_exc:
                    print(f"[real_env] perception capture failed: {fallback_exc}")

        # If there is still no scene_objects entry for the active object, fall
        # back to the geometry section of scene_context.
        for ent in self.scene_context.get("entities", []):
            name = ent["name"]
            if name in self.scene_objects:
                continue
            geom = self.scene_context.get("geometry", {}).get(name, {})
            if "xy" in geom:
                z = float(
                    geom.get(
                        "z",
                        self.scene_context.get("placement_defaults", {}).get(
                            "table_surface_z", self.table_z
                        ),
                    )
                )
                xyz = np.asarray(
                    [float(geom["xy"][0]), float(geom["xy"][1]), z],
                    dtype=np.float32,
                )
                self.update_scene_object(name, xyz)
                self.scene_object_debug.setdefault(
                    name,
                    {
                        "name": name,
                        "xyz": xyz.tolist(),
                        "xy": xyz[:2].tolist(),
                        "z": float(xyz[2]),
                        "source": "scene_context_geometry",
                    },
                )

        return self._build_obs()


    # ---------------------------------------------------- grasp debug images

    def _remember_perception_row(self, row: Dict[str, Any]) -> None:
        name = row.get("name")
        if not name:
            return
        meta = dict(row)
        if "xyz_robot" in meta and meta["xyz_robot"] is not None:
            try:
                xyz = np.asarray(meta["xyz_robot"], dtype=np.float32)
                meta["xyz"] = xyz.tolist()
                meta["xy"] = xyz[:2].tolist()
                meta["z"] = float(xyz[2])
            except Exception:
                pass
        self.scene_object_debug[str(name)] = meta

    def _save_intended_grasp_debug_image(
        self,
        object_name: str,
        *,
        object_xyz: np.ndarray,
        pregrasp_xyz: np.ndarray,
        grasp_xyz: np.ndarray,
        carry_xyz: np.ndarray,
    ) -> Optional[str]:
        """Save an image labeling the exact real-world grasp target.

        The saved overlay shows:
        - the perception source pixel / SAM mark if available
        - the projected 3D grasp point that the robot will actually move to
        - the pregrasp and carry projections
        This separates VLM/SAM grounding errors from projection/calibration or
        execution errors.
        """
        try:
            import cv2  # type: ignore
        except Exception:
            return None
        if self._latest_color is None:
            try:
                color, depth = self.capture_rgbd(save_dir=self.cwd / "outputs")
            except Exception:
                return None
        else:
            color = self._latest_color.copy()

        img = color.copy()  # BGR OpenCV image
        h, w = img.shape[:2]
        outdir = self.cwd / "outputs" / "grasp_debug"
        outdir.mkdir(parents=True, exist_ok=True)

        def _draw_projected(pt: np.ndarray, label: str, color_bgr: tuple[int, int, int], radius: int = 9):
            if self.primary_projector is None:
                return None
            try:
                px, py = self.primary_projector.project_to_image(np.asarray(pt, dtype=np.float32))
            except Exception:
                return None
            if px is None or py is None:
                return None
            px_i, py_i = int(px), int(py)
            if not (0 <= px_i < w and 0 <= py_i < h):
                return (px_i, py_i)
            cv2.drawMarker(
                img,
                (px_i, py_i),
                color_bgr,
                markerType=cv2.MARKER_CROSS,
                markerSize=2 * radius + 6,
                thickness=2,
            )
            cv2.circle(img, (px_i, py_i), radius, color_bgr, 2)
            cv2.putText(
                img,
                label,
                (px_i + 12, py_i - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_bgr,
                1,
                cv2.LINE_AA,
            )
            return (px_i, py_i)

        meta = self.scene_object_debug.get(object_name, {}) or {}
        source_px = None
        if isinstance(meta, dict):
            pix = meta.get("pixel")
            if pix is not None and len(pix) >= 2:
                try:
                    source_px = (int(pix[0]), int(pix[1]))
                except Exception:
                    source_px = None
        if source_px is not None and 0 <= source_px[0] < w and 0 <= source_px[1] < h:
            cv2.circle(img, source_px, 14, (0, 255, 255), 2)
            cv2.putText(
                img,
                f"perception pixel {object_name}",
                (source_px[0] + 14, source_px[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        grasp_px = _draw_projected(grasp_xyz, "PROJECTED GRASP TARGET", (0, 255, 0), radius=11)
        _draw_projected(pregrasp_xyz, "pregrasp", (255, 0, 0), radius=8)
        _draw_projected(carry_xyz, "carry", (255, 0, 255), radius=8)

        y0 = 24
        lines = [
            f"object: {object_name}",
            f"object_xyz: [{object_xyz[0]:+.3f}, {object_xyz[1]:+.3f}, {object_xyz[2]:+.3f}]",
            f"grasp_xyz: [{grasp_xyz[0]:+.3f}, {grasp_xyz[1]:+.3f}, {grasp_xyz[2]:+.3f}]",
            f"pregrasp_xyz: [{pregrasp_xyz[0]:+.3f}, {pregrasp_xyz[1]:+.3f}, {pregrasp_xyz[2]:+.3f}]",
            f"source_pixel: {source_px} mark={meta.get('mark') if isinstance(meta, dict) else None}",
            f"projected_grasp_pixel: {grasp_px}",
            f"top_grasp_depth_below_surface: {self.real_top_grasp_depth_below_surface:.3f}",
            f"top_grasp_floor_z: {self._real_grasp_floor_z():.3f}",
        ]
        for i, line in enumerate(lines):
            yy = y0 + 18 * i
            cv2.putText(img, line, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = outdir / f"intended_grasp_{object_name}.png"
        cv2.imwrite(str(out_path), img)
        print(f"[real_env] saved intended grasp debug image: {out_path}")
        return str(out_path)


    # ---------------------------------------------------- simple real motion

    def simple_stop_servo(self) -> None:
        try:
            self.controller.stop_servo()
        except Exception:
            pass

    def simple_current_pose(self) -> np.ndarray:
        return np.asarray(self.controller.get_eef_pose(), dtype=np.float64)

    def simple_mpc_move_to(
        self,
        target_xyz: np.ndarray,
        *,
        tolerance: float = 0.015,
        max_steps: int = 400,
        label: str = "move",
        sleep_extra: float = 0.0,
        step_callback: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """Move the real EE to a Cartesian target using the simple MPC loop.

        This intentionally mirrors mpc_waypoint_test.py:
            u = mpc.act(cur_xyz, ctx)
            next_xyz = cur_xyz + mpc.action_scale_m * u
            servo_to(next_xyz, current_rotvec)

        No orientation target is computed. The current gripper orientation is
        held fixed at every step.
        """
        from .nominal_mpc import STLStageContext

        target_raw = np.asarray(target_xyz, dtype=np.float32)
        target = self._clip_xyz_to_workspace(target_raw)
        if not np.allclose(target, target_raw):
            print(
                "[real_env/simple_mpc] target clipped to workspace: "
                f"raw={target_raw.round(4).tolist()} clipped={target.round(4).tolist()}"
            )
        self.nominal_mpc.reset()

        first_pose = self.simple_current_pose()
        start_xyz = first_pose[:3].astype(np.float32)
        ctx = STLStageContext(
            target_xyz=target.astype(np.float32),
            knife_xy=np.zeros(2, dtype=np.float32),
            knife_radius=0.0,
            table_z=float(min(start_xyz[2], target[2])) - 0.10,
            always_keepout=False,
            workspace_min_xyz=self.workspace_min_xyz,
            workspace_max_xyz=self.workspace_max_xyz,
            workspace_weight=self.mpc_workspace_weight,
        )

        last_dist = float("inf")
        steps = 0
        reached = False
        last_xyz = start_xyz.astype(np.float64)
        for step in range(int(max_steps)):
            cur_pose = self.simple_current_pose()
            cur_xyz = cur_pose[:3].astype(np.float32)
            cur_rotvec = cur_pose[3:6].astype(np.float64)
            dist = float(np.linalg.norm(cur_xyz - target))
            last_dist = dist
            last_xyz = cur_xyz.astype(np.float64)

            if dist <= float(tolerance):
                reached = True
                steps = step
                break

            u = self.nominal_mpc.act(cur_xyz, ctx)
            u = self._clip_delta_action_to_workspace(
                cur_xyz,
                u,
                scale_m=float(self.nominal_mpc.action_scale_m),
            )
            next_xyz = cur_xyz + self.nominal_mpc.action_scale_m * u
            next_xyz = self._clip_xyz_to_workspace(next_xyz)

            try:
                self.controller.servo_to(
                    next_xyz,
                    cur_rotvec,  # hold current orientation; do not rotate gripper
                    speed=self.tcp_speed,
                    accel=self.tcp_accel,
                )
            except Exception as exc:
                print(f"[real_env/simple_mpc] servoL failed while moving to {label}: {exc}")
                break

            try:
                time.sleep(float(getattr(self.controller, "servo_dt", 0.05)) + float(sleep_extra))
            except Exception:
                pass
            
            if step_callback is not None:
                try:
                    step_callback(self, {"step": step, "label": label})
                except Exception:
                    pass
            steps = step + 1

        self.simple_stop_servo()
        self.last_raw_obs = self._build_obs()
        return {
            "label": label,
            "target_xyz": target.astype(float).tolist(),
            "final_xyz": last_xyz.astype(float).tolist(),
            "steps": int(steps),
            "reached": bool(reached),
            "dist": float(last_dist),
            "action_scale_m": float(self.nominal_mpc.action_scale_m),
            "workspace": self.workspace_bounds_status(),
            "orientation_held": True,
        }

    def simple_open_gripper(self, *, settle_steps: int = 8) -> None:
        self._send_gripper("open")
        for _ in range(int(settle_steps)):
            try:
                time.sleep(float(getattr(self.controller, "servo_dt", 0.05)))
            except Exception:
                pass
            self._refresh_grasped_state()

    def simple_close_gripper(self, *, settle_steps: int = 14) -> bool:
        self._send_gripper("close")
        for _ in range(int(settle_steps)):
            try:
                time.sleep(float(getattr(self.controller, "servo_dt", 0.05)))
            except Exception:
                pass
            self._refresh_grasped_state()
        return bool(self._is_grasped())

    def simple_resolve_object_xyz(self, object_name: str | None = None) -> np.ndarray:
        if object_name is not None:
            self.set_active_object(object_name)
        self.last_raw_obs = self._build_obs()
        return self._cube_pos(self.last_raw_obs).astype(np.float32)

    def _real_grasp_floor_z(self) -> float:
        """Lowest safe z for the real top-grasp target.

        Prefer the configured table surface if it looks realistic. Otherwise,
        use the lower workspace z bound, which is usually the tabletop height
        in the UR3e real setup.
        """
        candidates: list[float] = []
        try:
            candidates.append(float(self.table_z))
        except Exception:
            pass
        try:
            z_bounds = self.workspace_bounds.get("z", None)
            if z_bounds is not None and len(z_bounds) >= 1:
                candidates.append(float(z_bounds[0]))
        except Exception:
            pass
        candidates = [z for z in candidates if z > 0.05]
        floor = max(candidates) if candidates else float(self.table_z)
        return float(floor + self.real_top_grasp_table_clearance)

    def simple_grasp_target(self, object_xyz: np.ndarray) -> np.ndarray:
        """Real top-grasp target.

        In the real perception path, object_xyz comes from SAM/cloud geometry
        and represents the visible top/surface point of the object. Move the TCP
        below that surface by a calibrated grasp depth so the fingers actually
        surround the object before closing.
        """
        obj = np.asarray(object_xyz, dtype=np.float32)
        target = obj.copy()
        if self.mode_name == "top":
            target[2] = max(
                float(obj[2]) - float(self.real_top_grasp_depth_below_surface),
                self._real_grasp_floor_z(),
            )
        return target.astype(np.float32)

    def simple_pick(
        self,
        object_name: str,
        *,
        tolerance: float = 0.01,
        max_steps: int = 600,
        step_callback: Optional[Callable] = None,
    ) -> dict[str, Any]:
        self.set_active_object(object_name)
        self.simple_open_gripper(settle_steps=4)
        obj = self.simple_resolve_object_xyz(object_name)

        pregrasp = self.simple_pregrasp_target(obj)
        grasp = self.simple_grasp_target(obj)
        carry = self.simple_carry_target(obj)
        grasp_debug_image = self._save_intended_grasp_debug_image(
            object_name,
            object_xyz=obj,
            pregrasp_xyz=pregrasp,
            grasp_xyz=grasp,
            carry_xyz=carry,
        )

        records = []
        records.append(self.simple_mpc_move_to(pregrasp, tolerance=tolerance, max_steps=max_steps, label=f"pregrasp:{object_name}", step_callback=step_callback))
        records.append(self.simple_mpc_move_to(grasp, tolerance=tolerance, max_steps=max_steps, label=f"grasp:{object_name}", step_callback=step_callback))
        grasped_after_close = self.simple_close_gripper(settle_steps=max(14, int(self.close_steps)))
        # Robotiq feedback can be noisy / too strict for small cubes. Always
        # attempt the carry lift; the executor can decide whether to require
        # gripper feedback or just require motion completion.
        self._refresh_grasped_state()
        records.append(self.simple_mpc_move_to(carry, tolerance=tolerance, max_steps=max_steps, label=f"carry:{object_name}", step_callback=step_callback))

        motion_success = all(bool(r.get("reached", False)) for r in records)
        gripper_reports_grasp = bool(self._is_grasped())
        pick_success = bool(gripper_reports_grasp if self.require_grasp_feedback else motion_success)

        return {
            "skill": "pick",
            "object": object_name,
            "grasped": gripper_reports_grasp,
            "reported_grasp_after_close": bool(grasped_after_close),
            "require_grasp_feedback": bool(self.require_grasp_feedback),
            "motion_success": bool(motion_success),
            "pick_success": bool(pick_success),
            "records": records,
            "grasp_debug_image": grasp_debug_image,
            "intended_grasp_xyz": grasp.astype(float).tolist(),
            "real_top_grasp_depth_below_surface": float(self.real_top_grasp_depth_below_surface),
            "real_top_grasp_floor_z": float(self._real_grasp_floor_z()),
            "perception_debug": self.scene_object_debug.get(object_name, {}),
            "final_status": self.status(),
        }

    def simple_transport(
        self,
        target_xyz: np.ndarray,
        *,
        tolerance: float = 0.015,
        max_steps: int = 600,
        label: str = "transport",
        step_callback: Optional[Callable] = None,
    ) -> dict[str, Any]:
        target = np.asarray(target_xyz, dtype=np.float32)
        rec = self.simple_mpc_move_to(target, tolerance=tolerance, max_steps=max_steps, label=label, step_callback=step_callback)
        return {"skill": "transport", "records": [rec], "final_status": self.status()}

    def simple_place(
        self,
        target_xyz: np.ndarray,
        *,
        tolerance: float = 0.015,
        max_steps: int = 600,
        step_callback: Optional[Callable] = None,
    ) -> dict[str, Any]:
        target = np.asarray(target_xyz, dtype=np.float32)
        over_goal = np.array([target[0], target[1], float(self.carry_z)], dtype=np.float32)
        descend = target
        retreat = np.array([target[0], target[1], float(self.retreat_z)], dtype=np.float32)
        records = []
        records.append(self.simple_mpc_move_to(over_goal, tolerance=tolerance, max_steps=max_steps, label="place:over_goal", step_callback=step_callback))
        records.append(self.simple_mpc_move_to(descend, tolerance=tolerance, max_steps=max_steps, label="place:descend", step_callback=step_callback))
        self.simple_open_gripper()
        records.append(self.simple_mpc_move_to(retreat, tolerance=tolerance, max_steps=max_steps, label="place:retreat", step_callback=step_callback))
        return {"skill": "place", "records": records, "final_status": self.status()}

    def _step_backend(self, full_action: np.ndarray):
        """Compatibility step for the old shared stage machine.

        The real-world path now prefers the simple executor in
        System1Executor._run_real_simple_subtasks(...). This fallback still
        follows the same convention as mpc_waypoint_test.py:

            next_xyz = current_xyz + action_scale_m * normalized_u

        Orientation is held fixed by default; no gripper rotation is commanded.
        """
        action = np.asarray(full_action, dtype=np.float64)
        pos_scale_m = float(self.nominal_mpc.action_scale_m)
        normalized_u = np.clip(action[:3], -1.0, 1.0)
        delta_xyz = pos_scale_m * normalized_u

        current_pose = self.controller.get_eef_pose()
        current_xyz = np.asarray(current_pose[:3], dtype=np.float64)
        current_rotvec = np.asarray(current_pose[3:6], dtype=np.float64)

        target_xyz = current_xyz + delta_xyz
        target_rotvec = current_rotvec.copy()  # hold orientation; do not rotate gripper

        info = {
            "real_normalized_u": normalized_u.astype(np.float32).copy(),
            "real_delta_xyz_m": delta_xyz.astype(np.float32).copy(),
            "real_action_scale_m": float(pos_scale_m),
            "real_servo_target_xyz": target_xyz.astype(np.float32).copy(),
            "real_orientation_held": True,
        }

        try:
            self.controller.servo_to(
                target_xyz,
                target_rotvec,
                speed=self.tcp_speed,
                accel=self.tcp_accel,
            )
        except Exception as exc:
            print(f"[real_env] servoL failed: {exc}")

        try:
            time.sleep(float(getattr(self.controller, "servo_dt", 0.05)))
        except Exception:
            pass

        gripper_cmd = float(action[6]) if action.shape[0] > 6 else 0.0
        if gripper_cmd > 0.5 and not self._gripper_closed_flag:
            self._send_gripper("close")
        elif gripper_cmd < -0.5 and self._gripper_closed_flag:
            self._send_gripper("open")

        self._refresh_grasped_state()
        return self._build_obs(), 0.0, False, info


    def _build_obs(self) -> Dict[str, Any]:
        try:
            pose = self.controller.get_eef_pose()
        except Exception:
            pose = np.array([0.0, -0.30, 0.30, 3.14, 0.0, 0.0])
        return {
            "robot0_eef_pos": pose[:3].astype(np.float32),
            "robot0_eef_rotvec": pose[3:6].astype(np.float32),
            "scene_objects": dict(self.scene_objects),
            "active_object_name": self.active_object_name,
        }

    def _eef_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)

    def _eef_rotmat(self, obs: Dict[str, Any]) -> np.ndarray:
        return rotvec_to_matrix(np.asarray(obs["robot0_eef_rotvec"], dtype=np.float64)).astype(np.float32)

    def _cube_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        # While grasped, the active object follows the EE through a fixed offset.
        if self._ee_to_object_offset is not None:
            ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            return ee + self._ee_to_object_offset
        # Otherwise read from scene_objects.
        name = self.active_object_name or obs.get("active_object_name", "")
        if name in self.scene_objects:
            return self.scene_objects[name].astype(np.float32).copy()
        # Sensible fallback: the goal anchor.
        gxy, gz = self._goal_anchor()
        return np.array([gxy[0], gxy[1], gz], dtype=np.float32)

    def _is_grasped(self) -> bool:
        return bool(self._grasped_flag)

    # ------------------------------------------------------ gripper control

    def _send_gripper(self, command: str) -> None:
        if self.gripper is None and not self.mock:
            return
        if command == "close":
            target = int(self.gripper_cfg.get("close_position", 249))
        else:
            target = int(self.gripper_cfg.get("open_position", 3))
        speed = int(self.gripper_cfg.get("speed", 100))
        force = int(self.gripper_cfg.get("force", 100))
        if self.gripper is not None:
            try:
                self.gripper.move(target, speed, force)
            except Exception as exc:
                print(f"[real_env] gripper command failed: {exc}")
        self._gripper_closed_flag = command == "close"

    def _refresh_grasped_state(self) -> None:
        if not self._gripper_closed_flag:
            self._grasped_flag = False
            self._ee_to_object_offset = None
            return
        if self.mock:
            # Pretend we grasped after the close command settled.
            self._grasped_flag = True
            self._capture_object_offset_if_needed()
            return
        if self.gripper is None:
            return
        try:
            obj_status = self.gripper._get_var(self.gripper.OBJ)  # 0 moving, 1/2 grasp, 3 at dest
        except Exception:
            return
        if obj_status in (1, 2):
            self._grasped_flag = True
            self._capture_object_offset_if_needed()
        elif obj_status == 3:
            # Closed all the way without an object.
            self._grasped_flag = False
            self._ee_to_object_offset = None

    def _capture_object_offset_if_needed(self) -> None:
        if self._ee_to_object_offset is not None:
            return
        try:
            ee = self.controller.get_eef_xyz().astype(np.float32)
        except Exception:
            return
        target_obj = self.scene_objects.get(self.active_object_name)
        if target_obj is None:
            self._ee_to_object_offset = np.zeros(3, dtype=np.float32)
            return
        self._ee_to_object_offset = (target_obj - ee).astype(np.float32)

    # ------------------------------------------------------ scene queries

    def _object_pos_by_name(self, object_name: Optional[str]) -> Optional[np.ndarray]:
        return super()._object_pos_by_name(object_name)

    # ------------------------------------------------------- rendering

    def render_rgb_frame(
        self,
        camera_name: str = "agentview",
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        # Internal RealSense frames are BGR. Generic render/video consumers
        # expect RGB arrays, so convert at this boundary only.
        if self._latest_color is not None:
            return self._latest_color[:, :, ::-1].copy()
        if self.primary_camera_name is None:
            return np.zeros((int(height), int(width), 3), dtype=np.uint8)
        try:
            color, depth = self.cameras.cameras[self.primary_camera_name].capture()
            self._latest_color = color
            self._latest_depth = depth
            return color[:, :, ::-1].copy()
        except Exception:
            return np.zeros((int(height), int(width), 3), dtype=np.uint8)

    def snapshot_scene_image(
        self,
        save_path: str,
        *,
        camera_name: str = "agentview",
        width: int = 640,
        height: int = 480,
    ) -> str:
        """Capture a fresh RGB frame from the primary RealSense and save it.

        Always flushes the camera buffer first so the saved frame reflects
        the post-motion state of the workspace.
        """
        if self.mock or self.primary_camera_name is None:
            return super().snapshot_scene_image(
                save_path, camera_name=camera_name, width=int(width), height=int(height)
            )
        try:
            color, _ = self.capture_rgbd(save_dir=None)
        except Exception:
            return super().snapshot_scene_image(
                save_path, camera_name=camera_name, width=int(width), height=int(height)
            )
        return save_color_image(color, save_path)

    @staticmethod
    def capture_initial_scene_image(
        save_path: str,
        scene_context: Dict[str, Any],
        camera_name: Optional[str] = None,
        cwd: Optional[Path] = None,
        mock: bool = False,
    ) -> str:
        """Capture an initial RGB frame for System 2 from the configured RealSense camera."""
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if mock:
            # Save a black placeholder.
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            return save_color_image(placeholder, save_path)
        platform = scene_context.get("platform", {})
        cameras_cfg = platform.get("cameras", {})
        if not cameras_cfg:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            return save_color_image(placeholder, save_path)
        if camera_name is None:
            for name, cfg in cameras_cfg.items():
                if cfg.get("is_primary_for_vlm", False):
                    camera_name = name
                    break
            if camera_name is None:
                camera_name = next(iter(cameras_cfg))
        cfg = cameras_cfg[camera_name]
        cam = RealSenseCamera(
            serial=str(cfg.get("serial", "")),
            name=camera_name,
            width=int(cfg.get("resolution", [640, 480])[0]),
            height=int(cfg.get("resolution", [640, 480])[1]),
            fps=int(cfg.get("fps", 30)),
        )
        try:
            cam.start()
            cam.warmup(30)
            color, _ = cam.capture()
            save_color_image(color, save_path)
        finally:
            cam.stop()
        return save_path

    # ------------------------------------------------------- shutdown

    def close(self):
        try:
            super().close()
        finally:
            try:
                # Open the gripper and move to a safe spot before disconnecting.
                if self._gripper_closed_flag:
                    self._send_gripper("open")
                self.controller.move_home(self.home_joints_rad, speed=0.5, accel=0.5)
            except Exception:
                pass
            self._disconnect_hardware()
