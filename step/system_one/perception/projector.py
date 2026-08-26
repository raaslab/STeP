"""2D pixel <-> 3D robot-base-frame projection for RealSense cameras.

Adapted from ``temp_real_robot_files/projection.py``. A :class:`Projector`
loads a calibration JSON of the form::

    {
      "T_base_cam": <4x4 list>,
      "camera_matrix": <3x3 list>,
      "dist_coeffs": <list>,
      "camera": "left",
      "serial": "..."
    }

and provides:

* :py:meth:`pixel_to_camera_frame`
* :py:meth:`camera_to_robot_frame`
* :py:meth:`pixel_to_robot_frame`            full pipeline with depth patch
* :py:meth:`project_to_image`                3D -> 2D pixel
* :py:meth:`is_in_workspace`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class Projector:
    def __init__(self, calibration_path: str | Path):
        with open(calibration_path) as f:
            calib = json.load(f)
        self.T_base_cam = np.array(calib["T_base_cam"])
        self.camera_matrix = np.array(
            calib.get(
                "camera_matrix",
                [[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]],
            )
        )
        self.dist_coeffs = np.array(
            calib.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
        )
        self.camera_name = calib.get("camera", "unknown")
        self.serial = calib.get("serial", "")

    def pixel_to_camera_frame(self, px: int, py: int, depth_m: float) -> np.ndarray:
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        x = (px - cx) * depth_m / fx
        y = (py - cy) * depth_m / fy
        return np.array([x, y, depth_m])

    def camera_to_robot_frame(self, point_cam: np.ndarray) -> np.ndarray:
        point_cam_h = np.append(point_cam, 1.0)
        point_base_h = self.T_base_cam @ point_cam_h
        return point_base_h[:3]

    def pixel_to_robot_frame(
        self,
        px: int,
        py: int,
        depth_image: np.ndarray,
        patch_size: int = 5,
        depth_scale: float = 1000.0,
        min_depth_m: float = 0.1,
        max_depth_m: float = 3.0,
    ) -> np.ndarray:
        """Pixel -> robot base frame using a small depth patch for robustness."""
        h, w = depth_image.shape[:2]
        half = max(1, int(patch_size)) // 2
        y_min = max(0, int(py) - half)
        y_max = min(h, int(py) + half + 1)
        x_min = max(0, int(px) - half)
        x_max = min(w, int(px) + half + 1)
        patch = depth_image[y_min:y_max, x_min:x_max]
        valid = patch[patch > 0]
        if len(valid) == 0:
            raise ValueError(f"No valid depth at pixel ({px}, {py})")
        depth_m = float(np.median(valid)) / float(depth_scale)
        if not (float(min_depth_m) <= depth_m <= float(max_depth_m)):
            raise ValueError(
                f"Depth {depth_m:.3f}m out of range [{min_depth_m}, {max_depth_m}] m"
            )
        return self.camera_to_robot_frame(
            self.pixel_to_camera_frame(int(px), int(py), depth_m)
        )

    def project_to_image(self, point_robot: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        T_cam_base = np.linalg.inv(self.T_base_cam)
        p_h = np.append(point_robot, 1.0)
        p_cam = T_cam_base @ p_h
        if p_cam[2] <= 0:
            return None, None
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        px = (p_cam[0] * fx / p_cam[2]) + cx
        py = (p_cam[1] * fy / p_cam[2]) + cy
        return int(px), int(py)

    def is_in_workspace(self, point: np.ndarray, workspace_bounds: dict) -> bool:
        x_range = workspace_bounds.get("x", [-1, 1])
        y_range = workspace_bounds.get("y", [-1, 1])
        z_range = workspace_bounds.get("z", [-1, 1])
        return (
            x_range[0] <= point[0] <= x_range[1]
            and y_range[0] <= point[1] <= y_range[1]
            and z_range[0] <= point[2] <= z_range[1]
        )
