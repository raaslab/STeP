"""RGB-D capture from the simulated cameras.

Reads only what a real calibrated RGB-D camera provides: pixels, metric
depth, intrinsics and the camera's pose (fixed for agentview, forward
kinematics for the wrist camera). Never the renderer's segmentation buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RGBDFrame:
    camera_name: str
    rgb: np.ndarray  # (H, W, 3) uint8, row 0 = top
    depth: np.ndarray  # (H, W) metres along the optical axis
    K: np.ndarray  # (3, 3)
    cam_pos: np.ndarray  # (3,) world
    cam_mat: np.ndarray  # (3, 3) world-from-camera, MuJoCo convention (camera looks down -Z, +Y up)

    def points_world(self) -> np.ndarray:
        """(H, W, 3) world-frame back-projection of every pixel."""
        h, w = self.depth.shape
        f, cx, cy = self.K[0, 0], self.K[0, 2], self.K[1, 2]
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        pts_cam = np.stack([(cols - cx) * self.depth / f, -(rows - cy) * self.depth / f, -self.depth], axis=-1)
        return pts_cam @ self.cam_mat.T + self.cam_pos

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """(N, 3) world points -> (N, 2) pixel (col, row); inverse of points_world."""
        pts_cam = (np.asarray(points_world) - self.cam_pos) @ self.cam_mat
        z = -pts_cam[:, 2]
        f, cx, cy = self.K[0, 0], self.K[0, 2], self.K[1, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            col = pts_cam[:, 0] * f / z + cx
            row = -pts_cam[:, 1] * f / z + cy
        return np.stack([col, row], axis=-1)


def capture_rgbd(sim, camera_name: str, height: int, width: int) -> RGBDFrame:
    from robosuite.utils.camera_utils import get_real_depth_map

    rgb, depth_normalized = sim.render(width=width, height=height, camera_name=camera_name, depth=True)
    # Raw sim.render() is OpenGL row order (row 0 = bottom).
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[::-1])
    depth = np.asarray(get_real_depth_map(sim, np.asarray(depth_normalized, dtype=np.float32)[::-1]), dtype=np.float32)

    cid = sim.model.camera_name2id(camera_name)
    fovy = float(sim.model.cam_fovy[cid])
    f = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy))
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    cam_pos = np.asarray(sim.data.cam_xpos[cid], dtype=np.float64).copy()
    cam_mat = np.asarray(sim.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3).copy()
    return RGBDFrame(camera_name, rgb, depth, K, cam_pos, cam_mat)
