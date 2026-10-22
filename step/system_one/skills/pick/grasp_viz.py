"""Shared grasp-candidate image rendering, used by both manual debugging
scripts and LiberoPickAdapter.run() so the two stay in sync.

Draws each candidate's position and closing axis: center candidates blue,
rim-edge orange, other/unlabeled yellow; the selected one is green and labeled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...motion.scene_query import quat_to_rotmat_wxyz
from ...motion.types import GraspCandidate


def camera_project(model: Any, data: Any, camera_name: str, height: int, width: int, points_world: np.ndarray) -> np.ndarray:
    """World (N,3) -> pixel (N,2) [row, col] for a MuJoCo camera, with the same
    up/down flip LiberoManipulationEnv.render_rgb_frame applies."""
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise KeyError(f"No camera named {camera_name!r} in model.")
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    fovy_deg = float(model.cam_fovy[cam_id])
    f = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg))

    local = (points_world - cam_pos[None, :]) @ cam_mat  # (N,3), camera frame
    depth = -local[:, 2]
    depth_safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    col = width / 2.0 + f * local[:, 0] / depth_safe
    row = height / 2.0 - f * local[:, 1] / depth_safe
    row_flipped = height - 1 - row
    pixels = np.stack([row_flipped, col], axis=1)
    pixels[depth <= 0] = np.nan  # behind the camera
    return pixels


def render_grasp_candidates_image(
    frame_rgb: np.ndarray,
    model: Any,
    data: Any,
    camera_name: str,
    candidates: list[GraspCandidate],
    selected_grasp: GraspCandidate | None = None,
):
    """A PIL Image of frame_rgb with every candidate's closing axis
    overlaid, the selected_grasp (if any, matched by pose) highlighted."""
    from PIL import Image, ImageDraw, ImageFont

    height, width = frame_rgb.shape[0], frame_rgb.shape[1]
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    selected_index = None
    if selected_grasp is not None:
        for i, c in enumerate(candidates):
            if np.allclose(c.grasp_pose_world, selected_grasp.grasp_pose_world):
                selected_index = i
                break

    finger_half_span = 0.04  # purely visual: half a typical gripper's max opening
    for i, cand in enumerate(candidates):
        R = quat_to_rotmat_wxyz(cand.grasp_quat_wxyz)
        close_axis_world = R @ np.array([1.0, 0.0, 0.0])  # gripper local +X is the closing axis
        p_center = cand.grasp_position
        p_a = p_center + finger_half_span * close_axis_world
        p_b = p_center - finger_half_span * close_axis_world

        try:
            pts = camera_project(model, data, camera_name, height, width, np.stack([p_center, p_a, p_b]))
        except KeyError:
            continue
        if np.isnan(pts).any():
            continue
        (r0, c0), (r1, c1), (r2, c2) = pts

        is_selected = i == selected_index
        strategy = cand.metadata.get("strategy", "center")
        if is_selected:
            color = (0, 220, 0)
        elif strategy == "rim_edge":
            color = (255, 140, 0)
        elif strategy == "center":
            color = (60, 140, 255)
        else:
            color = (230, 210, 30)  # learned-generator or otherwise unlabeled candidates
        width_px = 4 if is_selected else 2

        draw.line([(c1, r1), (c2, r2)], fill=color, width=width_px)
        radius = 6 if is_selected else 4
        draw.ellipse([c0 - radius, r0 - radius, c0 + radius, r0 + radius], outline=color, width=2)
        label = "SELECTED" if is_selected else str(i)
        draw.text((c0 + 6, r0 + 6), label, fill=color, font=font)

    return image


def save_grasp_candidates_image(
    frame_rgb: np.ndarray,
    model: Any,
    data: Any,
    camera_name: str,
    candidates: list[GraspCandidate],
    out_path: str | Path,
    selected_grasp: GraspCandidate | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_grasp_candidates_image(frame_rgb, model, data, camera_name, candidates, selected_grasp)
    image.save(out_path)
    return out_path
