"""Segment Anything (SAM) based mask + 3D geometry estimation.

Adapted from ``temp_real_robot_files/cloud_grasp.py``. Given a 2D pixel hint
from a VLM, this module:

1. Runs SAM at that pixel (with a small grid of jittered seeds) to find the
   tightest mask of the underlying object.
2. Backprojects every masked pixel through the depth image + camera calibration
   into the robot base frame (or, if a pre-computed Open3D point cloud is
   supplied, filters that cloud by reprojecting through the SAM mask).
3. Runs DBSCAN clustering on the resulting 3D points to drop background bleed
   and depth-edge noise.
4. Returns a robust ``(top_xyz, center_xy, radius, mask, surface_points)`` tuple.

Everything heavy (``torch``, ``segment_anything``, ``open3d``,
``sklearn.cluster``) is imported lazily so the sim-only side of the codebase
does not need any of these packages installed. Callers should treat all
returns as best-effort and fall back to single-pixel deprojection if SAM is
unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from .projector import Projector


# ---------------------------------------------------------------------------
# SAM checkpoint discovery
# ---------------------------------------------------------------------------

DEFAULT_SAM_CHECKPOINTS = {
    "vit_h": ["models/sam_vit_h_4b8939.pth", "checkpoints/sam_vit_h_4b8939.pth"],
    "vit_l": ["models/sam_vit_l_0b3195.pth", "checkpoints/sam_vit_l_0b3195.pth"],
    "vit_b": ["models/sam_vit_b_01ec64.pth", "checkpoints/sam_vit_b_01ec64.pth"],
}


def find_sam_checkpoint(model_type: str = "vit_b", search_dirs: Optional[List[Path]] = None) -> Optional[Path]:
    """Find a SAM checkpoint on disk. Returns ``None`` if not found.

    Search order:
    1. ``$SAM_CHECKPOINT`` environment variable.
    2. The relative paths in :data:`DEFAULT_SAM_CHECKPOINTS` for ``model_type``,
       resolved against ``search_dirs`` (default: cwd + repo root).
    """
    import os

    env_path = os.environ.get("SAM_CHECKPOINT", "").strip()
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    rels = DEFAULT_SAM_CHECKPOINTS.get(model_type, [])
    if not rels:
        return None
    if search_dirs is None:
        search_dirs = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for base in search_dirs:
        for rel in rels:
            cand = Path(base) / rel
            if cand.is_file():
                return cand
    return None


# ---------------------------------------------------------------------------
# SAM segmenter
# ---------------------------------------------------------------------------


@dataclass
class SAMSegmenter:
    """Lazy wrapper around the Segment Anything predictor.

    Construct once per process, then call :py:meth:`get_mask`. If SAM is not
    installed or the checkpoint is missing, ``available`` is False and the
    segmenter is a no-op (returns ``None`` from ``get_mask``).
    """

    model_type: str = "vit_b"
    checkpoint_path: Optional[str] = None
    device: Optional[str] = None
    _predictor: Any = None
    _available: Optional[bool] = None

    def __post_init__(self) -> None:
        # Resolve the checkpoint lazily; we don't try to load until first use
        # (so that import-only paths in tests stay cheap).
        if self.checkpoint_path is None:
            ckpt = find_sam_checkpoint(self.model_type)
            self.checkpoint_path = None if ckpt is None else str(ckpt)

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._try_load()
        return bool(self._available)

    def _try_load(self) -> bool:
        if not self.checkpoint_path or not Path(self.checkpoint_path).is_file():
            print(
                f"[sam] no checkpoint found for model_type={self.model_type!r}. "
                "Set SAM_CHECKPOINT or download to ./models/. SAM-based refinement disabled."
            )
            return False
        try:
            import torch  # noqa: F401
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        except Exception as exc:
            print(f"[sam] segment_anything / torch not importable ({exc}); SAM disabled.")
            return False
        try:
            print(f"[sam] loading SAM ({self.model_type}) from {self.checkpoint_path} ...")
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            if self.device is None:
                self.device = "cuda" if _cuda_available() else "cpu"
            sam.to(device=self.device)
            self._predictor = SamPredictor(sam)
            print(f"[sam] loaded on {self.device}.")
            return True
        except Exception as exc:
            print(f"[sam] load failed ({exc}); SAM disabled.")
            return False

    def get_mask(self, image: np.ndarray, px: int, py: int) -> Optional[np.ndarray]:
        """Return a boolean mask for the object at pixel ``(px, py)`` or None."""
        if not self.available:
            return None
        try:
            import cv2  # type: ignore

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            image_rgb = image
        try:
            self._predictor.set_image(image_rgb)
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array([[int(px), int(py)]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            return np.asarray(masks[best], dtype=bool)
        except Exception as exc:
            print(f"[sam] predict failed at ({px},{py}): {exc}")
            return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 3D geometry from a SAM mask
# ---------------------------------------------------------------------------


@dataclass
class GeometryEstimate:
    top_xyz: np.ndarray
    center_xy: np.ndarray
    radius: float
    mask: np.ndarray
    surface_points: np.ndarray  # Nx2 (XY) representative dots for keep-out shaping
    num_points: int

    def as_dict(self) -> dict:
        return {
            "top_xyz": self.top_xyz.tolist(),
            "center_xy": self.center_xy.tolist(),
            "radius": float(self.radius),
            "num_points": int(self.num_points),
            "surface_points": self.surface_points.tolist(),
        }


def estimate_geometry_from_mask(
    image: np.ndarray,
    depth_image: np.ndarray,
    px: int,
    py: int,
    *,
    sam: Optional[SAMSegmenter],
    projector: Projector,
    pcd: Any = None,
    label: str = "object",
    table_z: float = 0.0,
    max_z: float = 0.40,
    grid_offsets: Tuple[int, ...] = (0, -7, 7, -14, 14),
) -> Optional[GeometryEstimate]:
    """Backproject every masked pixel into 3D and return robust geometry.

    If ``sam`` is None or unavailable, falls back to a single-pixel
    deprojection at ``(px, py)``.

    Args:
        pcd: Optional Open3D point cloud (already in robot base frame). When
            provided, the function projects every cloud point through
            ``projector.project_to_image`` and keeps only those falling inside
            the SAM mask. This is the high-precision dual-camera fusion mode
            described in :mod:`temp_real_robot_files.cloud_grasp`.
    """
    h, w = image.shape[:2]

    # 1) Get a mask. Try a few jittered seeds to compensate for VLM error.
    best_mask = None
    if sam is not None and sam.available:
        best_score = -1
        for ox in grid_offsets:
            for oy in grid_offsets:
                cx, cy = int(px) + ox, int(py) + oy
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                mask = sam.get_mask(image, cx, cy)
                if mask is None:
                    continue
                num_pixels = int(np.sum(mask))
                if 100 < num_pixels < 50_000 and num_pixels > best_score:
                    best_score = num_pixels
                    best_mask = mask
            if best_score > 500:
                break

    # 2) Fallback: no SAM mask -> single-pixel deprojection.
    if best_mask is None:
        try:
            xyz = projector.pixel_to_robot_frame(int(px), int(py), depth_image)
        except Exception as exc:
            print(f"[sam] no mask and single-pixel deprojection failed for {label!r}: {exc}")
            return None
        return GeometryEstimate(
            top_xyz=np.asarray(xyz, dtype=np.float32),
            center_xy=np.asarray(xyz[:2], dtype=np.float32),
            radius=0.05,
            mask=np.zeros((h, w), dtype=bool),
            surface_points=np.asarray([xyz[:2]], dtype=np.float32),
            num_points=1,
        )

    # 3) Backproject masked pixels (or filter pcd) into 3D.
    points_3d: List[np.ndarray] = []
    if pcd is not None:
        try:
            all_pts = np.asarray(pcd.points)
            for pt in all_pts:
                u, v = projector.project_to_image(pt)
                if u is None:
                    continue
                if 0 <= int(u) < w and 0 <= int(v) < h and best_mask[int(v), int(u)]:
                    points_3d.append(np.asarray(pt, dtype=np.float32))
        except Exception as exc:
            print(f"[sam] pcd-mask filter failed for {label!r}: {exc}; falling back to depth backprojection.")
            points_3d = []
    if not points_3d:
        ys, xs = np.where(best_mask)
        if len(xs) == 0:
            return None
        step = 1 if len(xs) < 5_000 else max(1, len(xs) // 5_000)
        for i in range(0, len(xs), step):
            u, v = int(xs[i]), int(ys[i])
            try:
                p = projector.pixel_to_robot_frame(u, v, depth_image)
            except Exception:
                continue
            points_3d.append(np.asarray(p, dtype=np.float32))

    if len(points_3d) < 30:
        # Very few valid 3D points; try the centre of the mask.
        try:
            cy, cx = np.mean(np.argwhere(best_mask), axis=0)
            xyz = projector.pixel_to_robot_frame(int(cx), int(cy), depth_image)
        except Exception:
            return None
        return GeometryEstimate(
            top_xyz=np.asarray(xyz, dtype=np.float32),
            center_xy=np.asarray(xyz[:2], dtype=np.float32),
            radius=0.05,
            mask=best_mask,
            surface_points=np.asarray([xyz[:2]], dtype=np.float32),
            num_points=int(len(points_3d)),
        )

    pts = np.asarray(points_3d, dtype=np.float32)

    # 4) Drop points that obviously aren't part of the object: below table or
    #    way above the workspace.
    z_mask = (pts[:, 2] >= float(table_z) - 0.005) & (pts[:, 2] <= float(table_z) + float(max_z))
    if np.any(z_mask):
        pts = pts[z_mask]

    # 5) DBSCAN to drop bleed/noise. Lazy-import sklearn.
    try:
        from sklearn.cluster import DBSCAN  # type: ignore

        clustering = DBSCAN(eps=0.015, min_samples=30).fit(pts[:, :2])
        labels = clustering.labels_
        valid = labels[labels >= 0]
        if len(valid) > 0:
            uniq, counts = np.unique(valid, return_counts=True)
            best_label = int(uniq[int(np.argmax(counts))])
            pts = pts[labels == best_label]
    except Exception:
        pass

    if len(pts) == 0:
        return None

    center = np.mean(pts, axis=0)
    z_thresh = np.percentile(pts[:, 2], 95)
    top_points = pts[pts[:, 2] >= z_thresh]
    top_xyz = np.mean(top_points, axis=0) if len(top_points) > 0 else center

    radial = np.linalg.norm(pts[:, :2] - center[:2], axis=1)
    radius = float(np.percentile(radial, 95)) if len(radial) > 0 else 0.05

    num_surface = max(1, min(len(pts), 15))
    indices = np.linspace(0, len(pts) - 1, num_surface, dtype=int)
    surface_points = pts[indices][:, :2]

    return GeometryEstimate(
        top_xyz=np.asarray(top_xyz, dtype=np.float32),
        center_xy=np.asarray(center[:2], dtype=np.float32),
        radius=radius,
        mask=best_mask,
        surface_points=np.asarray(surface_points, dtype=np.float32),
        num_points=int(len(pts)),
    )


def create_point_cloud_from_rgbd(
    color_img: np.ndarray,
    depth_img: np.ndarray,
    intrinsics: Any,
    T_base_cam: np.ndarray,
    *,
    max_z: float = 1.5,
    depth_scale: float = 1000.0,
):
    """Build an Open3D point cloud transformed into the robot base frame.

    Returns ``None`` if Open3D is not installed.
    """
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return None
    try:
        import cv2  # type: ignore

        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = color_img
    try:
        o3d_color = o3d.geometry.Image(np.ascontiguousarray(rgb))
        o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth_img))
        intr = o3d.camera.PinholeCameraIntrinsic(
            int(intrinsics.width),
            int(intrinsics.height),
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.ppx),
            float(intrinsics.ppy),
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=float(depth_scale),
            depth_trunc=float(max_z),
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
        pcd.transform(np.asarray(T_base_cam))
        return pcd
    except Exception as exc:
        print(f"[sam] point cloud creation failed: {exc}")
        return None
