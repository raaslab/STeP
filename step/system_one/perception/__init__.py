"""Perception module for the real-robot backend.

* :mod:`realsense` wraps Intel RealSense pipelines (single + dual camera).
* :mod:`projector` deprojects pixel + depth into the robot base frame.
* :mod:`sam_segmenter` wraps Segment Anything (SAM) with depth/projector
  integration to produce robust 3D centroids + radii from a single 2D hint.
* :mod:`scene_perception` glues a VLM call (+ optional SAM) to the projector
  to produce 3D positions for every entity declared in scene_context.json.
* :mod:`scene_discovery` runs at startup to auto-enumerate scene entities
  via the VLM and build a fresh scene_context from scratch.
"""

from .projector import Projector
from .realsense import RealSenseCamera, RealSenseStream, capture_rgbd
from .sam_segmenter import (
    GeometryEstimate,
    SAMSegmenter,
    create_point_cloud_from_rgbd,
    estimate_geometry_from_mask,
    find_sam_checkpoint,
)
from .scene_discovery import DiscoveredEntity, SceneDiscovery
from .scene_perception import EntityLocalization, ScenePerception, save_color_image

__all__ = [
    "Projector",
    "RealSenseCamera",
    "RealSenseStream",
    "capture_rgbd",
    "SAMSegmenter",
    "GeometryEstimate",
    "estimate_geometry_from_mask",
    "create_point_cloud_from_rgbd",
    "find_sam_checkpoint",
    "ScenePerception",
    "EntityLocalization",
    "save_color_image",
    "SceneDiscovery",
    "DiscoveredEntity",
]
