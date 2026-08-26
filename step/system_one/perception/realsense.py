"""Intel RealSense capture utilities.

Wraps the :mod:`pyrealsense2` SDK in a small, opinionated API:

* :class:`RealSenseCamera` manages one device (start, warmup, capture aligned
  RGB + depth, expose intrinsics).
* :class:`RealSenseStream` holds a dictionary of cameras keyed by name (e.g.
  ``"left"`` and ``"right"``) and captures from all of them in lock-step.
* :func:`capture_rgbd` is a one-shot helper that flushes stale frames first
  (useful between motions where the buffer can be ~1 second behind reality).

All RealSense imports are deferred so the sim backend keeps running on
machines that do not have ``pyrealsense2`` installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


def _try_import_rs():
    try:
        import pyrealsense2 as rs  # type: ignore

        return rs
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "pyrealsense2 is required for the real-robot backend. Install with `pip install pyrealsense2`."
        ) from exc


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float

    @classmethod
    def from_rs(cls, intr) -> "CameraIntrinsics":
        return cls(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            ppx=float(intr.ppx),
            ppy=float(intr.ppy),
        )


@dataclass
class RealSenseCamera:
    serial: str
    name: str = "camera"
    width: int = 640
    height: int = 480
    fps: int = 30
    pipeline: object = field(default=None, repr=False)
    align: object = field(default=None, repr=False)
    intrinsics: Optional[CameraIntrinsics] = None

    def start(self) -> None:
        rs = _try_import_rs()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.serial))
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.pipeline = pipeline
        self.align = align
        self.intrinsics = CameraIntrinsics.from_rs(intr)

    def warmup(self, frames: int = 30) -> None:
        if self.pipeline is None:
            raise RuntimeError(f"Camera '{self.name}' is not started.")
        for _ in range(int(frames)):
            self.pipeline.wait_for_frames()

    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture one aligned (color_bgr, depth) pair.

        RealSense is configured with rs.format.bgr8 to match OpenCV and the
        legacy cloud_grasp / interactive_planner.py pipeline. Do NOT treat the
        returned color image as RGB unless you explicitly convert it with
        color[:, :, ::-1].
        """
        if self.pipeline is None or self.align is None:
            raise RuntimeError(f"Camera '{self.name}' is not started.")
        # Flush stale buffered frames so we get a fresh post-motion snapshot.
        for _ in range(15):
            self.pipeline.wait_for_frames()
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color = np.asanyarray(aligned.get_color_frame().get_data())
        depth = np.asanyarray(aligned.get_depth_frame().get_data())
        return color, depth

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.align = None


@dataclass
class RealSenseStream:
    """A bundle of RealSense cameras keyed by logical name (left / right / wrist)."""

    cameras: Dict[str, RealSenseCamera] = field(default_factory=dict)

    def add(self, camera: RealSenseCamera) -> None:
        self.cameras[camera.name] = camera

    def start_all(self, warmup_frames: int = 30) -> None:
        for cam in self.cameras.values():
            cam.start()
        for cam in self.cameras.values():
            cam.warmup(warmup_frames)

    def capture_all(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {name: cam.capture() for name, cam in self.cameras.items()}

    def stop_all(self) -> None:
        for cam in list(self.cameras.values()):
            cam.stop()


def capture_rgbd(camera: RealSenseCamera) -> Tuple[np.ndarray, np.ndarray]:
    """One-shot flush-and-capture helper."""
    return camera.capture()
