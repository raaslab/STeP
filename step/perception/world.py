"""The perceived world state: what the vision pipeline believes is in the scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import BoxGeom, ObjectShape


@dataclass
class PerceivedObject:
    name: str
    category: str  # the SAM3 prompt that finds it again on later frames
    role: str
    description: str = ""
    shape: ObjectShape | None = None
    score: float = 0.0
    last_seen_step: int = 0
    # Pose of the body origin; differs from shape.origin_world only once the
    # object has been carried somewhere.
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    # Set while held: object pose in the TCP frame, captured at grasp time.
    tcp_from_object: np.ndarray | None = None
    is_static_fixture: bool = False
    relation: str = ""  # e.g. "rests on ramekin", measured from depth at discovery

    def footprint_radius(self) -> float:
        if self.shape is None:
            return 0.0
        return float(max(np.linalg.norm(np.abs(b.center_local[:2]) + b.half_size[:2]) for b in self.shape.boxes))

    def centroid(self) -> np.ndarray:
        if self.shape is None:
            return self.pos.copy()
        return self.pos + np.array([0.0, 0.0, 0.5 * (self.shape.top_z - self.shape.bottom_z)])


@dataclass
class PerceivedDrawer:
    cabinet: str
    level: str  # "top" / "middle" / "bottom" / "" (single drawer)
    handle_center: np.ndarray  # world, at the drawer's *closed* position
    handle_half: np.ndarray  # (3,) in the drawer frame (x = along pull axis)
    panel_center: np.ndarray  # world, closed position
    panel_half: np.ndarray  # (3,) in the drawer frame
    pull_axis: np.ndarray  # unit, horizontal, pointing out of the cabinet
    travel_m: float
    qpos: float = 0.0  # current opening along pull_axis
    grasp_tcp_offset: float | None = None  # TCP projection - qpos at grasp, while held

    @property
    def body_name(self) -> str:
        return f"{self.cabinet}_cabinet_{self.level or 'drawer'}"

    @property
    def joint_name(self) -> str:
        return f"{self.cabinet}_{self.level or 'drawer'}_level"


@dataclass
class PerceivedKnob:
    fixture: str
    center: np.ndarray  # world pivot, on the mounting surface
    axis: np.ndarray  # unit, outward normal of the mounting surface
    half: np.ndarray  # (3,) radius, radius, half-height along axis
    range_rad: tuple[float, float]
    # The handle (raised bar on the base disc): offset of its midpoint from
    # `center` in the mounting plane, its long direction in that plane, and
    # half-extents (long, thin, half-height). None when no handle was seen.
    grip_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grip_dir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    grip_half: np.ndarray | None = None
    qpos: float = 0.0
    # While held: TCP rotation and knob angle at grasp time.
    grasp_tcp_rot: np.ndarray | None = None
    grasp_qpos: float = 0.0

    @property
    def body_name(self) -> str:
        # infer_stove_fixtures strips "_knob" to find the fixture entity.
        return f"{self.fixture}_knob"


@dataclass
class WorldEstimate:
    support_z: float | None = None
    support_xy_bounds: tuple[np.ndarray, np.ndarray] | None = None
    objects: dict[str, PerceivedObject] = field(default_factory=dict)
    drawers: list[PerceivedDrawer] = field(default_factory=list)
    knobs: list[PerceivedKnob] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialisable summary, stored in scene_context["perception"] so a
        later process (the execution env) re-identifies the same objects
        under the same names without asking the VLM again."""

        def f(v):
            return [round(float(x), 5) for x in np.asarray(v).reshape(-1)]

        return {
            "support_z": self.support_z,
            "objects": [
                {
                    "name": o.name,
                    "category": o.category,
                    "role": o.role,
                    "description": o.description,
                    "relation": o.relation,
                    "pos": f(o.pos),
                    "centroid": f(o.centroid()),
                    "radius": round(o.footprint_radius(), 4),
                    "top_z": None if o.shape is None else o.shape.top_z,
                    "is_static_fixture": o.is_static_fixture,
                    "score": round(o.score, 3),
                    "num_boxes": 0 if o.shape is None else len(o.shape.boxes),
                }
                for o in self.objects.values()
            ],
            "drawers": [
                {
                    "cabinet": d.cabinet,
                    "level": d.level,
                    "handle_center_closed": f(d.handle_center),
                    "pull_axis": f(d.pull_axis),
                    "travel_m": round(d.travel_m, 4),
                    "qpos": round(d.qpos, 4),
                }
                for d in self.drawers
            ],
            "knobs": [
                {"fixture": k.fixture, "center": f(k.center), "axis": f(k.axis), "radius": round(float(k.half[0]), 4), "qpos": round(k.qpos, 4)}
                for k in self.knobs
            ],
        }


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


__all__ = ["BoxGeom", "ObjectShape", "PerceivedObject", "PerceivedDrawer", "PerceivedKnob", "WorldEstimate", "yaw_quat_wxyz"]
