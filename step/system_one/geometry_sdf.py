"""Analytic 2D SDF primitives and constraints for System 1.

Sign convention at the constraint level:
    rho > 0  => satisfied
    rho = 0  => on the boundary
    rho < 0  => violated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import math

import numpy as np

EPS = 1e-8


def _as_xy(xy: Any) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected at least 2 coordinates, got {xy!r}")
    return arr[:2].astype(np.float32)


def _rot2(theta: float) -> np.ndarray:
    c, s = math.cos(float(theta)), math.sin(float(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _segment_distance_and_closest(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < EPS:
        return float(np.linalg.norm(p - a)), a.copy()
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), q.astype(np.float32)


def _point_in_polygon(p: np.ndarray, verts: np.ndarray) -> bool:
    inside = False
    n = len(verts)
    x, y = float(p[0]), float(p[1])
    for i in range(n):
        x0, y0 = float(verts[i, 0]), float(verts[i, 1])
        x1, y1 = float(verts[(i + 1) % n, 0]), float(verts[(i + 1) % n, 1])
        crosses = ((y0 > y) != (y1 > y)) and (
            x < (x1 - x0) * (y - y0) / (y1 - y0 + EPS) + x0
        )
        if crosses:
            inside = not inside
    return inside


@dataclass
class SDFPrimitive2D:
    name: str = "sdf"

    def robustness(self, xy: Any, *, inflation: float = 0.0) -> float:
        raise NotImplementedError

    def grad(self, xy: Any, *, inflation: float = 0.0) -> np.ndarray:
        p = _as_xy(xy)
        h = 1e-4
        g = np.zeros(2, dtype=np.float32)
        for j in range(2):
            e = np.zeros(2, dtype=np.float32)
            e[j] = h
            g[j] = (
                self.robustness(p + e, inflation=inflation)
                - self.robustness(p - e, inflation=inflation)
            ) / (2.0 * h)
        return g.astype(np.float32)

    def value_and_grad(self, xy: Any, *, inflation: float = 0.0) -> tuple[float, np.ndarray]:
        return float(self.robustness(xy, inflation=inflation)), self.grad(xy, inflation=inflation)


@dataclass
class CircleSDF(SDFPrimitive2D):
    center_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    radius: float = 0.05
    convention: str = "inside_positive"

    def robustness(self, xy: Any, *, inflation: float = 0.0) -> float:
        p = _as_xy(xy)
        d = float(np.linalg.norm(p - self.center_xy))
        r = float(self.radius) + float(inflation)
        return d - r if self.convention == "outside_positive" else r - d


@dataclass
class BoxSDF(SDFPrimitive2D):
    center_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    half_extents_xy: np.ndarray = field(default_factory=lambda: np.ones(2, dtype=np.float32) * 0.05)
    rotation_rad: float = 0.0
    convention: str = "inside_positive"

    def _outside_positive_sd(self, xy: Any, *, inflation: float = 0.0) -> float:
        p = _as_xy(xy)
        R = _rot2(float(self.rotation_rad))
        local = R.T @ (p - self.center_xy)
        half = np.asarray(self.half_extents_xy, dtype=np.float32) + float(inflation)
        q = np.abs(local) - half
        outside = float(np.linalg.norm(np.maximum(q, 0.0)))
        inside = float(min(max(float(q[0]), float(q[1])), 0.0))
        return outside + inside

    def robustness(self, xy: Any, *, inflation: float = 0.0) -> float:
        sd = self._outside_positive_sd(xy, inflation=inflation)
        return sd if self.convention == "outside_positive" else -sd


@dataclass
class CapsuleSDF(SDFPrimitive2D):
    p0_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    p1_xy: np.ndarray = field(default_factory=lambda: np.ones(2, dtype=np.float32))
    radius: float = 0.03
    convention: str = "outside_positive"

    def robustness(self, xy: Any, *, inflation: float = 0.0) -> float:
        p = _as_xy(xy)
        d, _ = _segment_distance_and_closest(p, self.p0_xy, self.p1_xy)
        r = float(self.radius) + float(inflation)
        return r - d if self.convention == "inside_positive" else d - r


@dataclass
class HalfPlaneSDF(SDFPrimitive2D):
    normal_xy: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    offset: float = 0.0
    convention: str = "inside_positive"

    def robustness(self, xy: Any, *, inflation: float = 0.0) -> float:
        p = _as_xy(xy)
        n = np.asarray(self.normal_xy, dtype=np.float32)
        n = n / (float(np.linalg.norm(n)) + EPS)
        signed = float(np.dot(n, p) - float(self.offset)) + float(inflation)
        return -signed if self.convention == "outside_positive" else signed


@dataclass
class PolygonSDF(SDFPrimitive2D):
    vertices_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    convention: str = "inside_positive"

    def robustness(self, xy: Any, *, inflation: float = 0.0) -> float:
        p = _as_xy(xy)
        verts = np.asarray(self.vertices_xy, dtype=np.float32)
        if len(verts) < 3:
            return -1e3
        min_d = float("inf")
        for i in range(len(verts)):
            d, _ = _segment_distance_and_closest(p, verts[i], verts[(i + 1) % len(verts)])
            min_d = min(min_d, d)
        inside = _point_in_polygon(p, verts)
        sd = -min_d if inside else min_d
        return sd - float(inflation) if self.convention == "outside_positive" else -sd + float(inflation)


@dataclass
class CompositeSDF(SDFPrimitive2D):
    op: str = "union"
    children: list[SDFPrimitive2D] = field(default_factory=list)

    def robustness(self, xy: Any, *, inflation: float = 0.0) -> float:
        vals = [c.robustness(xy, inflation=inflation) for c in self.children]
        if not vals:
            return -1e3
        if self.op == "intersection":
            return float(min(vals))
        if self.op == "difference":
            return float(min(vals[0], *[-v for v in vals[1:]])) if len(vals) > 1 else float(vals[0])
        return float(max(vals))


@dataclass
class SDFConstraint:
    name: str
    primitive: SDFPrimitive2D
    role: str = "constraint"
    applies_to: tuple[str, ...] = ("object",)
    weight: float = 10.0
    margin: float = 0.0
    active_stages: tuple[int, ...] | None = None
    entity: str | None = None
    notes: str | None = None
    always_active: bool = False

    def applies_to_kind(self, kind: str) -> bool:
        return kind in self.applies_to or "both" in self.applies_to or "all" in self.applies_to

    def active_for_stage(self, stage: int) -> bool:
        return self.active_stages is None or int(stage) in set(int(s) for s in self.active_stages)

    def robustness_xy(self, xy: Any, *, inflation: float = 0.0) -> float:
        return float(self.primitive.robustness(xy, inflation=inflation) - float(self.margin))

    def grad_xy(self, xy: Any, *, inflation: float = 0.0) -> np.ndarray:
        return self.primitive.grad(xy, inflation=inflation).astype(np.float32)


def _tuple_strs(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return ("ee", "object") if value == "both" else (value,)
    if isinstance(value, Iterable):
        out = tuple(str(v) for v in value)
        return ("ee", "object") if "both" in out else out
    return default


def primitive_from_config(cfg: dict[str, Any]) -> SDFPrimitive2D:
    g = dict(cfg.get("geometry", cfg))
    typ = str(g.get("type", g.get("shape", "circle"))).lower()
    convention = str(cfg.get("sdf_convention", g.get("sdf_convention", cfg.get("convention", "inside_positive"))))
    name = str(cfg.get("name", g.get("name", typ)))
    if typ in {"circle", "disk", "cylinder"}:
        center = g.get("center_xy", g.get("xy", g.get("center", [0.0, 0.0])))
        return CircleSDF(name=name, center_xy=_as_xy(center), radius=float(g.get("radius", 0.05)), convention=convention)
    if typ in {"box", "square", "rectangle", "oriented_box", "obb"}:
        center = g.get("center_xy", g.get("xy", g.get("center", [0.0, 0.0])))
        if "half_extents_xy" in g:
            half = _as_xy(g["half_extents_xy"])
        elif "size_xy" in g:
            half = 0.5 * _as_xy(g["size_xy"])
        elif "size" in g:
            half = 0.5 * _as_xy(g["size"])
        else:
            half = np.ones(2, dtype=np.float32) * float(g.get("half_extent", 0.05))
        return BoxSDF(name=name, center_xy=_as_xy(center), half_extents_xy=half, rotation_rad=float(g.get("rotation_rad", g.get("theta", g.get("yaw", 0.0)))), convention=convention)
    if typ in {"capsule", "segment"}:
        return CapsuleSDF(name=name, p0_xy=_as_xy(g.get("p0_xy", g.get("p0", [0.0, 0.0]))), p1_xy=_as_xy(g.get("p1_xy", g.get("p1", [0.1, 0.0]))), radius=float(g.get("radius", 0.03)), convention=convention)
    if typ in {"half_plane", "halfplane"}:
        return HalfPlaneSDF(name=name, normal_xy=_as_xy(g.get("normal_xy", g.get("normal", [1.0, 0.0]))), offset=float(g.get("offset", 0.0)), convention=convention)
    if typ in {"polygon", "poly"}:
        verts = np.asarray(g.get("vertices_xy", g.get("vertices", [])), dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError(f"Polygon SDF requires vertices_xy as Nx2, got {verts.shape}")
        return PolygonSDF(name=name, vertices_xy=verts, convention=convention)
    if typ in {"union", "intersection", "difference"}:
        return CompositeSDF(name=name, op=typ, children=[primitive_from_config(child) for child in g.get("children", [])])
    raise ValueError(f"Unknown SDF primitive type {typ!r}")


def constraint_from_config(cfg: dict[str, Any]) -> SDFConstraint:
    primitive = primitive_from_config(cfg)
    role = str(cfg.get("role", "constraint"))
    applies_default = ("ee", "object") if role in {"obstacle", "keepout"} else ("object",)
    active_raw = cfg.get("active_stages")
    return SDFConstraint(
        name=str(cfg.get("name", primitive.name)),
        primitive=primitive,
        role=role,
        applies_to=_tuple_strs(cfg.get("applies_to"), applies_default),
        weight=float(cfg.get("weight", cfg.get("mpc_weight", 10.0))),
        margin=float(cfg.get("margin", cfg.get("safety_margin", 0.0))),
        active_stages=None if active_raw is None else tuple(int(s) for s in active_raw),
        entity=None if cfg.get("entity") is None else str(cfg.get("entity")),
        notes=cfg.get("notes"),
        always_active=bool(cfg.get("always_active", False)),
    )


def parse_sdf_constraints(scene_context: dict[str, Any] | None) -> list[SDFConstraint]:
    if not isinstance(scene_context, dict):
        return []
    raw: list[dict[str, Any]] = []
    for item in scene_context.get("sdf_constraints", []) or []:
        if isinstance(item, dict):
            raw.append(item)
    platform = scene_context.get("platform", {})
    if isinstance(platform, dict):
        for item in platform.get("sdf_constraints", []) or []:
            if isinstance(item, dict):
                raw.append(item)

    entities = {str(e.get("name")): e for e in scene_context.get("entities", []) if isinstance(e, dict) and e.get("name")}
    for name, geom in (scene_context.get("geometry", {}) or {}).items():
        if not isinstance(geom, dict):
            continue
        sdf_geom = geom.get("sdf_geometry", geom.get("sdf", None))
        if not isinstance(sdf_geom, dict):
            continue
        ent = entities.get(str(name), {})
        role = str(geom.get("role", ent.get("role", "constraint")))
        convention = sdf_geom.get("sdf_convention")
        if convention is None:
            convention = "outside_positive" if role in {"obstacle", "keepout"} else "inside_positive"
        raw.append({
            "name": str(name),
            "entity": str(name),
            "role": role,
            "geometry": sdf_geom,
            "sdf_convention": convention,
            "applies_to": sdf_geom.get("applies_to", ["ee", "object"] if role in {"obstacle", "keepout"} else ["object"]),
            "weight": sdf_geom.get("weight", 24.0 if role in {"obstacle", "keepout"} else 8.0),
            "margin": sdf_geom.get("margin", 0.0),
            # Without this a geometry-derived constraint is only ever live while
            # its name appears in the env's active set, which is populated from a
            # subtask's applied_constraints -- something System 2 never emits. A
            # benchmark keep-out expressing G(Safe(obstacle)) must hold for the
            # whole episode, so it declares always_active itself.
            "always_active": bool(sdf_geom.get("always_active", False)),
        })

    out = []
    for cfg in raw:
        try:
            out.append(constraint_from_config(cfg))
        except Exception as exc:
            print(f"[sdf] skipping invalid SDF constraint {cfg.get('name', '<unnamed>')}: {exc}")
    return out
