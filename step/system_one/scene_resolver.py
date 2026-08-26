"""Scene-context name, alias, role, and geometry resolution utilities."""

from __future__ import annotations

from typing import Any
import numpy as np


class SceneResolver:
    """Maps task-spec strings to canonical scene-context entities and geometry."""

    def __init__(self, scene_context: dict[str, Any]):
        self.scene_context = scene_context
        self.entity_aliases: dict[str, str] = {}
        self.entity_roles: dict[str, str] = {}
        for entity in scene_context.get("entities", []):
            canonical = entity["name"]
            self.entity_aliases[canonical.lower()] = canonical
            self.entity_roles[canonical] = entity.get("role", "unknown")
            for alias in entity.get("aliases", []):
                self.entity_aliases[alias.lower()] = canonical
        self.geometry = scene_context.get("geometry", {})

    def role_of(self, name: str | None) -> str:
        if name is None:
            return "unknown"
        return self.entity_roles.get(self.canonical(name) or "", "unknown")

    def manipulated_object_names(self) -> set[str]:
        return {n for n, r in self.entity_roles.items() if r == "manipulated_object"}

    def support_object_names(self) -> set[str]:
        return {n for n, r in self.entity_roles.items() if r == "support_object"}

    def goal_region_names(self) -> set[str]:
        return {n for n, r in self.entity_roles.items() if r == "goal_region"}

    def obstacle_names(self) -> set[str]:
        return {n for n, r in self.entity_roles.items() if r == "obstacle"}

    def dynamic_goal_reference_names(self) -> set[str]:
        return self.manipulated_object_names() | self.support_object_names()

    def canonical(self, name: str | None) -> str | None:
        if name is None:
            return None
        lower = name.lower().strip()
        if lower in self.entity_aliases:
            return self.entity_aliases[lower]

        # Fuzzy fallbacks driven by entity object_class. This mirrors the old
        # color/keyword heuristics but stays role-agnostic.
        for entity in self.scene_context.get("entities", []):
            cls = (entity.get("object_class") or "").lower()
            ent_name = entity["name"].lower()
            if cls and cls in lower:
                return entity["name"]
            if ent_name and ent_name in lower:
                return entity["name"]
        return name

    def _first_geometry_for_role(self, role: str) -> tuple[str | None, dict]:
        for entity in self.scene_context.get("entities", []):
            if entity.get("role") == role:
                geom = self.geometry.get(entity["name"], {})
                if "xy" in geom:
                    return entity["name"], geom
        return None, {}

    def resolve_goal(
        self, name: str | None
    ) -> tuple[str, tuple[float, float], float, float]:
        canonical = self.canonical(name)
        geom = self.geometry.get(canonical, {})
        if not geom or "xy" not in geom:
            fallback_name, fallback_geom = self._first_geometry_for_role("goal_region")
            canonical = canonical or fallback_name or "goal_region"
            geom = fallback_geom or {
                "xy": list(self._workspace_center_xy()),
                "radius": 0.055,
                "descend_radius": 0.075,
            }
        xy = geom.get("xy", list(self._workspace_center_xy()))
        return (
            canonical or "goal_region",
            (float(xy[0]), float(xy[1])),
            float(geom.get("radius", 0.055)),
            float(geom.get("descend_radius", 0.075)),
        )

    def _workspace_center_xy(self) -> tuple[float, float]:
        """Return a sensible XY default that respects the platform's workspace.

        Used when the scene_context has no goal_region yet (e.g. the auto-scene
        config before discovery has populated the entities). For sim, this
        falls back to the legacy (0.22, 0.22).
        """
        bounds = (
            self.scene_context.get("platform", {})
            .get("workspace_bounds", {})
        )
        x = bounds.get("x")
        y = bounds.get("y")
        if x and y and len(x) == 2 and len(y) == 2:
            return (float(x[0] + x[1]) / 2.0, float(y[0] + y[1]) / 2.0)
        return (0.22, 0.22)

    def resolve_keepout(self, name: str | None) -> tuple[float, float, float]:
        canonical = self.canonical(name)
        geom = self.geometry.get(canonical, {}) if canonical else {}
        if not geom or "xy" not in geom:
            # Match legacy behavior: when a support_object is requested as a
            # keep-out, fall back to the obstacle region geometry first.
            obstacle_name, obstacle_geom = self._first_geometry_for_role("obstacle")
            if obstacle_geom:
                geom = obstacle_geom
            else:
                support_name, support_geom = self._first_geometry_for_role("support_object")
                if support_geom:
                    geom = support_geom
        xy = geom.get("xy", [0.0, 0.0])
        return float(xy[0]), float(xy[1]), float(geom.get("radius", 0.0))
