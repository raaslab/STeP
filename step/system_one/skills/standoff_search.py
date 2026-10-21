"""Standoff-position candidate search, shared by PickSkill and
OpenDrawerSkill: a single fixed standoff assumes the airspace along the
approach direction is clear, which clutter (e.g. an object on the table) can violate."""

from __future__ import annotations

import numpy as np


def standoff_candidates(
    grasp_position: np.ndarray,
    approach_direction_world: np.ndarray,
    standoff_m: float,
    standoff_distance_multipliers: tuple[float, ...],
    lateral_offsets_m: tuple[float, ...],
    n_directions: int,
) -> list[np.ndarray]:
    """Candidate standoff positions, in order: the straight-line standoff,
    then widening rings of lateral offsets and larger standoff distances (which can clear an obstacle blocking every lateral direction)."""
    approach = approach_direction_world / np.linalg.norm(approach_direction_world)

    # Any orthonormal basis for the plane perpendicular to approach.
    ref = np.array([0.0, 0.0, 1.0]) if abs(approach[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    lateral1 = np.cross(approach, ref)
    lateral1 /= np.linalg.norm(lateral1)
    lateral2 = np.cross(approach, lateral1)

    angles = np.linspace(0.0, 2.0 * np.pi, max(1, n_directions), endpoint=False)
    candidates: list[np.ndarray] = []
    for multiplier in standoff_distance_multipliers:
        straight = grasp_position - (standoff_m * multiplier) * approach
        candidates.append(straight)
        for radius in lateral_offsets_m:
            for angle in angles:
                offset = radius * (np.cos(angle) * lateral1 + np.sin(angle) * lateral2)
                candidates.append(straight + offset)
    return candidates
