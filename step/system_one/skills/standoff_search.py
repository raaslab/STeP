"""Standoff-position candidate search, shared by ``PickSkill`` and
``OpenDrawerSkill`` -- any skill whose final approach walks a fixed grasp
point down from a standoff along a single approach direction runs into the
same problem: a single fixed standoff position assumes the airspace along
that one direction is always clear, which real cluttered scenes routinely
violate (an object left on the table between the robot and the target).
"""

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
    """Candidate standoff positions to try, in order: the plain straight-line
    standoff first (directly opposite the approach direction at the nominal
    distance), then increasingly wide rings of positions offset sideways in
    the plane perpendicular to the approach, *and* at increasingly larger
    standoff distances -- see ``GraspSampleConfig.standoff_lateral_offsets_m``
    for why a single fixed standoff isn't always enough (something else can
    be sitting in that airspace).

    Distance matters independently of lateral direction: an obstacle
    hovering over the object can block every horizontal direction at the
    *nominal* standoff height, since reaching that height at all requires
    sweeping an arm link (not just the gripper) through the obstacle's
    footprint. A taller standoff can clear the obstacle's height entirely,
    letting the final descent come down past it instead of detouring around
    it at the same blocked altitude.
    """
    approach = approach_direction_world / np.linalg.norm(approach_direction_world)

    # Any orthonormal basis for the plane perpendicular to `approach`.
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
