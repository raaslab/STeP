
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Object-class substrings that require a rim grasp (offset above centroid,
# wider XY close-tolerance). Matching is case-insensitive substring search
# against the active object name (e.g. "akita_black_bowl_1").
RIM_GRASP_KEYWORDS: frozenset[str] = frozenset([
    "bowl",
    "mug",
    "ramekin",
    "frypan",
    "pan",
    "pot",
    "cup",
    "pitcher",
    "stein",
    "plate",
    "saucer",
])


@dataclass(frozen=True)
class GraspMode:
    name: str
    pregrasp_offset: np.ndarray
    grasp_offset: np.ndarray
    desired_approach_dir: np.ndarray
    world_rotvec_offset: np.ndarray
    local_wrist_roll: float = 0.0
    # Applied on top of cube_pos in the top-grasp branch of _targets().
    # [dx, dy, dz] in world frame — positive Z moves the grasp point upward
    # toward the rim instead of down into the bowl center.
    grasp_rim_offset: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "grasp_rim_offset",
            _arr([0.0, 0.0, 0.0]) if self.grasp_rim_offset is None else np.asarray(self.grasp_rim_offset, dtype=np.float32),
        )


def _arr(x: list[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


MODE_CONFIGS: dict[str, GraspMode] = {
    "left": GraspMode(
        name="left",
        # Side approach from +y toward the object. Keep the wrist near cube
        # mid-height rather than hovering 12 cm above as in the top-grasp
        # approach. The final side-grasp point is slightly outside the cube
        # face, not above the cube.
        pregrasp_offset=_arr([0.00, 0.105, 0.035]),
        grasp_offset=_arr([0.00, -0.025, -0.002]),
        desired_approach_dir=_arr([0.0, -1.0, -0.05]),
        world_rotvec_offset=_arr([-np.pi / 2.0, 0.0, 0.0]),
        local_wrist_roll=-np.pi / 2.0,
    ),
    "right": GraspMode(
        name="right",
        # Symmetric side approach from -y toward the object.
        pregrasp_offset=_arr([0.00, -0.105, 0.035]),
        grasp_offset=_arr([0.00, 0.0, 0.006]),
        desired_approach_dir=_arr([0.0, 1.0, -0.05]),
        world_rotvec_offset=_arr([np.pi / 2.0, 0.0, 0.0]),
        local_wrist_roll=np.pi / 2.0,
    ),
    "top": GraspMode(
        name="top",
        pregrasp_offset=_arr([0.00, 0.00, 0.15]),
        grasp_offset=_arr([0.00, 0.00, 0.0]),
        desired_approach_dir=_arr([0.0, 0.0, -1.0]),
        world_rotvec_offset=_arr([0.0, 0.0, 0.0]),
        local_wrist_roll=0.0,
        # Lift the final grasp target above the object centroid so the
        # gripper fingers land on the bowl rim rather than plunging into
        # the bowl interior.
        # cube_pos for LIBERO bowls is at the base of the mesh (table level).
        # The rim is ~7 cm above that, so lift the grasp target to the rim.
        grasp_rim_offset=_arr([0.00, 0.045, 0.05]),
    ),
}
