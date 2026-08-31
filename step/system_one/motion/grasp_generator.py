"""Grasp-candidate generation.

Starting point is intentionally simple, per Cap-X's privileged-state generator
(``capx/integrations/franka/libero_privileged.py``, MIT License, Copyright (c)
2026 Max Fu, https://github.com/capgym/cap-x): ``FrankaLiberoPrivilegedApi.
sample_grasp_pose()`` there is a single hardcoded top-down grasp at the
object's privileged pose with quaternion **WXYZ** ``[0, 1, 0, 0]`` (gripper
pointing straight down). This module keeps that same top-down, oracle-pose
starting point but extends it in two ways:

1. **Center-grasp candidates**: multiple wrist-yaw samples, a real pregrasp
   offset, and a documented, model-derived fingertip/hand transform instead
   of Cap-X's hardcoded ``FRANKA_HAND_TO_FINGERTIP_Z_M`` constant (see
   ``scene_query.SceneQuery.hand_to_tcp_transform``). This is the right
   strategy for anything narrower than the gripper's opening at *some* yaw
   (cans, boxes, bottles).

2. **Rim-edge candidates**: Cap-X has no object-specific grasp logic to
   borrow here -- its privileged generator is a single fixed pose, and its
   vision path (``libero.py``) delegates entirely to Contact-GraspNet, which
   finds rim/wall grasps only by sampling antipodal contacts over real
   geometry, not via a hand-coded rule. Since we don't have Contact-GraspNet
   available, this generator adds an explicit strategy for objects whose
   *whole* cross-section is too wide to encompass at any yaw (bowls, mugs,
   cups): grasp a single point around the object's rim instead, with the
   wrist yawed so the gripper's closing axis is *radial* at that point (one
   finger lands just outside the rim, the other just inside) rather than
   trying to span the object's full diameter. This is the same geometric
   idea as the legacy stage machine's fixed ``grasp_rim_offset`` (see
   ``base_env.py``'s ``_effective_grasp_rim_offset``/``modes.py``'s
   ``RIM_GRASP_KEYWORDS``) -- generalized from one hardcoded direction/object
   to *every* object, sampled at ``rim_num_angle_samples`` points around the
   full circumference instead of one fixed direction, so there's always a
   candidate on the side that's actually IK-reachable and collision-free.
   Verified against the real ``akita_black_bowl`` mesh: the legacy system's
   ``[0.0, 0.045, 0.05]`` offset (a Y-offset of 4.5cm) corresponds almost
   exactly to ``rim_offset_fraction=0.6`` of this generator's AABB-derived
   radius for that same bowl (measured rim ring radius ~0.047m vs `0.6 *
   min(half_extents[:2])` = ~0.047m) -- the default wasn't guessed, it was
   calibrated against the one object we already know a working offset for.

Multiple orientations across *both* strategies are handed to ``PickSkill``,
which is responsible for picking a working one via IK/collision/OMPL
feasibility and cost -- this generator's job is only to make sure a feasible
candidate exists somewhere in the set it proposes, not to guess which one
will work.

Quaternion convention: **WXYZ** everywhere (position of components:
``[qw, qx, qy, qz]``), matching Cap-X and MuJoCo's own native body/site quaternion
layout -- see ``scene_query.py``'s module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import GraspSampleConfig
from .scene_query import SceneQuery, quat_mul_wxyz
from .types import GraspCandidate

# Top-down grasp orientation in the *world* frame, expressed as a quaternion
# that rotates the gripper's canonical +Z (approach axis) to point straight
# down (-world Z). This is the same convention as Cap-X's [0, 1, 0, 0]: a 180
# degree rotation about the world X axis.
_TOP_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])


def yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    """Quaternion for a rotation of ``yaw_rad`` about the world +Z axis."""
    half = yaw_rad / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


class GraspGenerator(ABC):
    @abstractmethod
    def generate(self, scene: SceneQuery, object_names: list[str]) -> list[GraspCandidate]:
        """Return candidate grasps for every name in ``object_names``. Order is
        not meaningful -- ``PickSkill`` scores and ranks candidates itself."""


class PrivilegedTopDownGraspGenerator(GraspGenerator):
    """Oracle-pose, top-down, parallel-jaw grasp sampler.

    For each object: read its privileged world pose from MuJoCo, sample
    ``config.num_yaw_samples`` wrist yaws evenly over ``config.yaw_range``
    (default ``[0, pi)`` -- a parallel-jaw gripper is symmetric under a pi
    rotation about its approach axis, so sampling a full 2*pi would just
    duplicate every candidate), grasp at the object's geometric center (offset
    by half its Z half-extent so the TCP lands at the object's mid-height
    rather than its base -- see the ``Grasped(object, height)`` mid-height
    convention already used by System 2's prompt, `step/system_two/prompts.py`
    rule 7b), and build a pregrasp pose offset ``config.pregrasp_offset_m``
    back along the approach direction.
    """

    def __init__(
        self,
        config: GraspSampleConfig | None = None,
        hand_site: str = "gripper0_right_grip_site",
        gripper_max_opening_m: float = 0.08,
    ) -> None:
        self.config = config or GraspSampleConfig()
        self.hand_site = hand_site
        self.gripper_max_opening_m = gripper_max_opening_m

    def generate(self, scene: SceneQuery, object_names: list[str]) -> list[GraspCandidate]:
        candidates: list[GraspCandidate] = []
        for name in object_names:
            candidates.extend(self._generate_for_object(scene, name))
        return candidates

    def _generate_for_object(self, scene: SceneQuery, object_name: str) -> list[GraspCandidate]:
        obj_pos, _obj_quat = scene.body_pose(object_name)
        extent = scene.body_extent_world(object_name)
        # Mid-height in the object's own AABB, not just the body origin + half
        # its Z extent -- a body's collision geoms aren't always centered on
        # its own frame origin (this bowl's AABB center sits 2.8cm above its
        # body origin), and using center_offset fixes both strategies' grasp
        # height instead of just approximating it off the origin.
        height_offset = float(extent.center_offset[2]) if extent is not None else 0.0
        grasp_center = obj_pos + np.array([0.0, 0.0, height_offset])
        approach_dir_world = np.array([0.0, 0.0, -1.0])  # top-down: approach from above

        out: list[GraspCandidate] = []
        out.extend(self._center_grasp_candidates(object_name, grasp_center, approach_dir_world, extent))
        if extent is not None:
            out.extend(self._rim_edge_candidates(object_name, grasp_center, approach_dir_world, extent))
        return out

    def _center_grasp_candidates(
        self, object_name: str, grasp_center: np.ndarray, approach_dir_world: np.ndarray, extent
    ) -> list[GraspCandidate]:
        n = max(1, int(self.config.num_yaw_samples))
        yaw_lo, yaw_hi = self.config.yaw_range
        yaws = np.linspace(yaw_lo, yaw_hi, n, endpoint=False)

        out: list[GraspCandidate] = []
        for i, yaw in enumerate(yaws):
            grasp_quat = quat_mul_wxyz(yaw_quat_wxyz(float(yaw)), _TOP_DOWN_QUAT_WXYZ)
            pregrasp_center = grasp_center - self.config.pregrasp_offset_m * approach_dir_world

            width = self._estimate_width(extent, float(yaw)) if extent is not None else None
            score = self._score(width)

            out.append(
                GraspCandidate(
                    object_name=object_name,
                    grasp_pose_world=np.concatenate([grasp_center, grasp_quat]),
                    pregrasp_pose_world=np.concatenate([pregrasp_center, grasp_quat]),
                    approach_direction_world=approach_dir_world,
                    gripper_width=width,
                    generator_score=score,
                    metadata={
                        "generator": "PrivilegedTopDownGraspGenerator",
                        "strategy": "center",
                        "yaw_rad": float(yaw),
                        "yaw_index": i,
                    },
                )
            )
        return out

    def _rim_edge_candidates(
        self, object_name: str, grasp_center: np.ndarray, approach_dir_world: np.ndarray, extent
    ) -> list[GraspCandidate]:
        """Grasp a single point around the object's rim/perimeter, wrist
        yawed so the gripper closes radially (one finger just outside the
        rim, one just inside) -- see module docstring for why this exists and
        how the default offset fraction was calibrated."""
        n = max(0, int(self.config.rim_num_angle_samples))
        if n == 0:
            return []
        radius = self.config.rim_offset_fraction * min(extent.half_extents[0], extent.half_extents[1])
        width = float(self.config.rim_grasp_width_m)
        score = self._score(width)

        out: list[GraspCandidate] = []
        for i in range(n):
            theta = 2.0 * np.pi * i / n
            radial_dir = np.array([np.cos(theta), np.sin(theta)])
            point_center = grasp_center + np.array([radius * radial_dir[0], radius * radial_dir[1], 0.0])
            # _estimate_width's close_axis for yaw is [-sin(yaw), cos(yaw)];
            # solving close_axis == radial_dir gives yaw = theta - pi/2, so the
            # fingers close along the line through the object's center (i.e.
            # straddle the rim wall) instead of tangent to it.
            yaw = theta - np.pi / 2.0
            grasp_quat = quat_mul_wxyz(yaw_quat_wxyz(float(yaw)), _TOP_DOWN_QUAT_WXYZ)
            pregrasp_center = point_center - self.config.pregrasp_offset_m * approach_dir_world

            out.append(
                GraspCandidate(
                    object_name=object_name,
                    grasp_pose_world=np.concatenate([point_center, grasp_quat]),
                    pregrasp_pose_world=np.concatenate([pregrasp_center, grasp_quat]),
                    approach_direction_world=approach_dir_world,
                    gripper_width=width,
                    generator_score=score,
                    metadata={
                        "generator": "PrivilegedTopDownGraspGenerator",
                        "strategy": "rim_edge",
                        "yaw_rad": float(yaw),
                        "rim_angle_rad": float(theta),
                        "rim_radius_m": float(radius),
                    },
                )
            )
        return out

    def _estimate_width(self, extent, yaw_rad: float) -> float:
        # Width the gripper must span across is the object's AABB extent
        # projected onto the axis the fingers close along (gripper local X
        # after the yaw rotation). This is exact for a box-shaped object, but
        # a rotated-square-AABB projection *overestimates* a round object's
        # true (yaw-invariant) cross-section at oblique angles -- e.g. it
        # predicts the AABB's diagonal instead of the actual diameter for a
        # cylindrical can whose XY half-extents happen to be near-equal. Cap
        # at the object's narrowest overall horizontal extent (always
        # physically achievable at *some* yaw) so we never let that artifact
        # inflate the estimate past what's actually graspable.
        close_axis = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)])
        projected = 2.0 * abs(close_axis[0]) * extent.half_extents[0] + 2.0 * abs(
            close_axis[1]
        ) * extent.half_extents[1]
        narrowest = 2.0 * float(min(extent.half_extents[0], extent.half_extents[1]))
        return float(min(projected, narrowest))

    def _score(self, width: float | None) -> float:
        # Higher score = more preferred by the generator alone (before IK/
        # collision/downstream feasibility are folded in by PickSkill). Prefer
        # grasps whose required opening leaves comfortable clearance inside the
        # gripper's max opening; penalize widths that are too close to (or
        # exceed) the physical limit.
        if width is None:
            return 0.5
        if width >= self.gripper_max_opening_m:
            return 0.0
        return float(np.clip(1.0 - width / self.gripper_max_opening_m, 0.0, 1.0))


def hand_from_fingertip_pose(
    scene: SceneQuery,
    fingertip_pos: np.ndarray,
    fingertip_quat_wxyz: np.ndarray,
    hand_site: str = "gripper0_right_ft_frame",
    tcp_site: str = "gripper0_right_grip_site",
) -> np.ndarray:
    """Convert a grasp pose expressed in the fingertip/TCP frame (the
    convention most grasp detectors -- including Contact-GraspNet, per Cap-X's
    ``graspnet.py`` -- output) into the ``panda_hand``/flange frame that a
    motion planner's IK typically targets.

    Mirrors Cap-X's ``curobo_api._grasp_pose_fingertip_to_hand`` in spirit, but
    reads the offset live from the model (``SceneQuery.hand_to_tcp_transform``)
    instead of hardcoding ``FRANKA_HAND_TO_FINGERTIP_Z_M``. Returns a (7,)
    ``[x, y, z, qw, qx, qy, qz]`` pose.
    """
    offset_z = scene.hand_to_tcp_transform(hand_site, tcp_site)
    from .scene_query import quat_to_rotmat_wxyz

    R = quat_to_rotmat_wxyz(fingertip_quat_wxyz)
    hand_pos = np.asarray(fingertip_pos, dtype=np.float64) - R @ np.array([0.0, 0.0, offset_z])
    return np.concatenate([hand_pos, fingertip_quat_wxyz])
