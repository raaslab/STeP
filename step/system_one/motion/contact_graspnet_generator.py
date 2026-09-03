"""Learned grasp-candidate generator: Contact-GraspNet, called in-process.

Runs directly against ``contact_graspnet_pytorch.GraspEstimator`` -- the same
class Cap-X's own perception server (``capx/serving/
launch_contact_graspnet_server.py``, MIT License, https://github.com/capgym/
cap-x) wraps in a FastAPI endpoint -- rather than talking to a separate
server process. The user has a dedicated CUDA machine and asked to skip the
extra server: this module is meant to be imported by a ``run_libero_step.py``
process running *on that machine*, with ``contact_graspnet_pytorch`` and a
real checkpoint installed there. This dev environment (no CUDA) never
imports ``contact_graspnet_pytorch`` at all -- see ``_load_real_estimator``.

Feeds ``GraspGenerator``'s existing contract (see ``grasp_generator.py``):
returned candidates are handed to ``PickSkill`` alongside
``PrivilegedTopDownGraspGenerator``'s via ``CompositeGraspGenerator``, which
does its own IK/collision/OMPL feasibility filtering -- this generator's job
is only to propose plausible poses, not to guess which one will work.

Point cloud construction uses ground-truth mesh geometry (``SceneQuery.
body_surface_points_dense_local``/``nearby_body_names``), not a rendered
depth image -- the simplest buildable v1 given this is sim ground truth, not
a real camera. Known limitation: no table/floor representation (MuJoCo plane
geoms sample to zero points) and a full, unoccluded view rather than a
realistic single-camera partial one. If real testing shows disappointing
grasp quality, the documented next step is a simulated depth-camera capture
(mirroring Cap-X's own ``agentview``/``robot0_eye_in_hand`` multi-view
fusion) instead -- out of scope for this pass.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from .config import ContactGraspNetConfig
from .grasp_generator import GraspGenerator, hand_from_fingertip_pose
from .scene_query import SceneQuery, quat_to_rotmat_wxyz, rotmat_to_quat_wxyz
from .types import GraspCandidate

logger = logging.getLogger(__name__)

# Single-object segmap id used for every per-object predict_scene_grasps call
# -- pc_segments/return dicts only ever hold this one key here, since this
# generator is called one object at a time (matching PrivilegedTopDown
# GraspGenerator's own per-object loop in generate()).
_SEGMAP_ID = 1

EstimatorFactory = Callable[[ContactGraspNetConfig], Any]


def _load_real_estimator(config: ContactGraspNetConfig) -> Any:
    """Construct and load the real Contact-GraspNet PyTorch estimator.

    Imported lazily (only from inside this function body) so that merely
    importing this module -- or constructing a ``ContactGraspNetGenerator``
    with ``enabled=False`` -- never requires ``contact_graspnet_pytorch`` or
    CUDA to be installed. Mirrors ``scene_query.py``'s own pattern for a
    missing ``mujoco`` install: a clear, actionable error instead of a bare
    ``ModuleNotFoundError``.

    Call shape transcribed directly from Cap-X's server
    (``capx/serving/launch_contact_graspnet_server.py``) -- not yet executed
    against a real checkpoint, since this dev machine has no CUDA. Expect to
    fix up the exact YAML-loading/config shape once run for real once on the
    target GPU machine.
    """
    try:
        import yaml
        from contact_graspnet_pytorch.checkpoints import CheckpointIO
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    except ImportError as exc:
        raise ImportError(
            "ContactGraspNetGenerator requires 'contact_graspnet_pytorch' (+ CUDA + "
            "torch) to be installed -- this is expected to run only on a CUDA machine, "
            "never in this dev environment. See Cap-X (https://github.com/capgym/cap-x) "
            "for the vendored package and its 'contactgraspnet' extra."
        ) from exc

    if not config.checkpoint_dir:
        raise FileNotFoundError(
            "ContactGraspNetConfig.checkpoint_dir is empty -- set it to a directory "
            "containing Contact-GraspNet's model.pt and config.yaml before enabling "
            "contact_graspnet in the motion config."
        )

    import os

    with open(os.path.join(config.checkpoint_dir, "config.yaml")) as f:
        global_config = yaml.safe_load(f)

    estimator = GraspEstimator(global_config)
    checkpoint_io = CheckpointIO(checkpoint_dir=config.checkpoint_dir, model=estimator.model)
    checkpoint_io.load("model.pt")
    estimator.model.to(config.device)
    return estimator


class ContactGraspNetGenerator(GraspGenerator):
    """Learned grasp-candidate source. See module docstring."""

    def __init__(
        self,
        config: ContactGraspNetConfig,
        *,
        estimator_factory: EstimatorFactory | None = None,
        hand_site: str = "gripper0_right_ft_frame",
        tcp_site: str = "gripper0_right_grip_site",
        gripper_max_opening_m: float = 0.08,
    ) -> None:
        self.config = config
        self._estimator_factory = estimator_factory or _load_real_estimator
        self._estimator: Any | None = None
        self.hand_site = hand_site
        self.tcp_site = tcp_site
        self.gripper_max_opening_m = gripper_max_opening_m

    def _get_estimator(self) -> Any:
        # Lazy + cached: constructing this generator (e.g. inside
        # CompositeGraspGenerator) never by itself loads a checkpoint or
        # touches CUDA -- only the first real generate() call does.
        if self._estimator is None:
            self._estimator = self._estimator_factory(self.config)
        return self._estimator

    def generate(self, scene: SceneQuery, object_names: list[str]) -> list[GraspCandidate]:
        if not self.config.enabled:
            return []
        candidates: list[GraspCandidate] = []
        for name in object_names:
            try:
                candidates.extend(self._generate_for_object(scene, name))
            except Exception:
                # Never let one object's inference failure (bad checkpoint,
                # CUDA OOM, malformed model output, empty scene geometry)
                # block picks on other objects or crash the pick attempt --
                # the heuristic generator's candidates still carry it. See
                # module docstring / config.py's ContactGraspNetConfig.
                logger.exception("ContactGraspNetGenerator failed for object %r", name)
        return candidates

    def _generate_for_object(self, scene: SceneQuery, object_name: str) -> list[GraspCandidate]:
        pc_segment = scene.body_surface_points_dense_local(object_name)
        obj_pos, obj_quat = scene.body_pose(object_name)
        pc_segment_world = self._to_world(pc_segment, obj_pos, obj_quat)
        if len(pc_segment_world) < 4:
            return []

        nearby = scene.nearby_body_names(
            obj_pos, self.config.point_cloud_radius_m, exclude=(object_name,)
        )
        context_points = [pc_segment_world]
        for other in nearby:
            other_pos, other_quat = scene.body_pose(other)
            pts = scene.body_surface_points_dense_local(other)
            if len(pts):
                context_points.append(self._to_world(pts, other_pos, other_quat))
        pc_full_world = np.concatenate(context_points, axis=0)

        estimator = self._get_estimator()
        pred_grasps_cam, scores, _contact_pts, _ = estimator.predict_scene_grasps(
            pc_full_world,
            pc_segments={_SEGMAP_ID: pc_segment_world},
            local_regions=self.config.local_regions,
            filter_grasps=self.config.filter_grasps,
            forward_passes=self.config.forward_passes,
        )
        grasps = np.asarray(pred_grasps_cam[_SEGMAP_ID])
        raw_scores = np.asarray(scores[_SEGMAP_ID])
        if len(grasps) == 0:
            return []

        order = np.argsort(-raw_scores)[: self.config.max_candidates]
        out: list[GraspCandidate] = []
        for i in order:
            transform = grasps[i]
            fingertip_pos = transform[:3, 3]
            fingertip_quat = rotmat_to_quat_wxyz(transform[:3, :3])
            hand_pose = hand_from_fingertip_pose(
                scene, fingertip_pos, fingertip_quat, hand_site=self.hand_site, tcp_site=self.tcp_site
            )
            approach_direction_world = quat_to_approach_axis(hand_pose[3:7])
            out.append(
                GraspCandidate(
                    object_name=object_name,
                    grasp_pose_world=hand_pose,
                    approach_direction_world=approach_direction_world,
                    gripper_width=None,
                    generator_score=float(np.clip(raw_scores[i], 0.0, 1.0)),
                    metadata={"generator": "ContactGraspNetGenerator", "network_score": float(raw_scores[i])},
                )
            )
        return out

    @staticmethod
    def _to_world(points_local: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        R = quat_to_rotmat_wxyz(quat_wxyz)
        return points_local @ R.T + pos


def quat_to_approach_axis(quat_wxyz: np.ndarray) -> np.ndarray:
    """The hand frame's local +Z axis (this package's approach-axis
    convention -- see ``grasp_generator.py``'s module docstring) expressed in
    world coordinates, for a WXYZ quaternion."""
    R = quat_to_rotmat_wxyz(quat_wxyz)
    return R[:, 2]
