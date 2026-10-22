"""Learned grasp-candidate generator: Contact-GraspNet, called in-process
against contact_graspnet_pytorch.GraspEstimator rather than through a
separate server. Requires contact_graspnet_pytorch and a real checkpoint
installed; never imported unless explicitly enabled (see _load_real_estimator).

Returned candidates are handed to PickSkill via CompositeGraspGenerator
alongside the heuristic generator's -- this generator only proposes plausible
poses, feasibility filtering happens downstream.

Point clouds come from real agentview/robot0_eye_in_hand depth +
ground-truth segmentation, fused across both views -- a dense ground-truth
mesh sample looks nothing like the partial, occluded views the network was trained on.

A few correctness requirements:
- Every point cloud must be in OpenCV camera convention (+Y down, +Z into the
  scene), via _mujoco_to_opencv_cam_mat -- feeding world-frame or
  MuJoCo-convention points produces a systematic position offset.
- When filter_grasps leaves zero survivors, retry in a freshly sampled
  synthetic camera frame rather than loosening the filter (see _predict_with_viewpoint_retries).
- GraspCandidate.gripper_width must come from predict_scene_grasps's
  own gripper_openings, never None -- PickSkill closes fully shut otherwise.
- The grasp position is contact_pts, not transform[:3, 3] (offset
  behind the touch point by the checkpoint's gripper depth), used directly
  with no hand_from_fingertip_pose conversion -- GraspCandidate.grasp_pose_world's contract is already the TCP frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .grasp_generator import GraspGenerator
from ...motion.scene_query import SceneQuery, quat_to_rotmat_wxyz, rotmat_to_quat_wxyz
from ...motion.types import GraspCandidate

logger = logging.getLogger(__name__)


@dataclass
class ContactGraspNetConfig:
    """Tuning for ContactGraspNetGenerator. When enabled, LiberoPickAdapter
    uses it instead of the heuristic generator, not composited (PickSkill's
    cost function tends to prefer the heuristic's cheap-but-wrong candidates).
    Disabled by default; CUDA not required since its PointNet++ ops are pure PyTorch."""

    enabled: bool = False
    # Checkpoint directory (model.pt + config.yaml). Machine-specific --
    # CONTACT_GRASPNET_CHECKPOINT_DIR overrides whatever is set here.
    checkpoint_dir: str = ""
    # "auto" resolves to cuda when available, else cpu. An explicit "cuda"/"cpu"/
    # "mps" is honored as-is.
    device: str = "auto"
    forward_passes: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    # Capped at this many (by score) before PickSkill's IK/collision search,
    # so a large network output doesn't blow it up.
    max_candidates: int = 32
    # Cameras rendered and fused for the point cloud; a camera missing from a
    # given model is skipped, not an error.
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    render_height: int = 480
    render_width: int = 480
    # Depth far-clip (m) -- drops background rays (MuJoCo reports a large
    # depth, not a sentinel, where nothing is hit) before they pollute pc_full.
    depth_max_m: float = 3.0
    # If filter_grasps leaves zero survivors, retry in a freshly sampled
    # synthetic camera frame (no new rendering needed) up to this many times.
    max_viewpoint_retries: int = 10
    viewpoint_xy_extent_m: float = 0.25
    viewpoint_z_min_m: float = 0.4
    viewpoint_z_max_m: float = 0.9
    # None: a fresh (non-reproducible) RNG per generate() call, matching real
    # deployment. Tests pass a fixed seed for determinism.
    viewpoint_seed: int | None = None
    # Same purpose as GraspSampleConfig.grasp_height_margin_m: floor/palm clearance margin.
    grasp_height_margin_m: float = 0.01

# Single-object segmap id used for every per-object predict_scene_grasps call,
# since this generator is called one object at a time.
_SEGMAP_ID = 1

EstimatorFactory = Callable[[ContactGraspNetConfig], Any]


def _load_real_estimator(config: ContactGraspNetConfig) -> Any:
    """Construct and load the real Contact-GraspNet PyTorch estimator.

    Imported lazily so merely importing this module, or constructing a
    disabled generator, never requires contact_graspnet_pytorch or CUDA.
    CPU inference works fine since it ships no compiled CUDA extensions.
    """
    try:
        import yaml
        from contact_graspnet_pytorch.checkpoints import CheckpointIO
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    except ImportError as exc:
        raise ImportError(
            "ContactGraspNetGenerator requires the 'contact_graspnet_pytorch' package "
            "(https://github.com/elchun/contact_graspnet_pytorch, also vendored by Cap-X "
            "https://github.com/capgym/cap-x). CUDA is NOT required -- it runs on CPU. "
            "Install with: pip install -e <checkout> --no-deps --no-build-isolation "
            "(its setup.py imports numpy at build time, so build isolation breaks), plus "
            "'trimesh pyrender tqdm', which its Linux/CUDA-pinned conda env file does not "
            "provide in a usable form."
        ) from exc

    import os

    checkpoint_dir = os.environ.get("CONTACT_GRASPNET_CHECKPOINT_DIR") or config.checkpoint_dir
    if not checkpoint_dir:
        raise FileNotFoundError(
            "No Contact-GraspNet checkpoint directory configured -- set the "
            "CONTACT_GRASPNET_CHECKPOINT_DIR environment variable, or "
            "ContactGraspNetConfig.checkpoint_dir, to a directory containing this "
            "model's config.yaml and a checkpoints/model.pt."
        )
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Contact-GraspNet checkpoint directory {checkpoint_dir!r} does not exist on "
            "this machine. It is machine-specific: set CONTACT_GRASPNET_CHECKPOINT_DIR to "
            "your own checkout's checkpoints/contact_graspnet directory."
        )

    with open(os.path.join(checkpoint_dir, "config.yaml")) as f:
        global_config = yaml.safe_load(f)

    estimator = GraspEstimator(global_config)
    # CheckpointIO.load() joins checkpoint_dir with the filename, so this must
    # be the "checkpoints" subdir, not config.checkpoint_dir itself.
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(checkpoint_dir, "checkpoints"), model=estimator.model)

    import torch

    device = config.device
    if device in ("", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # torch.load is wrapped since CheckpointIO.load() passes no kwargs: torch>=2.6
    # needs weights_only=False (numpy scalars) and map_location (checkpoint saved on CUDA).
    original_torch_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        kwargs.setdefault("map_location", device)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        checkpoint_io.load("model.pt")
    finally:
        torch.load = original_torch_load

    estimator.model.to(device)
    return estimator


class ContactGraspNetGenerator(GraspGenerator):
    """Learned grasp-candidate source."""

    # Process-wide, so the fallback warning prints once per run rather than
    # once per object per pick.
    _fallback_warned: bool = False

    def __init__(
        self,
        config: ContactGraspNetConfig,
        *,
        estimator_factory: EstimatorFactory | None = None,
        gripper_max_opening_m: float = 0.08,
    ) -> None:
        self.config = config
        self._estimator_factory = estimator_factory or _load_real_estimator
        self._estimator: Any | None = None
        self.gripper_max_opening_m = gripper_max_opening_m

    def _get_estimator(self) -> Any:
        # Lazy + cached: constructing this generator never loads a checkpoint
        # or touches CUDA -- only the first real generate() call does.
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
            except Exception as exc:
                # Never let one object's inference failure block picks on other
                # objects or crash the attempt -- the heuristic generator's candidates still carry it.
                logger.exception("ContactGraspNetGenerator failed for object %r", name)
                # Also print once per process: a silent fallback produces plausible
                # numbers for the wrong grasp method, and an unconfigured checkpoint path is otherwise invisible in the logs.
                if not ContactGraspNetGenerator._fallback_warned:
                    ContactGraspNetGenerator._fallback_warned = True
                    print(
                        f"[contact_graspnet] WARNING: inference failed ({type(exc).__name__}: {exc}). "
                        "Falling back to the HEURISTIC grasp generator -- results are NOT "
                        "Contact-GraspNet's. Set CONTACT_GRASPNET_CHECKPOINT_DIR to fix."
                    )
        return candidates

    def _capture_point_clouds(self, scene: SceneQuery, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Real multi-camera depth capture -> (pc_full_world, pc_segment_world),
        fusing every camera in config.camera_names. A camera missing from this
        particular model is skipped rather than failing the whole capture."""
        target_bid = scene.body_id(object_name)
        full_chunks: list[np.ndarray] = []
        segment_chunks: list[np.ndarray] = []
        for camera_name in self.config.camera_names:
            try:
                depth, body_id = scene.render_depth_and_body_ids(
                    camera_name, self.config.render_height, self.config.render_width
                )
            except KeyError:
                continue
            valid = (depth > 0) & (depth < self.config.depth_max_m)
            if not np.any(valid):
                continue
            points = scene.deproject_camera_depth(camera_name, depth)
            full_chunks.append(points[valid])
            segment_mask = valid & (body_id == target_bid)
            if np.any(segment_mask):
                segment_chunks.append(points[segment_mask])

        pc_full_world = np.concatenate(full_chunks, axis=0) if full_chunks else np.zeros((0, 3))
        pc_segment_world = np.concatenate(segment_chunks, axis=0) if segment_chunks else np.zeros((0, 3))
        return pc_full_world, pc_segment_world

    def _generate_for_object(self, scene: SceneQuery, object_name: str) -> list[GraspCandidate]:
        pc_full_world, pc_segment_world = self._capture_point_clouds(scene, object_name)
        if len(pc_segment_world) < 4:
            return []

        estimator = self._get_estimator()
        grasps, raw_scores, gripper_openings, contact_pts = self._predict_with_viewpoint_retries(
            scene, estimator, pc_full_world, pc_segment_world
        )
        if len(grasps) == 0:
            return []

        # Same floor/palm clamp as PrivilegedTopDownGraspGenerator, so the
        # network can't pick a contact point the gripper would hit the table to reach.
        min_safe_z = None
        extent = scene.body_collision_extent_world(object_name)
        if extent is not None:
            obj_pos, _ = scene.body_pose(object_name)
            margin = self.config.grasp_height_margin_m
            object_center_z = obj_pos[2] + extent.center_offset[2]
            object_top_z = object_center_z + extent.half_extents[2]
            object_bottom_z = object_center_z - extent.half_extents[2]
            min_safe_for_palm = object_top_z - scene.palm_clearance_above_tcp_m() + margin
            min_safe_for_floor = object_bottom_z + scene.finger_reach_below_tcp_m() + margin
            min_safe_z = min(max(min_safe_for_palm, min_safe_for_floor), object_top_z - margin)

        order = np.argsort(-raw_scores)[: self.config.max_candidates]
        out: list[GraspCandidate] = []
        for i in order:
            transform = grasps[i]
            # Real touch point is contact_pts, not transform[:3, 3] (shifted
            # behind it by gripper depth) -- used directly, with no hand_from_fingertip_pose conversion (already TCP frame).
            fingertip_pos = contact_pts[i].copy()
            if min_safe_z is not None:
                fingertip_pos[2] = max(fingertip_pos[2], min_safe_z)
            fingertip_quat = rotmat_to_quat_wxyz(transform[:3, :3])
            grasp_pose = np.concatenate([fingertip_pos, fingertip_quat])
            approach_direction_world = quat_to_approach_axis(fingertip_quat)
            # The network's own predicted opening -- leaving this None makes
            # PickSkill close fully shut, colliding with any non-paper-thin object.
            width = float(np.clip(gripper_openings[i], 0.0, self.gripper_max_opening_m))
            out.append(
                GraspCandidate(
                    object_name=object_name,
                    grasp_pose_world=grasp_pose,
                    approach_direction_world=approach_direction_world,
                    gripper_width=width,
                    generator_score=float(np.clip(raw_scores[i], 0.0, 1.0)),
                    metadata={"generator": "ContactGraspNetGenerator", "network_score": float(raw_scores[i])},
                )
            )
        return out

    def _predict_with_viewpoint_retries(
        self, scene: SceneQuery, estimator: Any, pc_full_world: np.ndarray, pc_segment_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (grasps_world (N,4,4), scores (N,), gripper_openings_m (N,),
        contact_pts_world (N,3)). Every attempt uses a proper OpenCV-convention
        camera frame -- anything else produces a systematic position offset. The
        first attempt uses a real rendered camera pose; retries sample a synthetic viewpoint looking at the segment's centroid."""
        center_world = pc_segment_world.mean(axis=0)
        rng = np.random.default_rng(self.config.viewpoint_seed)

        cam_poses: list[tuple[np.ndarray, np.ndarray]] = []
        for camera_name in self.config.camera_names:
            try:
                cam_pos_world, cam_mat_mujoco = scene.camera_pose_world(camera_name)
            except KeyError:
                continue
            cam_poses.append((cam_pos_world, _mujoco_to_opencv_cam_mat(cam_mat_mujoco)))
            break  # one real camera pose as the deterministic first attempt
        for _ in range(self.config.max_viewpoint_retries):
            cam_poses.append(
                _sample_synthetic_opencv_camera_frame(
                    center_world,
                    rng,
                    self.config.viewpoint_xy_extent_m,
                    self.config.viewpoint_z_min_m,
                    self.config.viewpoint_z_max_m,
                )
            )

        for cam_pose in cam_poses:
            grasps, scores, openings, contact_pts = self._predict_scene_grasps_in_frame(
                estimator, pc_full_world, pc_segment_world, cam_pose
            )
            if len(grasps):
                return grasps, scores, openings, contact_pts
        return np.zeros((0, 4, 4)), np.zeros((0,)), np.zeros((0,)), np.zeros((0, 3))

    def _predict_scene_grasps_in_frame(
        self,
        estimator: Any,
        pc_full_world: np.ndarray,
        pc_segment_world: np.ndarray,
        cam_pose: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One predict_scene_grasps call in the given camera frame; resulting
        grasps and contact points are transformed back to world frame -- callers only ever see world-frame data."""
        cam_pos_world, cam_mat = cam_pose
        pc_full = _world_points_to_frame(pc_full_world, cam_pos_world, cam_mat)
        pc_segment = _world_points_to_frame(pc_segment_world, cam_pos_world, cam_mat)

        pred_grasps_cam, scores, contact_pts, gripper_openings = estimator.predict_scene_grasps(
            pc_full,
            pc_segments={_SEGMAP_ID: pc_segment},
            local_regions=self.config.local_regions,
            filter_grasps=self.config.filter_grasps,
            forward_passes=self.config.forward_passes,
        )
        grasps = np.asarray(pred_grasps_cam.get(_SEGMAP_ID, []))
        raw_scores = np.asarray(scores.get(_SEGMAP_ID, []))
        if len(grasps) == 0:
            return np.zeros((0, 4, 4)), np.zeros((0,)), np.zeros((0,)), np.zeros((0, 3))
        openings = np.asarray(gripper_openings.get(_SEGMAP_ID, [])) if gripper_openings else np.zeros((0,))
        if len(openings) != len(grasps):
            # filter_grasps can silently drop the openings entry for a segment --
            # fall back to this generator's own max opening rather than crash or misalign.
            openings = np.full(len(grasps), self.gripper_max_opening_m)
        contact_pts_frame = np.asarray(contact_pts.get(_SEGMAP_ID, [])) if contact_pts else np.zeros((0, 3))
        if len(contact_pts_frame) != len(grasps):
            # Same defensive fallback as openings above: use the grasp
            # translation (wrong by gripper_depth, but never absent).
            contact_pts_frame = grasps[:, :3, 3]
        grasps = np.stack([_frame_transform_to_world(g, cam_pos_world, cam_mat) for g in grasps])
        contact_pts_world = _frame_points_to_world(contact_pts_frame, cam_pos_world, cam_mat)
        return grasps, raw_scores, openings, contact_pts_world


def _mujoco_to_opencv_cam_mat(cam_mat_mujoco: np.ndarray) -> np.ndarray:
    """MuJoCo/OpenGL camera convention (+Y up, +Z away from view) -> OpenCV
    convention (+Y down, +Z into scene) that contact_graspnet_pytorch expects, by negating the local Y and Z basis vectors."""
    return cam_mat_mujoco @ np.diag([1.0, -1.0, -1.0])


def _sample_synthetic_opencv_camera_frame(
    center_world: np.ndarray, rng: np.random.Generator, xy_extent_m: float, z_min_m: float, z_max_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """A random (cam_pos_world, cam_mat) in OpenCV convention, at a random pose
    near center_world oriented to look at it -- built directly in OpenCV convention since there's no real MuJoCo camera to convert from."""
    offset = np.array(
        [
            rng.uniform(-xy_extent_m, xy_extent_m),
            rng.uniform(-xy_extent_m, xy_extent_m),
            rng.uniform(z_min_m, z_max_m),
        ]
    )
    cam_pos_world = center_world + offset
    z_axis = -offset / np.linalg.norm(offset)  # forward: camera -> target
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(z_axis, world_up)
    if np.linalg.norm(x_axis) < 1e-6:  # view direction (near-)parallel to world up
        world_up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, world_up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)  # down
    cam_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    return cam_pos_world, cam_mat


def _world_points_to_frame(points_world: np.ndarray, cam_pos_world: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    return (points_world - cam_pos_world) @ cam_mat


def _frame_points_to_world(points_frame: np.ndarray, cam_pos_world: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    return points_frame @ cam_mat.T + cam_pos_world


def _frame_transform_to_world(transform_frame: np.ndarray, cam_pos_world: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    """A 4x4 SE3 transform in a synthetic camera's frame -> world frame,
    applying world = frame @ cam_mat.T + cam_pos_world to both rotation and translation."""
    out = np.eye(4)
    out[:3, :3] = cam_mat @ transform_frame[:3, :3]
    out[:3, 3] = cam_mat @ transform_frame[:3, 3] + cam_pos_world
    return out


def quat_to_approach_axis(quat_wxyz: np.ndarray) -> np.ndarray:
    """The hand frame's local +Z axis (this package's approach-axis convention), expressed in world coordinates."""
    R = quat_to_rotmat_wxyz(quat_wxyz)
    return R[:, 2]
