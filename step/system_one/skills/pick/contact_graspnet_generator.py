"""Learned grasp-candidate generator: Contact-GraspNet, called in-process.

Runs directly against ``contact_graspnet_pytorch.GraspEstimator`` -- the same
class Cap-X's own perception server (``capx/serving/
launch_contact_graspnet_server.py``, MIT License, https://github.com/capgym/
cap-x) wraps in a FastAPI endpoint -- rather than talking to a separate
server process. This module is meant to be imported by a
``run_libero_step.py`` process running on a machine with
``contact_graspnet_pytorch`` and a real checkpoint installed; this dev
environment (no CUDA) never imports ``contact_graspnet_pytorch`` at all --
see ``_load_real_estimator``.

Feeds ``GraspGenerator``'s existing contract (see ``grasp_generator.py``):
returned candidates are handed to ``PickSkill`` alongside
``PrivilegedTopDownGraspGenerator``'s via ``CompositeGraspGenerator``, which
does its own IK/collision/OMPL feasibility filtering -- this generator's job
is only to propose plausible poses, not to guess which one will work.

Point cloud construction renders real ``agentview``/``robot0_eye_in_hand``
depth + ground-truth segmentation (``SceneQuery.render_depth_and_body_ids``/
``deproject_camera_depth``) and fuses both views, mirroring Cap-X's own
multi-camera capture for its LIBERO integration. Cap-X segments objects with
a vision model since it has no sim ground truth to rely on; this generator
uses MuJoCo's own ground-truth instance segmentation instead, consistent with
the rest of this "privileged" package.

A few non-obvious correctness requirements worth flagging for anyone
touching this file:

- The point cloud must be a real camera capture, not a dense ground-truth
  mesh sample -- a fully unoccluded 360-degree sample looks nothing like the
  partial, occluded views Contact-GraspNet was trained on and lands grasps
  well off the actual object.
- Every point cloud handed to the estimator must be expressed in **OpenCV**
  camera convention (+Y down, +Z into the scene), via
  ``_mujoco_to_opencv_cam_mat`` -- ``predict_grasps`` assumes this convention
  internally, and MuJoCo/OpenGL's own convention (+Y up, +Z away from view)
  differs on exactly those two axes. Feeding it world-frame or MuJoCo-
  convention points produces a systematic, viewpoint-independent offset.
- When ``filter_grasps``'s tight contact-to-segment-cloud filter leaves zero
  survivors, retry by re-expressing the same point cloud in a freshly
  sampled synthetic camera frame rather than loosening the filter (mirrors
  Cap-X's own server-side retry loop). ``_predict_with_viewpoint_retries``
  below implements this.
- ``GraspCandidate.gripper_width`` must come from ``predict_scene_grasps``'s
  own ``gripper_openings`` output, not ``None`` -- ``PickSkill`` closes the
  fingers fully shut whenever width is unknown, which collides with the
  object almost by construction.
- The grasp position is the network's own ``contact_pts``, not
  ``transform[:3, 3]`` (which ``build_6d_grasp`` returns pre-offset behind
  the true touch point by the checkpoint's gripper depth), and must be used
  directly with **no** ``hand_from_fingertip_pose`` conversion --
  ``GraspCandidate.grasp_pose_world``'s contract (``types.py``) is the TCP
  frame, which ``contact_pts`` already is; converting to the hand/flange
  frame shifts the target ~10cm along the approach axis into the wrong
  physical point.
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
    """Tuning for ``ContactGraspNetGenerator`` -- a learned grasp-candidate
    source, called in-process. When enabled, ``LiberoPickAdapter`` uses it
    *instead of* ``PrivilegedTopDownGraspGenerator``, not composited with it
    (see ``libero_pick_adapter.py``'s comment: PickSkill's OR-set cost
    function tends to prefer the heuristic's cheap-but-possibly-wrong
    candidates over the network's). Disabled by default so nothing changes
    for a machine without ``contact_graspnet_pytorch`` or a real checkpoint
    installed (that package is only ever imported lazily, inside
    ``_load_real_estimator``, never at module import time).

    CUDA is *not* required: ``contact_graspnet_pytorch`` ships no compiled CUDA
    extensions (its PointNet++ ops are pure PyTorch), so CPU inference works."""

    enabled: bool = False
    # Path to the Contact-GraspNet checkpoint directory (expects model.pt
    # plus its config.yaml alongside it, matching Cap-X's own server loader).
    # Machine-specific, so it must never be hardcoded to one person's checkout:
    # the CONTACT_GRASPNET_CHECKPOINT_DIR environment variable overrides
    # whatever is configured here (see _load_real_estimator).
    checkpoint_dir: str = ""
    # "auto" resolves to cuda when available, else cpu. An explicit "cuda"/"cpu"/
    # "mps" is honored as-is.
    device: str = "auto"
    forward_passes: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    # Returned candidates are capped at this many (by score) before being
    # handed to PickSkill, so a large network output doesn't blow up the
    # downstream IK/collision feasibility search.
    max_candidates: int = 32
    # Cameras rendered and fused for the point cloud (SceneQuery.
    # render_depth_and_body_ids/deproject_camera_depth) -- standard LIBERO/
    # robosuite camera names; mirrors Cap-X's own agentview + wrist-camera
    # fusion for its LIBERO integration. A camera not present in a given
    # model is skipped, not an error.
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    render_height: int = 480
    render_width: int = 480
    # Depth far-clip (m) -- drops background/sky rays (MuJoCo reports a very
    # large depth, not a sentinel, where a ray hits nothing) before they
    # pollute pc_full.
    depth_max_m: float = 3.0
    # Mirrors Cap-X's own synthetic-viewpoint retry: if filter_grasps's tight
    # contact-to-segment-cloud filter leaves zero survivors, re-express the
    # same captured points in a freshly sampled synthetic camera frame and
    # retry, up to this many times, before giving up on this object. No new
    # rendering needed -- Contact-GraspNet's predictions are sensitive to the
    # frame its input is expressed in, so re-expression alone changes the result.
    max_viewpoint_retries: int = 10
    viewpoint_xy_extent_m: float = 0.25
    viewpoint_z_min_m: float = 0.4
    viewpoint_z_max_m: float = 0.9
    # None: a fresh (non-reproducible) RNG per generate() call, matching real
    # deployment. Tests pass a fixed seed for determinism.
    viewpoint_seed: int | None = None
    # Same value and purpose as GraspSampleConfig.grasp_height_margin_m --
    # see ContactGraspNetGenerator._generate_for_object's floor/palm-
    # clearance clamp for why this generator needs it too.
    grasp_height_margin_m: float = 0.01

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
    (``capx/serving/launch_contact_graspnet_server.py``); CPU inference works
    fine since ``contact_graspnet_pytorch`` ships no compiled CUDA extensions.
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
    # Checkpoints ship as <checkpoint_dir>/config.yaml + <checkpoint_dir>/checkpoints/model.pt --
    # CheckpointIO.load() joins its checkpoint_dir with the filename, so this must be the
    # "checkpoints" subdir, not config.checkpoint_dir itself.
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(checkpoint_dir, "checkpoints"), model=estimator.model)

    import torch

    device = config.device
    if device in ("", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # CheckpointIO.load() calls torch.load(filename) with no extra arguments, so
    # the only way to pass these is to wrap torch.load for the duration of the
    # call. Both are required on any torch >= 2.6, on CPU *and* GPU machines:
    #   * weights_only: torch 2.6 flipped this default to True, which refuses to
    #     unpickle this checkpoint's numpy scalars.
    #   * map_location: the published checkpoint was saved from a CUDA device, so
    #     loading it anywhere without a GPU raises outright.
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
    """Learned grasp-candidate source. See module docstring."""

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
            except Exception as exc:
                # Never let one object's inference failure (bad checkpoint,
                # CUDA OOM, malformed model output, empty scene geometry)
                # block picks on other objects or crash the pick attempt --
                # the heuristic generator's candidates still carry it. See
                # module docstring / config.py's ContactGraspNetConfig.
                logger.exception("ContactGraspNetGenerator failed for object %r", name)
                # Also print, once per process: a run that silently falls back
                # to the heuristic generator produces perfectly plausible
                # numbers for a grasp method other than the one being reported.
                # An unconfigured checkpoint path is the common cause and is
                # otherwise invisible in the logs.
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
        fusing every camera in ``config.camera_names`` by concatenation (see
        module docstring). A camera missing from this particular MuJoCo model
        is skipped rather than failing the whole capture -- ``agentview``/
        ``robot0_eye_in_hand`` are standard on every LIBERO/robosuite env,
        but this generator isn't LIBERO-specific, and a differently-built
        scene may only have one of them."""
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

        # Same floor/palm-clearance clamp as PrivilegedTopDownGraspGenerator
        # (grasp_generator.py) -- without it, the network can correctly find a
        # low contact point on a short/thin object that the gripper's hardware
        # (palm or fingers extending past the TCP) can't actually reach
        # without hitting the table first.
        #
        # Two corrections beyond a direct port of that clamp:
        # 1. The floor reference must come from the object's own AABB bottom,
        #    not raw obj_pos[2] -- that assumes a body's origin coincides with
        #    its base, which isn't true for every asset.
        # 2. The clamp must also be capped from above (backed off from
        #    object_top_z by the same margin): for a sufficiently thin object,
        #    an uncapped floor-driven clamp can push the target above the
        #    object's own top surface entirely, and capping at exactly
        #    object_top_z with zero margin just moves the rejection to the
        #    opposite (too-tight) extreme.
        #
        # Uses collision-only extent, not the merged visual+collision extent:
        # the two can disagree (e.g. a tall decorative handle/spout in the
        # visual mesh sitting well above the real collision geometry), and
        # it's the collision geometry that determines what the gripper
        # actually touches.
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
            # The real touch point is contact_pts, not transform[:3, 3] --
            # build_6d_grasp's translation is already shifted behind the
            # contact point by the checkpoint's assumed gripper depth.
            #
            # No hand_from_fingertip_pose conversion on top of this:
            # GraspCandidate.grasp_pose_world's contract (types.py) is the
            # TCP frame, which contact_pts already is. Converting to the
            # hand/flange frame instead moves the target along the approach
            # axis to a different physical point, which IK then drives the
            # TCP to as if it were still TCP-frame -- landing the fingers
            # well off where the network intended them to close, even though
            # tracking itself converges with near-zero error.
            fingertip_pos = contact_pts[i].copy()
            if min_safe_z is not None:
                fingertip_pos[2] = max(fingertip_pos[2], min_safe_z)
            fingertip_quat = rotmat_to_quat_wxyz(transform[:3, :3])
            grasp_pose = np.concatenate([fingertip_pos, fingertip_quat])
            approach_direction_world = quat_to_approach_axis(fingertip_quat)
            # The network's own predicted opening, not a guess -- leaving this
            # None makes PickSkill close the fingers fully shut, which
            # collides with any object that isn't paper-thin.
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
        """(grasps_world (N,4,4), scores (N,), gripper_openings_m (N,),
        contact_pts_world (N,3)) -- see module docstring.

        Every attempt goes through a proper OpenCV-convention camera frame,
        never raw world-frame or MuJoCo-convention points --
        ``contact_graspnet_pytorch`` assumes OpenCV convention internally
        (``predict_grasps``/``data.center_pc_convert_cam``), and feeding it
        anything else produces a systematic, viewpoint-independent position
        offset that no amount of retrying fixes. ``_mujoco_to_opencv_cam_mat``
        below performs the needed axis negation.

        The first attempt uses ``config.camera_names``'s first real, rendered
        camera pose; every retry after that is a freshly sampled synthetic
        viewpoint looking at the segment's own centroid, mirroring Cap-X's
        own server-side retry loop."""
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
        """One predict_scene_grasps call in the given OpenCV-convention
        (cam_pos_world, cam_mat) camera's frame, with any resulting grasps
        *and contact points* transformed straight back to world frame --
        every caller only ever sees world-frame poses/points."""
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
            # contact_graspnet_pytorch's own filter_grasps step can silently drop
            # the openings entry for a segment -- fall back to this generator's
            # own max opening rather than crash or misalign scores/openings.
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
    """MuJoCo/OpenGL camera convention (local +X right, +Y up, +Z *away*
    from the view direction -- ``SceneQuery.camera_pose_world``'s
    convention) -> OpenCV convention (+X right, +Y down, +Z *into* the
    scene) that ``contact_graspnet_pytorch`` expects, by negating the local
    Y and Z basis vectors (columns 1, 2) -- exactly what
    ``contact_graspnet_pytorch.data.center_pc_convert_cam`` does to its own
    OpenGL-format camera poses."""
    return cam_mat_mujoco @ np.diag([1.0, -1.0, -1.0])


def _sample_synthetic_opencv_camera_frame(
    center_world: np.ndarray, rng: np.random.Generator, xy_extent_m: float, z_min_m: float, z_max_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """A random (cam_pos_world, cam_mat) in OpenCV convention -- a camera at
    a random pose near ``center_world``, oriented to look at it. Mirrors
    Cap-X's ``sample_random_camera_viewpoint``/``camera_so3_looking_at_origin``
    (``capx/integrations/vision/graspnet.py``), generalized from "looking at
    the world origin" to "looking at an arbitrary target", and built directly
    in OpenCV convention (+Z forward into the scene, +Y down) rather than via
    ``_mujoco_to_opencv_cam_mat`` -- there's no real MuJoCo camera here to
    convert from."""
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
    """A 4x4 SE3 transform expressed in a synthetic camera's frame -> world
    frame (inverts ``_world_points_to_frame``: world = frame @ cam_mat.T +
    cam_pos_world, applied to both the rotation and translation parts)."""
    out = np.eye(4)
    out[:3, :3] = cam_mat @ transform_frame[:3, :3]
    out[:3, 3] = cam_mat @ transform_frame[:3, 3] + cam_pos_world
    return out


def quat_to_approach_axis(quat_wxyz: np.ndarray) -> np.ndarray:
    """The hand frame's local +Z axis (this package's approach-axis
    convention -- see ``grasp_generator.py``'s module docstring) expressed in
    world coordinates, for a WXYZ quaternion."""
    R = quat_to_rotmat_wxyz(quat_wxyz)
    return R[:, 2]
