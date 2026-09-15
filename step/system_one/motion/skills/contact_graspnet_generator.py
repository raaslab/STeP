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

Point cloud construction renders real ``agentview``/``robot0_eye_in_hand``
depth + ground-truth segmentation (``SceneQuery.render_depth_and_body_ids``/
``deproject_camera_depth``) and fuses both views, mirroring Cap-X's own
multi-camera capture for its LIBERO integration (``capx/envs/simulators/
libero.py``'s ``FrankaLiberoEnv.get_observation``/``sample_grasp_pose``) --
concatenate both cameras' full-scene points for ``pc_full``, and both
cameras' object-masked points for ``pc_segment``. Cap-X segments objects with
a vision model (SAM3) on the RGB image, since it has no sim ground truth to
rely on; this generator uses MuJoCo's own ground-truth instance segmentation
instead, consistent with the rest of this "privileged" package (e.g.
``PrivilegedTopDownGraspGenerator`` already reads exact object poses/extents
directly) -- there's no reason to reintroduce vision-model noise/latency here
just to segment a sim object this code already knows the identity of.

Three real bugs were found and fixed getting this far, each confirmed by
real testing on this machine's CUDA + checkpoint, not guessed:

1. An earlier version fed Contact-GraspNet a dense *ground-truth mesh*
   sample (``SceneQuery.body_surface_points_dense_local``/
   ``nearby_body_names``) instead of a real camera capture -- simpler to
   build, but landed predicted grasps 12-18cm off the actual object: a real,
   fully unoccluded 360-degree mesh sample looks nothing like the partial,
   occluded single/dual-camera views Contact-GraspNet was trained on. This
   camera-based capture replaces it.
2. Even with a verified-correct camera-frame point cloud, raw predictions
   still landed ~0.11m off the segmented object -- consistently, across
   different objects, and un-rescued by resampling viewpoints. Root cause:
   ``predict_grasps``'s own docstring says it "Convert[s] from OpenCV to
   internal training camera coordinates" -- it assumes its input point cloud
   is already in **OpenCV** camera convention (+Y down, +Z into the scene),
   confirmed exactly by ``contact_graspnet_pytorch.data.center_pc_convert_cam``
   (``cam_poses[:,:3,1] = -cam_poses[:,:3,1]; cam_poses[:,:3,2] =
   -cam_poses[:,:3,2]``). MuJoCo/OpenGL's own convention (+Y up, +Z *away*
   from view -- ``SceneQuery.camera_pose_world``'s convention, the one every
   ``data.cam_xmat`` is expressed in) is different on exactly those two axes.
   Every point cloud handed to the estimator must go through
   ``_mujoco_to_opencv_cam_mat`` (or be built directly in OpenCV convention,
   for synthetic viewpoints) first -- world-frame points with no camera
   convention at all, or MuJoCo-convention camera-frame points, both produce
   this same systematic, viewpoint-independent offset.
3. Cap-X's real defense against an occasional bad candidate isn't a loose
   distance check -- it's the library's own tight, ~0.1mm ``filter_grasps``
   contact-to-segment-cloud filter (``contact_graspnet_pytorch``'s
   ``filter_segment``) combined with a **retry loop**
   (``launch_contact_graspnet_server.py``'s ``_do_plan_point_clouds``): if
   that strict filter leaves zero survivors, re-express the *same* point
   cloud in a freshly sampled synthetic camera frame (looking at the object
   from a random nearby pose) and try inference again, up to 10 times.
   ``_predict_with_viewpoint_retries`` below ports that loop -- keeping
   ``filter_grasps``'s real threshold rather than loosening it, since
   loosening it is what let bug #2's off-target candidates through
   undetected in the first place.
4. With bugs 1-3 fixed, candidates finally land on-object with sane scores
   -- but real ``PickSkill.execute()`` runs against them still rejected
   nearly everything, overwhelmingly with ``gripper_closure_in_collision``.
   Cause: this generator set ``gripper_width=None`` on every candidate (a
   deliberate v1 choice -- see the git history -- made before there was a
   live model output to validate the semantics against), and
   ``PickSkill.check_gripper_closure`` closes the fingers to fully-shut
   whenever a candidate's ``gripper_width`` is ``None``. Closing all the way
   to zero at a pose sized for a real object's width collides with that
   object almost by construction -- confirmed by reading Cap-X's own
   pipeline (it has the same "always fully close" behavior, but never runs a
   closed-gripper collision check at all, so it never surfaces this). Now
   validated: ``predict_scene_grasps``'s 4th return value, ``gripper_
   openings``, holds small, physically sane per-candidate widths (~1-2cm for
   a bowl's rim, confirmed against real inference output) -- passed through
   as ``GraspCandidate.gripper_width`` below instead of discarded.
5. With bugs 1-4 fixed, candidates were feasible and grasp-verification's
   ``gripper_to_target_error_m`` reported sub-mm tracking -- yet real
   rollout video showed the gripper closing on empty air well above the
   table. Two compounding position bugs, both now fixed:
   (a) an earlier version used ``transform[:3, 3]`` as the contact point,
   but ``contact_graspnet.py``'s own ``build_6d_grasp`` already returns that
   translation pre-offset behind the true touch point by the checkpoint's
   ``gripper_depth`` (0.1034m); fixed by using the network's own
   ``contact_pts`` instead.
   (b) even after (a), this generator ran ``contact_pts`` through
   ``grasp_generator.hand_from_fingertip_pose`` to convert it into the
   ``panda_hand``/flange frame -- but ``GraspCandidate.grasp_pose_world``'s
   own documented contract (``types.py``) is the **TCP**/
   ``gripper0_right_grip_site`` frame (what ``robot0_eef_pos`` reports),
   which is exactly what ``contact_pts`` already is: ``libero_pick_adapter``
   builds its ``IKSolver`` with ``site_name=self.ee_site`` (the TCP site),
   and ``PrivilegedTopDownGraspGenerator``'s own (working) candidates use
   the object's center/rim point directly with no hand offset at all.
   Converting to the hand frame moved the target ~9.7cm backward along the
   approach axis into a different physical point, which the TCP was then
   driven to as if it were still a TCP-frame target -- landing the real
   fingers ~10cm above where the network intended them to close. Fixed by
   dropping the ``hand_from_fingertip_pose`` conversion entirely.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from ..config import ContactGraspNetConfig
from .grasp_generator import GraspGenerator
from ..scene_query import SceneQuery, quat_to_rotmat_wxyz, rotmat_to_quat_wxyz
from ..types import GraspCandidate

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
    (``capx/serving/launch_contact_graspnet_server.py``), then verified by
    running it for real: CPU inference works (no CUDA needed --
    ``contact_graspnet_pytorch`` ships no compiled CUDA extensions) at
    ~1.6s/call.
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
    # Real checkpoints ship as <checkpoint_dir>/config.yaml + <checkpoint_dir>/checkpoints/model.pt
    # (confirmed against contact_graspnet_pytorch's own inference.py) -- CheckpointIO.load()
    # joins its checkpoint_dir with the filename, so this must be the "checkpoints" subdir,
    # not config.checkpoint_dir itself.
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
        # (grasp_generator.py) -- the network's raw contact_pts height is
        # otherwise used unclamped, which is fine for anything tall enough
        # that its natural contact height already clears the table, but for
        # a very short/thin object (e.g. cream_cheese_1, ~2cm tall) the
        # network can correctly identify a low contact point on the object
        # that this gripper's *hardware* can't reach without the palm or
        # fingers -- which extend past the TCP -- hitting the table first.
        # Confirmed via a real rollout: every rejected cream_cheese
        # candidate's grasp height sat within ~2mm of the object's own
        # resting/floor height, and PickSkill rejected them all with
        # grasp_pose_in_collision/grasp_ik_failed, never even reaching a
        # real grasp attempt.
        #
        # Two corrections beyond a direct port of that clamp, both confirmed
        # against cream_cheese_1's real numbers:
        # 1. "Floor height" must come from the object's own AABB *bottom*
        #    (object_center_z - half_extents[2]), not raw obj_pos[2] --
        #    PrivilegedTopDownGraspGenerator's own comment assumes obj_pos[2]
        #    IS the resting/floor height, which holds only when a body's
        #    origin coincides with its base. cream_cheese_1's origin instead
        #    sits at its geometric *center* (center_offset[2] is ~0), so
        #    using obj_pos[2] directly silently assumed a floor ~1cm too
        #    high.
        # 2. Even with (1) fixed, finger_reach_below_tcp_m() alone
        #    (~1.27cm) already exceeds this object's full height (~2cm) --
        #    for any object that thin, an *uncapped* clamp still overshoots
        #    past the object's own top surface entirely (measured: clamp
        #    landed at z=0.0316 vs the object's real top at z=0.0189, i.e.
        #    1.3cm of empty air above a 2cm-tall object -- confirmed via a
        #    real rollout as the reason the gripper closed on nothing despite
        #    perfect tracking to that target). A safety clamp that floats the
        #    grasp point above the object it's meant to protect defeats its
        #    own purpose, so cap it -- worst case, grasp within the object's
        #    own body rather than floating above it entirely. Capping at
        #    *exactly* object_top_z (tried first) pinned every candidate for
        #    this object to one identical height with zero margin below the
        #    boundary the collision checker itself is evaluating -- confirmed
        #    against a real rollout to still reject every candidate as
        #    grasp_pose_in_collision, just at the opposite (too-tight, not
        #    too-high) extreme. Backing off by the same margin used for the
        #    other two thresholds gives the checker real clearance instead of
        #    sitting exactly on its own decision boundary.
        # Collision-only, not the merged visual+collision extent used by an
        # earlier version of this clamp: the two can disagree, and when they
        # do it's the collision geometry that determines what the gripper can
        # actually touch. Confirmed against the real moka_pot_1 asset: its
        # visual mesh's own top (1.087m, from a tall rendered
        # handle/spout) sits 3.5cm above its real collision top (1.052m).
        # Every one of this generator's 32 candidates for that object clamped
        # to the visual-mesh ceiling -- identical z to four decimal places,
        # confirming the clamp (not the network) determined the height --
        # closing on empty air every time (zero object displacement, despite
        # near-zero tracking error to that target).
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
            # The real touch point, *not* transform[:3, 3] -- confirmed by
            # reading contact_graspnet.py's own build_6d_grasp: transform's
            # translation is already `contact_pts - gripper_depth *
            # approach_dirs` (gripper_depth=0.1034m default), i.e. already
            # shifted behind the contact point by ~this checkpoint's own
            # assumed gripper depth.
            #
            # No further hand_from_fingertip_pose offset on top of this --
            # GraspCandidate.grasp_pose_world's own documented contract
            # (types.py) is the TCP/``gripper0_right_grip_site`` frame (what
            # ``robot0_eef_pos`` reports), which is exactly what contact_pts
            # already is and exactly what IKSolver/PickSkill's servo loop
            # target (``libero_pick_adapter.py`` constructs
            # ``IKSolver(site_name=self.ee_site)`` with
            # ``ee_site="gripper0_right_grip_site"``, and
            # PrivilegedTopDownGraspGenerator's own candidates -- the ones
            # that actually work -- likewise use grasp_center/point_center
            # directly with no hand offset). Applying
            # hand_from_fingertip_pose here shifted the position backward
            # along the approach axis into the *hand/flange* frame instead
            # -- a real, different physical point ~9.7cm further from the
            # table -- and then had IK/the servo loop drive the TCP there
            # as if it were still a TCP-frame target, landing the fingers
            # ~10cm above where the network actually intended them to
            # close. Confirmed by a real rollout: settle converged to
            # sub-mm error against this generator's own computed target,
            # yet the gripper visibly closed on empty air well above the
            # table -- i.e. the *target itself*, not the tracking, was
            # wrong.
            fingertip_pos = contact_pts[i].copy()
            if min_safe_z is not None:
                fingertip_pos[2] = max(fingertip_pos[2], min_safe_z)
            fingertip_quat = rotmat_to_quat_wxyz(transform[:3, :3])
            grasp_pose = np.concatenate([fingertip_pos, fingertip_quat])
            approach_direction_world = quat_to_approach_axis(fingertip_quat)
            # The network's own predicted opening, not a guess -- confirmed
            # against real inference to be small, physically sane values
            # (e.g. ~1-2cm for a bowl's rim), not the placeholder v1 skipped
            # this: PickSkill.check_gripper_closure only ever closed the
            # fingers to fully-shut (gripper_width=None's fallback) for every
            # candidate this generator produced, colliding with any object
            # that isn't paper-thin -- confirmed as the dominant real
            # rejection reason (gripper_closure_in_collision) before this fix.
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

        Every attempt goes through a proper OpenCV-convention camera frame
        (never raw world-frame points) -- ``contact_graspnet_pytorch`` isn't
        frame-agnostic despite superficially round-tripping whatever frame
        it's given: ``predict_grasps``'s own docstring says it "Convert[s]
        from OpenCV to internal training camera coordinates", and
        ``data.center_pc_convert_cam`` spells out exactly what that means
        (``cam_poses[:,:3,1] = -cam_poses[:,:3,1]; cam_poses[:,:3,2] =
        -cam_poses[:,:3,2]`` -- OpenCV is OpenGL/MuJoCo's own camera
        convention with local Y and Z negated). A first pass of this
        generator fed it raw world-frame points (no camera convention at
        all) and, for retries, points re-expressed in MuJoCo's own
        convention (local Z *away* from view, Y up) -- neither is OpenCV,
        so the ~0.1mm ``filter_grasps`` check never had a chance: every
        candidate's predicted position was off by a roughly constant,
        viewpoint-independent amount (confirmed by real testing: ~0.11m on
        two different objects, un-rescued by 10 synthetic-viewpoint
        retries in the wrong convention -- resampling viewpoints doesn't
        fix a systematic axis-convention bug). ``_mujoco_to_opencv_cam_mat``
        below applies exactly that column negation.

        The first attempt uses ``config.camera_names``'s first real,
        rendered camera pose (an actual sensor view, not an arbitrary one --
        closer to Cap-X's own first attempt, which uses a real camera-derived
        frame too); every retry after that is a freshly sampled synthetic
        viewpoint looking at the segment's own centroid, mirroring Cap-X's
        ``sample_random_camera_viewpoint``/``camera_so3_looking_at_origin``
        (``capx/integrations/vision/graspnet.py``) and its server-side retry
        loop (``launch_contact_graspnet_server.py``'s
        ``_do_plan_point_clouds``)."""
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
            # Real, observed case: contact_graspnet_pytorch's own filter_grasps
            # step silently drops the openings entry for a segment whose
            # gripper-opening array failed to index (its own "skipped gripper
            # openings" fallback, contact_grasp_estimator.py) -- fall back to
            # this generator's own gripper_max_opening_m rather than crash or
            # silently misalign scores/openings with the wrong grasps.
            openings = np.full(len(grasps), self.gripper_max_opening_m)
        contact_pts_frame = np.asarray(contact_pts.get(_SEGMAP_ID, [])) if contact_pts else np.zeros((0, 3))
        if len(contact_pts_frame) != len(grasps):
            # Same defensive fallback as gripper_openings above, for the same
            # reason -- fall back to the (already wrong-by-gripper_depth, but
            # never absent) grasp translation rather than crash or misalign.
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
