import numpy as np
import pytest

from conftest import requires_libero
from step.system_one.skills.pick.contact_graspnet_generator import (
    ContactGraspNetConfig,
    ContactGraspNetGenerator,
    _load_real_estimator,
    _mujoco_to_opencv_cam_mat,
    quat_to_approach_axis,
)
from step.system_one.motion.scene_query import rotmat_to_quat_wxyz


class _FakeEstimator:
    """Stands in for contact_graspnet_pytorch.GraspEstimator, shaped exactly
    like the real predict_scene_grasps() call transcribed from Cap-X's
    server (see contact_graspnet_generator.py's module docstring) -- lets
    every non-CUDA part of this generator (point-cloud construction, pose
    conversion, scoring, truncation, error handling) be tested on a machine
    with no GPU and no contact_graspnet_pytorch installed."""

    def __init__(self, grasps, scores, contact_pts=None, gripper_openings=None):
        self.grasps = np.asarray(grasps)
        self.scores = np.asarray(scores)
        self.contact_pts = np.zeros((len(self.grasps), 3)) if contact_pts is None else contact_pts
        # None: real predict_scene_grasps signature -- some callers get back
        # a bare None instead of a per-segment dict (see contact_graspnet_
        # generator.py's own fallback for this exact case).
        self.gripper_openings = gripper_openings
        self.calls = []

    def predict_scene_grasps(self, pc_full, pc_segments, local_regions, filter_grasps, forward_passes):
        self.calls.append((pc_full, pc_segments, local_regions, filter_grasps, forward_passes))
        openings = None if self.gripper_openings is None else {1: np.asarray(self.gripper_openings)}
        return {1: self.grasps}, {1: self.scores}, {1: self.contact_pts}, openings


class _RaisingEstimator:
    def predict_scene_grasps(self, *args, **kwargs):
        raise RuntimeError("simulated CUDA OOM")


def _identity_transform(pos: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pos
    return T


def test_disabled_returns_empty_and_never_loads_estimator():
    load_calls = []

    def factory(cfg):
        load_calls.append(cfg)
        return _FakeEstimator(np.zeros((0, 4, 4)), np.zeros((0,)))

    gen = ContactGraspNetGenerator(ContactGraspNetConfig(enabled=False), estimator_factory=factory)
    out = gen.generate(scene=None, object_names=["anything"])
    assert out == []
    assert load_calls == []  # enabled=False must never even lazy-load the estimator


@requires_libero
def test_happy_path_converts_pose_and_clips_score(libero_scene):
    # The fake grasp is the identity transform *in the camera frame the
    # generator's first attempt actually uses* (agentview, converted to
    # OpenCV convention -- see _mujoco_to_opencv_cam_mat), i.e. sitting
    # exactly at that camera's own world position/orientation once converted
    # back -- deterministic and exact, unlike "at the object's position",
    # which would only be approximately recovered (real depth-camera capture
    # noise) now that every attempt round-trips through a camera frame.
    cam_pos_world, cam_mat_mujoco = libero_scene.camera_pose_world("agentview")
    cam_mat_cv = _mujoco_to_opencv_cam_mat(cam_mat_mujoco)

    fake = _FakeEstimator(
        grasps=[_identity_transform(np.zeros(3))], scores=[5.0], gripper_openings=[0.017]
    )  # 5.0: out-of-range on purpose
    gen = ContactGraspNetGenerator(
        ContactGraspNetConfig(enabled=True, max_candidates=10),
        estimator_factory=lambda cfg: fake,
    )
    out = gen.generate(libero_scene, ["milk_1_main"])

    assert len(out) == 1
    cand = out[0]
    assert cand.object_name == "milk_1_main"
    assert cand.gripper_width == pytest.approx(0.017)  # the network's own predicted opening, passed through
    assert cand.generator_score == 1.0  # clipped from 5.0

    # No hand-frame offset applied: grasp_pose_world's own contract is the
    # TCP/grip_site frame (see types.py), which is exactly what contact_pts
    # (here, the fake's zeroed-out contact point in camera frame, i.e. the
    # camera's own origin once transformed back to world) already is.
    expected_pose = np.concatenate([cam_pos_world, rotmat_to_quat_wxyz(cam_mat_cv)])
    assert np.allclose(cand.grasp_pose_world, expected_pose, atol=1e-9)
    assert len(fake.calls) == 1


@requires_libero
def test_missing_gripper_openings_falls_back_to_max_opening(libero_scene):
    # Real, observed case: the estimator can return a bare None for
    # gripper_openings instead of a per-segment dict (contact_graspnet_
    # pytorch's own "skipped gripper openings" fallback). Silently closing
    # to fully-shut for every such candidate was the actual bug this
    # generator used to have -- PickSkill.check_gripper_closure only ever
    # closes to gripper_width=None's fallback (fully shut), colliding with
    # any object that isn't paper-thin. Falling back to this generator's own
    # gripper_max_opening_m instead keeps candidates usable.
    fake = _FakeEstimator(grasps=[_identity_transform(np.zeros(3))], scores=[0.5], gripper_openings=None)
    gen = ContactGraspNetGenerator(
        ContactGraspNetConfig(enabled=True), estimator_factory=lambda cfg: fake, gripper_max_opening_m=0.08
    )
    out = gen.generate(libero_scene, ["milk_1_main"])
    assert len(out) == 1
    assert out[0].gripper_width == pytest.approx(0.08)


@requires_libero
def test_max_candidates_truncates_to_highest_scores(libero_scene):
    obj_pos, _ = libero_scene.body_pose("milk_1_main")
    transforms = [_identity_transform(obj_pos) for _ in range(5)]
    scores = [0.1, 0.9, 0.5, 0.3, 0.7]
    fake = _FakeEstimator(transforms, scores)
    gen = ContactGraspNetGenerator(
        ContactGraspNetConfig(enabled=True, max_candidates=2),
        estimator_factory=lambda cfg: fake,
    )
    out = gen.generate(libero_scene, ["milk_1_main"])
    assert sorted((c.generator_score for c in out), reverse=True) == [0.9, 0.7]


@requires_libero
def test_estimator_error_returns_empty_without_raising(libero_scene):
    gen = ContactGraspNetGenerator(
        ContactGraspNetConfig(enabled=True), estimator_factory=lambda cfg: _RaisingEstimator()
    )
    out = gen.generate(libero_scene, ["milk_1_main"])
    assert out == []


def test_load_real_estimator_missing_package_raises_actionable_error():
    # A real (not hypothetical) regression guard against a bare, confusing
    # ModuleNotFoundError leaking out of this path -- only meaningful on an
    # environment that actually lacks contact_graspnet_pytorch (e.g. this
    # package's own dev machine); skipped where it's installed for real
    # (e.g. a CUDA machine's contact_graspnet_pytorch/.venv-libero env).
    try:
        import contact_graspnet_pytorch  # noqa: F401

        pytest.skip("contact_graspnet_pytorch is installed in this environment")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="contact_graspnet_pytorch"):
        _load_real_estimator(ContactGraspNetConfig(enabled=True, checkpoint_dir="/nonexistent"))


def test_quat_to_approach_axis_matches_identity_convention():
    axis = quat_to_approach_axis(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(axis, [0.0, 0.0, 1.0])
