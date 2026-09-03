import numpy as np
import pytest

from conftest import requires_libero
from step.system_one.motion.config import ContactGraspNetConfig
from step.system_one.motion.contact_graspnet_generator import (
    ContactGraspNetGenerator,
    _load_real_estimator,
    quat_to_approach_axis,
)
from step.system_one.motion.grasp_generator import hand_from_fingertip_pose


class _FakeEstimator:
    """Stands in for contact_graspnet_pytorch.GraspEstimator, shaped exactly
    like the real predict_scene_grasps() call transcribed from Cap-X's
    server (see contact_graspnet_generator.py's module docstring) -- lets
    every non-CUDA part of this generator (point-cloud construction, pose
    conversion, scoring, truncation, error handling) be tested on a machine
    with no GPU and no contact_graspnet_pytorch installed."""

    def __init__(self, grasps, scores, contact_pts=None):
        self.grasps = np.asarray(grasps)
        self.scores = np.asarray(scores)
        self.contact_pts = np.zeros((len(self.grasps), 3)) if contact_pts is None else contact_pts
        self.calls = []

    def predict_scene_grasps(self, pc_full, pc_segments, local_regions, filter_grasps, forward_passes):
        self.calls.append((pc_full, pc_segments, local_regions, filter_grasps, forward_passes))
        return {1: self.grasps}, {1: self.scores}, {1: self.contact_pts}, None


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
    obj_pos, _ = libero_scene.body_pose("milk_1_main")
    fake = _FakeEstimator(grasps=[_identity_transform(obj_pos)], scores=[5.0])  # 5.0: out-of-range on purpose
    gen = ContactGraspNetGenerator(
        ContactGraspNetConfig(enabled=True, max_candidates=10),
        estimator_factory=lambda cfg: fake,
    )
    out = gen.generate(libero_scene, ["milk_1_main"])

    assert len(out) == 1
    cand = out[0]
    assert cand.object_name == "milk_1_main"
    assert cand.gripper_width is None
    assert cand.generator_score == 1.0  # clipped from 5.0

    expected_pose = hand_from_fingertip_pose(libero_scene, obj_pos, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(cand.grasp_pose_world, expected_pose, atol=1e-9)
    assert len(fake.calls) == 1


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
    # This dev environment genuinely lacks contact_graspnet_pytorch -- a real
    # (not hypothetical) regression guard against a bare, confusing
    # ModuleNotFoundError leaking out of this path.
    with pytest.raises(ImportError, match="contact_graspnet_pytorch"):
        _load_real_estimator(ContactGraspNetConfig(enabled=True, checkpoint_dir="/nonexistent"))


def test_quat_to_approach_axis_matches_identity_convention():
    axis = quat_to_approach_axis(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(axis, [0.0, 0.0, 1.0])
