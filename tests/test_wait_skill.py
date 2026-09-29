"""EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md, "wait skill" section):
tests for the not-before wait mechanism -- motion.servo.wait_steps and
executor._wait_for_stl_start -- using a stub adapter so no live LIBERO env
is needed. These are pure step-count/gripper-persistence checks; whether the
arm visibly holds still in a real rollout still needs a live test."""

import numpy as np

from step.system_one.executor import System1Executor
from step.system_one.motion.servo import wait_steps


class _StubAdapter:
    """Records every _step_env call instead of touching a real sim."""

    def __init__(self):
        self.calls = []

    def _step_env(self, cart_pos, cart_ori, gripper_cmd):
        self.calls.append((np.array(cart_pos), np.array(cart_ori), gripper_cmd))
        return {}


class _StubEnv:
    open_gripper_cmd = -1.0
    close_gripper_cmd = 1.0


class _StubExecutorSelf:
    """Minimal stand-in for `self` -- _wait_for_stl_start's internal
    step_callback calls self._capture_video_frame(env), which a bare `None`
    can't satisfy."""

    def _capture_video_frame(self, env, force=False):
        pass

    def _on_skill_step(self, env):
        # The real executor records a frame and ticks its STL monitor here. The
        # wait mechanism only cares that the callback is safe to call per step.
        pass


def test_wait_steps_holds_zero_delta_for_exact_count():
    adapter = _StubAdapter()
    used = wait_steps(adapter, 5, gripper_cmd=1.0)
    assert used == 5
    assert len(adapter.calls) == 5
    for cart_pos, cart_ori, gripper_cmd in adapter.calls:
        assert np.allclose(cart_pos, 0.0)
        assert np.allclose(cart_ori, 0.0)
        assert gripper_cmd == 1.0


def test_wait_steps_zero_or_negative_is_noop():
    adapter = _StubAdapter()
    assert wait_steps(adapter, 0, gripper_cmd=1.0) == 0
    assert wait_steps(adapter, -5, gripper_cmd=1.0) == 0
    assert adapter.calls == []


def test_wait_steps_invokes_step_callback_each_tick():
    adapter = _StubAdapter()
    ticks = []
    wait_steps(adapter, 3, gripper_cmd=-1.0, step_callback=lambda: ticks.append(1))
    assert len(ticks) == 3


def test_wait_for_stl_start_noop_without_deadline():
    adapter = _StubAdapter()
    env = _StubEnv()
    used = System1Executor._wait_for_stl_start(
        _StubExecutorSelf(),
        adapter,
        {"params": {"object": "cup_1"}},
        task_steps_so_far=0,
        gripper_cmd=env.close_gripper_cmd,
        env=env,
    )
    assert used == 0
    assert adapter.calls == []


def test_wait_for_stl_start_noop_when_already_at_or_past_start():
    adapter = _StubAdapter()
    env = _StubEnv()
    sub = {"stl_deadline": {"start_step": 90, "end_step": 270}}
    used = System1Executor._wait_for_stl_start(
        _StubExecutorSelf(), adapter, sub, task_steps_so_far=90, gripper_cmd=env.close_gripper_cmd, env=env
    )
    assert used == 0
    used_past = System1Executor._wait_for_stl_start(
        _StubExecutorSelf(), adapter, sub, task_steps_so_far=150, gripper_cmd=env.close_gripper_cmd, env=env
    )
    assert used_past == 0
    assert adapter.calls == []


def test_wait_for_stl_start_holds_exact_remaining_gap():
    adapter = _StubAdapter()
    env = _StubEnv()
    sub = {"stl_deadline": {"start_step": 340, "end_step": 460}}
    used = System1Executor._wait_for_stl_start(
        _StubExecutorSelf(), adapter, sub, task_steps_so_far=270, gripper_cmd=env.close_gripper_cmd, env=env
    )
    assert used == 70
    assert len(adapter.calls) == 70
    assert all(gripper_cmd == env.close_gripper_cmd for _, _, gripper_cmd in adapter.calls)


def test_wait_for_stl_start_uses_open_gripper_for_pre_grasp_wait():
    """A pick's own wait (before anything is grasped) must hold the gripper
    OPEN, not closed -- distinguishes it from transport/place's wait, which
    must keep the already-grasped object held."""
    adapter = _StubAdapter()
    env = _StubEnv()
    sub = {"stl_deadline": {"start_step": 50, "end_step": 140}}
    System1Executor._wait_for_stl_start(
        _StubExecutorSelf(), adapter, sub, task_steps_so_far=0, gripper_cmd=env.open_gripper_cmd, env=env
    )
    assert all(gripper_cmd == env.open_gripper_cmd for _, _, gripper_cmd in adapter.calls)
