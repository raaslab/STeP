"""The benchmark's STL scorer, on hand-made trajectories with known answers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "libero_constrained"))

import spec as S  # noqa: E402


class _Npz(dict):
    @property
    def files(self):
        return list(self.keys())


def _traj(obj_xyz, eef_xyz=None, held=None, quat=None, hz=10):
    """One object called 'cup' and a fixed obstacle 'jar' at (0, 0, 0)."""
    obj = np.asarray(obj_xyz, dtype=float)
    T = len(obj)
    jar = np.zeros((T, 3))
    data = _Npz(
        object_names=np.array(["cup", "jar"], dtype=object),
        object_pos=np.stack([obj, jar], axis=1),
        eef_pos=np.asarray(eef_xyz if eef_xyz is not None else obj, dtype=float),
        grasped=np.asarray(held if held is not None else np.zeros(T), dtype=float),
    )
    if quat is not None:
        data["object_quat"] = np.stack([np.asarray(quat, float), np.tile([0, 0, 0, 1.0], (T, 1))], axis=1)
    return S.Trajectory(data, hz)


def _line(x0, x1, T=11):
    return np.stack([np.linspace(x0, x1, T), np.zeros(T), np.zeros(T)], axis=1)


KEEPOUT = {"G": {"pred": [{"dist": ["cup", "jar"]}, ">=", 0.05]}}


def test_keepout_robustness_is_the_closest_approach_in_cm():
    # passes 3 cm from the jar at its closest; rule asks 5 cm -> -2 cm
    path = np.stack([np.linspace(-0.2, 0.2, 21), np.full(21, 0.03), np.zeros(21)], axis=1)
    assert np.isclose(S.evaluate(_traj(path), KEEPOUT), -2.0)


def test_keepout_held_when_never_close():
    path = np.stack([np.linspace(-0.2, 0.2, 21), np.full(21, 0.10), np.zeros(21)], axis=1)
    r = S.score(_traj(path), KEEPOUT)
    assert r["held"] and np.isclose(r["robustness"], 5.0)


def test_eventually_and_bounded_window():
    traj = _traj(_line(0.3, 0.0))          # reaches the jar at the last step, t = 1.0 s
    near = {"pred": [{"dist": ["cup", "jar"]}, "<=", 0.01]}
    assert S.evaluate(traj, {"F": near}) > 0
    assert S.evaluate(traj, {"F": near, "in": [0, 0.5]}) < 0   # deadline too early


def test_until_requires_left_to_hold_until_right():
    # order: cup must not be lifted until the jar... here: "held" is the right side
    T = 11
    held = np.array([0] * 5 + [1] * 6, dtype=float)
    traj = _traj(_line(0.3, 0.3, T), held=held)
    lifted = {"pred": [{"height": "cup"}, ">=", 0.02]}
    is_held = {"pred": [{"held": True}, ">=", 0.5]}
    # cup never lifts, and held arrives -> holds
    assert S.evaluate(traj, {"U": [{"not": lifted}, is_held]}) > 0
    # right side never comes -> fails
    never = {"pred": [{"held": True}, ">=", 2.0]}
    assert S.evaluate(traj, {"U": [{"not": lifted}, never]}) < 0


def test_until_fails_when_left_breaks_first():
    T = 11
    path = _line(0.3, 0.3, T)
    path[3:, 2] = 0.05                      # lifted at step 3
    held = np.array([0] * 6 + [1] * 5, dtype=float)
    traj = _traj(path, held=held)
    lifted = {"pred": [{"height": "cup"}, ">=", 0.02]}
    is_held = {"pred": [{"held": True}, ">=", 0.5]}
    assert S.evaluate(traj, {"U": [{"not": lifted}, is_held]}) < 0


def test_tilt_from_start_orientation():
    T = 5
    a = np.radians([0, 5, 10, 20, 30])       # tip about x
    quat = np.stack([np.sin(a / 2), np.zeros(T), np.zeros(T), np.cos(a / 2)], axis=1)  # xyzw
    traj = _traj(_line(0.3, 0.3, T), quat=quat)
    upright = {"G": {"pred": [{"tilt": "cup"}, "<=", 15.0]}}
    assert np.isclose(S.evaluate(traj, upright), -15.0)


def test_speed_limit_while_held():
    T = 21
    path = _line(0.0, 0.4, T)               # 0.4 m in 2 s at 10 Hz -> 0.2 m/s
    held = np.ones(T)
    traj = _traj(path, held=held)
    slow = {"G": {"implies": [{"pred": [{"held": True}, ">=", 0.5]},
                              {"pred": [{"speed": "cup"}, "<=", 0.10]}]}}
    assert np.isclose(S.evaluate(traj, slow), -10.0)   # 10 cm/s over the limit
    traj_free = _traj(path, held=np.zeros(T))
    assert S.evaluate(traj_free, slow) > 0             # not held -> no limit


def test_final_state_rule():
    traj = _traj(_line(0.3, 0.02))
    at_jar = {"end": {"pred": [{"dist": ["cup", "jar"]}, "<=", 0.03]}}
    assert np.isclose(S.evaluate(traj, at_jar), 1.0)


def test_missing_object_is_unscorable_not_a_violation():
    traj = _traj(_line(0.3, 0.0))
    r = S.score(traj, {"G": {"pred": [{"dist": ["cup", "plate"]}, ">=", 0.05]}})
    assert r["held"] is None
