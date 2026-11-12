"""Safe and Upright from the plan: the helpers that turn them into motion limits."""

import numpy as np

from step.system_one.motion.config import CollisionConfig, KeepoutZone
from step.system_one.motion.scene_query import quat_to_rotmat_wxyz
from step.system_one.motion.attached_object import level_object_quat as _level_quat
from step.system_two.stl_plan import find_all_predicate_args


def _pred(name, *args):
    return {"type": "predicate", "name": name, "args": list(args)}


def test_find_all_returns_every_safe_term():
    f = {"type": "and", "children": [
        {"type": "eventually", "interval": [0, 90], "child": _pred("Grasped", "bowl_1")},
        {"type": "always", "interval": None, "child": _pred("Safe", "bowl_1", "milk_1", 0.08)},
        {"type": "always", "interval": [90, 270], "child": _pred("Safe", "ee", "milk_1", 0.08)},
        {"type": "always", "interval": None, "child": _pred("Upright", "bowl_1", 5)},
    ]}
    assert find_all_predicate_args(f, "Safe") == [["bowl_1", "milk_1", 0.08], ["ee", "milk_1", 0.08]]
    assert find_all_predicate_args(f, "Upright") == [["bowl_1", 5]]
    assert find_all_predicate_args(f, "Missing") == []


def test_level_quat_removes_tilt_and_keeps_yaw():
    up_local = np.array([0.0, 0.0, 1.0])
    yaw = np.array([np.cos(0.4), 0.0, 0.0, np.sin(0.4)])                 # 46 deg about z
    tip = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])                 # 11.5 deg about x
    w0, x0, y0, z0 = tip
    w1, x1, y1, z1 = yaw
    tilted = np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1, w0*x1 + x0*w1 + y0*z1 - z0*y1,
                       w0*y1 - x0*z1 + y0*w1 + z0*x1, w0*z1 + x0*y1 - y0*x1 + z0*w1])
    level = _level_quat(tilted, up_local)
    R = quat_to_rotmat_wxyz(level)
    assert np.allclose(R @ up_local, [0, 0, 1], atol=1e-9)
    # heading of the object's x axis is unchanged to within the tilt's own effect
    heading = lambda q: np.arctan2(*(quat_to_rotmat_wxyz(q) @ [1, 0, 0])[[1, 0]])
    assert abs(heading(level) - heading(yaw)) < np.radians(2)


def test_level_quat_with_a_mesh_base_rotation():
    # LIBERO meshes rest with an arbitrary frame; up is whatever was up at rest
    rest = np.array([0.5, 0.5, 0.5, 0.5])
    up_local = quat_to_rotmat_wxyz(rest).T @ np.array([0, 0, 1.0])
    tip = np.array([np.cos(0.08), 0.0, np.sin(0.08), 0.0])
    w0, x0, y0, z0 = tip
    w1, x1, y1, z1 = rest
    tilted = np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1, w0*x1 + x0*w1 + y0*z1 - z0*y1,
                       w0*y1 - x0*z1 + y0*w1 + z0*x1, w0*z1 + x0*y1 - y0*x1 + z0*w1])
    level = _level_quat(tilted, up_local)
    assert np.allclose(quat_to_rotmat_wxyz(level) @ up_local, [0, 0, 1], atol=1e-9)


def test_defaults_leave_behaviour_unchanged():
    assert CollisionConfig().max_tilt_deg is None
    z = KeepoutZone(name="zone", center_xy=(0, 0), radius_m=0.05)
    assert z.subject is None and z.buffer_m == 0.0
