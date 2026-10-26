"""The perceived twin: a MuJoCo model containing the real robot and only
what perception reconstructed (support plane, objects, cabinets/drawers).

Every skill, planner and collision check reads the world through
SceneQuery.from_robosuite_sim; in vision mode that returns a
PerceivedSceneQuery over this model, so downstream code runs unchanged on
perceived geometry. Robot joint state is copied from the live sim each call
(proprioception); everything else comes from the WorldEstimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from ..system_one.motion.scene_query import SceneQuery
from .world import WorldEstimate, yaw_quat_wxyz

if TYPE_CHECKING:
    from .vision import VisionPerception

_ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")
_OBJECT_RGBA = [0.2, 0.6, 0.9, 1.0]
# Collision bits. Robot geoms keep LIBERO's own (contype, conaffinity) of
# (1, 1) and (0, 1), so every perceived geom keeps contype bit 1. Static
# geometry (table, fixtures) skips static-static pairs, which planning never
# needs and which made the twin's collision checks 6x slower than the real
# model's; movable objects still collide with everything, which the
# carried-object placement check relies on.
_STATIC_BITS = (1, 2)
_MOVABLE_BITS = (3, 3)


def robot_only_spec(world_xml: str) -> mujoco.MjSpec:
    """The task's MJCF with everything but the robot removed: objects,
    fixtures, the table and the arena's own static geoms. Cameras and lights
    stay, being sensor calibration rather than scene content."""
    spec = mujoco.MjSpec.from_string(world_xml)
    for body in list(spec.worldbody.bodies):
        if not body.name.startswith(_ROBOT_PREFIXES):
            spec.delete(body)
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    # Contact excludes, sensors and equalities can name bodies that are now
    # gone; none of them matter for planning.
    for item in list(spec.excludes) + list(spec.sensors) + list(spec.equalities):
        spec.delete(item)
    return spec


def _add_box(body, name: str, center, half, quat=None, rgba=None, static: bool = True) -> None:
    geom = body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[float(h) for h in half],
        pos=[float(c) for c in center],
        rgba=rgba or _OBJECT_RGBA,
    )
    if quat is not None:
        geom.quat = [float(q) for q in quat]
    geom.contype, geom.conaffinity = _STATIC_BITS if static else _MOVABLE_BITS


def _axis_yaw_quat(axis: np.ndarray) -> np.ndarray:
    return yaw_quat_wxyz(float(np.arctan2(axis[1], axis[0])))


def build_twin_model(world_xml: str, world: WorldEstimate) -> mujoco.MjModel:
    spec = robot_only_spec(world_xml)
    wb = spec.worldbody

    if world.support_z is not None:
        lo, hi = world.support_xy_bounds or (np.array([-0.5, -0.6]), np.array([0.6, 0.6]))
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        # 3 mm below the plane objects stand on: flush boxes would generate a
        # contact per object base box on every collision query.
        table = wb.add_body(name="table", pos=[float(center[0]), float(center[1]), float(world.support_z) - 0.028])
        _add_box(table, "table_top", [0, 0, 0], [float(half[0]), float(half[1]), 0.025], rgba=[0.7, 0.6, 0.5, 1.0])

    for obj in world.objects.values():
        if obj.shape is None:
            continue
        body = wb.add_body(name=obj.name, pos=[float(v) for v in obj.pos], quat=[float(v) for v in obj.quat_wxyz])
        if not obj.is_static_fixture:
            body.add_freejoint(name=f"{obj.name}_joint0")
        for i, box in enumerate(obj.shape.boxes):
            _add_box(body, f"{obj.name}_g{i}", box.center_local, box.half_size, static=obj.is_static_fixture)

        for d in (d for d in world.drawers if d.cabinet == obj.name):
            # Drawer bodies hang off the cabinet so MuJoCo's parent/child
            # contact exclusion stops the drawer colliding with its own carcass.
            rel = d.panel_center - obj.pos
            R = _yaw_rotmat(obj.quat_wxyz)
            drawer = body.add_body(name=d.body_name, pos=[float(v) for v in R.T @ rel])
            joint = drawer.add_joint(
                name=d.joint_name,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[float(v) for v in R.T @ d.pull_axis],
            )
            joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            joint.range = [0.0, float(d.travel_m)]
            q_local = _axis_yaw_quat(R.T @ d.pull_axis)
            _add_box(drawer, f"{d.body_name}_panel", [0, 0, 0], d.panel_half, quat=q_local)
            handle_rel = R.T @ (d.handle_center - d.panel_center)
            _add_box(drawer, f"{d.body_name}_handle", handle_rel, d.handle_half, quat=q_local)

        for k in (k for k in world.knobs if k.fixture == obj.name):
            R = _yaw_rotmat(obj.quat_wxyz)
            axis_local = R.T @ k.axis
            knob = body.add_body(name=k.body_name, pos=[float(v) for v in R.T @ (k.center - obj.pos)])
            joint = knob.add_joint(name=k.body_name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[float(v) for v in axis_local])
            joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            joint.range = [float(k.range_rad[0]), float(k.range_rad[1])]
            # A box rather than a cylinder: estimate_knob_radius_m reads the
            # AABB, and a box spins visibly in the twin's FK.
            _add_box(knob, f"{k.body_name}_g0", axis_local * k.half[2], k.half, quat=_quat_aligning_z(axis_local))

    return spec.compile()


def _quat_aligning_z(axis: np.ndarray) -> np.ndarray:
    """Quaternion rotating local +Z onto axis."""
    z = np.array([0.0, 0.0, 1.0])
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    v = np.cross(z, a)
    c = float(z @ a)
    if np.linalg.norm(v) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    q = np.array([1.0 + c, *v])
    return q / np.linalg.norm(q)


def _yaw_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


class PerceivedSceneQuery(SceneQuery):
    """SceneQuery over the perceived twin. Rendering returns real camera
    depth with perception's masks in place of the simulator's segmentation."""

    def __init__(self, model, data, *, perception: "VisionPerception", live_sim: Any) -> None:
        super().__init__(model, data, _render_sim=None)
        self._perception = perception
        self._live_sim = live_sim

    def clone(self) -> "PerceivedSceneQuery":
        new_data = mujoco.MjData(self.model)
        new_data.qpos[:] = self.data.qpos
        new_data.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.model, new_data)
        return PerceivedSceneQuery(self.model, new_data, perception=self._perception, live_sim=self._live_sim)

    def render_depth_and_body_ids(self, camera_name: str, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        return self._perception.render_depth_and_body_ids(self, camera_name, height, width)


class TwinSync:
    """Copies proprioception from the live sim into a twin MjData, and
    perceived object/drawer state from the WorldEstimate."""

    def __init__(self, twin_model: mujoco.MjModel, live_model: mujoco.MjModel, world: WorldEstimate) -> None:
        self.model = twin_model
        self.robot_pairs: list[tuple[int, int, int, int]] = []
        for j in range(twin_model.njnt):
            name = mujoco.mj_id2name(twin_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if not name.startswith(_ROBOT_PREFIXES):
                continue
            live_j = mujoco.mj_name2id(live_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if live_j < 0:
                continue
            n = 1 if twin_model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) else 0
            if n:
                self.robot_pairs.append(
                    (
                        int(twin_model.jnt_qposadr[j]),
                        int(live_model.jnt_qposadr[live_j]),
                        int(twin_model.jnt_dofadr[j]),
                        int(live_model.jnt_dofadr[live_j]),
                    )
                )
        self.world = world

    def snapshot(self, live_data) -> mujoco.MjData:
        data = mujoco.MjData(self.model)
        for twin_q, live_q, twin_v, live_v in self.robot_pairs:
            data.qpos[twin_q] = live_data.qpos[live_q]
            data.qvel[twin_v] = live_data.qvel[live_v]
        for obj in self.world.objects.values():
            if obj.is_static_fixture or obj.shape is None:
                continue
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj.name}_joint0")
            if jid < 0:
                continue
            adr = int(self.model.jnt_qposadr[jid])
            data.qpos[adr : adr + 3] = obj.pos
            data.qpos[adr + 3 : adr + 7] = obj.quat_wxyz
        for d in self.world.drawers:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, d.joint_name)
            if jid >= 0:
                data.qpos[int(self.model.jnt_qposadr[jid])] = d.qpos
        for k in self.world.knobs:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, k.body_name)
            if jid >= 0:
                data.qpos[int(self.model.jnt_qposadr[jid])] = k.qpos
        mujoco.mj_forward(self.model, data)
        return data
