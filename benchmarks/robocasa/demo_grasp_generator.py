"""GraspGenerator backed by RoboCasa's human demonstrations.

Implements MimicGen's transfer rule -- T_grasp = T_object_now @ T_offset -- over
offsets mined from the demos by mimicgen_grasps.extract. Drops into PickSkill
wherever the privileged or Contact-GraspNet generators would go.

Filtered to the exact mesh in the scene, because that is as far as the transfer
rule reaches: it re-anchors a grasp to a new pose of the same object, not to a
different one.

Two levels of "different" bit here in turn. RoboCasa samples a new object
category every episode, so the unfiltered pool offered a carrot's and an egg's
offsets for grasping an eggplant -- measured, that closed the gripper 2.4 cm away
and shoved the eggplant 6 cm across the counter. Filtering to "eggplant" was not
enough either: the category ships three meshes (eggplant_0/_1/_2), each with its
own body origin, and an eggplant_1 offset applied to eggplant_0 sent the hand
9.6 cm from the object to close on air. The mesh is the unit.
"""
import json
import pathlib

import numpy as np

from mimicgen_grasps import CACHE, apply
from step.system_one.motion.types import GraspCandidate


class DemoGraspGenerator:
    """Emits the grasps humans used, re-anchored to this episode's object pose."""

    def __init__(self, task, model=None, gripper_max_opening_m=0.08, width_m=0.04):
        self.task = task
        self.model = model
        self.gripper_max_opening_m = float(gripper_max_opening_m)
        self._width = float(width_m)

        # Every task's cache, not just this one. How to grasp an eggplant is a
        # property of the eggplant, not of where it is being carried to, and the
        # same meshes recur across the pick-place tasks. Pooling matters because
        # the per-mesh counts are thin: one task's 108 episodes cover 74 meshes,
        # so eggplant_0 has 2 grasps there and 4 across all five tasks.
        allo = []
        for path in sorted(CACHE.glob("*.json")):
            try:
                allo.extend(json.loads(path.read_text()))
            except Exception:
                continue
        self.offsets = [o for o in allo if o.get("model") == model] if model else allo
        if model and not self.offsets:
            # Better to say so than to silently fall back to every other object's
            # grasps, which is the failure this filter exists to stop. Sibling
            # meshes of the same category are named so the caller can see how
            # near a miss it was.
            kin = sorted({o.get("model") for o in allo
                          if o.get("model") and o.get("object")
                          and o["object"] == model.rsplit("_", 1)[0]})
            print(f"[demo-grasp] no demonstration of mesh {model!r} in {task} "
                  f"({len(allo)} cached). Same category: {', '.join(kin) or 'none'}")
        else:
            tasks = sorted({o.get("task") for o in self.offsets if o.get("task")})
            print(f"[demo-grasp] {len(self.offsets)} offsets for {model or 'any mesh'} "
                  f"({len(allo)} cached over {len(list(CACHE.glob('*.json')))} tasks"
                  + (f", from {', '.join(tasks)}" if tasks else "") + ")")

    def generate(self, scene, object_names):
        out = []
        for name in object_names:
            try:
                o_pos, o_quat = scene.body_pose(name)
            except Exception:
                continue
            for off in self.offsets:
                pos, quat = apply(off, np.asarray(o_pos, float), np.asarray(o_quat, float))
                approach = _quat_to_mat(quat) @ np.array([0.0, 0.0, 1.0])
                out.append(GraspCandidate(
                    object_name=name,
                    grasp_pose_world=np.concatenate([pos, quat]),
                    approach_direction_world=approach,
                    # The demos do not record a width; the recorded pose already
                    # encodes a grasp a human closed successfully, so this is a
                    # nominal value that keeps PickSkill's width filter from
                    # rejecting it rather than a measurement.
                    gripper_width=self._width,
                    # Ranked by how close the grip site sits to the object's
                    # origin. Episode index was standing in for quality here,
                    # which is no signal at all -- it let a 9.6 cm offset
                    # outrank a 1.9 cm one on an object a few centimetres wide.
                    generator_score=float(1.0 / (1.0 + 10.0 * off.get("offset_dist", 0.0))),
                    metadata={"generator": "DemoGraspGenerator", "strategy": "demo",
                              "episode": off.get("episode"), "tilt_deg": off.get("tilt_deg")},
                ))
        return out


def _quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
