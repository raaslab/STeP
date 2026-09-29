"""Contact-GraspNet on a RoboCasa kitchen object.

The generator is backend-agnostic -- it renders depth from named cameras and
deprojects to a world point cloud -- so the only RoboCasa-specific part is which
cameras to name. LIBERO's defaults ("agentview") do not exist here; RoboCasa
offers robot0_agentview_* plus the same robot0_eye_in_hand.

Checks in order: does depth render at all, does the object segment out of the
scene cloud, and does the network return usable grasps (rather than silently
falling back to the heuristic generator, which is what happened on LIBERO when
the checkpoint path was unset).
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys, os, time
sys.path.insert(0, str(_HERE))
sys.path.insert(0, _STEP_ROOT)
import numpy as np
from constrained_task import make_env, build_constrained_context
from robocasa_scene import object_categories
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.config import MotionConfig

env = make_env()
scene = SceneQuery.from_robosuite_sim(env.sim)
cats = object_categories(env)
print("episode:", env.get_ep_meta()["lang"], "| target handle 'obj' =", cats["obj"])

CAMS = ["robot0_agentview_center", "robot0_eye_in_hand"]
target_body = env.sim.model.body(env.obj_body_id["obj"]).name
print("target body:", target_body)
for cam in CAMS:
    t0 = time.time()
    depth, bid = scene.render_depth_and_body_ids(cam, 240, 320)
    valid = (depth > 0) & (depth < 3.0)
    on_target = valid & (bid == scene.body_id(target_body))
    pts = scene.deproject_camera_depth(cam, depth)
    print(f"  {cam:<26} depth ok, {valid.sum():6d} valid px, "
          f"{on_target.sum():5d} on target, {time.time()-t0:.2f}s")
    if on_target.sum():
        seg = pts[on_target]
        print(f"      segment centroid {np.round(seg.mean(axis=0),3)}  "
              f"(true obj {np.round(env.sim.data.body_xpos[env.obj_body_id['obj']],3)})")

mc = MotionConfig.from_scene_context(build_constrained_context(env, enforce=True))
mc.contact_graspnet.enabled = True          # off by default
mc.contact_graspnet.camera_names = CAMS
mc.contact_graspnet.render_height, mc.contact_graspnet.render_width = 240, 320
print("\ncheckpoint dir:", os.environ.get("CONTACT_GRASPNET_CHECKPOINT_DIR", "<unset>"))
from step.system_one.motion.skills.contact_graspnet_generator import ContactGraspNetGenerator
gen = ContactGraspNetGenerator(mc.contact_graspnet)
t0 = time.time()
cands = gen.generate(scene, [target_body])   # takes a LIST of names
print(f"grasps returned: {len(cands)} in {time.time()-t0:.1f}s")
for c in cands[:3]:
    pos = getattr(c, "position", getattr(c, "translation", None))
    print("   ", np.round(pos, 3) if pos is not None else c)
