"""Can STeP's motion stack read a RoboCasa kitchen?

Three questions, in order of how likely they are to fail:
  1. does SceneQuery bind to the robosuite 1.5 sim at all,
  2. does the collision checker report a sane state at the home configuration,
  3. is the kitchen's clutter (walls, cabinets, floor) going to make every
     configuration read as in-collision -- the tabletop assumption I flagged.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys
sys.path.insert(0, _STEP_ROOT)
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.config import MotionConfig
from step.system_one.motion import panda

cfg = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(env_name="PickPlaceCounterToStove", robots="PandaOmron",
                     controller_configs=cfg, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False,
                     layout_ids=[1], style_ids=[1], seed=0)
env.reset()

scene = SceneQuery.from_robosuite_sim(env.sim)
print("1. SceneQuery bound OK")
ee_pos, ee_quat = scene.site_pose(panda.EE_SITE)
print(f"   EE position: {np.round(ee_pos, 3)}")
q = scene.get_qpos(panda.JOINT_NAMES)
print(f"   arm qpos   : {np.round(q, 3)}")

mc = MotionConfig.from_scene_context({})
checker = CollisionChecker(scene, panda.JOINT_NAMES, panda.EE_SITE,
                           panda.ROBOT_BODIES, panda.GRIPPER_BODIES,
                           config=mc.collision)
res = checker.check(q, phase=CollisionPhase.FREE_SPACE)
print("2. collision check at home config")
print(f"   in_collision={res.in_collision}  n_contacts={len(res.contacts)}  min_clearance={res.min_clearance:.4f}")
for c in res.contacts[:6]:
    print(f"     {c}")
print("3. scene size")
print(f"   nbody={env.sim.model.nbody}  ngeom={env.sim.model.ngeom}  (LIBERO is ~60 bodies / ~200 geoms)")
