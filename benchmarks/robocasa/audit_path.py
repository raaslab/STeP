"""Independent audit: does the 0.005 path really clear the microwave?

Tightening the resolution could in principle just move the samples either side of
the same 3.8 mm interpenetration. So re-check the returned path at a much finer
step than it was planned with, and report the worst clearance anywhere on it.
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
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.config import MotionConfig
from step.system_one.motion import panda

cfg = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(env_name="PickPlaceCounterToStove", robots="PandaOmron",
                     controller_configs=cfg, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False,
                     layout_ids=[1], style_ids=[1], seed=0)
env.reset()
scene = SceneQuery.from_robosuite_sim(env.sim)
q0 = scene.get_qpos(panda.JOINT_NAMES)
lo, hi = scene.joint_qpos_range(panda.JOINT_NAMES)
obj_xyz = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]], dtype=float)
sol = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig()).solve(
    scene.clone(), obj_xyz + np.array([0, 0, 0.12]), np.array([0.0, 1.0, 0.0, 0.0]), seed_q=q0)
q_goal = np.asarray(sol.q, dtype=float)

mc = MotionConfig.from_scene_context({})
mc.planner.validity_resolution = 0.005
checker = CollisionChecker(scene, panda.JOINT_NAMES, panda.EE_SITE,
                           panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
plan = MotionPlanner(panda.JOINT_NAMES, lo, hi, checker, config=mc.planner).plan(
    q0, [q_goal], phase=CollisionPhase.FREE_SPACE)
path = plan.geometric_path
print(f"planned at 0.005 -> success={plan.success}, {len(path)} waypoints")

worst, worst_i, offenders = float("inf"), -1, {}
for i in range(len(path) - 1):
    r = checker.check_edge(path[i], path[i + 1], resolution=0.002, phase=CollisionPhase.FREE_SPACE)
    if r.min_clearance < worst:
        worst, worst_i = r.min_clearance, i
    for c in r.contacts:
        offenders[(c.geom1, c.geom2)] = min(offenders.get((c.geom1, c.geom2), 0.0), c.distance)
print(f"audit at 0.002 (2.5x finer): worst clearance {worst:.5f} m at edge {worst_i}")
print("contacts found:", offenders if offenders else "none")

sc = scene.clone()
mw = [g for g in (env.sim.model.geom(i).name for i in range(env.sim.model.ngeom))
      if g and "microwave" in g]
print(f"\nclosest approach to the microwave ({len(mw)} geoms) along the path:")
best = float("inf")
for q in path[::10]:
    sc.set_qpos(panda.JOINT_NAMES, q)
    ee = sc.site_pose(panda.EE_SITE)[0]
    for gname in ("microwave_main_group_door_main",):
        gid = env.sim.model.geom(gname).id
        d = float(np.linalg.norm(ee - env.sim.data.geom_xpos[gid]))
        best = min(best, d)
print(f"   min EE-to-door-centre distance: {best:.3f} m")
