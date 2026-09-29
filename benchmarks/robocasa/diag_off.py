"""Why does the UNCONSTRAINED arm fail to plan the transport leg?

Removing the keep-out should only make planning easier. If the OFF arm fails, the
direct route is physically blocked by something -- and the A/B is then not a clean
constraint comparison but a geometry one. Capture the rejected edge and name it.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys
sys.path.insert(0, str(_HERE))
sys.path.insert(0, _STEP_ROOT)
import numpy as np
from constrained_task import make_env, build_constrained_context, burner_xy
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.config import MotionConfig
from step.system_one.motion import panda

env = make_env()
scene = SceneQuery.from_robosuite_sim(env.sim)
lo, hi = scene.joint_qpos_range(panda.JOINT_NAMES)
qh = scene.get_qpos(panda.JOINT_NAMES)
obj = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]], dtype=float)
pan = np.asarray(env.sim.data.body_xpos[env.obj_body_id["container"]], dtype=float)
down = np.array([0.0, 1.0, 0.0, 0.0])
ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
q0 = np.asarray(ik.solve(scene.clone(), obj + [0, 0, 0.15], down, seed_q=qh).q, float)
qg = np.asarray(ik.solve(scene.clone(), pan + [0, 0, 0.15], down, seed_q=qh).q, float)

ctx = build_constrained_context(env, enforce=False)
mc = MotionConfig.from_scene_context(ctx)
mc.planner.validity_resolution = 0.001
checker = CollisionChecker(scene, panda.JOINT_NAMES, panda.EE_SITE,
                           panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
print("zones (should be 0):", len(mc.collision.keepout_zones))
for name, q in (("start (above eggplant)", q0), ("goal (above pan)", qg)):
    r = checker.check(q, phase=CollisionPhase.FREE_SPACE)
    print(f"  {name}: in_collision={r.in_collision} contacts={[ (c.geom1,c.geom2) for c in r.contacts[:3]]}")

captured = {}
real = checker.check_edge
def spy(a, b, **kw):
    r = real(a, b, **kw)
    if r.in_collision and "f" not in captured:
        captured["f"] = r
    class Ok:
        in_collision = False; contacts = []; min_clearance = float("inf")
    return Ok()
checker.check_edge = spy
p = MotionPlanner(panda.JOINT_NAMES, lo, hi, checker, config=mc.planner).plan(
    q0, [qg], phase=CollisionPhase.FREE_SPACE)
print("with veto disabled -> success:", p.success,
      "waypoints:", 0 if p.geometric_path is None else len(p.geometric_path))
if "f" in captured:
    r = captured["f"]
    print(f"first rejected edge: min_clearance={r.min_clearance:.5f}")
    for c in r.contacts[:6]:
        print("   ", c)
