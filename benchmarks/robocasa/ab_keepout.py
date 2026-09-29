"""Does the keep-out reach System 1 in RoboCasa? Plan with it on and off.

The measure is the closest the end effector comes to the lit burner anywhere on
the planned path. If enforcement works, the ON arm clears the 10 cm zone and the
OFF arm cuts straight through it -- the same A/B that defines the LIBERO-
Constrained spatial family, now on geometry that blocks the route by construction
rather than by hand placement.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys, time
sys.path.insert(0, str(_HERE))
sys.path.insert(0, _STEP_ROOT)
import numpy as np

RETRIES = 8

from constrained_task import make_env, build_constrained_context, burner_xy, ZONE_RADIUS_M
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.config import MotionConfig
from step.system_one.motion import panda

env = make_env()
scene = SceneQuery.from_robosuite_sim(env.sim)
q0 = scene.get_qpos(panda.JOINT_NAMES)
lo, hi = scene.joint_qpos_range(panda.JOINT_NAMES)
bxy = burner_xy(env)[:2]
pan = np.asarray(env.sim.data.body_xpos[env.obj_body_id["container"]], dtype=float)

# The leg that actually crosses the stove is the TRANSPORT: from above the
# eggplant on the left counter to above the pan on the right burner. Planning
# from the home pose instead measures nothing -- home already sits at x=3.13,
# right next to the pan, so that path never approaches the left burner at all.
ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
obj = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]], dtype=float)
down = np.array([0.0, 1.0, 0.0, 0.0])
s_start = ik.solve(scene.clone(), obj + np.array([0, 0, 0.15]), down, seed_q=q0)
s_goal = ik.solve(scene.clone(), pan + np.array([0, 0, 0.15]), down, seed_q=q0)
assert s_start.success and s_goal.success, (s_start.reason, s_goal.reason)
q0 = np.asarray(s_start.q, dtype=float)      # start ABOVE THE EGGPLANT
q_goal = np.asarray(s_goal.q, dtype=float)   # end ABOVE THE PAN
print(f"transport leg: EE from {np.round(obj[:2],2)} to {np.round(pan[:2],2)}, "
      f"lit burner at {np.round(bxy,2)}")

def run(enforce, trials=4):
    ctx = build_constrained_context(env, enforce=enforce)
    mc = MotionConfig.from_scene_context(ctx)
    mc.planner.validity_resolution = 0.001
    checker = CollisionChecker(scene, panda.JOINT_NAMES, panda.EE_SITE,
                               panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
    planner = MotionPlanner(panda.JOINT_NAMES, lo, hi, checker, config=mc.planner)
    sc = scene.clone()
    out = []
    for _ in range(trials):
        # Retry on failure. In a kitchen RRTConnect regularly returns a path
        # that grazes a cabinet door by a fraction of a millimetre, and the
        # post-hoc veto throws the whole path away rather than repairing it.
        # Without retries the arms are not comparable: the unconstrained arm
        # failed 4/4 on a 0.13 mm link6-vs-door contact that has nothing to do
        # with the keep-out.
        p = None
        for _attempt in range(RETRIES):
            p = planner.plan(q0, [q_goal], phase=CollisionPhase.FREE_SPACE)
            if p.success:
                break
        if not p.success:
            out.append(None); continue
        d = np.inf
        for q in p.geometric_path:
            sc.set_qpos(panda.JOINT_NAMES, q)
            ee = sc.site_pose(panda.EE_SITE)[0][:2]
            d = min(d, float(np.linalg.norm(ee - bxy)))
        out.append(d)
    return mc, out

for enforce in (False, True):
    mc, ds = run(enforce)
    n_zones = len(mc.collision.keepout_zones)
    ok = [d for d in ds if d is not None]
    label = "STL ON (enforced)" if enforce else "STL OFF (monitored)"
    print(f"\n{label}:  zones in motion config = {n_zones}")
    for i, d in enumerate(ds):
        if d is None:
            print(f"   trial {i}: plan failed")
        else:
            print(f"   trial {i}: closest approach to lit burner = {d*100:6.1f} cm  "
                  f"{'VIOLATION' if d < ZONE_RADIUS_M else 'clear'}")
    if ok:
        viol = sum(1 for d in ok if d < ZONE_RADIUS_M)
        print(f"   -> {viol}/{len(ok)} plans enter the {ZONE_RADIUS_M*100:.0f} cm zone")
