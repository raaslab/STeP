"""Can the arm reach a standoff above the carrot at all?

7 of 16 grasp candidates failed on approach_standoff_ik_failed /
approach_standoff_in_collision. Either the pose is out of the arm's reach from
where the base is parked, or something is in the way. IK + collision, no
planning -- seconds, not a 30-minute rollout.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys
sys.path.insert(0, str(_HERE)); sys.path.insert(0, _STEP_ROOT)
import numpy as np
from constrained_task import make_env, build_constrained_context
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.config import MotionConfig
from step.system_one.motion import panda

env = make_env(); ctx = build_constrained_context(env, enforce=True)
sc = SceneQuery.from_robosuite_sim(env.sim)
mc = MotionConfig.from_scene_context(ctx)
ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                      panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
q0 = sc.get_qpos(panda.JOINT_NAMES)
carrot = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]], float)
base = np.asarray(env.sim.data.body_xpos[env.sim.model.body("mobilebase0_base").id], float)
ee = sc.site_pose(panda.EE_SITE)[0]
print("carrot:", np.round(carrot,3), " base:", np.round(base,3), " ee:", np.round(ee,3))
print("base->carrot horizontal distance: %.3f m  (Panda reach ~0.85)" %
      np.linalg.norm(carrot[:2] - base[:2]))
print()
down = np.array([0.0, 1.0, 0.0, 0.0])
ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
for h in (0.05, 0.08, 0.12, 0.15, 0.20):
    tgt = carrot + np.array([0, 0, h])
    sol = ik.solve(sc.clone(), tgt, down, seed_q=q0)
    if not sol.success:
        print(f"  standoff +{h*100:4.0f}cm: IK FAILED ({sol.reason}) pos_err={sol.position_error:.3f}")
        continue
    r = ck.check(np.asarray(sol.q,float), phase=CollisionPhase.FREE_SPACE)
    c = [f"{x.geom1}|{x.geom2}" for x in r.contacts[:2]]
    print(f"  standoff +{h*100:4.0f}cm: IK ok, collision={r.in_collision} {c if c else ''}")
