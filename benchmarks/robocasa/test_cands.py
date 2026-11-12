"""Per-candidate breakdown: width, IK, collision -- without a rollout."""
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
from step.system_one.motion.skills.grasp_generator import PrivilegedTopDownGraspGenerator
from step.system_one.motion import panda

env = make_env(); ctx = build_constrained_context(env, enforce=True)
sc = SceneQuery.from_robosuite_sim(env.sim)
mc = MotionConfig.from_scene_context(ctx)
ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                      panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
q0 = sc.get_qpos(panda.JOINT_NAMES)
body = env.sim.model.body(env.obj_body_id["obj"]).name
gen = PrivilegedTopDownGraspGenerator(mc.grasp)
cands = gen.generate(sc, [body])
ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
maxw = gen.gripper_max_opening_m
print(f"gripper max opening: {maxw:.3f} m,  {len(cands)} candidates")
print(f"{'#':>2} {'strategy':<12} {'width':>7} {'fits':>5} {'graspIK':>8} {'collision':>10}")
ok = 0
for i, c in enumerate(cands):
    w = c.gripper_width or 0.0
    fits = w <= maxw
    row = [f"{i:>2}", f"{(c.metadata or {}).get('strategy','?'):<12}", f"{w:7.3f}", f"{str(fits):>5}"]
    if not fits:
        print(" ".join(row) + f" {'-':>8} {'-':>10}")
        continue
    sol = ik.solve(sc.clone(), c.grasp_pose_world[:3], c.grasp_pose_world[3:], seed_q=q0)
    row.append(f"{str(sol.success):>8}")
    if sol.success:
        r = ck.check(np.asarray(sol.q, float), phase=CollisionPhase.TERMINAL_APPROACH,
                     target_object_body=body)
        row.append(f"{str(r.in_collision):>10}")
        if not r.in_collision:
            ok += 1
        else:
            for ct in r.contacts[:3]:
                print(f"      hits: {ct.geom1} <-> {ct.geom2}  d={ct.distance:.4f}")
    else:
        row.append(f"{'-':>10}")
    print(" ".join(row))
print(f"\nfully feasible: {ok}/{len(cands)}")
