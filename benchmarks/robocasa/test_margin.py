"""Does raising the grasp height clear the finger/plate collisions?"""
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
body = env.sim.model.body(env.obj_body_id["obj"]).name
q0 = sc.get_qpos(panda.JOINT_NAMES)
ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
print(f"{'margin':>7} {'tol':>6} {'feasible':>9}   first blocker")
for margin in (0.01, 0.02, 0.03):
    for tol in (0.006, 0.015):
        mc = MotionConfig.from_scene_context(ctx)
        mc.grasp.grasp_height_margin_m = margin
        mc.collision.finger_overlap_tolerance_m = tol
        ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                              panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
        gen = PrivilegedTopDownGraspGenerator(mc.grasp)
        cands = gen.generate(sc, [body])
        ok, blocker = 0, ""
        for c in cands:
            if (c.gripper_width or 9) > gen.gripper_max_opening_m:
                continue
            sol = ik.solve(sc.clone(), c.grasp_pose_world[:3], c.grasp_pose_world[3:], seed_q=q0)
            if not sol.success:
                continue
            r = ck.check(np.asarray(sol.q, float), phase=CollisionPhase.TERMINAL_APPROACH,
                         target_object_body=body)
            if not r.in_collision:
                ok += 1
            elif not blocker and r.contacts:
                blocker = f"{r.contacts[0].geom1}<->{r.contacts[0].geom2}"
        print(f"{margin:7.3f} {tol:6.3f} {ok:>9}   {blocker}")
