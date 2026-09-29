"""Does a different IK seed find an elbow posture that clears the furniture?

A 7-DoF arm reaches the same gripper pose with many arm postures. Our IK solves
from one seed (the current configuration) and takes what it gets; if that
posture puts link5/link6 into an overhead cabinet, the candidate is rejected
even though a clear posture exists. Randomized seeds is what
GraspSampleConfig.pose_selection_attempts already does for the standoff -- this
checks whether it helps at the grasp pose too.
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
import numpy as np, robosuite, robocasa
from robosuite.controllers import load_composite_controller_config
from precheck_tasks import motion_config_for
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.skills.grasp_generator import PrivilegedTopDownGraspGenerator
from step.system_one.motion import panda

rng = np.random.default_rng(0)
print(f"{'task':<28} {'1 seed':>7} {'12 seeds':>9}")
for task in ("PickPlaceCounterToCabinet", "PickPlaceToasterToCounter", "PickPlaceCounterToStove"):
    cfg = load_composite_controller_config(robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cfg,
                         has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, layout_ids=[1], style_ids=[1], seed=0)
    env.reset()
    sc = SceneQuery.from_robosuite_sim(env.sim)
    mc = motion_config_for(env)
    ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                          panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
    q0 = sc.get_qpos(panda.JOINT_NAMES)
    lo, hi = sc.joint_qpos_range(panda.JOINT_NAMES)
    body = env.sim.model.body(env.obj_body_id["obj"]).name
    gen = PrivilegedTopDownGraspGenerator(mc.grasp)
    ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
    maxw = gen.gripper_max_opening_m
    free1 = freeN = 0
    for c in gen.generate(sc, [body]):
        if c.gripper_width is not None and c.gripper_width > maxw:
            continue
        seeds = [q0] + [np.clip(rng.uniform(lo, hi), lo, hi) for _ in range(11)]
        for i, sd in enumerate(seeds):
            sol = ik.solve(sc.clone(), c.grasp_pose_world[:3], c.grasp_pose_world[3:], seed_q=sd)
            if not sol.success:
                continue
            r = ck.check(np.asarray(sol.q, float), phase=CollisionPhase.TERMINAL_APPROACH,
                         target_object_body=body)
            if not r.in_collision:
                if i == 0:
                    free1 += 1
                freeN += 1
                break
    print(f"{task:<28} {free1:>7} {freeN:>9}")
    env.close()
