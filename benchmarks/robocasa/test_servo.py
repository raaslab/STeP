"""Does the arm actually move where we command it?

A full rollout takes 30 minutes to tell us this. Instead: command a known
world-frame Cartesian delta and measure where the end effector actually goes.
If the base-frame rotation fix is right, commanding +x should move +x.
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
from robocasa_env import RoboCasaManipulationEnv

raw = make_env()
env = RoboCasaManipulationEnv(scene_context=build_constrained_context(raw, enforce=True), env=raw)
env.reset()

def probe(axis, name, rotate):
    env.reset()
    start = env._eef_pos(env._last_obs).copy()
    a = np.zeros(7, dtype=np.float32); a[axis] = 0.5; a[-1] = -1.0
    env._rotate_override = rotate
    for _ in range(40):
        env._step_backend(a)
    end = env._eef_pos(env._last_obs)
    d = end - start
    print(f"  commanded world {name}: moved {np.round(d,3)}  "
          f"|dominant axis = {'xyz'[int(np.argmax(np.abs(d)))]}|")
    return d

# with the fix active
print("WITH base-frame rotation:")
for ax, nm in ((0, "+x"), (1, "+y")):
    probe(ax, nm, True)

# disable the rotation to show the difference
orig = RoboCasaManipulationEnv._base_yaw_rotation
RoboCasaManipulationEnv._base_yaw_rotation = lambda self: None
print("WITHOUT rotation (old behaviour):")
for ax, nm in ((0, "+x"), (1, "+y")):
    probe(ax, nm, False)
RoboCasaManipulationEnv._base_yaw_rotation = orig
