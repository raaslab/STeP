"""Smoke-test the RoboCasa adapter against the contract the executor uses."""
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
from constrained_task import make_env, build_constrained_context
from robocasa_env import RoboCasaManipulationEnv

raw = make_env()
ctx = build_constrained_context(raw, enforce=True)
env = RoboCasaManipulationEnv(scene_context=ctx, env=raw)
obs = env._reset_backend()
print("reset ok, obs keys:", len(obs))
print("eef_pos      :", np.round(env._eef_pos(obs), 3))
print("eef_rotmat   : shape", env._eef_rotmat(obs).shape)
print("cube_pos(obj):", np.round(env._cube_pos(obs), 3))
print("is_grasped   :", env._is_grasped())
print("is_success   :", env._is_success())
print("action_dim   :", raw.action_dim)
for nm in ("obj", "eggplant", "pan", "stove_main_group"):
    p = env._object_pos_by_name(nm)
    print(f"  lookup {nm:<18}", "None" if p is None else np.round(p, 3))
env.set_active_object("eggplant")
print("active handle after set('eggplant'):", env._active_handle)
a = np.zeros(raw.action_dim, dtype=np.float32); a[-1] = -1.0
o, r, d, i = env._step_backend(a[:7])   # short action on purpose -> padded
print("step with short action ok, done =", d)
print("eef after step:", np.round(env._eef_pos(o), 3))
img = env.render_rgb_frame(width=320, height=240)
print("render frame:", img.shape, img.dtype)
