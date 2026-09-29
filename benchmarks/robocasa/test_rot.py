"""Does a commanded rotation produce the rotation we asked for?

The position channel was verified by measurement (command +x, EE moves +x). The
orientation channel never was. The controller declares input_ref_frame "base"
for BOTH, and the base is yawed 90 degrees -- so if the orientation delta is
being interpreted in the wrong frame, the wrist spins instead of turning, which
is what a 360-degree wander looks like.
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

def rotvec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-6:
        return np.zeros(3), 0.0
    ax = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(ang))
    return ax, float(np.degrees(ang))

raw = make_env()
env = RoboCasaManipulationEnv(scene_context=build_constrained_context(raw, enforce=True), env=raw)

for axis, name in ((3, "roll  (world +x)"), (4, "pitch (world +y)"), (5, "yaw   (world +z)")):
    env.reset()
    R0 = env._eef_rotmat(env._last_obs).copy()
    a = np.zeros(7, dtype=np.float32); a[axis] = 0.4; a[-1] = -1.0
    for _ in range(30):
        env._step_backend(a)
    R1 = env._eef_rotmat(env._last_obs)
    ax, ang = rotvec(R1 @ R0.T)
    want = np.zeros(3); want[axis-3] = 1.0
    align = float(np.dot(ax, want))
    print(f"{name}: rotated {ang:6.1f} deg about {np.round(ax,2)}  "
          f"alignment with commanded axis = {align:+.2f}  {'OK' if align > 0.8 else 'WRONG'}")
