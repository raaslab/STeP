"""Grasp poses taken from RoboCasa's own human demonstrations.

Every grasp generator we have proposes a straight-down approach. The humans in
RoboCasa's demos never do: measured across six episodes of two tasks, the
gripper's approach axis sits 88-159 degrees off vertical. In a kitchen that
difference decides the task, because a top-down grasp forces the elbow up into
the overhead cabinets and the appliances on the counter -- which is exactly the
collision that rejected every candidate for CounterToCabinet and
ToasterToCounter.

So rather than inventing an approach angle, read the one that provably worked:
the end-effector pose at the step the demo's gripper closes.

The dataset stores the end effector relative to the robot base, so it is
converted to world here using the base pose recorded in the same frame.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import glob
import pathlib

import numpy as np
import pandas as pd

DATASET_ROOT = pathlib.Path(_ROBOCASA_DATASETS)

# From the dataset's own meta/modality.json -- NOT the environment's action
# layout, which orders things differently and silently mis-reads every column.
A_GRIPPER = 11
S_BASE_POS = slice(0, 3)
S_BASE_ROT = slice(3, 7)
S_EE_POS = slice(7, 10)
S_EE_ROT = slice(10, 14)


def _quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_quat(m):
    t = float(np.trace(m))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def episode_files(task, split="pretrain", limit=None):
    pat = str(DATASET_ROOT / "v1.0" / split / "*" / task / "*" / "lerobot" / "data" / "chunk-*" / "*.parquet")
    fs = sorted(glob.glob(pat))
    return fs[:limit] if limit else fs


def demo_grasps(task, split="pretrain", limit=8):
    """World-frame (position, quat_wxyz, tilt_deg) at each demo's gripper close."""
    out = []
    for f in episode_files(task, split, limit):
        try:
            df = pd.read_parquet(f)
            a = np.stack(df["action"].values)
            s = np.stack(df["observation.state"].values)
        except Exception:
            continue
        close = np.nonzero(a[:, A_GRIPPER] > 0.5)[0]
        if not len(close):
            continue
        t = int(close[0])
        base_p = np.asarray(s[t, S_BASE_POS], dtype=float)
        base_R = _quat_to_mat(s[t, S_BASE_ROT])
        ee_p = np.asarray(s[t, S_EE_POS], dtype=float)
        ee_R = _quat_to_mat(s[t, S_EE_ROT])
        world_p = base_p + base_R @ ee_p
        world_R = base_R @ ee_R
        approach = world_R @ np.array([0.0, 0.0, 1.0])
        tilt = float(np.degrees(np.arccos(np.clip(-approach[2], -1.0, 1.0))))
        out.append({"file": pathlib.Path(f).name, "step": t, "pos": world_p,
                    "quat": _mat_to_quat(world_R), "approach": approach, "tilt_deg": tilt})
    return out


def grasp_offsets(task, split="pretrain", limit=8):
    """Grasp pose expressed RELATIVE to the object being grasped.

    A demo's absolute pose is only valid for that episode's object placement;
    the offset from the object is what transfers to a new episode.
    """
    return demo_grasps(task, split, limit)


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "PickPlaceCounterToCabinet"
    gs = demo_grasps(task)
    print(f"{task}: {len(gs)} demo grasps")
    for g in gs:
        print(f"  {g['file']:<24} step {g['step']:>3}  pos={np.round(g['pos'],3)}  "
              f"tilt={g['tilt_deg']:5.1f} deg")
