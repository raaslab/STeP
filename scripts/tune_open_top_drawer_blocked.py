"""Standalone repro for a "drawer physically blocked partway" scene: force
the bottom drawer of libero_10 task 3's cabinet closed, teleport an existing
prop (wine_bottle_1) into its slide path at some fraction of full travel,
and run the real `open` skill/adapter to see how far it actually gets.

This is step 1 of the few-shot-replanning demo: confirm the physical setup
genuinely caps open_frac partway (via real contact resistance, not a fixed
number) before wiring it into a full System2 recall scenario.

Usage:
    /home/amisha/Documents/step/cap-x/.venv-libero/bin/python \\
      scripts/tune_open_drawer_blocked.py outputs/open_drawer_blocked_tune \\
      --blocker-travel-frac 0.5 --blocker-extra-m 0.0

--blocker-travel-frac: where along the drawer's [closed, open] travel to
    place the front face of the blocker (0=at the closed front face,
    1=at the fully-open front face).
--blocker-extra-m: extra gap (+) or overlap (-) between the drawer's
    computed front face at that fraction and the blocker's own near face,
    for fine-tuning once the rough fraction is in the right ballpark.

Prints open_frac_post_retreat (LIBERO's own open_threshold_frac is 0.8 here
-- below that, `open` itself reports failure) and saves a video.
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("outdir", nargs="?", default="outputs/open_drawer_blocked_tune")
parser.add_argument("--blocker-travel-frac", type=float, default=0.5)
parser.add_argument("--blocker-extra-m", type=float, default=0.0)
parser.add_argument("--blocker-body", default="cream_cheese_1_main")
parser.add_argument("--blocker-joint", default="cream_cheese_1_joint0")
args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
scene_context = json.load(open("outputs/open_top_drawer_blocked_scene/libero_scene_context.json"))

plan = {
    "system2_plan": {
        "type": "skill_graph", "task_name": "open_bottom_drawer_blocked",
        "scene_summary": "The bottom drawer is closed; something may be blocking it from opening fully.",
        "subtasks": [{
            "subtask_id": 0, "skill": "open",
            "description": "Open the bottom drawer.",
            "params": {"object": "top_drawer"},
            "formula": {"type": "eventually", "interval": [0, 150],
                        "child": {"type": "predicate", "name": "Opened", "args": ["top_drawer"]}},
        }],
    }
}

from step.system_one.executor import System1Executor
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.articulation import resolve_drawer, measure_pull_axis_and_travel

executor = System1Executor(
    scene_context=scene_context, render=True,
    video_path=str(outdir / "rollout.mp4"),
    video_prompt="Open the bottom drawer (may be blocked).",
    original_prompt="Open the bottom drawer of the cabinet.",
)
subtasks = plan["system2_plan"]["subtasks"]
runtime = executor._extract_runtime_config(subtasks)
runtime["_computed_env_horizon"] = executor._computed_env_horizon(len(subtasks))
env = executor._make_env(runtime, debug_dir=outdir)
env.reset()
for _ in range(5):
    env.env.step([0.0] * 7)

sim = env.env.sim
scene = SceneQuery.from_robosuite_sim(sim)
drawer = resolve_drawer(scene, "top_drawer")
pull_axis, travel_m, qpos_closed, qpos_open = measure_pull_axis_and_travel(scene, drawer)
print(f"pull_axis={pull_axis} travel_m={travel_m} qpos_closed={qpos_closed} qpos_open={qpos_open}")

jid = sim.model.joint_name2id(drawer.joint_name)
qpos_addr = sim.model.jnt_qposadr[jid]
sim.data.qpos[qpos_addr] = qpos_closed
sim.forward()

# Drawer body position/extent at the forced-closed qpos, to find its own
# front face (the side facing the room, i.e. further along pull_axis from
# the body origin) in world space.
scene = SceneQuery.from_robosuite_sim(sim)
drawer_pos_closed, _ = scene.body_pose(drawer.drawer_body)
extent = scene.body_extent_world(drawer.drawer_body)
half_extent_along_axis = float(np.abs(extent.half_extents) @ np.abs(pull_axis)) if extent is not None else 0.15
front_face_closed = np.asarray(drawer_pos_closed) + pull_axis * half_extent_along_axis

target_travel_m = args.blocker_travel_frac * travel_m
blocker_near_face = front_face_closed + pull_axis * (target_travel_m + args.blocker_extra_m)

# Guess the blocker's own half-extent along pull_axis so we place its
# CENTER, not its near face, at blocker_near_face.
blocker_scene = SceneQuery.from_robosuite_sim(sim)
blocker_extent = blocker_scene.body_extent_world(args.blocker_body)
blocker_half_along_axis = (
    float(np.abs(blocker_extent.half_extents) @ np.abs(pull_axis)) if blocker_extent is not None else 0.03
)
blocker_center = blocker_near_face + pull_axis * blocker_half_along_axis
blocker_center[2] = float(drawer_pos_closed[2])  # same height as the drawer body

print(f"drawer front face (closed): {front_face_closed}")
print(f"placing blocker center at: {blocker_center} (target_travel_m={target_travel_m:.3f} of {travel_m:.3f})")

bjid = sim.model.joint_name2id(args.blocker_joint)
bqadr = sim.model.jnt_qposadr[bjid]
sim.data.qpos[bqadr:bqadr + 3] = blocker_center
sim.data.qvel[sim.model.jnt_dofadr[bjid]:sim.model.jnt_dofadr[bjid] + 6] = 0.0
sim.forward()
for _ in range(20):
    env.env.step([0.0] * 7)

scene = SceneQuery.from_robosuite_sim(sim)
blocker_settled, _ = scene.body_pose(args.blocker_body)
print(f"blocker settled at: {blocker_settled} | drawer qpos: {scene.get_qpos([drawer.joint_name])[0]}")

try:
    summary = executor.execute_on_env(env, plan, outdir=outdir, reset_first=False, capture_initial_video=True)
finally:
    executor._close_video_writer()
    env.close()

json.dump(summary, open(outdir / "execution_summary.json", "w"), indent=1, default=str)

info = summary["records"][0]["info"]
dg = info.get("diagnostics", {}) or {}
ov = dg.get("open_verification", {})
print("\n=== RESULT ===")
print("success:", info.get("success"), "| reason:", info.get("failure_reason"))
print("open_frac:", ov.get("open_frac"), "open_frac_post_retreat:", ov.get("open_frac_post_retreat"))
print("(LIBERO open_threshold_frac here is 0.8 -- below that, `open` itself reports failure)")
print("detail:", dg.get("detail"))
print("video:", outdir / "rollout.mp4")
