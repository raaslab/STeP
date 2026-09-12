"""Standalone repro for tuning CloseSkill's linear strategy against
libero_10 task 3 (KITCHEN_SCENE4: bowl in bottom drawer + close it).

Sets up the scene exactly like the real task would leave it (drawer open,
bowl manually placed inside -- approximating a completed place), then runs
the real `close` skill through the real executor/adapter and saves a video.

The orientation is built from three independent rotations, applied in this
order to the arm's own home orientation:
  1. --yaw   degrees about world +Z      (which world direction the
                                           approach axis points)
  2. --roll  degrees about local +Z      (the approach axis itself -- which
                                           way the fingers point around it)
  3. --tilt  degrees about local +X      (perpendicular to both -- how far
                                           the approach leans off dead-level)
plus:
  --height  meters, offset above (+) or below (-) the handle's own z.

These three rotations, composed in this order, span the full orientation
space, so any approach orientation is reachable by some (yaw, roll, tilt)
triple -- there is no "missing axis".

Usage:
    /home/amisha/Documents/step/cap-x/.venv-libero/bin/python \\
      scripts/tune_close_drawer.py outputs/close_drawer_<label> \\
      --yaw 90 --roll 270 --tilt 0 --height 0.0

Then check the printed open_frac (lower = more closed; LIBERO's own success
check needs qpos > 0.0, which is open_frac <= 0.0588 here) and watch
outputs/close_drawer_<label>/rollout.mp4.
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("outdir", nargs="?", default="outputs/close_drawer_tune")
parser.add_argument("--yaw", type=float, default=90.0, help="degrees about world +Z")
parser.add_argument("--roll", type=float, default=270.0, help="degrees about local +Z (approach axis)")
parser.add_argument("--tilt", type=float, default=0.0, help="degrees about local +X (perpendicular to both)")
parser.add_argument("--height", type=float, default=0.0, help="meters above (+) / below (-) the handle's own z")
args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
scene_context = json.load(open("outputs/close_drawer_e2e/libero_scene_context.json"))

# Inject the CLI-chosen values into scene_context, the real path
# MotionConfig.from_scene_context reads CloseConfig overrides from
# (see step/system_one/motion/config.py's own from_scene_context) --
# no source file edits needed for any of the four values below.
scene_context.setdefault("platform", {}).setdefault("libero", {}).setdefault("motion", {})["close"] = {
    "linear_yaws_deg": [args.yaw],
    "linear_roll_deg": args.roll,
    "linear_tilt_deg": args.tilt,
    "linear_height_offsets_m": [args.height],
}
print(f"config: yaw={args.yaw} roll={args.roll} tilt={args.tilt} height={args.height}")

plan = {
    "system2_plan": {
        "type": "skill_graph", "task_name": "close_bottom_drawer_only",
        "scene_summary": "Bowl manually placed inside the already-open bottom drawer; attempt to close it.",
        "subtasks": [{
            "subtask_id": 0, "skill": "close",
            "description": "Close the bottom drawer.",
            "params": {"object": "bottom_drawer"},
            "formula": {"type": "eventually", "interval": [0, 150],
                        "child": {"type": "predicate", "name": "Closed", "args": ["bottom_drawer"]}},
        }],
    }
}

from step.system_one.executor import System1Executor
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.articulation import resolve_drawer, measure_pull_axis_and_travel

executor = System1Executor(
    scene_context=scene_context, render=True,
    video_path=str(outdir / "rollout.mp4"),
    video_prompt="Close the bottom drawer (bowl already inside).",
    original_prompt="Close the bottom drawer (bowl already inside).",
)
subtasks = plan["system2_plan"]["subtasks"]
runtime = executor._extract_runtime_config(subtasks)
runtime["_computed_env_horizon"] = executor._computed_env_horizon(len(subtasks))
env = executor._make_env(runtime, debug_dir=outdir)
env.reset()
for _ in range(5):
    env.env.step([0.0] * 7)

scene = SceneQuery.from_robosuite_sim(env.env.sim)
drawer = resolve_drawer(scene, "bottom_drawer")
sim = env.env.sim
jid = sim.model.joint_name2id(drawer.joint_name)
qpos_addr = sim.model.jnt_qposadr[jid]
_, _, qpos_closed, qpos_open = measure_pull_axis_and_travel(scene, drawer)
sim.data.qpos[qpos_addr] = qpos_open
bowl_jid = sim.model.joint_name2id("akita_black_bowl_1_joint0")
bowl_qpos_addr = sim.model.jnt_qposadr[bowl_jid]
sim.data.qpos[bowl_qpos_addr:bowl_qpos_addr + 3] = [0.0, 0.15, 0.95]
sim.data.qvel[sim.model.jnt_dofadr[bowl_jid]:sim.model.jnt_dofadr[bowl_jid] + 6] = 0.0
sim.forward()
for _ in range(20):
    env.env.step([0.0] * 7)

scene = SceneQuery.from_robosuite_sim(env.env.sim)
bowl_pos, _ = scene.body_pose("akita_black_bowl_1_main")
print("bowl settled at:", bowl_pos, "| drawer qpos:", scene.get_qpos([drawer.joint_name])[0])

try:
    summary = executor.execute_on_env(env, plan, outdir=outdir, reset_first=False, capture_initial_video=True)
finally:
    executor._close_video_writer()
    env.close()

json.dump(summary, open(outdir / "execution_summary.json", "w"), indent=1, default=str)

info = summary["records"][0]["info"]
dg = info.get("diagnostics", {}) or {}
cv = dg.get("close_verification", {})
print("\n=== RESULT ===")
print("success:", info.get("success"), "| reason:", info.get("failure_reason"))
print("open_frac:", cv.get("open_frac"), "(LIBERO counts it closed only if qpos > 0.0, i.e. open_frac <= 0.0588 here)")
print("detail:", dg.get("detail"))
print("video:", outdir / "rollout.mp4")
