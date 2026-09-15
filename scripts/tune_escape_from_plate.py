"""Deterministic repro: force the bowl into contact with the plate (as if a
prior place subtask had left it there), then attempt transport(bowl,
top_drawer) -- checks whether escape_start_collision's new "away from the
actual obstacle" direction gets it clear, instead of walking further into
the plate via "toward top_drawer's own center" (the previously-only
non-vertical option).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

outdir = Path("outputs/escape_from_plate_tune")
outdir.mkdir(parents=True, exist_ok=True)

plan = {
    "system2_plan": {
        "type": "skill_graph", "task_name": "pick_then_transport_to_drawer",
        "scene_summary": "Pick the bowl (starting in contact with the plate) and move it into the top drawer.",
        "subtasks": [
            {
                "subtask_id": 0, "skill": "pick",
                "description": "Pick the bowl.",
                "params": {"object": "akita_black_bowl_1"},
                "formula": {"type": "and", "children": [
                    {"type": "eventually", "interval": [0, 40],
                     "child": {"type": "predicate", "name": "PreGrasp", "args": ["akita_black_bowl_1", "top"]}},
                    {"type": "eventually", "interval": [0, 90],
                     "child": {"type": "predicate", "name": "Grasped", "args": ["akita_black_bowl_1"]}},
                ]},
            },
            {
                "subtask_id": 1, "skill": "transport",
                "description": "Move the bowl into the top drawer.",
                "params": {"object": "akita_black_bowl_1", "goal": "top_drawer", "place_surface": "inside"},
                "formula": {"type": "eventually", "interval": [90, 270],
                            "child": {"type": "predicate", "name": "OverGoal", "args": ["akita_black_bowl_1", "top_drawer"]}},
            },
        ],
    }
}

from step.system_one.executor import System1Executor
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.articulation import resolve_drawer, measure_pull_axis_and_travel

scene_context = json.load(open("outputs/graspnet_llm_plan/libero_scene_context.json"))

executor = System1Executor(
    scene_context=scene_context, render=True,
    video_path=str(outdir / "rollout.mp4"),
    video_prompt="Pick the bowl and move it into the top drawer.",
    original_prompt="Move the bowl into the top drawer.",
)
subtasks = plan["system2_plan"]["subtasks"]
runtime = executor._extract_runtime_config(subtasks)
runtime["_computed_env_horizon"] = executor._computed_env_horizon(len(subtasks))
env = executor._make_env(runtime, debug_dir=outdir)
env.reset()

# Open the top drawer first (transport target needs it open to be feasible)
sim = env.env.sim
scene = SceneQuery.from_robosuite_sim(sim)
drawer = resolve_drawer(scene, "top_drawer")
_, _, qpos_closed, qpos_open = measure_pull_axis_and_travel(scene, drawer)
jid = sim.model.joint_name2id(drawer.joint_name)
qadr = sim.model.jnt_qposadr[jid]
sim.data.qpos[qadr] = qpos_open
sim.forward()

# Force the bowl to overlap the plate (simulating a prior place landing too
# close), matching the real diagnosed failure's contact geoms
# (plate_1_g2/g3/g4/g9/g10).
plate_bid = sim.model.body_name2id("plate_1_main")
plate_pos = sim.data.body_xpos[plate_bid].copy()
bowl_jid = sim.model.joint_name2id("akita_black_bowl_1_joint0")
bqadr = sim.model.jnt_qposadr[bowl_jid]
sim.data.qpos[bqadr:bqadr + 3] = [plate_pos[0], plate_pos[1], plate_pos[2] + 0.02]
sim.data.qvel[sim.model.jnt_dofadr[bowl_jid]:sim.model.jnt_dofadr[bowl_jid] + 6] = 0.0
sim.forward()
for _ in range(10):
    env.env.step([0.0] * 7)

scene = SceneQuery.from_robosuite_sim(sim)
bowl_pos, _ = scene.body_pose("akita_black_bowl_1_main")
print("bowl forced near plate at:", bowl_pos, "| plate at:", plate_pos)

try:
    summary = executor.execute_on_env(env, plan, outdir=outdir, reset_first=False, capture_initial_video=True)
finally:
    executor._close_video_writer()
    env.close()

json.dump(summary, open(outdir / "execution_summary.json", "w"), indent=1, default=str)

for r in summary["records"]:
    info = r["info"]
    print(f"\n=== {r['skill']} ===")
    print("success:", info.get("success"), "| reason:", info.get("failure_reason"))
    print("detail:", (info.get("diagnostics") or {}).get("detail"))
print("\nvideo:", outdir / "rollout.mp4")
