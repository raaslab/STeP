"""Standalone repro: pick the bowl, then transport it to 'table' (the
"move it out of the way before opening the drawer" subtask), for
libero_goal task_id=3. Isolates the transport leg to see exactly why the
settle loop stalls near-zero progress (position_error_m ~= planned path
length) instead of reaching the far hover target.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

outdir = Path("outputs/transport_table_tune")
outdir.mkdir(parents=True, exist_ok=True)

plan = {
    "system2_plan": {
        "type": "skill_graph", "task_name": "pick_and_move_bowl_to_table",
        "scene_summary": "Move the bowl to the table, out of the way.",
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
                "description": "Move the bowl to the table.",
                "params": {"object": "akita_black_bowl_1", "goal": "table"},
                "formula": {"type": "eventually", "interval": [90, 270],
                            "child": {"type": "predicate", "name": "OverGoal", "args": ["akita_black_bowl_1", "table"]}},
            },
        ],
    }
}

from libero.benchmark import get_benchmark
from step.system_one.executor import System1Executor

scene_context = json.load(open("outputs/graspnet_llm_plan/libero_scene_context.json"))
print("suite/task:", scene_context["platform"]["libero"].get("suite"), scene_context["platform"]["libero"].get("task_id"))

executor = System1Executor(
    scene_context=scene_context, render=True,
    video_path=str(outdir / "rollout.mp4"),
    video_prompt="Move the bowl to the table.",
    original_prompt="Move the bowl to the table.",
)
subtasks = plan["system2_plan"]["subtasks"]
runtime = executor._extract_runtime_config(subtasks)
runtime["_computed_env_horizon"] = executor._computed_env_horizon(len(subtasks))
env = executor._make_env(runtime, debug_dir=outdir)
env.reset()

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
    dg = info.get("diagnostics", {}) or {}
    print("detail:", dg.get("detail"))
    if r["skill"] == "transport":
        print("hover_hand_pos:", dg.get("hover_hand_pos"))
        print("planned_hand_path_m:", dg.get("planned_hand_path_m"))
        print("settle:", dg.get("settle"))
        print("plan_solve_status:", dg.get("plan_solve_status"))
print("\nvideo:", outdir / "rollout.mp4")
