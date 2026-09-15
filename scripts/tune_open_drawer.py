"""Standalone repro for tuning OpenDrawerSkill's approach-search grid against
the toy drawer-only scene (configs/toy_drawer_only.bddl -- just the cabinet,
no other clutter). Baseline (this project's own measured number, see
libero_open_drawer_adapter.py's comment history) is ~3/8-3/10 grip success
with the full clutter-routing search grid; the goal here is checking whether
narrowing that grid (favoring precision over clutter-routing robustness,
appropriate for a scene with no clutter at all) improves it.

Usage:
    /home/amisha/Documents/step/cap-x/.venv-libero/bin/python \\
      scripts/tune_open_drawer.py outputs/open_drawer_tune \\
      --tilts 0.0 --offsets 0,0

Prints open_frac/open_frac_post_retreat, handle_grip_verification, and
descent/pull path_efficiency, and saves a video.
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("outdir", nargs="?", default="outputs/open_drawer_tune")
parser.add_argument("--tilts", type=float, nargs="+", default=[0.0], help="approach_tilts_rad")
parser.add_argument("--rolls", type=float, nargs="+", default=[0.0], help="approach_rolls_rad")
parser.add_argument(
    "--offsets",
    type=str,
    nargs="+",
    default=["0,0"],
    help="grasp_offsets_m as 'depth,vert' pairs, e.g. 0,0 -0.02,0.02",
)
parser.add_argument("--n-runs", type=int, default=3, help="repeat this many times to check reliability, not just one lucky seed")
args = parser.parse_args()

offsets = [tuple(float(x) for x in pair.split(",")) for pair in args.offsets]

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
scene_context = json.load(open("outputs/close_drawer_e2e/libero_scene_context.json"))
scene_context.setdefault("platform", {}).setdefault("libero", {})["bddl_file"] = "configs/toy_drawer_only.bddl"
scene_context["platform"]["libero"].pop("suite", None)
scene_context["platform"]["libero"].pop("task_id", None)
scene_context["platform"]["libero"].pop("init_state_id", None)
scene_context["platform"]["libero"].setdefault("motion", {})["drawer"] = {
    "approach_tilts_rad": args.tilts,
    "approach_rolls_rad": args.rolls,
    "grasp_offsets_m": offsets,
}
print(f"config: tilts={args.tilts} rolls={args.rolls} offsets={offsets}")

plan = {
    "system2_plan": {
        "type": "skill_graph", "task_name": "open_top_drawer_only",
        "scene_summary": "Toy scene: only the cabinet, nothing else. Open the top drawer.",
        "subtasks": [{
            "subtask_id": 0, "skill": "open",
            "description": "Open the top drawer.",
            "params": {"object": "top_drawer"},
            "formula": {"type": "eventually", "interval": [0, 150],
                        "child": {"type": "predicate", "name": "Opened", "args": ["top_drawer"]}},
        }],
    }
}

from step.system_one.executor import System1Executor

results = []
for run_idx in range(args.n_runs):
    run_outdir = outdir / f"run{run_idx}"
    run_outdir.mkdir(parents=True, exist_ok=True)
    executor = System1Executor(
        scene_context=scene_context, render=(run_idx == 0),
        video_path=str(run_outdir / "rollout.mp4") if run_idx == 0 else None,
        video_prompt="Open the top drawer.",
        original_prompt="Open the top drawer of the cabinet.",
    )
    subtasks = plan["system2_plan"]["subtasks"]
    runtime = executor._extract_runtime_config(subtasks)
    runtime["_computed_env_horizon"] = executor._computed_env_horizon(len(subtasks))
    env = executor._make_env(runtime, debug_dir=run_outdir)
    env.reset()

    try:
        summary = executor.execute_on_env(env, plan, outdir=run_outdir, reset_first=False, capture_initial_video=(run_idx == 0))
    finally:
        executor._close_video_writer()
        env.close()

    json.dump(summary, open(run_outdir / "execution_summary.json", "w"), indent=1, default=str)
    info = summary["records"][0]["info"]
    dg = info.get("diagnostics", {}) or {}
    ov = dg.get("open_verification", {})
    mt = dg.get("motion_trace", {})
    row = {
        "run": run_idx,
        "success": info.get("success"),
        "gripped": (dg.get("handle_grip_verification") or {}).get("gripped"),
        "open_frac": ov.get("open_frac"),
        "open_frac_post_retreat": ov.get("open_frac_post_retreat"),
        "descent_path_efficiency": (mt.get("descent") or {}).get("path_efficiency"),
        "descent_path_length_m": (mt.get("descent") or {}).get("actual_path_length_m"),
        "detail": dg.get("detail"),
    }
    results.append(row)
    print(f"run {run_idx}: {row}")

print("\n=== SUMMARY ===")
n_gripped = sum(1 for r in results if r["gripped"])
n_open = sum(1 for r in results if r["success"])
print(f"gripped: {n_gripped}/{len(results)} | success (open_frac>=0.8): {n_open}/{len(results)}")
avg_eff = sum(r["descent_path_efficiency"] or 0 for r in results) / len(results)
print(f"avg descent path_efficiency: {avg_eff:.3f}")
print(f"video (run 0 only): {outdir / 'run0' / 'rollout.mp4'}")
