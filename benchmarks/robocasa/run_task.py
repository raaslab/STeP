"""One STeP rollout on a RoboCasa task as it ships -- no added constraint.

constrained_task.py bolts a lit-burner keep-out onto PickPlaceCounterToStove to
mirror LIBERO-Constrained. That is a separate experiment. This runs the
benchmark task itself: RoboCasa's own language for the episode, its own objects,
nothing added to the scene and nothing modified in the environment.

    python run_task.py PickPlaceCounterToStove runs/out
    python run_task.py OpenDrawer runs/out --seed 3
"""
import argparse, json, os, pathlib, sys

_HERE = pathlib.Path(__file__).resolve().parent
_STEP_ROOT = os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
sys.path.insert(0, str(_HERE)); sys.path.insert(0, _STEP_ROOT)

for line in (pathlib.Path(_STEP_ROOT) / ".env").read_text().splitlines():
    line = line.strip()
    if line.startswith("#") or "=" not in line:
        continue
    if line.startswith("export "):
        line = line[len("export "):]
    k, v = line.split("=", 1)
    k = k.strip()
    if k == "MUJOCO_GL":          # egl is invalid on macOS
        continue
    os.environ.setdefault(k, v.strip().strip('"').strip("'"))

import robosuite
import robocasa  # noqa: F401  (registers the kitchen environments)
from robosuite.controllers import load_composite_controller_config

from step.system_one.env_factory import register_backend
from robocasa_env import make_robocasa_env
from robocasa_scene import build_scene_context
register_backend("robocasa", make_robocasa_env)

ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("outdir")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--layout", type=int, default=1)
ap.add_argument("--style", type=int, default=1)
args = ap.parse_args()

outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

# --- 1. scene context from a throwaway env, read-only ------------------------
cfg = load_composite_controller_config(robot="PandaOmron")
probe = robosuite.make(env_name=args.task, robots="PandaOmron", controller_configs=cfg,
                       has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
                       camera_names=["robot0_frontview"], camera_heights=480, camera_widths=640,
                       layout_ids=[args.layout], style_ids=[args.style], seed=args.seed)
probe.reset()
ctx = build_scene_context(probe, task_name=args.task, layout_id=args.layout,
                          style_id=args.style, seed=args.seed)

# The task's OWN instruction, exactly as RoboCasa states it for this episode.
prompt = probe.get_ep_meta().get("lang", "").strip()
if not prompt:
    raise SystemExit(f"[task] {args.task} episode has no language")

import imageio.v2 as imageio
img_path = outdir / "system2_scene.png"
imageio.imwrite(img_path, probe.sim.render(width=640, height=480,
                                           camera_name="robot0_frontview")[::-1])
probe.close()
(outdir / "libero_scene_context.json").write_text(json.dumps(ctx, indent=2))
(outdir / "prompt.txt").write_text(prompt)
print(f"[task] {args.task}  seed={args.seed}  layout={args.layout} style={args.style}")
print(f"[task] prompt: {prompt}")

# --- 2. System 2 first, then runtime, then the env (libero_runner's order) ----
from step.system_two.planner import DirectSTLPlanner
from step.system_two.stl_plan import decompose_and_validate
schema = json.loads((pathlib.Path(_STEP_ROOT) / "schemas" / "stl_global_plan_schema.json").read_text())
plan, raw = DirectSTLPlanner.for_mode("stl_spec", schema=schema, scene_context=ctx).plan(
    instruction=prompt, image_path=img_path)
(outdir / "raw_model_output.txt").write_text(raw)
plan, result = decompose_and_validate(plan)
if plan is None:
    (outdir / "decomposition_errors.txt").write_text("\n".join(result.errors))
    raise SystemExit(f"[task] plan failed decomposition: {result.errors}")
(outdir / "direct_stl_plan.json").write_text(json.dumps(plan, indent=2))
subtasks = plan["system2_plan"]["subtasks"]
print(f"[task] System 2 -> {len(subtasks)} subtasks:", ", ".join(s["skill"] for s in subtasks))

from step.system_one.executor import System1Executor
ex = System1Executor(scene_context=ctx, video_path=str(outdir / "rollout.mp4"),
                     video_prompt=prompt)
runtime = ex._extract_runtime_config(subtasks)
runtime["_computed_env_horizon"] = ex._computed_env_horizon(len(subtasks))
print("[task] runtime:", {k: v for k, v in runtime.items() if not k.startswith("_")})
env = ex._make_env(runtime, debug_dir=outdir)
env.reset()

summary = ex.execute_on_env(env, plan, outdir=str(outdir), reset_first=False)
summary["provenance"] = {"model": os.environ.get("OPENAI_MODEL", "gpt-5.4"),
                         "backend": "robocasa", "stl_mode": "stl_spec",
                         "task": args.task, "seed": args.seed, "constrained": False}
(outdir / "execution_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("[task] success =", summary.get("success"),
      "| subtasks run =", len(summary.get("records") or []))
env.close()
