"""One STeP rollout in RoboCasa, following libero_runner's sequence exactly.

The order matters and an earlier version of this script got it wrong. LIBERO
plans FIRST, then derives the env's runtime configuration from that plan
(`_extract_runtime_config`: grasp mode, goal region, goal xy/z/radius, carry
height, descend radius, keep-out), and only then builds the env. Building the
env up front with `runtime={}` leaves the executor servoing toward defaults
instead of the task's actual goal, which is what made the arm wander.

Usage: run_rollout.py <outdir> [--no-enforce]
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys, os, json, pathlib
sys.path.insert(0, str(_HERE))
sys.path.insert(0, _STEP_ROOT)
import numpy as np

for line in pathlib.Path(_os.path.join(_STEP_ROOT, ".env")).read_text().splitlines():
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

from step.system_one.env_factory import register_backend
from robocasa_env import make_robocasa_env
from constrained_task import make_env, build_constrained_context
register_backend("robocasa", make_robocasa_env)

outdir = pathlib.Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
enforce = "--no-enforce" not in sys.argv

# --- 1. scene context, from a throwaway env used only to read the episode -----
probe = make_env()
ctx = build_constrained_context(probe, enforce=enforce)
prompt = ctx["platform"]["robocasa"]["constrained_language"]
import imageio.v2 as imageio
img_path = outdir / "system2_scene.png"
imageio.imwrite(img_path, probe.sim.render(width=640, height=480,
                                           camera_name="robot0_frontview")[::-1])
probe.close()
(outdir / "libero_scene_context.json").write_text(json.dumps(ctx, indent=2))
(outdir / "prompt.txt").write_text(prompt)
print("[rollout] enforce_keepout =", enforce)
print("[rollout] prompt:", prompt)

# --- 2. System 2 FIRST (libero_runner order) ---------------------------------
from step.system_two.planner import DirectSTLPlanner
from step.system_two.stl_plan import decompose_and_validate
schema = json.loads(pathlib.Path(_os.path.join(_STEP_ROOT, "schemas", "stl_global_plan_schema.json")).read_text())
plan, raw = DirectSTLPlanner.for_mode("stl_spec", schema=schema, scene_context=ctx).plan(
    instruction=prompt, image_path=img_path)
(outdir / "raw_model_output.txt").write_text(raw)
plan, result = decompose_and_validate(plan)
if plan is None:
    (outdir / "decomposition_errors.txt").write_text("\n".join(result.errors))
    raise SystemExit(f"[rollout] plan failed decomposition: {result.errors}")
(outdir / "direct_stl_plan.json").write_text(json.dumps(plan, indent=2))
subtasks = plan["system2_plan"]["subtasks"]
print(f"[rollout] System 2 -> {len(subtasks)} subtasks:", ", ".join(s["skill"] for s in subtasks))

# --- 3. runtime derived FROM the plan, then env built by the executor ---------
from step.system_one.executor import System1Executor
ex = System1Executor(scene_context=ctx,
                     video_path=str(outdir / "rollout.mp4"),
                     video_prompt=prompt)
runtime = ex._extract_runtime_config(subtasks)
runtime["_computed_env_horizon"] = ex._computed_env_horizon(len(subtasks))
print("[rollout] runtime:", {k: v for k, v in runtime.items() if not k.startswith("_")})
env = ex._make_env(runtime, debug_dir=outdir)
env.reset()

summary = ex.execute_on_env(env, plan, outdir=str(outdir), reset_first=False)
summary["provenance"] = {"model": os.environ.get("OPENAI_MODEL", "gpt-5.4"),
                         "backend": "robocasa", "stl_mode": "stl_spec",
                         "enforce_keepout": enforce}
(outdir / "execution_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("[rollout] success =", summary.get("success"),
      "| subtasks run =", len(summary.get("records") or []))
env.close()
