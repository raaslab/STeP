import json
import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa
from robocasa_scene import build_scene_context

TASK = "PickPlaceCounterToStove"
cfg = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(env_name=TASK, robots="PandaOmron", controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, layout_ids=[1], style_ids=[1], seed=0)
env.reset()
ctx = build_scene_context(env, task_name=TASK, layout_id=1, style_id=1)
json.dump(ctx, open("robocasa_scene_context.json", "w"), indent=2)
print("language :", ctx["platform"]["robocasa"]["language"])
print("entities :", len(ctx["entities"]))
for e in ctx["entities"]:
    if e["role"] != "fixture":
        print(f"   {e['name']:<16} {e['role']:<20} class={e['object_class']}")
fx = [e['name'] for e in ctx["entities"] if e["role"] == "fixture"]
print("fixtures :", len(fx), "->", fx[:8])
wb = ctx["platform"]["workspace_bounds"]
print("workspace bounds:")
for ax in "xyz":
    lo, hi = wb[ax]; print(f"   {ax}: [{lo:+.3f}, {hi:+.3f}]  span {hi-lo:.3f} m")
import numpy as np
from robocasa_scene import robot_base_xyz
print("robot base      :", np.round(robot_base_xyz(env), 3))
for h in ("obj", "obj_container", "container"):
    bid = env.obj_body_id.get(h)
    if bid is not None:
        print(f"   {h:<14}", np.round(env.sim.data.body_xpos[bid], 3))
print("wrote robocasa_scene_context.json")
