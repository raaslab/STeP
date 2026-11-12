"""Do STeP's robosuite-1.4 name constants still hold on RoboCasa (robosuite 1.5)?

Everything in step/system_one/motion/panda.py is a literal MuJoCo name. If those
names changed between 1.4 and 1.5, the planner, the collision checker and the MPC
all silently address nothing, so this is the first thing to establish.
"""
import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa

STEP_NAMES = {
    "joints": [f"robot0_joint{i}" for i in range(1, 8)],
    "fingers": ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"],
    "ee_site": "gripper0_right_grip_site",
    "hand_site": "gripper0_right_ft_frame",
}

cfg = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(
    env_name="PickPlaceCounterToStove", robots="PandaOmron", controller_configs=cfg,
    has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    layout_ids=[1], style_ids=[1],
)
env.reset()
m = env.sim.model
jnts = {m.joint(i).name for i in range(m.njnt)}
sites = {m.site(i).name for i in range(m.nsite)}
bodies = {m.body(i).name for i in range(m.nbody)}

print("=== STeP name constants vs RoboCasa/robosuite 1.5 ===")
for n in STEP_NAMES["joints"]:
    print(f"  {n:<34} {'OK' if n in jnts else 'MISSING'}")
for n in STEP_NAMES["fingers"]:
    print(f"  {n:<34} {'OK' if n in jnts else 'MISSING'}")
for k in ("ee_site", "hand_site"):
    n = STEP_NAMES[k]
    print(f"  {n:<34} {'OK' if n in sites else 'MISSING'}")
print()
print("base/mobile joints:", sorted(j for j in jnts if "base" in j or "omron" in j.lower())[:8])
print("task language     :", env.get_ep_meta().get("lang"))
print("n objects         :", len(getattr(env, "objects", {})))
print("object names      :", list(getattr(env, "objects", {}).keys())[:8])
