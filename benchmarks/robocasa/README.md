# RoboCasa backend for STeP

Runs the STeP pipeline (System 2 -> STL -> System 1 motion) inside RoboCasa's
kitchens, with a keep-out constraint over a lit stove burner.

RoboCasa is a different world from LIBERO in the ways that matter to System 1:
a mobile-base PandaOmron whose arm controller reads deltas in the base frame, a
counter instead of a table, real fixtures to read geometry from, and an object
sampled fresh from 200+ meshes every episode. Nothing here modifies the
environment -- everything is read from privileged simulator state.

## Layout

| file | what it is |
|---|---|
| `constrained_task.py` | the lit-burner keep-out task, and the seeded env it runs in |
| `robocasa_scene.py` | scene context: entities, geometry, motion config, grasp routing |
| `robocasa_env.py` | `BaseManipulationEnv` adapter; registers the `robocasa` backend |
| `robocasa_fixtures.py` | surface height, workspace bounds and base yaw, read from fixtures |
| `run_rollout.py` | one rollout, in `libero_runner`'s order (plan, then runtime, then env) |
| `mimicgen_grasps.py` | mines object-relative grasps from RoboCasa's human demos |
| `demo_grasp_generator.py` | serves those grasps, filtered to the scene's exact mesh |
| `grasp_offsets/` | the mined cache: 504 offsets over 212 meshes, 5 pick-place tasks |
| `precheck_*.py` | grasp feasibility without a rollout (seconds, not minutes) |

## Running

Needs the RoboCasa venv (robosuite 1.5.2, mujoco 3.3.1), not STeP's LIBERO one.

```
cd benchmarks/robocasa
MUJOCO_GL=cgl python run_rollout.py runs/out
```

Paths resolve relative to this directory. `STEP_ROOT` overrides the STeP
checkout, `ROBOCASA_DATASETS` the demonstration dataset.

## Two things worth knowing before trusting a number

**Seed the episode.** RoboCasa samples a new object mesh and placement every
episode. The scene context records its seed and `make_robocasa_env` rebuilds
that episode; without it the planner and the simulator hold different objects
(measured: context said `eggplant_0`, the executor built `garlic_1`, then
`LemonWedge004` a metre away). Every rollout taken before that was fixed is void.

**A demo grasp transfers within one mesh, not one category.** `eggplant` ships
as `eggplant_0/_1/_2` with different body origins, so the routing carries the
mesh (`demo:<task>:<mesh>`). Coverage is the limit of this source: 118 of the
212 meshes have a single demonstrated grasp.

## Where it stands

Contact-GraspNet is the grasp source (`routing: graspnet_first`). Its
closer-look stage finds ~32 feasible 6-DoF candidates where the top-down
heuristic finds none -- an eggplant's AABB reads 8.1-12.3 cm at every yaw
against an 8 cm gripper, while every collision-free CGN candidate comes in
tilted 9-73 degrees off vertical. Planning and motion work. The grasp itself
does not yet hold: the arm reaches to ~3 cm, closes, and the object does not
lift.
