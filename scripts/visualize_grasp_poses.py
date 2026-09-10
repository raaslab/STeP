"""Render a LIBERO scene with every candidate grasp pose overlaid, for
debugging the grasp generator/PickSkill's candidate selection directly
(e.g. "is the rim-edge candidate actually landing on the bowl's rim?")
instead of only inferring it from position/orientation-error numbers.

Usage:
    python scripts/visualize_grasp_poses.py --suite libero_spatial --task-id 0 \
        --object akita_black_bowl_1_main --out outputs/grasp_viz.png

Markers: a colored dot at each candidate's grasp position, plus a short
line along the gripper's closing axis (the direction the fingers move
together) so a rim grasp's radial straddle is visible, not just a point.
Center-grasp candidates are drawn in blue, rim-edge candidates in orange;
the candidate PickSkill would actually select (via full IK/collision/
planning scoring, not just the generator's own score) is highlighted in
green and labeled "SELECTED" -- pass --score-with-pick-skill (default on)
to run that full scoring pass; pass --no-score-with-pick-skill to skip it
(faster, generator-only view) if OMPL planning isn't needed for what you're
debugging.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

STEP_ROOT = Path(__file__).resolve().parents[1]
VLM_STL_ROOT = (STEP_ROOT / ".." / "vlm-stl").resolve()
for _candidate in (VLM_STL_ROOT / "LIBERO", VLM_STL_ROOT):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
if str(STEP_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP_ROOT))


def _configure_libero_paths() -> None:
    """Point LIBERO at this checkout's asset/bddl/init-state directories via
    a fresh config.yaml -- mirrors ``step/libero_runner.py``'s
    ``_configure_libero_paths``. Needed because LIBERO otherwise falls back
    to its own cached ``~/.libero/config.yaml``, which can point at a stale
    path from before this repo was last moved/renamed."""
    benchmark_root = VLM_STL_ROOT / "LIBERO" / "libero" / "libero"
    if not benchmark_root.is_dir():
        return
    config_dir = STEP_ROOT / "outputs" / "_libero_config_scratch"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"assets: {benchmark_root / 'assets'}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"benchmark_root: {benchmark_root}",
                f"datasets: {benchmark_root.parent / 'datasets'}",
                f"init_states: {benchmark_root / 'init_files'}",
                "",
            ]
        )
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


_configure_libero_paths()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--object", default=None, help="Real MuJoCo body name (e.g. akita_black_bowl_1_main). Defaults to every manipulated/support object in the scene.")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--out", default="outputs/grasp_viz.png")
    parser.add_argument("--score-with-pick-skill", dest="score", action="store_true", default=True)
    parser.add_argument("--no-score-with-pick-skill", dest="score", action="store_false")
    args = parser.parse_args()

    from step.system_one.libero_scene import _import_libero_benchmark, _import_libero_env  # sets MUJOCO_GL (step/__init__.py) before mujoco is ever imported

    import mujoco
    import torch  # noqa: F401  -- LIBERO's benchmark module expects torch importable

    benchmark, get_libero_path = _import_libero_benchmark()

    from step.system_one.motion.collision import CollisionChecker
    from step.system_one.motion.config import MotionConfig
    from step.system_one.motion.skills.grasp_generator import PrivilegedTopDownGraspGenerator
    from step.system_one.motion.skills.grasp_viz import save_grasp_candidates_image
    from step.system_one.motion.kinematics import IKSolver
    from step.system_one.motion.ompl_planner import MotionPlanner
    from step.system_one.motion.panda import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES
    from step.system_one.motion.skills.pick_skill import PickSkill
    from step.system_one.motion.scene_query import SceneQuery

    bm = benchmark.get_benchmark_dict()[args.suite]()
    task = bm.get_task(args.task_id)
    bddl_file = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)

    OffScreenRenderEnv = _import_libero_env()

    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=args.height, camera_widths=args.width, camera_names=args.camera)
    env.reset()
    try:
        init_states = bm.get_task_init_states(args.task_id)
        env.set_init_state(init_states[args.init_state_id])
        zero_action = np.zeros(7, dtype=np.float32)
        for _ in range(15):  # let objects settle onto the table, matching LiberoManipulationEnv._reset_backend
            env.step(zero_action)
    except Exception as exc:  # init states may not exist for every task; fall back to the bare reset
        print(f"[visualize_grasp_poses] could not apply init_state_id={args.init_state_id}: {exc}", file=sys.stderr)

    raw_frame = env.sim.render(camera_name=args.camera, height=args.height, width=args.width)
    frame = np.asarray(raw_frame)[::-1]

    model = env.env.sim.model._model
    data = env.env.sim.data._data

    scene = SceneQuery.from_robosuite_sim(env.env.sim)
    object_names = [args.object] if args.object else _default_object_names(scene, model)
    if not object_names:
        print("[visualize_grasp_poses] no object bodies found to visualize", file=sys.stderr)
        return 1

    gen = PrivilegedTopDownGraspGenerator(MotionConfig().grasp)
    candidates = gen.generate(scene, object_names)
    print(f"[visualize_grasp_poses] {len(candidates)} candidates across {object_names}")

    selected_grasp = None
    if args.score:
        checker = CollisionChecker(scene, JOINT_NAMES, EE_SITE, ROBOT_BODIES, GRIPPER_BODIES, MotionConfig().collision)
        lo, hi = scene.joint_qpos_range(JOINT_NAMES)
        ik = IKSolver(joint_names=JOINT_NAMES, site_name=EE_SITE)
        planner = MotionPlanner(JOINT_NAMES, lo, hi, checker, MotionConfig().planner)
        skill = PickSkill(gen, ik, planner, checker, MotionConfig())
        start_q = scene.get_qpos(JOINT_NAMES)
        result = skill.execute(scene, object_names, start_q, candidates=candidates)
        print(f"[visualize_grasp_poses] PickSkill result: success={result.success} failure_reason={result.failure_reason}")
        if result.success:
            selected_grasp = result.selected_grasp

    out_path = save_grasp_candidates_image(frame, model, data, args.camera, candidates, args.out, selected_grasp)
    print(f"[visualize_grasp_poses] wrote {out_path} ({len(candidates)} candidates, "
          f"{'selected candidate highlighted in green' if selected_grasp is not None else 'no candidate selected/highlighted'})")
    return 0


def _default_object_names(scene, model) -> list[str]:
    import mujoco

    names = []
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if name and name.endswith("_main"):
            names.append(name)
    return names


if __name__ == "__main__":
    raise SystemExit(main())
