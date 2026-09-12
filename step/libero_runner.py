from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .io import load_json, save_json
from .system_two.stl_plan import normalize_legacy_pregrasp_grasp, print_formulas, validate_stl_plan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run minimal STeP on a LIBERO task.")
    p.add_argument("--legacy-vlm-stl-root", default="../vlm-stl")
    p.add_argument("--base-scene-context", default="configs/libero_template.json")
    p.add_argument("--schema", default="schemas/stl_plan_schema.json")
    p.add_argument("--suite", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--init-state-id", type=int, default=None)
    p.add_argument("--bddl-file", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--outdir", default="outputs/libero")
    p.add_argument("--model", default=None)
    p.add_argument("--compiled-plan", default=None, help="Execute an existing direct STL plan JSON.")
    p.add_argument("--dry-run", action="store_true", help="Stop after writing scene/prompt/plan files.")
    p.add_argument("--render", action="store_true")
    p.add_argument("--save-video", default=None, help="Optional path for an execution video, e.g. outputs/libero/pick.mp4.")
    p.add_argument(
        "--max-pipeline-retries",
        type=int,
        default=3,
        help=(
            "On a failed execution (crash or success=False), recall System 2 for a "
            "new plan and retry, this many times total -- WITHOUT resetting the "
            "env: whatever the failure left behind (e.g. a dropped object sitting "
            "wherever it landed) is exactly what the next attempt starts from and "
            "what the recalled plan is generated against, using a freshly captured "
            "image/object-position snapshot of the live scene, not the original "
            "one. Skipped when --compiled-plan is given (there is no System 2 to "
            "recall in that mode); the same fixed plan is simply re-run against the "
            "same live, unreset env instead. Pass 1 to disable."
        ),
    )
    return p.parse_args()


def _add_legacy_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    for candidate in (root / "LIBERO", root):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return root


def _configure_libero_paths(legacy_root: Path, outdir: Path) -> None:
    benchmark_root = legacy_root / "LIBERO" / "libero" / "libero"
    if not benchmark_root.is_dir():
        return

    config_dir = outdir / "libero_config"
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
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))


def _resolve_prompt(args: argparse.Namespace, scene_context: dict[str, Any]) -> str:
    if args.prompt:
        return args.prompt.strip()

    try:
        from step.system_one.libero_scene import load_libero_task_from_scene_context

        task = load_libero_task_from_scene_context(scene_context)
        if getattr(task, "language", None):
            return str(task.language)
    except Exception:
        pass

    return "Execute the LIBERO manipulation task."


def _prepare_libero_scene(args: argparse.Namespace, outdir: Path) -> tuple[dict[str, Any], Path, str, Path]:
    scene_context = load_json(args.base_scene_context)
    platform = scene_context.setdefault("platform", {})
    platform["backend"] = "libero"
    libero = platform.setdefault("libero", {})
    if args.suite is not None:
        libero["suite"] = args.suite
    if args.task_id is not None:
        libero["task_id"] = int(args.task_id)
    if args.init_state_id is not None:
        libero["init_state_id"] = int(args.init_state_id)
    if args.bddl_file is not None:
        libero["bddl_file"] = args.bddl_file

    try:
        from step.system_one.libero_env import LiberoManipulationEnv

        scene_context = LiberoManipulationEnv.snapshot_scene_context(scene_context)
    except Exception as exc:
        print(f"[step] warning: LIBERO scene snapshot failed: {exc}")

    prompt = _resolve_prompt(args, scene_context)
    outdir.mkdir(parents=True, exist_ok=True)
    scene_path = outdir / "libero_scene_context.json"
    prompt_path = outdir / "prompt.txt"
    save_json(scene_context, scene_path)
    prompt_path.write_text(prompt)
    return scene_context, scene_path, prompt, prompt_path


def _capture_image(args: argparse.Namespace, scene_context: dict[str, Any], outdir: Path) -> str:
    if args.image:
        return args.image

    from step.system_one.libero_env import LiberoManipulationEnv

    capture_path = outdir / "system2_scene.png"
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    return LiberoManipulationEnv.capture_initial_scene_image(
        save_path=str(capture_path),
        scene_context=scene_context,
        camera_name="agentview",
    )


def _capture_live_scene_and_image(env: Any, scene_context: dict[str, Any], outdir: Path, attempt: int) -> tuple[dict[str, Any], str]:
    """Snapshot the *live, unreset* env's current object positions and a
    fresh camera frame -- for recalling System 2 mid-run against wherever
    the scene actually is after a failure (e.g. a dropped can still sitting
    where it fell), not the original init-state layout.

    Mirrors ``LiberoManipulationEnv.snapshot_scene_context``/
    ``capture_initial_scene_image``, which do the same thing but always
    build and reset a brand new env -- exactly what must NOT happen here.
    """
    from step.system_one.libero_scene import infer_objects_from_mujoco, infer_objects_from_obs, update_scene_context_from_objects

    obs = getattr(env, "last_raw_obs", None)
    objects = (infer_objects_from_obs(obs) if obs is not None else {}) or infer_objects_from_mujoco(env.env)
    fresh_context = update_scene_context_from_objects(scene_context, objects, source="pipeline_recall_live")

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.image as mpimg

    image_path = outdir / f"recall_attempt_{attempt:02d}_scene.png"
    frame = env.render_rgb_frame(camera_name="agentview", width=512, height=512)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(str(image_path), frame)
    return fresh_context, str(image_path)


def _annotate_already_satisfied_objects(
    fresh_context: dict[str, Any], env: Any, previous_plan: dict[str, Any]
) -> list[str]:
    """Stamp a "goal already reached" note onto any object the previous
    (just-failed) plan's own place subtasks targeted, if it's now actually
    resting inside its named goal container.

    Exists because a pipeline recall only replans after *some* subtask
    failed (e.g. a second can dropped in transport) -- but the LLM building
    the new plan has no way to tell a first object is already done from raw
    coordinates alone, and nothing in the prompt told it to check. Confirmed
    via a real two-object rollout (alphabet soup + tomato sauce -> basket):
    after the tomato sauce's transport failed, the recalled plan re-picked
    the alphabet soup too even though it was already sitting in the basket
    from a prior successful attempt -- it failed a second time for real
    (its post-place pose has no feasible grasp candidate) and burned the
    entire retry on a no-op instead of just finishing the tomato sauce.
    Mirrors ``libero_scene.py``'s ``_annotate_drawer_clearance_conflicts``:
    precompute the geometric fact in code and stamp the verdict directly
    onto the entity, rather than trust the LLM to (re-)derive it itself.
    """
    from .system_one.motion.scene_query import SceneQuery, object_within_container_footprint, resolve_body_name

    subtasks = ((previous_plan.get("system2_plan") or {}).get("subtasks")) or []
    place_targets = [
        ((sub.get("params") or {}).get("object"), (sub.get("params") or {}).get("goal_region"))
        for sub in subtasks
        if sub.get("skill") == "place"
    ]
    if not place_targets:
        return []

    scene = SceneQuery.from_robosuite_sim(env.env.sim)
    entities = fresh_context.get("entities")
    if not isinstance(entities, list):
        return []
    by_name = {e.get("name"): e for e in entities if isinstance(e, dict) and e.get("name")}

    satisfied: list[str] = []
    for object_name, goal_region in place_targets:
        if not object_name or not goal_region:
            continue
        # `goal_region` is not always a named entity string -- rule 6 also
        # allows an explicit point target ({"type": "point", "xy": [...],
        # "z": ..., "radius": ...}, e.g. "the back compartment of the
        # caddy", which has no body of its own to resolve). Confirmed via a
        # real recall on exactly that task: resolve_body_name got the dict
        # itself as its `name` argument and crashed inside mujoco's own
        # mj_name2id (a TypeError, not the KeyError already handled below)
        # rather than just reporting this object's goal as unresolvable.
        if isinstance(goal_region, dict):
            try:
                object_body = resolve_body_name(scene, object_name)
                object_pos, _ = scene.body_pose(object_body)
            except KeyError:
                continue
            xy = goal_region.get("xy")
            z = goal_region.get("z")
            radius = goal_region.get("radius")
            if not isinstance(xy, (list, tuple)) or len(xy) != 2 or z is None or radius is None:
                continue
            dx, dy = object_pos[0] - float(xy[0]), object_pos[1] - float(xy[1])
            within_xy = (dx * dx + dy * dy) ** 0.5 <= float(radius)
            within_z = abs(object_pos[2] - float(z)) <= 0.10
            if not (within_xy and within_z):
                continue
        elif isinstance(goal_region, str):
            try:
                object_body = resolve_body_name(scene, object_name)
                container_body = resolve_body_name(scene, goal_region)
            except KeyError:
                continue
            if object_within_container_footprint(scene, object_body, container_body) is not True:
                continue
        else:
            continue
        satisfied.append(object_name)
        goal_desc = (
            goal_region
            if isinstance(goal_region, str)
            else f"point(xy={goal_region.get('xy')}, z={goal_region.get('z')}, r={goal_region.get('radius')})"
        )
        flag = (
            f" [Precomputed: {object_name} already satisfies In({object_name}, {goal_desc}) -- "
            "its goal is already reached. Do NOT emit any pick/transport/place subtasks for it; "
            "plan only for the remaining, not-yet-satisfied objects/subtasks.]"
        )
        entity = by_name.get(object_name)
        if entity is not None:
            entity["notes"] = entity.get("notes", "") + flag
        else:
            entities.append({"name": object_name, "role": "object", "notes": flag.strip()})
    return satisfied


def _load_or_create_plan(
    args: argparse.Namespace,
    *,
    scene_context: dict[str, Any],
    prompt: str,
    outdir: Path,
) -> dict[str, Any]:
    if args.compiled_plan:
        plan = load_json(args.compiled_plan)
        save_json(plan, outdir / "compiled_stl_plan.json")
        return plan

    schema = load_json(args.schema)
    image_path = _capture_image(args, scene_context, outdir)
    print(f"[step] System 2 image: {image_path}")

    from .system_two.planner import DirectSTLPlanner

    planner = DirectSTLPlanner(schema=schema, scene_context=scene_context, model=args.model)
    plan, raw_text = planner.plan(instruction=prompt, image_path=image_path)
    save_json(plan, outdir / "direct_stl_plan.json")
    save_json(plan, outdir / "compiled_stl_plan.json")
    (outdir / "raw_model_output.txt").write_text(raw_text)
    return plan


def main() -> int:
    args = parse_args()
    legacy_root = _add_legacy_root(args.legacy_vlm_stl_root)
    outdir = Path(args.outdir)
    _configure_libero_paths(legacy_root, outdir)

    scene_context, scene_path, prompt, prompt_path = _prepare_libero_scene(args, outdir)
    print(f"[step] scene context: {scene_path}")
    print(f"[step] prompt: {prompt}")
    print(f"[step] prompt file: {prompt_path}")

    plan = _load_or_create_plan(args, scene_context=scene_context, prompt=prompt, outdir=outdir)
    if isinstance(plan.get("system2_plan"), dict) and isinstance(plan["system2_plan"].get("subtasks"), list):
        plan["system2_plan"]["subtasks"] = normalize_legacy_pregrasp_grasp(plan["system2_plan"]["subtasks"])
    result = validate_stl_plan(plan)
    if not result.ok:
        (outdir / "validation_errors.txt").write_text("\n".join(result.errors))
        print(f"[step] direct STL plan validation failed: {outdir / 'validation_errors.txt'}", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print_formulas(plan)
    if args.dry_run:
        return 0

    from step.system_one.executor import System1Executor

    executor = System1Executor(
        scene_context=scene_context,
        render=bool(args.render),
        video_path=args.save_video,
        video_prompt=prompt,
        original_prompt=prompt,
    )

    # Build the env ONCE and never reset it on a retry: a dropped object
    # mid-transport leaves the scene in a real, physical state (the can
    # sitting wherever it landed, the arm wherever it ended up) -- resetting
    # would throw that away and start over from the original init-state
    # layout, which is not a "retry" of the failure at all, just a
    # different roll of the dice. On failure, recall System 2 with a fresh
    # image/object-position snapshot of the CURRENT (still-unreset) scene,
    # so the new plan is written against reality -- e.g. "pick up the can"
    # from wherever it actually is now, not its original spot. This is the
    # practical version of System-2 recall: the real mid-execution recall
    # path in recall_controller.py is a stubbed NotImplementedError, never
    # finished when the codebase moved to direct STL plans.
    subtasks = plan["system2_plan"]["subtasks"]
    runtime = executor._extract_runtime_config(subtasks)
    runtime["_computed_env_horizon"] = executor._computed_env_horizon(len(subtasks))
    env = executor._make_env(runtime, debug_dir=outdir)
    max_attempts = max(1, int(args.max_pipeline_retries))
    summary: dict[str, Any] = {}
    try:
        env.reset()
        for attempt in range(1, max_attempts + 1):
            is_last_attempt = attempt == max_attempts
            if attempt > 1:
                if args.compiled_plan:
                    print(f"[step] attempt {attempt}/{max_attempts}: re-running the same --compiled-plan, env NOT reset")
                else:
                    print(f"[step] attempt {attempt}/{max_attempts}: recalling System 2 from the current (unreset) scene")
                    fresh_context, image_path = _capture_live_scene_and_image(env, scene_context, outdir, attempt)
                    already_satisfied = _annotate_already_satisfied_objects(fresh_context, env, plan)
                    if already_satisfied:
                        print(f"[step] attempt {attempt}/{max_attempts}: already at goal, excluding from replan: {already_satisfied}")
                    schema = load_json(args.schema)
                    from .system_two.planner import DirectSTLPlanner

                    planner = DirectSTLPlanner(schema=schema, scene_context=fresh_context, model=args.model)
                    plan, raw_text = planner.plan(instruction=prompt, image_path=image_path)
                    save_json(plan, outdir / f"recall_attempt_{attempt:02d}_plan.json")
                    (outdir / f"recall_attempt_{attempt:02d}_raw_model_output.txt").write_text(raw_text)
                    if isinstance(plan.get("system2_plan"), dict) and isinstance(plan["system2_plan"].get("subtasks"), list):
                        plan["system2_plan"]["subtasks"] = normalize_legacy_pregrasp_grasp(plan["system2_plan"]["subtasks"])
                    retry_validation = validate_stl_plan(plan)
                    if not retry_validation.ok:
                        summary = {"success": False, "attempt": attempt, "error": "plan_validation_failed", "errors": retry_validation.errors}
                        print(f"[step] attempt {attempt}/{max_attempts}: recalled plan failed validation: {retry_validation.errors}", file=sys.stderr)
                        continue
                    print_formulas(plan)
                    scene_context = fresh_context  # later recalls build on the latest known state, not the original

            try:
                summary = executor.execute_on_env(
                    env, plan, outdir=outdir, reset_first=False, capture_initial_video=(attempt == 1)
                )
            except Exception as exc:
                summary = {"success": False, "attempt": attempt, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[step] attempt {attempt}/{max_attempts} raised {type(exc).__name__}: {exc}", file=sys.stderr)

            if summary.get("success"):
                if attempt > 1:
                    print(f"[step] succeeded on attempt {attempt}/{max_attempts}")
                break
            if not is_last_attempt:
                print(f"[step] attempt {attempt}/{max_attempts} failed (success={summary.get('success')!r}); recalling and retrying without resetting the env")
    finally:
        executor._close_video_writer()
        env.close()

    summary["pipeline_attempts_used"] = attempt
    save_json(summary, outdir / "execution_summary.json")
    print(f"[step] execution summary: {outdir / 'execution_summary.json'}")
    if args.save_video:
        print(f"[step] execution video: {args.save_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
