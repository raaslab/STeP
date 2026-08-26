from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .io import load_json, save_json
from .system_two.stl_plan import print_formulas, validate_stl_plan


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
    summary = executor.execute(plan, outdir=outdir)
    save_json(summary, outdir / "execution_summary.json")
    print(f"[step] execution summary: {outdir / 'execution_summary.json'}")
    if args.save_video:
        print(f"[step] execution video: {args.save_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
