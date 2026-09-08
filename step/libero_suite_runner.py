from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import load_json, save_json


LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run STeP over a LIBERO benchmark suite.")
    p.add_argument("--legacy-vlm-stl-root", default="../vlm-stl")
    p.add_argument("--base-scene-context", default="configs/libero_template.json")
    p.add_argument("--schema", default="schemas/stl_plan_schema.json")
    p.add_argument("--safety-json", default="configs/libero_safety.json")
    p.add_argument("--suite", default="libero_spatial", choices=LIBERO_SUITES + ("all",))
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-states", type=int, default=1)
    p.add_argument("--state-ids", nargs="+", type=int, default=None)
    p.add_argument("--all-states", action="store_true")
    p.add_argument(
        "--outdir",
        default=None,
        help="Defaults to outputs/<suite>_<timestamp> (e.g. outputs/libero_object_20260907_143000) when omitted.",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--compiled-plan", default=None, help="Debug only: reuse one direct STL plan for every task.")
    p.add_argument("--save-video", default="rollout.mp4", help="Video filename inside each rollout directory.")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--use-safety-prompts", action="store_true", help="Use augmented prompts from configs/libero_safety.json when available.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _add_paths(legacy_root_arg: str | Path, outdir: Path) -> Path:
    legacy_root = Path(legacy_root_arg).expanduser().resolve()
    for candidate in (Path.cwd(), legacy_root / "LIBERO", legacy_root):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    benchmark_root = legacy_root / "LIBERO" / "libero" / "libero"
    if benchmark_root.is_dir():
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
    return legacy_root


def _suite_names(selected: str) -> list[str]:
    if selected == "all":
        return list(LIBERO_SUITES)
    return [selected]


def _load_safety_prompts(path: str | Path) -> dict[tuple[str, int], str]:
    p = Path(path)
    if not p.exists():
        return {}
    data = load_json(p)
    prompts: dict[tuple[str, int], str] = {}
    for task in data.get("tasks", []):
        suite = task.get("suite")
        task_id = task.get("task_id")
        prompt = task.get("augmented_prompt") or task.get("base_prompt")
        if suite is not None and task_id is not None and prompt:
            prompts[(str(suite), int(task_id))] = str(prompt)
    return prompts


def _enumerate_tasks(suites: list[str]) -> list[dict[str, Any]]:
    try:
        from libero.benchmark import get_benchmark
    except ImportError:
        from libero.libero.benchmark import get_benchmark  # type: ignore

    tasks: list[dict[str, Any]] = []
    for suite in suites:
        benchmark = get_benchmark(suite)()
        for task_id in range(benchmark.get_num_tasks()):
            task = benchmark.get_task(task_id)
            tasks.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task.name,
                    "language": task.language,
                    "num_init_states": 50,
                }
            )
    return tasks


def _state_ids(args: argparse.Namespace, num_init_states: int) -> list[int]:
    valid = list(range(max(1, int(num_init_states))))
    if args.state_ids is not None:
        return [sid for sid in args.state_ids if sid in valid]
    if args.all_states:
        return valid
    return valid[: max(1, int(args.max_states))]


def _run_dir(outdir: Path, suite: str, task_id: int, state_id: int) -> Path:
    return outdir / f"{suite}__task{task_id:02d}" / f"state_{state_id:03d}"


def _build_command(
    args: argparse.Namespace,
    *,
    task: dict[str, Any],
    state_id: int,
    run_dir: Path,
    prompt: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_libero_step.py",
        "--legacy-vlm-stl-root",
        str(Path(args.legacy_vlm_stl_root)),
        "--base-scene-context",
        args.base_scene_context,
        "--schema",
        args.schema,
        "--suite",
        task["suite"],
        "--task-id",
        str(task["task_id"]),
        "--init-state-id",
        str(state_id),
        "--prompt",
        prompt,
        "--outdir",
        str(run_dir),
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.compiled_plan:
        cmd += ["--compiled-plan", args.compiled_plan]
    if not args.no_video:
        cmd += ["--save-video", str(run_dir / args.save_video)]
    return cmd


def _print_running_tally(results: list[dict[str, Any]]) -> None:
    ran = [r for r in results if not r.get("skipped")]
    successes = [r for r in ran if r.get("success") is True]
    rate = len(successes) / max(1, len(ran))
    print(f"[suite] running success rate: {len(successes)}/{len(ran)} ({rate:.1%})")


def _read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "execution_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"summary_read_error": str(exc)}


def main() -> int:
    args = parse_args()
    if args.outdir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = f"outputs/{args.suite}_{timestamp}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _add_paths(args.legacy_vlm_stl_root, outdir)

    safety_prompts = _load_safety_prompts(args.safety_json) if args.use_safety_prompts else {}
    tasks = _enumerate_tasks(_suite_names(args.suite))
    if args.task_id is not None:
        tasks = [t for t in tasks if int(t["task_id"]) == int(args.task_id)]
    if args.max_tasks is not None:
        tasks = tasks[: max(0, int(args.max_tasks))]
    if not tasks:
        print("[suite] no matching tasks", file=sys.stderr)
        return 1

    runs: list[tuple[dict[str, Any], int, str]] = []
    for task in tasks:
        prompt = safety_prompts.get((task["suite"], task["task_id"]), task["language"])
        for state_id in _state_ids(args, task["num_init_states"]):
            runs.append((task, state_id, prompt))

    print(f"[suite] tasks={len(tasks)} rollouts={len(runs)} outdir={outdir}")
    if args.compiled_plan:
        print("[suite] warning: reusing one compiled plan for every task; use only for runner debugging.")

    results: list[dict[str, Any]] = []
    for idx, (task, state_id, prompt) in enumerate(runs, start=1):
        run_dir = _run_dir(outdir, task["suite"], task["task_id"], state_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        label = f"[suite {idx}/{len(runs)}] {task['suite']}/task{task['task_id']:02d}/state{state_id:03d}"

        if args.skip_existing and (run_dir / "execution_summary.json").exists():
            print(f"{label} skip")
            summary = _read_summary(run_dir)
            results.append(_result_row(task, state_id, run_dir, prompt, 0, 0.0, summary, skipped=True))
            _print_running_tally(results)
            continue

        cmd = _build_command(args, task=task, state_id=state_id, run_dir=run_dir, prompt=prompt)
        if args.dry_run:
            print(f"{label} dry-run: {' '.join(cmd)}")
            continue

        print(f"{label} running")
        t0 = time.time()
        ret = subprocess.call(cmd)
        elapsed = time.time() - t0
        summary = _read_summary(run_dir)
        print(f"{label} exit={ret} success={summary.get('success')} elapsed={elapsed:.1f}s")
        results.append(_result_row(task, state_id, run_dir, prompt, ret, elapsed, summary, skipped=False))
        _print_running_tally(results)

    if args.dry_run:
        print("[suite] dry-run complete")
        return 0

    ran = [r for r in results if not r.get("skipped")]
    successes = [r for r in ran if r.get("success") is True]
    aggregate = {
        "suite": args.suite,
        "total_tasks": len(tasks),
        "total_rollouts": len(runs),
        "total_run": len(ran),
        "total_success": len(successes),
        "success_rate": round(len(successes) / max(1, len(ran)), 4),
        "results": results,
    }
    save_json(aggregate, outdir / "results_summary.json")
    print(f"[suite] summary: {outdir / 'results_summary.json'}")
    print(f"[suite] success: {len(successes)}/{len(ran)}")
    return 0


def _result_row(
    task: dict[str, Any],
    state_id: int,
    run_dir: Path,
    prompt: str,
    exit_code: int,
    elapsed: float,
    summary: dict[str, Any],
    *,
    skipped: bool,
) -> dict[str, Any]:
    return {
        "suite": task["suite"],
        "task_id": task["task_id"],
        "state_id": state_id,
        "task_name": task["task_name"],
        "prompt": prompt,
        "success": summary.get("success"),
        "exit_code": exit_code,
        "elapsed_sec": round(float(elapsed), 1),
        "skipped": bool(skipped),
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    raise SystemExit(main())
