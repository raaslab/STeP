"""Run LIBERO-Constrained tasks from the manifest, with a per-rollout timeout.

A small number of rollouts take pathologically long -- measured between 26s and
13 minutes on the same task, driven by how hard the planner finds that particular
scene. Left alone they dominate the wall clock without adding information, so a
rollout that exceeds --timeout is killed and recorded as `timeout`. That is a
result, not a gap: a method that cannot produce a trajectory inside the budget has
not solved the task.

Resumable: a rollout whose summary already exists is skipped, so an interrupted
sweep continues where it stopped.

  python benchmarks/libero_constrained/run_benchmark.py --family spatial --states 1
  python benchmarks/libero_constrained/run_benchmark.py --arms on --timeout 420
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = pathlib.Path(__file__).parent / "tasks.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", nargs="*", default=None,
                   help="spatial / logical / temporal / unconstrained. Default: all.")
    p.add_argument("--kind", nargs="*", default=None, help="e.g. S-zone T-dwell")
    p.add_argument("--suite", nargs="*", default=None)
    p.add_argument("--task", nargs="*", default=None, help="explicit manifest ids")
    p.add_argument("--arms", nargs="*", default=["off", "on"], choices=["off", "on"],
                   help="'off' monitors the constraint only, 'on' enforces it in System 1.")
    p.add_argument("--states", type=int, default=3, help="init states per task, from 0")
    p.add_argument("--timeout", type=float, default=360.0,
                   help="seconds before a rollout is killed and recorded as timeout (default 360)")
    p.add_argument("--outdir", default="outputs/benchmark")
    p.add_argument("--retries", type=int, default=0, help="--max-pipeline-retries for each rollout")
    p.add_argument("--video", action="store_true", help="record a video per rollout")
    p.add_argument("--dry-run", action="store_true", help="list what would run, then stop")
    return p.parse_args()


def select(manifest, args):
    tasks = manifest["tasks"]
    if args.task:
        want = set(args.task)
        return [t for t in tasks if t["id"] in want]
    if args.family:
        tasks = [t for t in tasks if t["family"] in set(args.family)]
    if args.kind:
        tasks = [t for t in tasks if t["kind"] in set(args.kind)]
    if args.suite:
        tasks = [t for t in tasks if t["suite"] in set(args.suite)]
    return tasks


def run_one(task, arm, state, args, outdir):
    """One rollout. Returns a result row; never raises on a failed rollout."""
    slug = task["id"].replace("/", "_")
    run_dir = outdir / f"{slug}_s{state}_{arm}"
    summary = run_dir / "execution_summary.json"
    if summary.exists():
        return {"id": task["id"], "arm": arm, "state": state, "status": "skipped",
                "dir": str(run_dir)}

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "step.libero_runner",
        "--benchmark-task", task["id"],
        "--init-state-id", str(state),
        "--max-pipeline-retries", str(args.retries),
        "--outdir", str(run_dir),
    ]
    # The keep-out only exists for tasks that carry a zone; for the rest the two
    # arms would be identical, so 'off' is meaningless and never requested.
    if arm == "off" and task.get("zone"):
        cmd.append("--no-enforce-keepout")
    if args.video:
        cmd += ["--save-video", str(run_dir / "r.mp4")]

    log = outdir / f"{slug}_s{state}_{arm}.log"
    started = time.time()
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=args.timeout)
            status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            status = "timeout"
    elapsed = time.time() - started

    row = {"id": task["id"], "arm": arm, "state": state, "status": status,
           "seconds": round(elapsed, 1), "dir": str(run_dir)}
    if summary.exists():
        data = json.loads(summary.read_text())
        report = data.get("constraint_report") or {}
        row.update({
            "success": bool(data.get("success")),
            "violated": report.get("violated"),
            "min_distance_m": report.get("min_distance_m"),
        })
    return row


def main():
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text())
    tasks = select(manifest, args)
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    plan = []
    for task in tasks:
        arms = args.arms if task.get("zone") else [a for a in args.arms if a == "on"] or ["on"]
        for state in range(args.states):
            for arm in arms:
                plan.append((task, arm, state))

    print(f"{len(plan)} rollouts over {len(tasks)} tasks "
          f"(timeout {args.timeout:.0f}s each, worst case {len(plan)*args.timeout/3600:.1f} h)")
    if args.dry_run:
        for task, arm, state in plan:
            print(f"  {task['id']:34s} state={state} arm={arm}")
        return 0

    done = 0
    with results_path.open("a") as out:
        for task, arm, state in plan:
            done += 1
            print(f"[{done}/{len(plan)}] {task['id']} state={state} arm={arm}", flush=True)
            row = run_one(task, arm, state, args, outdir)
            out.write(json.dumps(row) + "\n")
            out.flush()
            note = row.get("status")
            if note == "skipped":
                print("      already done", flush=True)
            else:
                print(f"      {note} in {row.get('seconds', 0):.0f}s  "
                      f"success={row.get('success')} violated={row.get('violated')}", flush=True)

    print(f"\nwrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
