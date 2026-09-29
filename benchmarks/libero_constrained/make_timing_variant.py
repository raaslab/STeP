"""Re-issue the temporal tasks with a longer hold, as a separate manifest.

The temporal family asks the robot to hold: 2 seconds over the goal before
releasing (T-dwell), 4 seconds between two stages (T-wait). Both bars turned out
to sit under what the robot does anyway -- the measured dwells in the main run
were 58 to 91 steps against a 40-step requirement -- so a rollout could clear the
constraint without the constraint having changed anything. A longer hold puts the
bar above the incidental behaviour, which is the only way the number says
something about enforcement rather than about the planner's pace.

Only the duration moves. The task, the scene, the init state, the objects and
every other family are untouched, so the variant stays comparable to the frozen
manifest task-for-task.

    python benchmarks/libero_constrained/make_timing_variant.py --seconds 8

Then point both the runner and the scorer at the result:

    export LIBERO_CONSTRAINED_MANIFEST=benchmarks/libero_constrained/tasks_t8.json
"""

import argparse
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent
SOURCE = HERE / "tasks.json"

# Scoring types that carry a duration. `stage` and `clear_path` are ordering
# constraints with no clock, so they ride along unchanged -- dropping them would
# make the family's own aggregate incomparable to the main run.
TIMED = {"dwell", "wait"}


def retime(task, seconds, hz):
    """One task with its duration replaced. Returns (task, old_seconds) or None."""
    spec = task.get("scoring", {})
    if spec.get("type") not in TIMED or "seconds" not in spec:
        return None
    old = int(spec["seconds"])
    if old == seconds:
        return None

    out = json.loads(json.dumps(task))
    out["scoring"]["seconds"] = seconds
    out["scoring"]["steps"] = int(round(seconds * hz))

    # The instruction is what System 2 actually reads, so the number has to move
    # in the sentence too, not only in the scoring spec.
    out["prompt"] = re.sub(rf"\b{old} seconds\b", f"{seconds} seconds", task["prompt"])
    if out["prompt"] == task["prompt"]:
        raise SystemExit(f"{task['id']}: prompt does not say '{old} seconds'; refusing to guess")

    intent = task.get("stl_intent", "")
    intent = intent.replace(f"[0,{int(old * hz)}]", f"[0,{int(seconds * hz)}]")
    intent = intent.replace(f"Waited({old})", f"Waited({seconds})")
    out["stl_intent"] = intent
    return out, old


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=int, required=True, help="new hold duration")
    ap.add_argument("--out", default=None, help="default: tasks_t<seconds>.json")
    args = ap.parse_args()

    manifest = json.loads(SOURCE.read_text())
    hz = manifest["control_hz"]

    changed = []
    for i, task in enumerate(manifest["tasks"]):
        got = retime(task, args.seconds, hz)
        if got:
            manifest["tasks"][i], old = got
            changed.append((task["id"], old))

    manifest["variant"] = {
        "of": "tasks.json",
        "change": f"temporal hold raised to {args.seconds} s ({int(args.seconds * hz)} steps)",
        "retimed": [t for t, _ in changed],
    }

    out = pathlib.Path(args.out) if args.out else HERE / f"tasks_t{args.seconds}.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"retimed {len(changed)} tasks to {args.seconds} s ({int(args.seconds * hz)} steps)")
    for task_id, old in changed:
        print(f"  {task_id:28s} {old} s -> {args.seconds} s")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
