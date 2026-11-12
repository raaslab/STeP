"""Gather one temporal sweep into the rows a report needs.

Pulls together what is otherwise spread across three files per rollout: the
manifest's requirement, the scorer's verdict, the executor's own record of how
long it actually held, and the STL windows System 2 drew. Emits JSON.

    LIBERO_CONSTRAINED_MANIFEST=benchmarks/libero_constrained/tasks_t8.json \
    python benchmarks/libero_constrained/collect_temporal.py outputs/t8 --out /tmp/t8.json
"""

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import score as S  # noqa: E402


def stl_windows(run_dir):
    """The per-subtask formulas as System 2 wrote them, from the run's log."""
    logs = list(run_dir.parent.glob(run_dir.name + ".log"))
    if not logs:
        return []
    text = logs[0].read_text(errors="replace")
    block = re.search(r"=== STL formulas by subtask ===\n(.*?)=== End STL formulas ===",
                      text, re.S)
    if not block:
        return []
    out, skill = [], None
    for line in block.group(1).splitlines():
        head = re.match(r"\[subtask (\d+) \| skill=(\w+)\]", line.strip())
        if head:
            skill = head.group(2)
        elif line.strip() and skill:
            out.append({"skill": skill, "formula": line.strip()})
            skill = None
    return out


def holds(run_dir):
    """Steps the executor held before each subtask, and the total."""
    path = run_dir / "execution_summary.json"
    if not path.exists():
        return [], 0
    records = json.loads(path.read_text()).get("records", [])
    per = [{"skill": r.get("skill"),
            "steps": int(r.get("steps") or 0),
            "held": int((r.get("info") or {}).get("timing_hold_steps") or 0)}
           for r in records]
    return per, sum(p["held"] for p in per)


def measured(task, roll, total_held):
    """The quantity the task's own constraint is measured in, in steps.

    Each temporal kind holds a different thing, so the number that belongs on a
    chart differs too: a dwell is time spent over the goal still holding the
    object, a wait is time spent holding to reach the next subtask's window.
    """
    kind = task["scoring"]["type"]
    if kind == "wait":
        return total_held, "steps held between the stages"
    if kind == "dwell":
        if roll is None or "over_goal" not in roll:
            return None, "no trajectory logged"
        held = (roll["over_goal"] > 0) & (roll["grasped"] > 0)
        best = run = 0
        for v in held:
            run = run + 1 if v else 0
            best = max(best, run)
        return int(best), "longest hold over the goal, gripper closed"
    return None, "no duration -- ordering only"


def caveat(task, roll, per, total_held):
    """Why a row reads as a violation when the timing itself was not the cause.

    Two things can put a 'broken' on a temporal row without the delay mechanism
    having done anything wrong, and both are worth saying out loud rather than
    letting the column imply a timing failure.
    """
    import numpy as np

    kind = task["scoring"]["type"]
    failed = [p for p in per if p["skill"] and p["steps"]]

    # The rollout died at a skill before the delay was ever due.
    if kind == "wait" and total_held == 0 and len(per) <= 1 and per:
        return (f"the rollout ended at the {per[0]['skill']} skill after {per[0]['steps']} steps, "
                f"so the gap the plan declared was never reached")

    # The robot held, but the goal test never fired: over_goal is a signed
    # distance, and an object parked just outside the goal radius reads as
    # never having been over it however long the arm waited there.
    if kind == "dwell" and total_held > 0 and roll is not None and "over_goal" in roll:
        og = np.asarray(roll["over_goal"], dtype=float)
        if og.size and og.max() <= 0:
            return (f"the robot did hold for {total_held} steps, but the object sat "
                    f"{abs(og.max()) * 1000:.0f} mm outside the goal radius the whole time, "
                    f"so the dwell measure never started")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = pathlib.Path(args.outdir)
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        row = S.score_run(d)
        if not row or row["family"] != "temporal":
            continue
        task = S.TASKS[row["id"]]
        per, total_held = holds(d)
        roll = S.load_rollout(d)
        got, measure = measured(task, roll, total_held)
        video = d / "r.mp4"
        rows.append({
            "measured_steps": got,
            "measures": measure,
            "caveat": caveat(task, roll, per, total_held),
            **row,
            "prompt": task["prompt"],
            "scoring_type": task["scoring"]["type"],
            "seconds": task["scoring"].get("seconds"),
            "steps_required": task["scoring"].get("steps"),
            "subtasks": per,
            "held_steps": total_held,
            "windows": stl_windows(d),
            "video": str(video) if video.exists() and video.stat().st_size > 1000 else None,
        })

    agg = S.aggregate(rows)
    payload = {"control_hz": S.MANIFEST["control_hz"],
               "manifest": str(S.MANIFEST_PATH.name),
               "rows": rows, "aggregate": agg}
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"{len(rows)} temporal rollouts -> {args.out}")
    if agg:
        print(f"  TaskSucc {agg['task_success']:.0f}%  Viol {agg['violation']:.0f}%  "
              f"SafeSucc {agg['safe_success']:.0f}%  (n={agg['n']})")


if __name__ == "__main__":
    main()
