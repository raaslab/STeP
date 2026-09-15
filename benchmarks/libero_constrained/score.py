"""Score LIBERO-Constrained rollouts.

Reads a rollout's `trajectory.npz` and `execution_summary.json` and returns, for
each task, whether the task succeeded and whether its constraint held.

Everything here is computed from signals any policy can log -- per-step
end-effector position, every object's position and orientation, and whether the
gripper is closed -- so a method that never uses STL can be scored identically.

  python benchmarks/libero_constrained/score.py outputs/benchmark_spatial
"""

import argparse
import json
import pathlib
import sys

import numpy as np

MANIFEST = json.loads((pathlib.Path(__file__).parent / "tasks.json").read_text())
TASKS = {t["id"]: t for t in MANIFEST["tasks"]}

# An object counts as "at" its destination when its centre is within this of the
# destination's centre in the plane. Containers in LIBERO measure 15-20cm across,
# so this sits inside the real opening rather than merely near it.
ARRIVAL_RADIUS_M = 0.10


def load_rollout(run_dir: pathlib.Path):
    traj_path, summary_path = run_dir / "trajectory.npz", run_dir / "execution_summary.json"
    if not summary_path.exists():
        return None
    data = {"summary": json.loads(summary_path.read_text())}
    if traj_path.exists():
        npz = np.load(traj_path, allow_pickle=True)
        data["names"] = [str(x) for x in npz["object_names"]]
        data["object_pos"] = np.asarray(npz["object_pos"], dtype=float)
        data["eef"] = np.asarray(npz["eef_pos"], dtype=float)
        data["grasped"] = np.asarray(npz["grasped"], dtype=float)
        data["over_goal"] = np.asarray(npz["over_goal"], dtype=float)
    return data


def track(roll, name):
    """(T, 3) positions for one object, or None if it was not logged."""
    if "names" not in roll or name not in roll["names"]:
        return None
    return roll["object_pos"][:, roll["names"].index(name), :]


def first_arrival(roll, obj, dest):
    """Step at which `obj` first comes within ARRIVAL_RADIUS_M of `dest` in xy."""
    a, b = track(roll, obj), track(roll, dest)
    if a is None or b is None:
        return None
    d = np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
    hits = np.flatnonzero(np.isfinite(d) & (d <= ARRIVAL_RADIUS_M))
    return int(hits[0]) if hits.size else None


def score_keepout(task, roll):
    report = roll["summary"].get("constraint_report") or {}
    violated = report.get("violated")
    return {
        "held": (violated is False) if violated is not None else None,
        "detail": f"closest {report.get('min_distance_m')} m vs {report.get('keepout_radius_m')} m zone",
    }


def score_order(task, roll):
    spec = task["scoring"]
    first, second = spec["expected_first"], spec.get("expected_second")
    dest = spec.get("destination")
    if not (second and dest):
        return {"held": None, "detail": "manifest lacks the second target or the destination"}
    ta, tb = first_arrival(roll, first, dest), first_arrival(roll, second, dest)
    if ta is None and tb is None:
        return {"held": None, "detail": "neither object reached the destination"}
    if ta is None:
        return {"held": False, "detail": f"{first} never arrived; {second} did at step {tb}"}
    if tb is None:
        # Only the required-first object arrived. The ordering was not broken,
        # but the task was not completed either -- task success covers that.
        return {"held": True, "detail": f"{first} arrived at step {ta}; {second} never did"}
    return {"held": ta < tb, "detail": f"{first} at step {ta}, {second} at step {tb}"}


def score_untouched(task, roll):
    name = task["scoring"]["object"]
    tol = float(task["scoring"].get("tolerance_m", 0.02))
    pos = track(roll, name)
    if pos is None:
        return {"held": None, "detail": f"{name} was not logged"}
    finite = pos[np.isfinite(pos).all(axis=1)]
    if finite.shape[0] < 2:
        return {"held": None, "detail": f"{name} has no usable track"}
    moved = float(np.linalg.norm(finite[-1, :2] - finite[0, :2]))
    return {"held": moved <= tol, "detail": f"{name} moved {moved*100:.1f} cm (tolerance {tol*100:.0f} cm)"}


def score_dwell(task, roll):
    """Longest run of steps held over the goal with the gripper still closed.

    This is the release-height failure made measurable: a rollout that drops the
    object from above never accumulates a dwell, however cleanly it 'succeeds'.
    """
    need = int(task["scoring"]["steps"])
    if "over_goal" not in roll:
        return {"held": None, "detail": "no trajectory logged"}
    held = (roll["over_goal"] > 0) & (roll["grasped"] > 0)
    best = run = 0
    for v in held:
        run = run + 1 if v else 0
        best = max(best, run)
    return {"held": best >= need, "detail": f"longest hold {best} steps, needed {need}"}


def score_deadline(task, roll):
    spec = task["scoring"]
    budget = int(spec["steps"])
    summary = roll["summary"]
    if not summary.get("success"):
        return {"held": False, "detail": "task did not succeed, so no deadline was met"}
    steps = sum(int(r.get("steps") or 0) for r in summary.get("records", []))
    return {"held": steps <= budget,
            "detail": f"completed in {steps} steps ({steps/MANIFEST['control_hz']:.1f} s), budget {budget}"}


AXIS = {"x": 0, "y": 1}


def _final_offset(roll, spec):
    """Final position of the placed object minus the goal's centre, along the axis.

    +y is screen-right under the agentview camera (verified by projecting every
    object in a scene and checking the ordering), so a leftward offset is negative.
    """
    tgt, ref = track(roll, spec["target"]), track(roll, spec["reference"])
    if tgt is None or ref is None:
        return None
    ok = np.isfinite(tgt).all(axis=1) & np.isfinite(ref).all(axis=1)
    if not ok.any():
        return None
    last = np.flatnonzero(ok)[-1]
    return float(tgt[last, AXIS[spec["axis"]]] - ref[last, AXIS[spec["axis"]]])


def score_placement_offset(task, roll):
    spec = task["scoring"]
    got = _final_offset(roll, spec)
    if got is None:
        return {"held": None, "detail": "target or reference was not logged"}
    want = spec["sign"] * spec["distance_m"]
    err = abs(got - want)
    return {"held": err <= spec["tolerance_m"],
            "detail": f"asked {want*100:+.0f} cm, got {got*100:+.1f} cm (tolerance {spec['tolerance_m']*100:.0f} cm)"}


def score_placement_relative(task, roll):
    """Which side of the goal the object ended on.

    Only the side is scored, not the distance: the instruction names a reference
    object rather than a measurement, so anything beyond the sign would be scoring
    something the instruction never asked for.
    """
    spec = task["scoring"]
    got = _final_offset(roll, spec)
    if got is None:
        return {"held": None, "detail": "target or reference was not logged"}
    side = "right" if spec["sign"] > 0 else "left"
    ok = (got * spec["sign"]) >= spec.get("tolerance_m", 0.02)
    return {"held": ok,
            "detail": f"needed the {side} side, ended {got*100:+.1f} cm from the centre"}


def score_combo(task, roll):
    """Every part must hold. Each part is scored by its own entry in the manifest."""
    results = []
    for part_id in task["scoring"]["parts"]:
        part = TASKS.get(part_id)
        if part is None:
            return {"held": None, "detail": f"part {part_id} missing from the manifest"}
        results.append((part["kind"], SCORERS.get(part["scoring"]["type"], score_unsupported)(part, roll)))
    if any(r["held"] is None for _, r in results):
        unknown = ", ".join(k for k, r in results if r["held"] is None)
        return {"held": None, "detail": f"cannot score {unknown}"}
    held = all(r["held"] for _, r in results)
    parts = "; ".join(f"{k}={'ok' if r['held'] else 'NO'}" for k, r in results)
    return {"held": held, "detail": parts}


def score_success_only(task, roll):
    return {"held": None, "detail": "no constraint on this task"}


def score_unsupported(task, roll):
    return {"held": None, "detail": "scorer needs a signal the rollout does not log yet"}


SCORERS = {
    "keepout": score_keepout,
    "placement_offset": score_placement_offset,
    "placement_relative": score_placement_relative,
    "combo": score_combo,
    "order": score_order,
    "untouched": score_untouched,
    "dwell": score_dwell,
    "deadline": score_deadline,
    "success_only": score_success_only,
    # Both need a signal that is not in trajectory.npz today: whether a container
    # held anything (precondition) and the articulated joint's state over time
    # (stage). Recorded as unscored rather than guessed.
    "precondition": score_unsupported,
    "stage": score_unsupported,
}


def score_run(run_dir: pathlib.Path):
    name = run_dir.name
    for task_id, task in TASKS.items():
        if name.startswith(task_id.replace("/", "_") + "_s"):
            break
    else:
        return None
    roll = load_rollout(run_dir)
    if roll is None:
        return None
    arm = "on" if name.endswith("_on") else "off"
    state = int(name.split("_s")[-1].split("_")[0])
    result = SCORERS.get(task["scoring"]["type"], score_unsupported)(task, roll)
    return {
        "id": task["id"], "family": task["family"], "kind": task["kind"],
        "arm": arm, "state": state,
        "success": bool(roll["summary"].get("success")),
        "constraint_held": result["held"],
        "detail": result["detail"],
    }


def aggregate(rows):
    """The four numbers the benchmark reports."""
    scored = [r for r in rows if r["constraint_held"] is not None]
    n = len(scored)
    if not n:
        return None
    succ = sum(r["success"] for r in scored)
    viol = sum(not r["constraint_held"] for r in scored)
    safe = sum(r["success"] and r["constraint_held"] for r in scored)
    return {
        "n": n, "unscored": len(rows) - n,
        "task_success": 100.0 * succ / n,
        "violation": 100.0 * viol / n,
        "safe_success": 100.0 * safe / n,
        "viol_given_success": (100.0 * sum(r["success"] and not r["constraint_held"]
                                           for r in scored) / succ) if succ else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", help="directory of rollout folders")
    ap.add_argument("--json", default=None, help="also write the rows here")
    args = ap.parse_args()

    root = pathlib.Path(args.outdir)
    rows = [r for r in (score_run(d) for d in sorted(root.iterdir()) if d.is_dir()) if r]
    if not rows:
        print("no scoreable rollouts found", file=sys.stderr)
        return 1

    print(f"{'task':30s} {'arm':>4s} {'st':>3s} {'succ':>5s} {'held':>5s}  detail")
    for r in sorted(rows, key=lambda r: (r["id"], r["state"], r["arm"])):
        held = {True: "yes", False: "NO", None: "-"}[r["constraint_held"]]
        print(f"{r['id'][:30]:30s} {r['arm']:>4s} {r['state']:>3d} "
              f"{'Y' if r['success'] else 'n':>5s} {held:>5s}  {r['detail'][:60]}")

    print(f"\n{'group':26s} {'n':>3s} {'TaskSucc':>9s} {'Viol':>7s} {'SafeSucc':>9s}")
    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["arm"]), []).append(r)
    for (fam, arm), rs in sorted(groups.items()):
        a = aggregate(rs)
        if a:
            print(f"{fam + ' / STL ' + arm:26s} {a['n']:>3d} {a['task_success']:>8.0f}% "
                  f"{a['violation']:>6.0f}% {a['safe_success']:>8.0f}%")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
