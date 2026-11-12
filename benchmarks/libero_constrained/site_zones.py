"""Place keep-out zones on measured evidence instead of by hand.

A zone is only a constraint if the robot would otherwise go through it, and it
is only a *fair* constraint if there is somewhere else to go. The zones in
tasks.json satisfy neither reliably:

  - Six of twenty are clipped by less than a third of their radius by the
    unconstrained path, so avoiding them costs nothing and can happen by
    accident.
  - Several sit beside the natural path rather than on it, close enough that
    the arm clips the rim while going past. Five of the fourteen reviewer
    verdicts in adjudication.json are exactly that graze, which is why the
    benchmark still needs a human to score.

Both come from the same mistake, so both have the same fix: put the zone ON the
path, and only where there is room to go around it.

The fields `route_depth_m` and `object_clearance_m` already in the manifest
record this, but nothing reads them and the tool that wrote them is gone. This
recomputes them from a real rollout and reports what it would change.

    # 1. record where the arm goes with the constraint removed
    python benchmarks/libero_constrained/run_benchmark.py --kind S-zone \
        --stl-mode stl_spec --strip-constraint --outdir outputs/freepaths
    # 2. propose zones against those paths
    python benchmarks/libero_constrained/site_zones.py outputs/freepaths

Nothing is written unless --apply is passed.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
MANIFEST = HERE / "tasks.json"

# A zone the arm can only skirt is a zone it will clip. The detour has to be
# wider than the controller's own tracking error or a graze is the expected
# outcome, not a mistake -- replan.tracking_error_m is 0.03 in the LIBERO
# template, so this is that plus a margin.
MIN_DETOUR_M = 0.045
# Keep the zone off the pick and the place, or the task stops being solvable.
MIN_ENDPOINT_CLEARANCE_M = 0.10
# Below this a zone is too small to be seen or to matter.
MIN_RADIUS_M = 0.030
MAX_RADIUS_M = 0.070


def load_path(run_dir: pathlib.Path):
    """End-effector xy over the rollout, and the objects present."""
    f = run_dir / "trajectory.npz"
    if not f.exists():
        return None, None, None
    z = np.load(f, allow_pickle=True)
    ee = np.asarray(z["eef_pos"], dtype=float)
    ok = np.isfinite(ee).all(axis=1)
    names = [str(x) for x in z["object_names"]]
    obj = np.asarray(z["object_pos"], dtype=float)
    return ee[ok][:, :2], names, obj[ok]


def propose(ee_xy, obj_xy_per_step):
    """The point on the path with the most room around it.

    Walks the middle of the path -- the ends are the pick and the place, and a
    zone there makes the task unsolvable rather than constrained -- and scores
    each point by how far it sits from every object in the scene. The winner is
    the point with the widest free corridor, which is where a zone both blocks
    the natural route and leaves a way around.
    """
    n = len(ee_xy)
    if n < 20:
        return None
    lo, hi = int(0.25 * n), int(0.75 * n)
    start, end = ee_xy[0], ee_xy[-1]
    best = None
    for i in range(lo, hi):
        p = ee_xy[i]
        if min(np.linalg.norm(p - start), np.linalg.norm(p - end)) < MIN_ENDPOINT_CLEARANCE_M:
            continue
        objs = obj_xy_per_step[i][:, :2]
        clear = float(np.min(np.linalg.norm(objs - p, axis=1))) if len(objs) else 1.0
        if best is None or clear > best[0]:
            best = (clear, i, p)
    if best is None:
        return None
    clear, i, p = best
    # Radius: as large as the corridor allows once the detour is reserved.
    radius = float(np.clip(clear - MIN_DETOUR_M, MIN_RADIUS_M, MAX_RADIUS_M))
    feasible = clear - radius >= MIN_DETOUR_M
    return {"center_xy": [float(p[0]), float(p[1])], "radius_m": round(radius, 4),
            "route_depth_m": round(radius, 4),   # centred on the path by construction
            "object_clearance_m": round(clear, 4),
            "detour_margin_m": round(clear - radius, 4), "feasible": bool(feasible),
            "path_fraction": round(i / n, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir", help="rollouts recorded with --strip-constraint")
    ap.add_argument("--apply", action="store_true", help="write the proposals into tasks.json")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    by_id = {t["id"]: t for t in manifest["tasks"]}
    root = pathlib.Path(args.rundir)

    print(f"{'task':<28} {'old r':>6} {'old depth/r':>12} | "
          f"{'new r':>6} {'detour':>7} {'clearance':>10}  verdict")
    proposals, improved = {}, 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        tid = next((k for k in by_id if d.name.startswith(k.replace("/", "_") + "_s")), None)
        if tid is None or not by_id[tid].get("zone"):
            continue
        ee, names, obj = load_path(d)
        if ee is None:
            print(f"{tid:<28} (no trajectory logged)")
            continue
        got = propose(ee, obj)
        old = by_id[tid]["zone"]
        old_ratio = (old.get("route_depth_m") or 0) / (old.get("radius_m") or 1)
        if not got:
            print(f"{tid:<28} {old.get('radius_m'):>6} {old_ratio:>11.2f}  | (no usable site found)")
            continue
        verdict = "OK" if got["feasible"] else "NO ROOM TO GO AROUND"
        if got["feasible"] and old_ratio < 0.5:
            improved += 1
            verdict += "  <-- replaces a weak zone"
        print(f"{tid:<28} {old.get('radius_m'):>6} {old_ratio:>11.2f}  | "
              f"{got['radius_m']:>6} {got['detour_margin_m']:>7.3f} {got['object_clearance_m']:>10.3f}  {verdict}")
        proposals[tid] = got

    print(f"\n{len(proposals)} proposals, {improved} replace a zone the path barely clipped")
    if args.apply:
        for tid, z in proposals.items():
            if z["feasible"]:
                by_id[tid]["zone"].update({k: v for k, v in z.items() if k != "feasible"})
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote {MANIFEST}")
    else:
        print("(dry run -- pass --apply to write them)")


if __name__ == "__main__":
    main()
