"""Give the VLM-MPC-Cost baseline the tuning budget the method's own weights had.

The baseline writes its cost weights over Cartesian predicate distances while
the controller's native terms are joint-space and hand-tuned over many runs, so
the raw sum let an arbitrary magnitude decide how hard the avoid-term fought the
reach-term. Measured on the first sweep: where max(avoid)/max(reach) >= 0.5 the
baseline's task success was 27%, and below 0.5 it was 50% -- a scale artefact,
not a property of cost functions.

Normalising (CollisionConfig.mpc_cost_normalize) preserves every ratio the VLM
chose and leaves one free number, the gain. This sweeps it so the baseline can
be reported at its best setting rather than at an arbitrary one.

    python benchmarks/libero_constrained/sweep_mpc_gain.py --gains 0.25 0.5 1 2 4
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "libero_template.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--family", nargs="+", default=["spatial"],
                    help="families to sweep on. spatial carries the keep-out zones, which is "
                         "where the avoid/reach tradeoff actually bites.")
    ap.add_argument("--kind", nargs="*", default=None,
                    help="restrict to these kinds, e.g. S-zone. A sweep over every spatial task "
                         "costs 5x the family; the gain can be chosen from a subset.")
    ap.add_argument("--task", nargs="*", default=None, help="explicit manifest ids")
    ap.add_argument("--states", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--outroot", default="outputs/mpcgain")
    args = ap.parse_args()

    base = json.loads(TEMPLATE.read_text())
    tmpdir = ROOT / "outputs" / "_gain_templates"
    tmpdir.mkdir(parents=True, exist_ok=True)

    for g in args.gains:
        cfg = json.loads(json.dumps(base))
        # The template keeps its motion config under platform.libero.motion, and
        # that is the only place MotionConfig.from_scene_context reads. An
        # earlier version of this sweep wrote to a top-level "motion" block
        # instead -- nothing read it, every "gain" ran at the default of 1.0,
        # and the result was identical across an 8x range of gains.
        coll = (cfg.setdefault("platform", {}).setdefault("libero", {})
                   .setdefault("motion", {}).setdefault("collision", {}))
        coll["mpc_cost_normalize"] = True
        coll["mpc_cost_weight_gain"] = g
        path = tmpdir / f"gain_{g}.json"
        path.write_text(json.dumps(cfg, indent=2))

        out = f"{args.outroot}_{g}"
        print(f"\n=== gain {g} -> {out}", flush=True)
        subprocess.run([
            sys.executable, str(pathlib.Path(__file__).parent / "run_benchmark.py"),
            *(["--task", *args.task] if args.task else ["--family", *args.family]),
            *(["--kind", *args.kind] if args.kind else []),
            "--arms", "on", "--states", str(args.states),
            "--stl-mode", "mpc_cost", "--timeout", str(args.timeout),
            "--base-scene-context", str(path), "--outdir", out,
        ], cwd=ROOT)

    print("\n\n=== results by gain ===")
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import score as S
    print(f"{'gain':>6} {'n':>4} {'TaskSucc':>9} {'Viol':>7} {'SafeSucc':>9}")
    for g in args.gains:
        d = ROOT / f"{args.outroot}_{g}"
        if not d.exists():
            continue
        rows = [r for r in (S.score_run(x) for x in sorted(d.iterdir()) if x.is_dir()) if r]
        a = S.aggregate(rows)
        if a:
            print(f"{g:>6} {a['n']:>4} {a['task_success']:>8.0f}% "
                  f"{a['violation']:>6.0f}% {a['safe_success']:>8.0f}%")


if __name__ == "__main__":
    main()
