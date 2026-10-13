from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import load_json, save_json
from .libero_suite_runner import LIBERO_PRO_PERTURBATIONS, LIBERO_SUITES

# Cap-X's own LIBERO-Pro protocol, cross-referenced from their codebase
# (cap-x/capx/envs/scripts/run_libero_batch.py and
# cap-x/capx/scripts/start_servers_and_eval.sh, which states the full run as
# "6 suites x 10 tasks x 50 trials = 3000 total trials"): libero_spatial,
# libero_object, libero_goal (NOT libero_10 -- absent from their default suite
# list) x {swap ("Position"), task ("Task")}, every task in each suite, every
# one of the 50 init-state trials per task. Mirrored here as the default grid
# so STeP's numbers are directly comparable to Cap-X's own runs on the same
# benchmark suites.
CAPX_SUITES = ("libero_spatial", "libero_object", "libero_goal")
CAPX_PERTURBATIONS = ("swap", "task")

# "none" runs plain LIBERO (no perturbation suffix) through the same grid/
# aggregation machinery -- e.g. --suites libero_spatial libero_object
# libero_goal libero_10 --perturbations none for the standard 4-suite
# vanilla protocol.
PERTURBATION_CHOICES = list(LIBERO_PRO_PERTURBATIONS) + ["none"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run STeP across a grid of LIBERO-Pro (suite x perturbation) "
            "combinations and aggregate into one summary table. Defaults to "
            "Cap-X's own LIBERO-Pro protocol: libero_spatial/object/goal x "
            "swap/task, every task, all 50 init-state trials per task."
        )
    )
    p.add_argument("--base-scene-context", default="configs/libero_template.json")
    p.add_argument("--schema", default="schemas/stl_plan_schema.json")
    p.add_argument("--suites", nargs="+", default=list(CAPX_SUITES), choices=list(LIBERO_SUITES))
    p.add_argument(
        "--perturbations",
        nargs="+",
        default=list(CAPX_PERTURBATIONS),
        choices=PERTURBATION_CHOICES,
        help=(
            "Which LIBERO-Pro dimensions to run, or 'none' for plain LIBERO "
            "(no perturbation). Default: swap + task (Cap-X's own protocol). "
            "For the standard 4-suite vanilla protocol: --suites libero_spatial "
            "libero_object libero_goal libero_10 --perturbations none."
        ),
    )
    p.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="Trials per task. Default: all init states (50/task, matching Cap-X's own full protocol). Pass a small number (e.g. 10) for a quick pilot before committing to the full grid.",
    )
    p.add_argument("--max-tasks", type=int, default=None, help="Debug only: cap tasks/suite.")
    p.add_argument(
        "--rollout-timeout",
        type=float,
        default=300.0,
        help="Kill and count as a failure any single rollout that runs longer than this many seconds. Pass 0 to disable.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Defaults to outputs/libero_pro_<timestamp>. Each (suite, perturbation) combo gets its own subdirectory underneath.",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _combo_outdir(root: Path, suite: str, perturbation: str) -> Path:
    if perturbation == "none":
        return root / suite
    return root / f"{suite}_{perturbation}"


def _build_command(args: argparse.Namespace, suite: str, perturbation: str, combo_outdir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "step.libero_suite_runner",
        "--base-scene-context",
        args.base_scene_context,
        "--schema",
        args.schema,
        "--suite",
        suite,
        "--outdir",
        str(combo_outdir),
    ]
    if perturbation != "none":
        cmd += ["--perturbation", perturbation]
    if args.max_states is not None:
        cmd += ["--max-states", str(args.max_states)]
    else:
        cmd += ["--all-states"]
    if args.max_tasks is not None:
        cmd += ["--max-tasks", str(args.max_tasks)]
    cmd += ["--rollout-timeout", str(args.rollout_timeout)]
    if args.model:
        cmd += ["--model", args.model]
    if args.skip_existing:
        cmd += ["--skip-existing"]
    if args.no_video:
        cmd += ["--no-video"]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def main() -> int:
    args = parse_args()
    if args.outdir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = f"outputs/libero_pro_{timestamp}"
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    n_combos = len(args.suites) * len(args.perturbations)
    print(f"[libero-pro] grid: {len(args.suites)} suites x {len(args.perturbations)} perturbations = {n_combos} combos")
    print(f"[libero-pro] outdir: {root}")

    grid: dict[str, dict[str, Any]] = {suite: {} for suite in args.suites}
    for suite in args.suites:
        for perturbation in args.perturbations:
            combo_outdir = _combo_outdir(root, suite, perturbation)
            cmd = _build_command(args, suite, perturbation, combo_outdir)
            print(f"\n[libero-pro] === {suite} / {perturbation} ===")
            print(f"[libero-pro] {' '.join(cmd)}")
            if args.dry_run:
                continue

            t0 = time.time()
            ret = subprocess.call(cmd)
            elapsed = time.time() - t0
            summary_path = combo_outdir / "results_summary.json"
            summary = load_json(summary_path) if summary_path.exists() else {}
            grid[suite][perturbation] = {
                "success_rate": summary.get("success_rate"),
                "total_run": summary.get("total_run"),
                "total_success": summary.get("total_success"),
                "exit_code": ret,
                "elapsed_sec": round(elapsed, 1),
                "outdir": str(combo_outdir),
            }
            print(
                f"[libero-pro] {suite}/{perturbation} "
                f"success_rate={grid[suite][perturbation]['success_rate']} "
                f"({grid[suite][perturbation]['total_success']}/{grid[suite][perturbation]['total_run']}) "
                f"elapsed={elapsed:.1f}s"
            )

    if args.dry_run:
        print("\n[libero-pro] dry-run complete")
        return 0

    per_perturbation_rates: dict[str, list[float]] = {p: [] for p in args.perturbations}
    per_suite_rates: dict[str, list[float]] = {s: [] for s in args.suites}
    all_rates: list[float] = []
    for suite, row in grid.items():
        for perturbation, cell in row.items():
            rate = cell.get("success_rate")
            if rate is not None:
                per_perturbation_rates[perturbation].append(rate)
                per_suite_rates[suite].append(rate)
                all_rates.append(rate)

    aggregate = {
        "suites": list(args.suites),
        "perturbations": list(args.perturbations),
        "grid": grid,
        "per_suite_average": {s: _avg(v) for s, v in per_suite_rates.items()},
        "per_perturbation_average": {p: _avg(v) for p, v in per_perturbation_rates.items()},
        "total_average": _avg(all_rates),
    }
    save_json(aggregate, root / "libero_pro_summary.json")
    print(f"\n[libero-pro] summary: {root / 'libero_pro_summary.json'}")

    print("\n[libero-pro] === Results grid (rows=suite, cols=perturbation) ===")
    header = "suite".ljust(18) + "".join(p.ljust(10) for p in args.perturbations) + "avg"
    print(header)
    for suite in args.suites:
        row_str = suite.ljust(18)
        for perturbation in args.perturbations:
            rate = grid[suite].get(perturbation, {}).get("success_rate")
            row_str += (f"{rate:.2f}" if rate is not None else "-").ljust(10)
        row_avg = aggregate["per_suite_average"][suite]
        row_str += f"{row_avg:.2f}" if row_avg is not None else "-"
        print(row_str)
    footer = "avg".ljust(18)
    for perturbation in args.perturbations:
        col_avg = aggregate["per_perturbation_average"][perturbation]
        footer += (f"{col_avg:.2f}" if col_avg is not None else "-").ljust(10)
    total_avg = aggregate["total_average"]
    footer += f"{total_avg:.2f}" if total_avg is not None else "-"
    print(footer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
