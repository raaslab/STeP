"""Download LIBERO-Pro's perturbed bddl/init files for the swap (Position)
and task (Task) dimensions from the official HuggingFace dataset
(zhouxueyang/LIBERO-Pro) into a LIBERO-PRO checkout.

These files are NOT committed to the LIBERO-PRO git repo (they sit as
untracked working-tree files on the dev machine, downloaded the same way),
so a fresh git clone/submodule checkout won't have them -- this script
re-fetches them directly, which is more robust than trying to transfer the
untracked files themselves.

Usage:
    python cluster/fetch_libero_pro_data.py --libero-pro-root /path/to/cap-x/capx/third_party/LIBERO-PRO

Requires `huggingface_hub` to be installed in the active environment (it's
in requirements-libero.txt).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10")
DIMENSIONS = ("swap", "task", "object", "lan")  # swap=Position, task=Task, object=Object, lan=Semantic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--libero-pro-root",
        required=True,
        help="Path to the LIBERO-PRO checkout, e.g. cap-x/capx/third_party/LIBERO-PRO",
    )
    p.add_argument(
        "--dimensions",
        nargs="+",
        default=["swap", "task"],
        choices=list(DIMENSIONS),
        help="Which perturbation dimensions to fetch (default: swap + task, matching Cap-X's own protocol).",
    )
    p.add_argument(
        "--suites",
        nargs="+",
        default=list(SUITES),
        choices=list(SUITES),
        help="Which base suites to fetch data for (default: all 4).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.libero_pro_root).expanduser().resolve()
    libero_dir = root / "libero" / "libero"
    if not libero_dir.is_dir():
        raise SystemExit(f"not a LIBERO-PRO checkout (missing libero/libero/): {root}")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    api_files = api.list_repo_files("zhouxueyang/LIBERO-Pro", repo_type="dataset")

    needed = []
    for suite in args.suites:
        for dim in args.dimensions:
            name = f"{suite}_{dim}"
            bddl_dir = libero_dir / "bddl_files" / name
            if bddl_dir.is_dir() and any(bddl_dir.glob("*.bddl")):
                print(f"[fetch] {name}: already present, skipping")
                continue
            needed.append(name)

    if not needed:
        print("[fetch] nothing to do -- all requested suite/dimension combos already present")
        return 0

    print(f"[fetch] downloading: {needed}")
    count = 0
    for name in needed:
        for kind in ("bddl_files", "init_files"):
            prefix = f"{kind}/{name}/"
            matches = [f for f in api_files if f.startswith(prefix)]
            if not matches:
                print(f"[fetch] warning: no files found for {prefix} in the HF dataset -- skipping")
                continue
            for f in matches:
                hf_hub_download(
                    repo_id="zhouxueyang/LIBERO-Pro",
                    repo_type="dataset",
                    filename=f,
                    local_dir=str(libero_dir),
                )
                count += 1
    print(f"[fetch] downloaded {count} files into {libero_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
