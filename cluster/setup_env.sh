#!/usr/bin/env bash
# One-time environment setup for running STeP's LIBERO-Pro rollouts on a
# cluster (e.g. UMIACS Nexus). Run this FROM INSIDE a freshly cloned STeP
# checkout (this script lives at STeP/cluster/setup_env.sh).
#
# What it does:
#   1. Clones cap-x (with only the submodules the LIBERO pipeline needs) as
#      a sibling directory of STeP, unless --cap-x-root points at an
#      existing checkout.
#   2. Creates the `step-libero` conda environment from environment.yml.
#   3. Editable-installs the local packages (capx, contact_graspnet_pytorch,
#      LIBERO-PRO's libero, robosuite, sam3) into that environment.
#   4. Downloads LIBERO-Pro's swap/task perturbation bddl+init files.
#
# Usage:
#   cd STeP && bash cluster/setup_env.sh [--cap-x-root /path/to/cap-x]
#
# Re-running is safe -- each step is skipped if its result already exists.
set -euo pipefail

STEP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPX_ROOT="$(dirname "$STEP_ROOT")/cap-x"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cap-x-root) CAPX_ROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "[setup] STeP root:  $STEP_ROOT"
echo "[setup] cap-x root: $CAPX_ROOT"

# --- 1. Clone cap-x (only the submodules the LIBERO pipeline needs) ---
if [[ -d "$CAPX_ROOT/.git" ]]; then
  echo "[setup] cap-x already present at $CAPX_ROOT, skipping clone"
else
  echo "[setup] cloning cap-x into $CAPX_ROOT"
  git clone \
    --recurse-submodules=capx/third_party/LIBERO-PRO \
    --recurse-submodules=capx/third_party/contact_graspnet_pytorch \
    --recurse-submodules=capx/third_party/libero_dependencies/robosuite \
    --recurse-submodules=capx/third_party/sam3 \
    https://github.com/capgym/cap-x "$CAPX_ROOT"
fi

LIBERO_PRO_ROOT="$CAPX_ROOT/capx/third_party/LIBERO-PRO"
CONTACT_GRASPNET_ROOT="$CAPX_ROOT/capx/third_party/contact_graspnet_pytorch"
ROBOSUITE_ROOT="$CAPX_ROOT/capx/third_party/libero_dependencies/robosuite"
SAM3_ROOT="$CAPX_ROOT/capx/third_party/sam3"

for d in "$LIBERO_PRO_ROOT" "$CONTACT_GRASPNET_ROOT" "$ROBOSUITE_ROOT" "$SAM3_ROOT"; do
  if [[ ! -d "$d" ]]; then
    echo "[setup] error: expected submodule directory missing: $d" >&2
    echo "[setup] try: (cd $CAPX_ROOT && git submodule update --init -- ${d#$CAPX_ROOT/})" >&2
    exit 1
  fi
done

# --- 2. Conda environment ---
if conda env list | grep -qE '^\s*step-libero\s'; then
  echo "[setup] conda env step-libero already exists, skipping creation"
else
  echo "[setup] creating conda env step-libero"
  (cd "$STEP_ROOT/cluster" && conda env create -f environment.yml)
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate step-libero

# --- 3. Editable installs ---
echo "[setup] editable-installing local packages"
pip install -e "$CAPX_ROOT"
pip install -e "$CONTACT_GRASPNET_ROOT"
pip install -e "$LIBERO_PRO_ROOT/libero"
pip install -e "$ROBOSUITE_ROOT"
pip install -e "$SAM3_ROOT"

# --- 4. LIBERO-Pro perturbation data (swap + task) ---
echo "[setup] fetching LIBERO-Pro swap/task perturbation data"
python "$STEP_ROOT/cluster/fetch_libero_pro_data.py" \
  --libero-pro-root "$LIBERO_PRO_ROOT" \
  --dimensions swap task

# --- 5. Contact-GraspNet checkpoint path ---
# configs/libero_template.json intentionally ships with contact_graspnet.
# checkpoint_dir = "" (no personal absolute path baked into git) -- it's
# meant to be supplied via this env var instead. Without it, every pick
# silently falls back to the heuristic grasp generator (a caught exception,
# not a hard failure) rather than the real model -- confirmed missing on
# the dev machine's own shell profile too, so write it out here rather than
# assume it's set.
echo "export CONTACT_GRASPNET_CHECKPOINT_DIR=\"$CONTACT_GRASPNET_ROOT/checkpoints/contact_graspnet\"" > "$STEP_ROOT/cluster/env.sh"
echo "[setup] wrote $STEP_ROOT/cluster/env.sh -- source it (or the SLURM scripts, which already do) before running anything"

echo "[setup] done. Sanity check:"
echo "  conda activate step-libero"
echo "  source $STEP_ROOT/cluster/env.sh"
echo "  cd $STEP_ROOT"
echo "  python -c \"from libero.benchmark import get_benchmark; print(get_benchmark('libero_goal_swap')().n_tasks)\""
