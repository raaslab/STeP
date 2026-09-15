#!/usr/bin/env bash
#SBATCH --job-name=step-libero-vanilla
#SBATCH --array=0-3
#SBATCH --partition=tron
#SBATCH --qos=default
#SBATCH --account=CHANGE_ME          # your UMIACS/CSD-sponsored account; `show_partitions` lists what you can use
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=3-00:00:00            # default QoS max; use --qos=scavenger (+ --account=scavenger) for longer, preemptible runs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Plain (unperturbed) LIBERO -- one array task per standard suite:
# libero_spatial, libero_object, libero_goal, libero_10. No --perturbation
# flag is passed, so this is the ordinary benchmark, not LIBERO-Pro.
#
# Submit from the STeP repo root:
#   mkdir -p logs
#   sbatch cluster/submit_libero_vanilla.slurm
#
# Override trial count for a quick pilot before committing to the full 50/task:
#   sbatch --export=ALL,MAX_STATES=10 cluster/submit_libero_vanilla.slurm

set -euo pipefail

STEP_ROOT="$SLURM_SUBMIT_DIR"
cd "$STEP_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate step-libero

export MUJOCO_GL=egl

SUITES=(libero_spatial libero_object libero_goal libero_10)
SUITE="${SUITES[$SLURM_ARRAY_TASK_ID]}"
MAX_STATES="${MAX_STATES:-}"   # empty => --all-states (50/task)

OUTDIR="outputs/libero_vanilla_slurm_${SLURM_ARRAY_JOB_ID}/${SUITE}"
mkdir -p "$OUTDIR"

echo "[slurm] task $SLURM_ARRAY_TASK_ID: suite=$SUITE (no perturbation) outdir=$OUTDIR"

if [[ -n "$MAX_STATES" ]]; then
  STATE_FLAG=(--max-states "$MAX_STATES")
else
  STATE_FLAG=(--all-states)
fi

python scripts/run_libero_suite.py \
  --suite "$SUITE" \
  "${STATE_FLAG[@]}" \
  --rollout-timeout 300 \
  --skip-existing \
  --outdir "$OUTDIR"
