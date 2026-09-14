#!/usr/bin/env bash
#SBATCH --job-name=step-libero-pro
#SBATCH --array=0-5
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
# Runs STeP over one (suite, perturbation) combo per array task, mirroring
# Cap-X's own LIBERO-Pro protocol: libero_spatial/object/goal x swap/task,
# all 10 tasks, all 50 init-state trials/task (see step/libero_pro_eval.py
# for where these defaults are cross-referenced from Cap-X's own code).
#
# Submit from the STeP repo root:
#   mkdir -p logs
#   sbatch cluster/submit_libero_pro.slurm
#
# Override trial count for a quick pilot before committing to the full 50/task:
#   sbatch --export=ALL,MAX_STATES=10 cluster/submit_libero_pro.slurm

set -euo pipefail

STEP_ROOT="$SLURM_SUBMIT_DIR"
cd "$STEP_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate step-libero

export MUJOCO_GL=egl

# suite/perturbation grid -- must match CAPX_SUITES x CAPX_PERTURBATIONS in
# step/libero_pro_eval.py. Index with $SLURM_ARRAY_TASK_ID (0-5).
SUITES=(libero_spatial libero_spatial libero_object libero_object libero_goal libero_goal)
PERTURBATIONS=(swap task swap task swap task)

SUITE="${SUITES[$SLURM_ARRAY_TASK_ID]}"
PERTURBATION="${PERTURBATIONS[$SLURM_ARRAY_TASK_ID]}"
MAX_STATES="${MAX_STATES:-}"   # empty => --all-states (50/task, the full protocol)

OUTDIR="outputs/libero_pro_slurm_${SLURM_ARRAY_JOB_ID}/${SUITE}_${PERTURBATION}"
mkdir -p "$OUTDIR"

echo "[slurm] task $SLURM_ARRAY_TASK_ID: suite=$SUITE perturbation=$PERTURBATION outdir=$OUTDIR"

if [[ -n "$MAX_STATES" ]]; then
  STATE_FLAG=(--max-states "$MAX_STATES")
else
  STATE_FLAG=(--all-states)
fi

python scripts/run_libero_suite.py \
  --suite "$SUITE" \
  --perturbation "$PERTURBATION" \
  "${STATE_FLAG[@]}" \
  --rollout-timeout 300 \
  --skip-existing \
  --outdir "$OUTDIR"
