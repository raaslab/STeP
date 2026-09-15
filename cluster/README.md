# Running STeP's LIBERO-Pro eval on UMIACS Nexus

This mirrors the local dev setup (`cap-x/.venv-libero`) as a conda environment,
plus a SLURM array job for the 6 (suite, perturbation) combos Cap-X itself
evaluates on: `libero_spatial`/`libero_object`/`libero_goal` x `swap`
(Position) / `task` (Task).

## 1. Repos to clone on the cluster

| Repo | URL | Notes |
| --- | --- | --- |
| **STeP** (the big one) | `https://github.com/raaslab/STeP`, branch `main` | The planner/executor + everything in `cluster/`. |
| cap-x | `https://github.com/capgym/cap-x` | Cloned automatically by `setup_env.sh`, with only the submodules the LIBERO pipeline needs (see below) -- you don't need to clone it yourself. |

cap-x pulls in the rest as **git submodules**; `setup_env.sh` initializes only
the four this pipeline actually uses (skips `b1k`, `curobo`, `verl`, and the
second unrelated `robosuite` checkout, which aren't needed here and are large):

- `capx/third_party/LIBERO-PRO` (fork of `Zxy-MLlab/LIBERO-PRO`) -- the `libero` package itself
- `capx/third_party/contact_graspnet_pytorch` -- grasp generation (its model checkpoint is git-tracked, no separate download)
- `capx/third_party/libero_dependencies/robosuite` (Max-Fu fork, `maxf/egl_context` branch) -- needed for headless (EGL) rendering
- `capx/third_party/sam3`

Both repos may be private -- make sure you have SSH keys or a PAT set up for
`github.com/raaslab` and `github.com/capgym` on the cluster before cloning.

**Not needed here:** `/home/amisha/Documents/step/LIBERO-Constrained` on the
dev machine is a separate, unrelated LIBERO fork for a different feature --
nothing in this pipeline imports it, don't bother cloning it.

## 2. Storage

Nexus home (`/nfshomes/<user>`) is only 30GB and will not fit the conda env
(torch + CUDA wheels alone are several GB) plus rollout outputs/videos. Do
everything under network scratch instead:

```bash
mkdir -p /fs/nexus-scratch/$USER/step
cd /fs/nexus-scratch/$USER/step
git clone https://github.com/raaslab/STeP
```

Also point conda's own package cache/envs there if your `~/.condarc` doesn't
already, e.g. `conda config --add envs_dirs /fs/nexus-scratch/$USER/conda/envs`.

## 3. Environment setup

```bash
cd /fs/nexus-scratch/$USER/step/STeP
bash cluster/setup_env.sh
```

This clones cap-x (with the trimmed submodule set) as a sibling of STeP,
creates the `step-libero` conda env (`cluster/environment.yml` +
`cluster/requirements-libero.txt`, both dumped from the working dev
environment), editable-installs the local packages, and downloads the
LIBERO-Pro swap/task perturbation bddl+init files from the official
HuggingFace dataset (`zhouxueyang/LIBERO-Pro` -- the same source Cap-X's own
data came from). Re-running it is safe; each step skips if already done.

Sanity-check before submitting anything:

```bash
conda activate step-libero
cd /fs/nexus-scratch/$USER/step/STeP
python -c "from libero.benchmark import get_benchmark; print(get_benchmark('libero_goal_swap')().n_tasks)"
```

Should print `10` with no errors.

## 4. Submitting the SLURM job

```bash
mkdir -p logs
sbatch cluster/submit_libero_pro.slurm
```

This is an array job (`--array=0-5`), one array task per (suite,
perturbation) combo, requesting 1 GPU / 4 CPUs / 32GB / up to 3 days each
(the `default` QoS limits -- see below). Each task's rollouts go to
`outputs/libero_pro_slurm_<jobid>/<suite>_<perturbation>/`.

**Nexus specifics used in the script** (from the
[Nexus wiki](https://wiki.umiacs.umd.edu/umiacs/index.php/Nexus)):
- `--partition=tron` (default general-use partition) + `--qos=default` (1
  GPU, 4 CPU, 32GB, 3-day max per job) -- edit `--account=CHANGE_ME` in
  `submit_libero_pro.slurm` to your actual UMIACS/CSD-sponsored account
  first; run `show_partitions` to see what you're allowed to use.
- For longer or higher-throughput runs at the cost of preemption, switch to
  `--qos=scavenger --account=scavenger` (uncapped CPU/GPU/RAM, still 3-day
  max) instead of editing the per-job resource limits.
- GPUs are requested via `--gres=gpu:1`, standard Nexus syntax.

**Pilot before the full grid:** each combo's full protocol is 10 tasks x 50
states = 500 rollouts; six combos is 3,000 total. Try a small slice first:

```bash
sbatch --export=ALL,MAX_STATES=10 cluster/submit_libero_pro.slurm
```

Drop `MAX_STATES` (or unset it) to run the full 50 states/task once you're
happy with the pilot.

**Per-rollout timeout:** the script already passes `--rollout-timeout 300`
(5 minutes, per your existing preference) -- any single stuck rollout gets
killed and counted as a failure rather than eating your walltime budget.

## 4b. Plain (unperturbed) LIBERO

To run the standard 4-suite LIBERO benchmark instead of/alongside LIBERO-Pro
(no perturbation applied):

```bash
mkdir -p logs
sbatch cluster/submit_libero_vanilla.slurm
```

Array job over `libero_spatial`/`libero_object`/`libero_goal`/`libero_10`,
same resource requests and `MAX_STATES` pilot override as
`submit_libero_pro.slurm`, just without a `--perturbation` flag. Results land
in `outputs/libero_vanilla_slurm_<jobid>/<suite>/`.

The same grid runner handles vanilla runs too, if you'd rather not use SLURM
arrays -- pass `--perturbations none`:

```bash
python scripts/run_libero_pro_eval.py \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --perturbations none \
  --outdir outputs/libero_vanilla
```

You can also mix vanilla and perturbed dimensions in one grid, e.g.
`--perturbations none swap task`, to get a vanilla baseline column alongside
the perturbed ones in the same summary table.

## 5. Aggregating results

Once all 6 array tasks finish (`sacct -j <jobid>` to check), fold their
`results_summary.json` files into one suite x perturbation grid, matching
the LIBERO-Pro leaderboard's shape:

```bash
python scripts/run_libero_pro_eval.py \
  --outdir outputs/libero_pro_slurm_<jobid> \
  --skip-existing
```

`--skip-existing` makes this pick up the already-completed rollouts instead
of re-running them; it writes `outputs/libero_pro_slurm_<jobid>/libero_pro_summary.json`
plus a printed table.

## Consistency with Cap-X

The suite set (`libero_spatial`/`libero_object`/`libero_goal`, no
`libero_10`), the two dimensions (`swap`/`task`), and the 50-trials-per-task
protocol are all cross-referenced from Cap-X's own
`capx/envs/scripts/run_libero_batch.py` and
`capx/scripts/start_servers_and_eval.sh` (which states the full run as
"6 suites x 10 tasks x 50 trials = 3000 total trials") -- see
`step/libero_pro_eval.py`'s module docstring/comments for the exact citation.
The perturbed bddl/init data itself is the same official HuggingFace release
(`zhouxueyang/LIBERO-Pro`) Cap-X's own checkout pulled from, not a
re-derivation.
