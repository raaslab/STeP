# STeP

Minimal refactor target for the old `vlm-stl` codebase.

This first slice keeps only the LIBERO benchmark path and removes the old
task-spec-to-STL compiler from the main System 2 flow. The LLM now produces the
same STL formula JSON that System 1 already executes.

## Current Flow

1. Build or load a local LIBERO `scene_context` from `configs/libero_template.json`.
2. Capture a LIBERO scene image unless `--image` is supplied.
3. Ask the LLM for a direct STL skill graph JSON.
4. Validate the direct STL plan shape.
5. Execute it with System 1 (`step/system_one/`).

## Installation

STeP's own [pyproject.toml](pyproject.toml) only declares `openai` -- all the
heavy lifting (LIBERO, robosuite, MuJoCo, contact-graspnet, OMPL) comes from
a sibling [cap-x](https://github.com/capgym/cap-x) checkout that every
`scripts/run_libero_*.py` adds to `sys.path` at runtime. Concretely, that
means:

- STeP and cap-x are expected to live next to each other, e.g.
  `~/Documents/step/{STeP,cap-x}`.
- Everything LIBERO-related runs with cap-x's `.venv-libero` (a uv venv,
  Python 3.12) active -- not a venv inside STeP itself.

Requires a Linux machine with a CUDA-capable GPU (for LIBERO/MuJoCo
rendering and contact-graspnet) and Python 3.12.

### Quick setup

```bash
cd STeP
bash scripts/setup_env.sh
```

This clones cap-x as a sibling of STeP if it isn't already there, creates
its `.venv-libero` uv venv with the `libero` + `contactgraspnet` extras,
bridges in OMPL (see below), and writes `scripts/env.sh` with
`CONTACT_GRASPNET_CHECKPOINT_DIR` pointed at the checkpoint bundled in
cap-x's `contact_graspnet_pytorch` submodule -- `configs/libero_template.json`
ships that setting blank on purpose, and without it every pick silently
falls back to a heuristic grasp generator instead of the real model. Pass
`--cap-x-root /path/to/cap-x` if you already have a checkout elsewhere.
Re-running is safe; each step skips if already done.

**OMPL** has no prebuilt PyPI wheel, so `uv sync` alone can't get you its
Python bindings. The script bridges it in from a conda-forge build instead:
it creates (or reuses) a small `ompl_bridge` conda env just to fetch
prebuilt binaries, then copies the `ompl` package plus its own
libstdc++/libgcc_s/boost into `.venv-libero` so it never touches your
system's libstdc++. This needs `conda` on `PATH` once; nothing conda-related
is required afterwards to run STeP. If `conda` isn't available, you'll need
to get OMPL's Python bindings into `.venv-libero` some other way (e.g.
building it from source against that venv's Python).

### Before running anything

```bash
source ../cap-x/.venv-libero/bin/activate
source scripts/env.sh
export OPENAI_API_KEY=...
```

Sanity check:

```bash
python -c "from libero.benchmark import get_benchmark; print(get_benchmark('libero_spatial')().n_tasks)"
```

Should print `10`.

Local LIBERO configs:

- `configs/libero_template.json`: default LIBERO scene/runtime config.
- `configs/libero_safety.json`: copied LIBERO safety benchmark config.

Package layout:

- `step/system_one/`: tuned execution, environment, MPC, monitoring, and LIBERO adapter code ported from the legacy implementation.
- `step/system_two/`: new minimal direct-STL planning code.
- `step/libero_runner.py`: thin LIBERO entry point connecting System 2 plans to System 1 execution.

## Direct STL Plan

The plan schema is intentionally small and lives at:

```text
schemas/stl_plan_schema.json
```

The formula language is JSON:

```json
{
  "type": "eventually",
  "interval": [0, 90],
  "child": {
    "type": "predicate",
    "name": "Grasped",
    "args": ["object"]
  }
}
```

Each subtask only needs `subtask_id`, `skill`, `description`, `params`, and
`formula`. Supported formula nodes are `true`, `predicate`, `not`, `and`,
`implies`, `always`, `eventually`, and `until`.

`Grasped` takes an optional second argument: the object's height in meters,
e.g. `Grasped(object, 0.09)`. System 1 uses it to grasp at roughly the
object's mid-height instead of its base pose (table level), which matters for
tall non-rim objects like cans and bottles. Rim-grasp objects (bowls, mugs,
...) should keep using `params.grasp_offset_xyz` (see the developer prompt's
rule 7a) instead.

## Run A Local Smoke Test

This validates and prints a prebuilt direct STL plan without calling the model:

```bash
python scripts/run_libero_step.py \
  --compiled-plan examples/pick_place_direct_stl_plan.json \
  --dry-run
```

## Run Direct Planning

Set `OPENAI_API_KEY`, then run:

```bash
python scripts/run_libero_step.py \
  --suite libero_spatial \
  --task-id 0 \
  --init-state-id 0 \
  --dry-run
```

Remove `--dry-run` to execute the generated plan in LIBERO.

## Run A LIBERO Suite

Dry-run the first state of every task in `libero_spatial`:

```bash
python scripts/run_libero_suite.py --suite libero_spatial --dry-run
```

Run the suite with one rollout per task:

```bash
python scripts/run_libero_suite.py --suite libero_spatial
```

Use `--max-states N` or `--all-states` to run more initial states per task.
Outputs are grouped under `outputs/libero_suite/<suite>__taskXX/state_YYY/`.
