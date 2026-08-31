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
5. Execute it with the legacy LIBERO System 1 executor.

The legacy executor is still reused from `../vlm-stl` for now, but STeP owns
its default LIBERO configuration in `configs/`.

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
