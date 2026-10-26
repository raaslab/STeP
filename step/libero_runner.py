from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

from .io import load_json, save_json
from .system_two.stl_plan import normalize_legacy_pregrasp_grasp, print_formulas, validate_stl_plan

# LIBERO-Pro's locally-available generalization dimensions (see
# libero/libero/bddl_files/<suite>_<dim>/ in the vendored LIBERO-PRO checkout).
# "swap" = Position, "task" = Task, "object" = Object, "lan" = Semantic.
# LIBERO-Pro's fifth dimension, Environment, has no pre-built bddl/init files
# in this checkout, so it isn't offered here.
LIBERO_PRO_PERTURBATIONS = ("swap", "task", "object", "lan")

# EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md): schema picked by --stl-mode
# when --schema isn't given explicitly.
_DEFAULT_SCHEMA_FOR_MODE = {
    "stl_spec": "schemas/stl_global_plan_schema.json",
    "skill_graph": "schemas/stl_plan_schema.json",
    # The baseline emits the same skill_graph shape, minus every specification
    # term, so System 1 needs no special case to execute it.
    "no_stl": "schemas/stl_plan_schema.json",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run minimal STeP on a LIBERO task.")
    p.add_argument("--base-scene-context", default="configs/libero_template.json")
    p.add_argument(
        "--schema",
        default=None,
        help="STL plan JSON schema. Defaults to the schema matching --stl-mode (see _DEFAULT_SCHEMA_FOR_MODE).",
    )
    p.add_argument(
        "--stl-mode",
        default="stl_spec",
        choices=["stl_spec", "skill_graph", "no_stl"],
        help=(
            "System 2 output mode. 'stl_spec' (default, EXPERIMENTAL -- see "
            "STL_DECOMPOSITION_PLAN.md): the VLM emits one global STL formula and "
            "step/system_two/stl_plan.py:decompose() deterministically derives the "
            "subtask sequence, instead of the VLM deciding subtask boundaries "
            "itself. Pass --stl-mode skill_graph to revert to the original "
            "behavior where the VLM emits the pre-split per-subtask plan directly."
        ),
    )
    p.add_argument("--suite", default=None)
    p.add_argument(
        "--perception",
        default="privileged",
        choices=["privileged", "vision"],
        help=(
            "How System 1 and System 2 learn object names, poses and geometry. "
            "'privileged' (default) reads them from the simulator. 'vision' "
            "rebuilds the scene from the agentview + wrist RGB-D cameras and "
            "proprioception only (SAM3 + VLM naming, see step/perception/)."
        ),
    )
    p.add_argument(
        "--perturbation",
        default=None,
        choices=list(LIBERO_PRO_PERTURBATIONS) + ["random"],
        help=(
            "Run LIBERO-Pro's perturbed variant of --suite instead of plain LIBERO. "
            "'swap' (Position), 'task' (Task), 'object' (Object), 'lan' (Semantic) "
            "resolve --suite '<suite>' to '<suite>_<perturbation>' (e.g. "
            "libero_goal_swap). 'random' draws one of those four for this run and "
            "records which one was drawn under platform.libero.perturbation in the "
            "saved scene context. Omit for plain LIBERO (default)."
        ),
    )
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--init-state-id", type=int, default=None)
    p.add_argument("--bddl-file", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--outdir", default="outputs/libero")
    p.add_argument("--model", default=None)
    p.add_argument("--compiled-plan", default=None, help="Execute an existing direct STL plan JSON.")
    p.add_argument(
        "--avoid-region",
        default=None,
        help=(
            "LIBERO-Constrained: name of the body to treat as a keep-out zone "
            "(e.g. cookies_1). Materializes a real SDF constraint the MPC and the "
            "safety robustness terms both see, rather than only mentioning it in "
            "the prompt."
        ),
    )
    p.add_argument(
        "--avoid-margin",
        type=float,
        default=0.06,
        help="Keep-out radius in metres around --avoid-region. Difficulty knob.",
    )
    p.add_argument(
        "--benchmark-task",
        default=None,
        help=(
            "Run one task from benchmarks/libero_constrained/tasks.json by id "
            "(e.g. libero_10/0/S-zone). Sets the suite, task, init state, prompt and "
            "keep-out zone from the frozen manifest. Anything given explicitly on the "
            "command line wins."
        ),
    )
    p.add_argument(
        "--zone-center-xy",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help=(
            "Place the --avoid-region keep-out at this world xy as a VIRTUAL zone -- a "
            "marked region with no object behind it. Without this the region must name a "
            "real scene object, whose position then decides where the zone lands."
        ),
    )
    p.add_argument(
        "--place-obstacle-in-path",
        action="store_true",
        help=(
            "LIBERO-Constrained scene construction: at reset, move --avoid-region "
            "to the midpoint between the manipulated object and its goal, so the "
            "keep-out is on the transport path and must be planned around. "
            "Without this most tasks' obstacles sit far enough away that the "
            "constraint never binds."
        ),
    )
    p.add_argument(
        "--no-enforce-keepout",
        action="store_true",
        help=(
            "RQ2 ablation. Declare and SCORE the --avoid-region keep-out exactly "
            "as normal, but do not let System 1 plan or control around it, so the "
            "STL constraint is monitored only. This is the 'STL does not reach "
            "low-level control' arm; both arms stay directly comparable because "
            "the scene and the metric are identical."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Stop after writing scene/prompt/plan files.")
    p.add_argument("--render", action="store_true")
    p.add_argument("--save-video", default=None, help="Optional path for an execution video, e.g. outputs/libero/pick.mp4.")
    p.add_argument(
        "--max-pipeline-retries",
        type=int,
        default=3,
        help=(
            "On a failed execution (crash or success=False), recall System 2 for a "
            "new plan and retry, this many times total -- WITHOUT resetting the "
            "env: whatever the failure left behind (e.g. a dropped object sitting "
            "wherever it landed) is exactly what the next attempt starts from and "
            "what the recalled plan is generated against, using a freshly captured "
            "image/object-position snapshot of the live scene, not the original "
            "one. Skipped when --compiled-plan is given (there is no System 2 to "
            "recall in that mode); the same fixed plan is simply re-run against the "
            "same live, unreset env instead. Pass 1 to disable."
        ),
    )
    return p.parse_args()


def _resolve_prompt(args: argparse.Namespace, scene_context: dict[str, Any]) -> str:
    if args.prompt:
        return args.prompt.strip()

    try:
        from step.system_one.libero_scene import load_libero_task_from_scene_context

        task = load_libero_task_from_scene_context(scene_context)
        if getattr(task, "language", None):
            return str(task.language)
    except Exception:
        pass

    return "Execute the LIBERO manipulation task."


_BENCHMARK_MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks" / "libero_constrained" / "tasks.json"


def _load_benchmark_task(task_id: str) -> dict[str, Any]:
    """One entry from the LIBERO-Constrained manifest, by its id.

    The manifest is the reviewed, frozen definition: prompt, zone geometry and
    scoring spec all come from it, so a run cannot quietly differ from the
    benchmark it claims to be running.
    """
    if not _BENCHMARK_MANIFEST.exists():
        raise SystemExit(f"benchmark manifest not found at {_BENCHMARK_MANIFEST}")
    manifest = load_json(_BENCHMARK_MANIFEST)
    for task in manifest.get("tasks", []):
        if task.get("id") == task_id:
            return task
    ids = [t.get("id") for t in manifest.get("tasks", [])]
    raise SystemExit(
        f"no benchmark task {task_id!r}. {len(ids)} available, e.g. " + ", ".join(ids[:3])
    )


def _apply_benchmark_task(args: argparse.Namespace) -> dict[str, Any] | None:
    """Fill in suite/task/prompt/zone from --benchmark-task, without overriding
    anything the caller set explicitly on the command line."""
    if not getattr(args, "benchmark_task", None):
        return None
    task = _load_benchmark_task(args.benchmark_task)
    if args.suite is None:
        args.suite = task["suite"]
    if args.task_id is None:
        args.task_id = int(task["task_id"])
    if args.init_state_id is None:
        args.init_state_id = int(task.get("init_state_id", 0))
    if not args.prompt:
        args.prompt = task["prompt"]
    zone = task.get("zone")
    if zone and not args.avoid_region:
        # A virtual zone: a named region at a fixed point with no backing object.
        # Named to match how the prompt refers to it ("the red zone marked on the
        # table"), since System 2 has to resolve the phrase to this entity.
        args.avoid_region = "marked_zone_1"
        args.avoid_margin = float(zone["radius_m"])
        args.zone_center_xy = [float(c) for c in zone["center_xy"]]
    print(f"[benchmark] {task['id']}  family={task['family']}  scoring={task['scoring']['type']}")
    return task


def _prepare_libero_scene(args: argparse.Namespace, outdir: Path) -> tuple[dict[str, Any], Path, str, Path]:
    scene_context = load_json(args.base_scene_context)
    platform = scene_context.setdefault("platform", {})
    platform["backend"] = "libero"
    libero = platform.setdefault("libero", {})
    if args.suite is not None:
        perturbation = args.perturbation
        if perturbation == "random":
            perturbation = random.choice(LIBERO_PRO_PERTURBATIONS)
            print(f"[step] LIBERO-Pro random perturbation selected: {perturbation}")
        libero["suite"] = f"{args.suite}_{perturbation}" if perturbation else args.suite
        libero["perturbation"] = perturbation
    if args.task_id is not None:
        libero["task_id"] = int(args.task_id)
    if args.init_state_id is not None:
        libero["init_state_id"] = int(args.init_state_id)
    if args.bddl_file is not None:
        libero["bddl_file"] = args.bddl_file
    perception = platform.setdefault("perception", {})
    perception["mode"] = args.perception
    if args.perception == "vision":
        perception["instruction"] = _resolve_prompt(args, scene_context)
        perception["debug_dir"] = str(outdir / "perception")

    try:
        from step.system_one.libero_env import LiberoManipulationEnv

        scene_context = LiberoManipulationEnv.snapshot_scene_context(scene_context)
    except Exception as exc:
        print(f"[step] warning: LIBERO scene snapshot failed: {exc}")

    # Declare the keep-out zone *after* the snapshot: apply_libero_experiment_
    # geometry() needs the obstacle's discovered xy to place the SDF circle, and
    # that only exists once the scene has been snapshotted. The real execution
    # env then materializes it during construction.
    if args.avoid_region and args.perception == "vision" and not getattr(args, "zone_center_xy", None):
        perceived = {e.get("name") for e in scene_context.get("entities", [])}
        if args.avoid_region not in perceived:
            raise SystemExit(
                f"--avoid-region {args.avoid_region!r} names a simulator body, which vision mode cannot see "
                f"by that name. Perceived entities: {sorted(n for n in perceived if n)}. Pass one of those, "
                "or a virtual zone via --zone-center-xy / --benchmark-task."
            )
    if args.avoid_region:
        avoid: dict[str, Any] = {
            "region": args.avoid_region,
            "margin": float(args.avoid_margin),
        }
        if getattr(args, "zone_center_xy", None):
            avoid["center_xy"] = [float(c) for c in args.zone_center_xy]
        platform.setdefault("libero_experiment", {})["avoid_region"] = dict(avoid)
        scene_context.setdefault("platform", {}).setdefault("libero_experiment", {})[
            "avoid_region"
        ] = dict(avoid)
        # --no-enforce-keepout is the RQ2 ablation: the zone is still declared,
        # still turned into an SDF constraint, and still scored exactly the
        # same -- System 1 just never plans around it. Without this the "STL
        # off" arm would also be an unscored arm, and the two conditions would
        # not be comparable.
        for block in (platform["libero_experiment"], scene_context["platform"]["libero_experiment"]):
            # The no-STL baseline has no specification for the constraint to live
            # in, so it cannot reach the planner either. Leaving enforcement on
            # would hand the baseline the very mechanism it is meant to lack.
            block["enforce_keepout"] = not args.no_enforce_keepout and args.stl_mode != "no_stl"
            # Scene construction, not enforcement: the obstacle is relocated in
            # BOTH arms of the ablation, so the two see an identical scene and
            # differ only in whether System 1 plans around the zone.
            block["avoid_region"]["place_in_path"] = bool(args.place_obstacle_in_path)

    prompt = _resolve_prompt(args, scene_context)
    outdir.mkdir(parents=True, exist_ok=True)
    scene_path = outdir / "libero_scene_context.json"
    prompt_path = outdir / "prompt.txt"
    save_json(scene_context, scene_path)
    prompt_path.write_text(prompt)
    return scene_context, scene_path, prompt, prompt_path


def _capture_image(args: argparse.Namespace, scene_context: dict[str, Any], outdir: Path) -> str:
    if args.image:
        return args.image

    from step.system_one.libero_env import LiberoManipulationEnv

    capture_path = outdir / "system2_scene.png"
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    return LiberoManipulationEnv.capture_initial_scene_image(
        save_path=str(capture_path),
        scene_context=scene_context,
        camera_name="agentview",
    )


def _capture_live_scene_and_image(env: Any, scene_context: dict[str, Any], outdir: Path, attempt: int) -> tuple[dict[str, Any], str]:
    """Snapshot the *live, unreset* env's current object positions and a
    fresh camera frame -- for recalling System 2 mid-run against wherever
    the scene actually is after a failure (e.g. a dropped can still sitting
    where it fell), not the original init-state layout.

    Mirrors LiberoManipulationEnv.snapshot_scene_context/
    capture_initial_scene_image, which do the same thing but always
    build and reset a brand new env -- exactly what must NOT happen here.
    """
    from step.system_one.libero_scene import (
        infer_objects_from_mujoco,
        infer_objects_from_obs,
        refresh_drawer_blocking_flags,
        update_scene_context_from_objects,
    )

    if getattr(env, "perception", None) is not None:
        from step.system_one.libero_env import perceived_scene_context

        env.perception_keyframe(reason="recall")
        fresh_context = perceived_scene_context(scene_context, env)
    else:
        obs = getattr(env, "last_raw_obs", None)
        objects = (infer_objects_from_obs(obs) if obs is not None else {}) or infer_objects_from_mujoco(env.env)
        fresh_context = update_scene_context_from_objects(scene_context, objects, source="pipeline_recall_live")
    try:
        fresh_context = refresh_drawer_blocking_flags(fresh_context, env.env)
    except Exception as exc:
        print(f"[step] warning: refresh_drawer_blocking_flags failed: {exc}")

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.image as mpimg

    image_path = outdir / f"recall_attempt_{attempt:02d}_scene.png"
    frame = env.render_rgb_frame(camera_name="agentview", width=512, height=512)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(str(image_path), frame)
    return fresh_context, str(image_path)


def _annotate_already_satisfied_objects(
    fresh_context: dict[str, Any], env: Any, previous_plan: dict[str, Any]
) -> list[str]:
    """Stamp a "goal already reached" note onto any object the previous
    (just-failed) plan's own place subtasks targeted, if it's now actually
    resting inside its named goal container.

    Exists because a pipeline recall only replans after *some* subtask
    failed (e.g. a second can dropped in transport) -- but the LLM building
    the new plan has no way to tell a first object is already done from raw
    coordinates alone, and nothing in the prompt told it to check. Confirmed
    via a real two-object rollout (alphabet soup + tomato sauce -> basket):
    after the tomato sauce's transport failed, the recalled plan re-picked
    the alphabet soup too even though it was already sitting in the basket
    from a prior successful attempt -- it failed a second time for real
    (its post-place pose has no feasible grasp candidate) and burned the
    entire retry on a no-op instead of just finishing the tomato sauce.
    Mirrors libero_scene.py's _annotate_drawer_clearance_conflicts:
    precompute the geometric fact in code and stamp the verdict directly
    onto the entity, rather than trust the LLM to (re-)derive it itself.
    """
    from .system_one.motion.scene_query import SceneQuery, object_within_container_footprint, resolve_body_name

    subtasks = ((previous_plan.get("system2_plan") or {}).get("subtasks")) or []
    place_targets = [
        ((sub.get("params") or {}).get("object"), (sub.get("params") or {}).get("goal_region"))
        for sub in subtasks
        if sub.get("skill") == "place"
    ]
    if not place_targets:
        return []

    scene = SceneQuery.from_robosuite_sim(env.env.sim)
    entities = fresh_context.get("entities")
    if not isinstance(entities, list):
        return []
    by_name = {e.get("name"): e for e in entities if isinstance(e, dict) and e.get("name")}

    satisfied: list[str] = []
    for object_name, goal_region in place_targets:
        if not object_name or not goal_region:
            continue
        # goal_region is not always a named entity string -- rule 6 also
        # allows an explicit point target ({"type": "point", "xy": [...],
        # "z": ..., "radius": ...}, e.g. "the back compartment of the
        # caddy", which has no body of its own to resolve). Confirmed via a
        # real recall on exactly that task: resolve_body_name got the dict
        # itself as its name argument and crashed inside mujoco's own
        # mj_name2id (a TypeError, not the KeyError already handled below)
        # rather than just reporting this object's goal as unresolvable.
        if isinstance(goal_region, dict):
            try:
                object_body = resolve_body_name(scene, object_name)
                object_pos, _ = scene.body_pose(object_body)
            except KeyError:
                continue
            xy = goal_region.get("xy")
            z = goal_region.get("z")
            radius = goal_region.get("radius")
            if not isinstance(xy, (list, tuple)) or len(xy) != 2 or z is None or radius is None:
                continue
            dx, dy = object_pos[0] - float(xy[0]), object_pos[1] - float(xy[1])
            within_xy = (dx * dx + dy * dy) ** 0.5 <= float(radius)
            within_z = abs(object_pos[2] - float(z)) <= 0.10
            if not (within_xy and within_z):
                continue
        elif isinstance(goal_region, str):
            try:
                object_body = resolve_body_name(scene, object_name)
                container_body = resolve_body_name(scene, goal_region)
            except KeyError:
                continue
            if object_within_container_footprint(scene, object_body, container_body) is not True:
                continue
        else:
            continue
        satisfied.append(object_name)
        goal_desc = (
            goal_region
            if isinstance(goal_region, str)
            else f"point(xy={goal_region.get('xy')}, z={goal_region.get('z')}, r={goal_region.get('radius')})"
        )
        flag = (
            f" [Precomputed: {object_name} already satisfies In({object_name}, {goal_desc}) -- "
            "its goal is already reached. Do NOT emit any pick/transport/place subtasks for it; "
            "plan only for the remaining, not-yet-satisfied objects/subtasks.]"
        )
        entity = by_name.get(object_name)
        if entity is not None:
            entity["notes"] = entity.get("notes", "") + flag
        else:
            entities.append({"name": object_name, "role": "object", "notes": flag.strip()})
    return satisfied


def _annotate_timing_overruns(fresh_context: dict[str, Any], previous_summary: dict[str, Any]) -> list[str]:
    """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md).

    Stamp a note onto fresh_context for every subtask that ran past its
    own declared STL deadline in the just-failed attempt, so the recalled
    plan can adjust timing with real numbers instead of guessing blind. Reads
    the timing_violation fact executor.py's _annotate_timing_violation
    writes onto each SkillExecutionRecord.info -- itself derived from
    sub["stl_deadline"], which only --stl-mode stl_spec plans carry (see
    step/system_two/stl_plan.py:decompose()), so this is a no-op for
    --stl-mode skill_graph.

    Deliberately does nothing more than log/annotate: per the explicit design
    discussion in STL_DECOMPOSITION_PLAN.md, a declared time budget that turns
    out to be infeasible is not something System 1 should try to enforce or
    repair -- "if it fails, it would most likely be over the time limit, so
    just log it" and let the next System-2 call decide what to do with that
    information (a larger budget, a different approach, or leaving it as-is).
    """
    records = previous_summary.get("records") or []
    violations = [
        r
        for r in records
        if isinstance(r, dict) and isinstance((r.get("info") or {}).get("timing_violation"), dict)
    ]
    if not violations:
        return []

    entities = fresh_context.setdefault("entities", [])
    by_name = {e.get("name"): e for e in entities if isinstance(e, dict) and e.get("name")}

    annotated: list[str] = []
    for record in violations:
        tv = record["info"]["timing_violation"]
        obj = tv.get("object")
        skill = tv.get("skill")
        outcome = "did not succeed" if not tv.get("subtask_succeeded") else "succeeded, but later than planned"
        flag = (
            f" [Precomputed: a previous attempt's {skill} subtask"
            + (f" for {obj}" if obj else "")
            + f" took {tv.get('actual_absolute_step')} controller-steps, exceeding its declared "
            f"deadline of {tv.get('declared_deadline_step')} by {tv.get('overrun_steps')} steps, "
            f"and {outcome}. The requested overall timing may not be achievable as specified -- "
            "consider a larger time budget for this step, or a different approach.]"
        )
        entity = by_name.get(obj) if obj else None
        if entity is not None:
            entity["notes"] = entity.get("notes", "") + flag
        else:
            entities.append({"name": obj or f"{skill}_timing", "role": "note", "notes": flag.strip()})
        annotated.append(obj or str(skill))
    return annotated


def _annotate_skill_failure_hints(fresh_context: dict[str, Any], previous_summary: dict[str, Any]) -> list[str]:
    """Stamp a note onto fresh_context describing *why* each failed
    subtask in the just-failed attempt actually failed (its
    failure_reason/diagnostics.detail), so the recalled plan has a
    textual signal to reason from instead of just an unchanged scene image.

    Exists because, unlike a pick/place failure (which usually leaves a
    visible trace -- a dropped object, an arm stuck somewhere), a rejected
    open attempt that never finds a feasible candidate pull leaves the
    scene looking *identical* to the very first frame: the drawer never
    moved, nothing visibly changed. Confirmed directly: without this, a real
    recall reasoned "no precomputed blockage flag is present, so no object
    relocation is needed" -- it had no way to know the previous attempt hit
    final_descent_in_collision at all, let alone connect that to whatever is
    sitting in front of the handle in the image. Mirrors
    _annotate_timing_overruns's own pattern (precompute the fact in code,
    stamp the verdict as text) for the general "why did this fail" case."""
    records = previous_summary.get("records") or []
    failed = [r for r in records if isinstance(r, dict) and (r.get("info") or {}).get("success") is False]
    if not failed:
        return []

    entities = fresh_context.setdefault("entities", [])
    by_name = {e.get("name"): e for e in entities if isinstance(e, dict) and e.get("name")}

    annotated: list[str] = []
    for record in failed:
        info = record.get("info") or {}
        skill = record.get("skill")
        obj = (record.get("params") or {}).get("object")
        reason = info.get("failure_reason")
        detail = (info.get("diagnostics") or {}).get("detail")
        pieces = [f"a previous attempt's {skill} subtask" + (f" for {obj}" if obj else "") + " did not succeed"]
        if reason:
            pieces.append(f"failure_reason={reason}")
        if detail:
            pieces.append(f"detail: {detail}")
        flag = (
            " [Precomputed: " + "; ".join(pieces) + " -- if this looks like the arm could not reach or "
            "move past something, check the scene image for an object sitting near the target and "
            "consider relocating it before retrying.]"
        )
        entity = by_name.get(obj) if obj else None
        if entity is not None:
            entity["notes"] = entity.get("notes", "") + flag
        else:
            entities.append({"name": obj or f"{skill}_failure", "role": "note", "notes": flag.strip()})
        annotated.append(obj or str(skill))
    return annotated


def _formula_is_trivially_true(formula: Any) -> bool:
    """A "true" leaf, or an "and" node whose children are all trivially
    true -- the model sometimes wraps a bare "true" in an "and" instead of
    emitting it directly, and both mean the same thing: nothing left to do."""
    if not isinstance(formula, dict):
        return False
    if formula.get("type") == "true":
        return True
    if formula.get("type") == "and":
        children = formula.get("children")
        return isinstance(children, list) and len(children) > 0 and all(
            _formula_is_trivially_true(c) for c in children
        )
    return False


def _plan_has_nothing_to_do(plan: dict[str, Any]) -> bool:
    """True when System 2 declares the task already satisfied: a stl_spec
    formula that reduces to "true" (possibly wrapped in "and"), or an
    explicitly empty skill_graph subtasks list."""
    system2_plan = plan.get("system2_plan")
    if not isinstance(system2_plan, dict):
        return False
    if _formula_is_trivially_true(system2_plan.get("formula")):
        return True
    subtasks = system2_plan.get("subtasks")
    return isinstance(subtasks, list) and len(subtasks) == 0


def _release_and_return_home(env, n_steps: int = 60, step_callback=None) -> int:
    """Open the gripper in place, then drive the arm back to its reset
    configuration -- so a run never ends mid-reach or holding an object."""
    import numpy as np

    from .system_one.motion.panda import EE_SITE, JOINT_NAMES
    from .system_one.motion.servo import _ServoMixin

    class _HomeServo(_ServoMixin):
        def __init__(self, env):
            self.env = env
            self.joint_names = JOINT_NAMES
            self.ee_site = EE_SITE

    servo = _HomeServo(env)
    # Release where the arm is before moving, so the object drops at the
    # hover pose instead of being dragged toward home by half-open fingers.
    steps = 0
    zero = np.zeros(3, dtype=np.float32)
    for _ in range(int(env.close_steps) * 2):
        servo._step_env(zero, zero, env.open_gripper_cmd)
        steps += 1
        if step_callback is not None:
            step_callback()
    return steps + servo.servo_to_home(n_steps=n_steps, step_callback=step_callback)


def _decompose_or_annotate_errors(plan: dict[str, Any], errors_path: Path) -> dict[str, Any]:
    """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md): turns a single-global-
    formula ("stl_spec") plan into the standard subtasks shape.

    On failure, writes the specific validation/decomposition error to
    errors_path and returns the plan UNCHANGED, so the caller's normal
    validate_stl_plan(...) call still rejects it (missing/invalid
    subtasks) through the existing validation_errors.txt path -- this
    function only adds a more specific error message alongside that, it
    doesn't replace the existing validation control flow.
    """
    from .system_two.stl_plan import decompose_and_validate

    system2_plan = plan.get("system2_plan")
    if (
        isinstance(system2_plan, dict)
        and isinstance(system2_plan.get("formula"), dict)
        and not system2_plan.get("subtasks")
        and system2_plan.get("type") != "stl_spec"
    ):
        # The model sometimes labels a global-formula plan "skill_graph";
        # go by the content, or the decomposer rejects a valid plan.
        plan = {**plan, "system2_plan": {**system2_plan, "type": "stl_spec"}}
    decomposed_plan, result = decompose_and_validate(plan)
    if decomposed_plan is None:
        errors_path.write_text("\n".join(result.errors))
        return plan
    return decomposed_plan


def _load_or_create_plan(
    args: argparse.Namespace,
    *,
    scene_context: dict[str, Any],
    prompt: str,
    outdir: Path,
) -> dict[str, Any]:
    if args.compiled_plan:
        plan = load_json(args.compiled_plan)
        save_json(plan, outdir / "compiled_stl_plan.json")
        return plan

    schema_path = args.schema or _DEFAULT_SCHEMA_FOR_MODE[args.stl_mode]
    schema = load_json(schema_path)
    image_path = _capture_image(args, scene_context, outdir)
    print(f"[step] System 2 image: {image_path}")

    from .system_two.planner import DirectSTLPlanner

    planner = DirectSTLPlanner.for_mode(args.stl_mode, schema=schema, scene_context=scene_context, model=args.model)
    plan, raw_text = planner.plan(instruction=prompt, image_path=image_path)
    save_json(plan, outdir / "direct_stl_plan.json")
    (outdir / "raw_model_output.txt").write_text(raw_text)

    if args.stl_mode == "stl_spec":
        plan = _decompose_or_annotate_errors(plan, outdir / "decomposition_errors.txt")
        save_json(plan, outdir / "decomposed_stl_plan.json")

    save_json(plan, outdir / "compiled_stl_plan.json")
    return plan


def main() -> int:
    args = parse_args()
    benchmark_task = _apply_benchmark_task(args)
    outdir = Path(args.outdir)

    scene_context, scene_path, prompt, prompt_path = _prepare_libero_scene(args, outdir)
    print(f"[step] scene context: {scene_path}")
    print(f"[step] prompt: {prompt}")
    print(f"[step] prompt file: {prompt_path}")

    plan = _load_or_create_plan(args, scene_context=scene_context, prompt=prompt, outdir=outdir)
    if isinstance(plan.get("system2_plan"), dict) and isinstance(plan["system2_plan"].get("subtasks"), list):
        plan["system2_plan"]["subtasks"] = normalize_legacy_pregrasp_grasp(plan["system2_plan"]["subtasks"])
    result = validate_stl_plan(plan)
    if not result.ok:
        (outdir / "validation_errors.txt").write_text("\n".join(result.errors))
        print(f"[step] direct STL plan validation failed: {outdir / 'validation_errors.txt'}", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print_formulas(plan)
    if args.dry_run:
        return 0

    from step.system_one.executor import System1Executor

    executor = System1Executor(
        scene_context=scene_context,
        render=bool(args.render),
        video_path=args.save_video,
        video_prompt=prompt,
        original_prompt=prompt,
    )

    # Build the env ONCE and never reset it on a retry: a dropped object
    # mid-transport leaves the scene in a real, physical state (the can
    # sitting wherever it landed, the arm wherever it ended up) -- resetting
    # would throw that away and start over from the original init-state
    # layout, which is not a "retry" of the failure at all, just a
    # different roll of the dice. On failure, recall System 2 with a fresh
    # image/object-position snapshot of the CURRENT (still-unreset) scene,
    # so the new plan is written against reality -- e.g. "pick up the can"
    # from wherever it actually is now, not its original spot. This is the
    # practical version of System-2 recall: the real mid-execution recall
    # path in recall_controller.py is a stubbed NotImplementedError, never
    # finished when the codebase moved to direct STL plans.
    subtasks = plan["system2_plan"]["subtasks"]
    runtime = executor._extract_runtime_config(subtasks)
    runtime["_computed_env_horizon"] = executor._computed_env_horizon(len(subtasks))
    env = executor._make_env(runtime, debug_dir=outdir)
    max_attempts = max(1, int(args.max_pipeline_retries))
    summary: dict[str, Any] = {}
    try:
        env.reset()
        for attempt in range(1, max_attempts + 1):
            is_last_attempt = attempt == max_attempts
            if attempt > 1:
                if args.compiled_plan:
                    print(f"[step] attempt {attempt}/{max_attempts}: re-running the same --compiled-plan, env NOT reset")
                else:
                    print(f"[step] attempt {attempt}/{max_attempts}: recalling System 2 from the current (unreset) scene")
                    fresh_context, image_path = _capture_live_scene_and_image(env, scene_context, outdir, attempt)
                    already_satisfied = _annotate_already_satisfied_objects(fresh_context, env, plan)
                    if already_satisfied:
                        print(f"[step] attempt {attempt}/{max_attempts}: already at goal, excluding from replan: {already_satisfied}")
                    timing_overruns = _annotate_timing_overruns(fresh_context, summary)
                    if timing_overruns:
                        print(f"[step] attempt {attempt}/{max_attempts}: previous attempt exceeded its declared timing for: {timing_overruns}")
                    failure_hints = _annotate_skill_failure_hints(fresh_context, summary)
                    if failure_hints:
                        print(f"[step] attempt {attempt}/{max_attempts}: previous attempt failed for: {failure_hints}")
                    schema_path = args.schema or _DEFAULT_SCHEMA_FOR_MODE[args.stl_mode]
                    schema = load_json(schema_path)
                    from .system_two.planner import DirectSTLPlanner

                    planner = DirectSTLPlanner.for_mode(args.stl_mode, schema=schema, scene_context=fresh_context, model=args.model)
                    plan, raw_text = planner.plan(instruction=prompt, image_path=image_path)
                    save_json(plan, outdir / f"recall_attempt_{attempt:02d}_plan.json")
                    (outdir / f"recall_attempt_{attempt:02d}_raw_model_output.txt").write_text(raw_text)
                    if _plan_has_nothing_to_do(plan):
                        # A bare "true" formula (or an explicitly empty
                        # subtasks list) means System 2 considers the task
                        # already satisfied -- decomposing/validating it as a
                        # normal plan would always fail (there's nothing to
                        # infer a skill from), so that's not a real failure.
                        print(f"[step] attempt {attempt}/{max_attempts}: recall says nothing left to do; releasing and returning home")
                        home_steps = _release_and_return_home(
                            env, step_callback=lambda: executor._capture_video_frame(env)
                        )
                        summary = {
                            "success": True,
                            "attempt": attempt,
                            "detail": "recall determined the task is already satisfied; released and returned home",
                            "home_steps": home_steps,
                        }
                        break
                    if args.stl_mode == "stl_spec":
                        plan = _decompose_or_annotate_errors(
                            plan, outdir / f"recall_attempt_{attempt:02d}_decomposition_errors.txt"
                        )
                    if isinstance(plan.get("system2_plan"), dict) and isinstance(plan["system2_plan"].get("subtasks"), list):
                        plan["system2_plan"]["subtasks"] = normalize_legacy_pregrasp_grasp(plan["system2_plan"]["subtasks"])
                    retry_validation = validate_stl_plan(plan)
                    if not retry_validation.ok:
                        summary = {"success": False, "attempt": attempt, "error": "plan_validation_failed", "errors": retry_validation.errors}
                        print(f"[step] attempt {attempt}/{max_attempts}: recalled plan failed validation: {retry_validation.errors}", file=sys.stderr)
                        continue
                    print_formulas(plan)
                    scene_context = fresh_context  # later recalls build on the latest known state, not the original

            try:
                summary = executor.execute_on_env(
                    env, plan, outdir=outdir, reset_first=False, capture_initial_video=(attempt == 1)
                )
            except Exception as exc:
                summary = {"success": False, "attempt": attempt, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[step] attempt {attempt}/{max_attempts} raised {type(exc).__name__}: {exc}", file=sys.stderr)

            if summary.get("success"):
                if attempt > 1:
                    print(f"[step] succeeded on attempt {attempt}/{max_attempts}")
                break
            if not is_last_attempt:
                print(f"[step] attempt {attempt}/{max_attempts} failed (success={summary.get('success')!r}); recalling and retrying without resetting the env")
        if not summary.get("success"):
            print("[step] run ended without success; releasing and returning home")
            try:
                summary["release_home_steps"] = _release_and_return_home(
                    env, step_callback=lambda: executor._capture_video_frame(env)
                )
            except Exception as exc:
                summary["release_home_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # No-op (returns None) for tasks with no <bddl>.constraints.json sidecar,
        # e.g. plain LIBERO/LIBERO-Pro suites -- only LIBERO-Constrained's suites
        # populate this. Checked independently of summary["success"]: a run can
        # complete the task while still violating its constraint.
        if hasattr(env, "check_constraints"):
            constraint_check = env.check_constraints()
            if constraint_check is not None:
                summary["constraint_check"] = constraint_check
        executor._close_video_writer()
        env.close()

    summary["pipeline_attempts_used"] = attempt
    # Record what produced this result. The model is otherwise only implied by a
    # default in planner.py and an environment variable, so a summary could not
    # say which model it came from -- and a shell with OPENAI_MODEL set would
    # silently change it with nothing in the output to show that.
    summary["provenance"] = {
        "model": args.model or os.environ.get("OPENAI_MODEL", "gpt-5.4"),
        "benchmark_task": getattr(args, "benchmark_task", None),
        "max_pipeline_retries": int(args.max_pipeline_retries),
        "enforce_keepout": not bool(getattr(args, "no_enforce_keepout", False)),
        "perception": args.perception,
    }
    if args.perception == "vision":
        from step.perception import audit

        summary["privileged_access_audit"] = audit.report()
    save_json(summary, outdir / "execution_summary.json")
    print(f"[step] execution summary: {outdir / 'execution_summary.json'}")
    if args.save_video:
        print(f"[step] execution video: {args.save_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
