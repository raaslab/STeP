# Algorithmic STL formula decomposition (subtask boundaries derived from the formula, not the LLM)

## Context

Today, System 2 (`step/system_two/planner.py`'s `DirectSTLPlanner.plan`) asks the VLM to
emit the *entire* `system2_plan.subtasks` list in one JSON call. Each subtask already
carries its own `skill`, `params`, **and** a per-subtask STL `formula` with subtask-local
time intervals -- e.g. `build_developer_prompt` (`step/system_two/prompts.py`) rule 7 says
"For a normal pick, use `F[0,40](PreGrasp(object, grasp_mode))` and
`F[0,90](Grasped(object))`", rule 8 says transport gets `F[0,180](OverGoal(object, goal))`,
etc. Rules 1-18 in that prompt are, in effect, hand-encoded decomposition and
skill-grounding logic (where to draw subtask boundaries, in what order, with what timing
budget) baked into natural-language instructions to the model, rather than derived from
any formal structure of the specification itself.

`step/schemas/stl_plan_schema.json` bakes this same assumption into the JSON Schema: the
per-subtask `formula` lives inside `$defs/subtask`, which lives inside
`system2_plan.subtasks` -- there is no schema shape today for "one formula for the whole
task." Changing this is a schema change, not just a prompt change.

`step/system_two/stl_plan.py`'s `_validate_formula` defines today's formula grammar:
`true, predicate, not, and, implies, always, eventually, until`. There is no `or` yet --
worth noting since the paper's own STL grammar (icra2027/main.tex, Background section)
does include `\vee`.

**Paper direction** (icra2027/main.tex, Introduction, as of 2026-09-12): System 2 should
translate the instruction into *one* STL specification for the whole task; a separate,
deterministic decomposition procedure should then derive the ordered subtask sequence from
that formula's own temporal structure, rather than the VLM deciding subtask boundaries and
per-subtask timing directly. This plan is what should actually get built to make that true.

## Goal

Replace "VLM decides subtask boundaries + per-subtask timing via 18 natural-language
rules" with:

1. VLM emits **one global STL formula** over the full task horizon (objects/skills are
   still grounded through predicate names and args, exactly as today -- nothing changes
   about how objects are referenced).
2. A new, deterministic **decomposition function** walks that formula and produces the
   ordered `subtasks` list that `stl_plan.py`, the System 1 executor, and `stl_monitor.py`
   already expect -- so nothing downstream of "we have a subtasks list" needs to change.

## Algorithm: formula progression / syntax-tree splitting by temporal-operator intervals

This is a known technique, not something novel to invent from scratch -- it's the STL
analogue of Bacchus & Kabanza's LTL *formula progression* (Bacchus & Kabanza, "Using
Temporal Logics to Express Search Control Knowledge for Planning," *Artificial
Intelligence* 116(1-2), 2000, pp. 123-191, doi:10.1016/S0004-3702(99)00071-5 -- no arXiv
version exists; this predates arXiv adoption in AI planning), and it's essentially what
Raman et al.'s and Sadraddini & Belta's STL-MPC formulations do implicitly when they slice
a receding horizon by a specification's time bounds (both already in the paper's
bibliography).

**Closest existing algorithm to adapt directly: SLICE**, Algorithm 3 in Kapoor, Kang &
Meira-Góes, "Safe Planning Through Incremental Decomposition of Signal Temporal Logic
Specifications," NFM 2024 (arXiv: **https://arxiv.org/abs/2403.10554**, full text:
https://arxiv.org/html/2403.10554v1; published version: Springer LNCS vol. 14627,
doi:10.1007/978-3-031-60698-4_23). SLICE takes a set of time-bounded constraints, finds
the min/max time bound across all of them, walks the timeline to determine which
constraints are active at each point, partitions the timeline into "slices" of consistent
constraint sets, and merges same-type propositions within each slice into atomic
reachability/invariance task formulas -- this is the same G-vs-F split described above
(invariance constraints persist across slices; reachability constraints become one atomic
task per slice), already worked out and worth reading/adapting before writing
`decompose(...)` from scratch. Also relevant: Yu, Wang, Yuan, Li & Yin, "Model Predictive
Control for Signal Temporal Logic Specifications with Time Interval Decomposition," 2022,
arXiv: **https://arxiv.org/abs/2211.08031** -- decomposes an STL formula into sub-formulae
with disjoint time horizons and solves each with shrinking-horizon MPC, with terminal
constraints linking successive sub-formulae.

Core idea -- split the formula into two parts:

- **(a) Persistent/global part**: top-level `always` (`G`) conjuncts that must hold for the
  *entire* task (safety, invariance, e.g. "never enter the red zone"). These are **not**
  assigned to any single subtask -- they get carried into every subtask's formula (and
  hence every subtask's MPC cost / STL monitor) as an always-active background constraint.
- **(b) Sequential/achievement part**: the top-level conjunction (or `until` chain) of
  `eventually` (`F`)-guarded goals, taken in source order. Each one becomes exactly one
  subtask.

```python
def decompose(formula: Formula) -> list[Subtask]:
    global_constraints, sequential_goals = split_top_level_and(formula)
    # global_constraints: every top-level 'always' conjunct (nothing time-bounded to "finish")
    # sequential_goals: every top-level 'eventually'/'until' conjunct, in source order
    subtasks = []
    for goal in sequential_goals:
        skill = infer_skill_from_predicates(goal)  # see predicate->skill table below
        subtasks.append(Subtask(
            skill=skill,
            formula={"type": "and", "children": [goal, *global_constraints]},
        ))
    return subtasks
```

Handling each operator:

- **`until` (`phi1 U[a,b] phi2`)**: progresses like Bacchus-Kabanza LTL progression --
  while `phi1` continues to hold, the goal is "not yet due"; once `phi2` is satisfied, the
  goal is complete. An `until` node becomes **one** subtask, monitored with `phi1` folded
  in as an ongoing (safety-like) condition and `phi2` as its completion predicate -- it is
  not split further.
- **`implies` (conditional clauses, e.g. the paper's own `\varphi_1 \Rightarrow \varphi_2`
  grammar rule)**: has no clean sequential split, since the antecedent may or may not ever
  become true. Two options, need a decision (see open questions below): (1) fold it into
  whichever subtask's global-constraint set is active when the antecedent can occur, or
  (2) introduce a distinct "guarded subtask" type that System 1 only activates once the
  antecedent's robustness crosses zero.
- **`or` (disjunction)**: has no single decomposition -- two independent ways to satisfy
  something can't be assigned disjoint time windows. The current grammar/schema doesn't
  even support `or` yet; if it's added, treat a top-level `or` as non-decomposable (pass
  the whole disjunction to one subtask) rather than trying to split it.

## Concrete integration points (confirmed by reading the current code, not guessed)

- **`step/schemas/stl_plan_schema.json`**: needs a new top-level schema shape for "one
  formula, no subtasks" (or a variant `system2_plan.type`, e.g. `"stl_spec"` vs. today's
  `"skill_graph"`), since the subtask nesting is currently baked directly into the schema.
- **`step/system_two/prompts.py`**: `build_developer_prompt` needs a new prompt variant
  asking for the single global formula. Rules 7, 8, 9, 12, 13, 16 (the ones that assign a
  fixed `F[...]` template per skill type) go away entirely -- that timing structure is now
  *derived*, not prescribed. Rules that encode real scene-conditioning logic (15, 15a/b,
  17, 18 -- the drawer-clearance / already-satisfied-goal precomputed-flag checks) are
  **not** decomposition logic and likely need to stay, but now shape which predicates go
  into the single global formula rather than which subtasks get emitted.
- **`step/system_two/planner.py`**: `DirectSTLPlanner.plan` itself likely doesn't need
  logic changes -- it already just forwards whatever schema it's given to the model -- but
  its call site needs to invoke `decompose(...)` on the returned formula before handing
  anything to System 1.
- **`step/system_two/stl_plan.py`**: needs (1) a new `validate_global_stl_plan` (or a mode
  flag on the existing validator) for the single-formula shape, and (2) the new
  `decompose(...)` / `split_top_level_and(...)` / `infer_skill_from_predicates(...)`
  functions described above. `_collect_predicate_names`, `formula_to_string`, and
  `find_predicate_args` are all reusable as-is -- they already operate on a bare formula
  tree, independent of any subtask wrapping.
- **Predicate -> skill table**: rules 7-9/12/13/16 in `prompts.py` already define an
  *implicit* mapping (`PreGrasp`/`Grasped` -> `pick`, `OverGoal` -> `transport`,
  `Placed`+`Released` -> `place`, `Opened` -> `open`, `Closed` -> `close`, `TurnedOn` ->
  `turn_on`). This should become an explicit `PREDICATE_TO_SKILL` dict that
  `infer_skill_from_predicates` reads, replacing the natural-language version of the same
  mapping that currently only exists inside the prompt text.
- **`normalize_legacy_pregrasp_grasp`**: becomes unnecessary for newly-generated plans (a
  pick's `PreGrasp`+`Grasped` will naturally live inside one sequential goal, matching
  today's rule 7 behavior) but its docstring already frames it as a compatibility shim for
  external/legacy plans, so it should stay as-is.

## Real open questions to resolve before implementing

1. Controller steps vs. seconds: rule 4 currently mandates controller-step intervals
   (LIBERO defaults to 20 Hz) for per-subtask formulas. Does the single global formula stay
   in controller steps for the whole episode, or move to seconds and get converted at
   decomposition time?
2. What happens to the scene-specific precomputed-flag rules (15, 15a/b, 17, 18) that
   depend on injected scene notes (e.g. "`[Precomputed clearance to <cabinet>: ...]`")? These
   are scene-conditioning logic, not temporal decomposition, and most likely need to stay as
   a pre-processing step that shapes which predicates/constraints appear in the single
   global formula -- separate from, and upstream of, the new decomposition step.
3. Does `implies` (conditional subgoals) actually show up in any current task specs in
   practice, or is it currently unused? That decides how urgent the "guarded subtask"
   design question above really is versus something to defer.
4. Ordering: is source order in the formula's top-level conjunction always the intended
   execution order, or does anything (e.g. `until` scoping, nested conjunctions) need a
   topological sort instead of a flat left-to-right walk?
