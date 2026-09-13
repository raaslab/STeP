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

## Implementation status (2026-09-12, now the default)

Built as an additive path, then promoted to the default after real-scene testing held up
(see results below). `step/libero_runner.py` now defaults to `--stl-mode stl_spec` --
`DirectSTLPlanner.for_mode("stl_spec", ...)` + `decompose_and_validate(...)` -- for both the
initial plan and every System-2 recall on retry. `DirectSTLPlanner()` called with no args
(any other caller) still defaults to the original `build_developer_prompt`/skill_graph
behavior; only `libero_runner.py`'s own default flipped. **To revert:** pass
`--stl-mode skill_graph` on the CLI -- one flag, no code changes, restores the exact
pre-decomposition behavior (VLM emits `subtasks` directly). To remove the feature entirely:
drop `schemas/stl_global_plan_schema.json`, `tests/test_stl_decompose.py`,
`outputs/_stl_decomposition_experiment/`, `build_global_developer_prompt` in `prompts.py`,
`DirectSTLPlanner.for_mode`/`developer_prompt_fn` in `planner.py`, everything from
`validate_global_stl_plan` through `decompose_and_validate` in `stl_plan.py`, and
`_DEFAULT_SCHEMA_FOR_MODE`/`_decompose_or_annotate_errors`/the `--stl-mode` flag in
`libero_runner.py` (reverting `--schema`'s default back to `"schemas/stl_plan_schema.json"`).

This has still only been run through plan creation, never through System 1/the MPC executor
-- `libero_runner.py`'s `--dry-run` flag exercises the new default end-to-end (scene capture
-> global-formula VLM call -> decompose -> validate -> `print_formulas`) without executing
any skill, which is as far as this has been verified.

**Open questions, resolved pragmatically for this spike:**

1. Controller steps: kept as controller-step intervals per achievement goal, unchanged.
   These per-goal-type budgets (`F[0,40]` PreGrasp, `F[0,90]` Grasped, etc.) are a property
   of the achievement itself (how long that predicate should take once attempted), not a
   subtask-boundary decision, so they still live in the prompt -- what moved to `decompose()`
   is only which achievements get grouped into one subtask and in what order.
2. Scene-conditioning rules (15/15a/15b/17/18 in the old numbering): kept in
   `build_global_developer_prompt` as rules 9-12, reworded from "emit these subtasks" to
   "emit these top-level achievement goals" -- same logic, upstream of decomposition, exactly
   as anticipated.
3. `implies`: unused in every real plan checked (`grep -rl '"always"' outputs/*/direct_stl_plan.json`
   also found zero hits -- no plan in this repo has ever emitted `always` or `implies`). Not
   implemented beyond raising `DecompositionError` on an unrecognized top-level predicate set;
   the new prompt explicitly tells the model not to emit `implies`/`or` at all, deferring the
   "guarded subtask" design question until a real task needs it.
4. Ordering: flat left-to-right source order, confirmed sufficient -- every real scene tested
   (including the 7- and 8-subtask blocking-object-aside branches) preserved intended order
   with no topological-sort cases found.

**Design gap found while implementing:** per-subtask `params` fields with no predicate-arg
home (`grasp_target="rim"`, `grasp_offset_xyz`, `place_surface="top"`) can't be derived from
the formula alone. Added `system2_plan.object_hints: {<entity_name>: {...}}` as a schema
sibling of `formula` -- decomposition merges an object's hints into whichever subtask
`params` that object ends up in. This is scene/object metadata, not decomposition logic, so
it doesn't weaken the "decomposition is derived, not authored" property the plan is going for.

**Test results:** `tests/test_stl_decompose.py` (15 deterministic unit tests, no API calls)
covers pick/transport/place grouping, height/rim hints, global `always` constraints folded
into every subtask, single-atomic-goal tasks, the full open->pick->transport->place->close
chain, the `DecompositionError` path for an unrecognized predicate set, and the
`decompose_and_validate` end-to-end wrapper -- all pass, as does the full existing `tests/`
suite (101 passed, one pre-existing unrelated failure in `test_transport_skill.py` confirmed
present on `main` before this change too).

**Correction (2026-09-12, same day): the first version of rule 2a was wrong.** It told the
VLM that each goal's window "starts fresh at 0" once active -- i.e. a per-goal-local clock,
matching the legacy per-subtask convention this whole project exists to move away from.
Went back to the actual SLICE paper (Kapoor/Kang/Meira-Goes, NFM 2024, arXiv:2403.10554,
Algorithm 3) to check: it operates on one **absolute global timeline** -- constraints are
`(l, h, p)` tuples with absolute bounds, and the atomic tasks SLICE emits keep those absolute
bounds; they are never rebased to a local 0. A pick spanning [0,90] followed by a transport
must produce `F[90,270](...)`, not `F[0,180](...)` again. This is also exactly what "pick up
the bowl within 5 seconds" / "complete between 5 and 8 seconds" (the eventual MPC-cost-term
goal, see below) requires: those numbers are only meaningful as positions on one shared
timeline, not as six unrelated per-skill durations.

Fixed both sides:
- **Prompt** (`build_global_developer_prompt` rule 2a): now explicitly states intervals are
  absolute-on-one-timeline, gives a full worked 3-goal cumulative example, and rule 5's
  per-skill templates are now labeled as *durations* to be offset by the running total, not
  literal windows.
- **`decompose()`**: added `goal_absolute_window()` (reads the [min-start, max-end] a goal's
  own time-bounded sub-formulas span) and `_validate_sequential_timing()` (SLICE's own
  invariant: each goal's absolute window must start at or after the previous goal's window
  ended -- non-overlapping, non-decreasing, touching boundaries allowed), raising
  `DecompositionError` on violation. Absolute intervals are never rewritten by `decompose()`
  -- only validated -- per explicit instruction not to discard them: the goal is to eventually
  feed them into the MPC as a real cost term (e.g. splitting "8 seconds total" into "4s for
  pick, 4s for place"), which is future work and intentionally NOT done here -- `max_subtask_steps`
  (the executor's hard per-skill step cap) was explicitly left untouched.

Re-ran all 8 real-scene cases against the corrected prompt: every case now produces a
genuinely cumulative, non-overlapping absolute-time chain (e.g. the 7-goal blocking scenario:
pick[0,90] -> transport[90,270] -> place[270,390] -> open[390,510] -> pick[510,600] ->
transport[600,780] -> place[780,900]), `_validate_sequential_timing` passes on all 8 with zero
errors, and skill sequences still match the legacy planner exactly. 8 new unit tests added
covering `goal_absolute_window`, rejecting overlapping windows, rejecting the literal
two-goals-both-reset-to-zero bug, allowing touching boundaries, and a full 7-goal cumulative
scenario (26 tests total in `tests/test_stl_decompose.py`, 108 passed across the full suite).

**Second correction (2026-09-12, same day, requested re-read against the primary source):**
fetched the actual paper (not just the plan doc's secondhand summary) a second time to check
the fix above against Algorithm 3's real pseudocode and worked example. Found one real gap
and confirmed one design decision:

- **Gap (fixed):** the paper's `(l, h, p)` constraints are "active" during a slice only when
  the slice's range falls inside `[l, h]` -- an invariance (`G`) constraint is not
  automatically global just for being top-level. My `decompose()` folded *every* top-level
  `always` into *every* subtask unconditionally, regardless of its own interval. Harmless
  today only because the prompt exclusively asks for `interval: null` ("holds for the whole
  task"), which correctly covers everything -- but a scoped constraint like "avoid the red
  zone only during steps 200-400" would have been silently over-applied. Added
  `_constraint_covers_window()`: a bounded `always[a,b]` now only folds into subtasks whose
  own `goal_absolute_window()` overlaps `[a,b]`; `interval: null` (or a goal with no timed
  nodes) still covers everything, unchanged. Added two tests for this (a scoped constraint
  covering only the subtasks that overlap it, and an unbounded one still covering all of
  them) -- 20 tests total in `tests/test_stl_decompose.py`, 110 passed across the full suite
  (one unrelated, pre-existing flaky OMPL-planner test excluded, confirmed to pass standalone).
- **Confirms open question #3 with primary-source evidence, not inference:** the paper's own
  grammar (Section 4.1) is `phi ::= F[a,b](phi) | G[a,b](phi) | phi1 AND phi2 | p` -- no
  `until`, no top-level disjunction, no negated temporal formulas. Deferring `until`/`implies`
  isn't just "seems rare in practice" -- SLICE itself doesn't cover them. Tightened rule 1 in
  `build_global_developer_prompt` to explicitly forbid `not`/`until` too (previously only
  `implies`/`or` were forbidden), matching the algorithm's actual proven fragment exactly.
- Re-ran all 8 real-scene cases again after both changes: still 8/8 skill-sequence match, zero
  validation errors, genuinely cumulative absolute timing confirmed in the raw model output.

Also ran the real VLM (`gpt-5.4`, live `OPENAI_API_KEY` call) in `stl_spec` mode against 8
real saved scenes/images/prompts from `outputs/` spanning every skill and every branch of
the scene-conditioning rules (plain pick/place, `turn_on`+pick+place, pick+transport+place+
close, the 7- and 8-subtask blocking-object-aside paths, and a two-object plan) --
see `outputs/_stl_decomposition_experiment/` (`run_experiment.py` + one directory per case
with `global_plan.json`, `decomposed_plan.json`, `raw_model_output.txt`, and
`experiment_log.txt` for the full run). All 8 decomposed to the exact same skill sequence as
the legacy per-subtask planner's saved plan for the same scene, and all passed
`validate_stl_plan`. One real issue was found and fixed this way: the first run of
`task0_regression_check` had the model add a spurious 3rd arg to `OverGoal` (naming the
containing basket) that no legacy plan had ever produced; `decompose()` silently ignored it
(harmless, since System 1's `stl_monitor.py` never reads predicate args for margins), but the
prompt was tightened ("Exactly these two args, nothing more") and a re-run confirmed the
clean 2-arg shape came back with `grasp_mode`/`place_surface`/height estimates otherwise
matching the legacy plan closely (small numeric height-estimate deltas like 0.09 vs. 0.1 m
are ordinary VLM vision-estimate noise, present in the legacy pipeline too).

## Related work: insight from the second cited paper (2026-09-12, read-through only, no code change)

The original plan document also cited Yu, Wang, Yuan, Li & Yin, "Model Predictive Control for
Signal Temporal Logic Specifications with Time Interval Decomposition," 2022, arXiv:2211.08031
(authors: Xinyi Yu, Chuwei Wang, Dingran Yuan, Shaoyuan Li, Xiang Yin). Read it in full at the
user's request to check for insight beyond SLICE. This is a much heavier robust-MPC/MILP paper
than SLICE -- most of its machinery does not map onto STeP's architecture (a sampling-based
OMPL planner + physics-sim environment, not a bounded-disturbance linear/affine dynamics model
solved via mixed-integer programming), but two things are worth keeping:

**1. Independent confirmation of the disjoint-time-window invariant.** The paper requires
exactly the same structural precondition as SLICE for its decomposition to be valid: `Phi =
Phi_1^[S1,T1] AND ... AND Phi_n^[Sn,Tn]` with `T_i <= S_{i+1}` for every i -- non-overlapping,
non-decreasing sub-formula windows. They are explicit that this doesn't come for free: "we can
always take N=1... but no computational reduction will be gained" for a formula that doesn't
actually factor this way. This is exactly `_validate_sequential_timing`'s invariant in
`stl_plan.py`, and exactly why `decompose()` raises `DecompositionError` rather than forcing a
bad split when it's violated -- two independent papers landing on the same requirement is
good evidence the invariant is the right one to enforce, not an SLICE-specific artifact.

**2. A genuinely new idea, not yet acted on: terminal constraints between sub-problems.**
Their decomposition doesn't just chain sub-formulae in sequence -- at the end of each
sub-problem's time window, the state must land in a terminal set `T_i` from which *every
remaining* sub-formula is still satisfiable under worst-case future disturbance, not just
"did this sub-problem hit its own goal." This is a formal version of a failure mode STeP has
already hit and patched ad hoc, more than once, each time as a separate bugfix rather than a
recognized instance of one general problem:

- [post_grasp_carry_lift](post_grasp_carry_lift.md) (memory) -- finishing a pick without
  lifting first left OMPL unable to find any transport plan from that state.
- [open_drawer_return_home](open_drawer_return_home.md) (memory) -- finishing an open without
  returning the arm home blocked the next subtask's own planning.
- [transport_carry_height_rim_clip](transport_carry_height_rim_clip.md) (memory) -- finishing
  a transport at a fixed carry height clipped the goal container's rim for tall objects.

Each is the same shape: a subtask satisfied *its own* goal predicate but left the robot in a
state that made the *next* subtask infeasible or unsafe. Today that's only caught when a
rollout fails and someone patches that specific skill pairing. The paper's framing suggests
making this structural instead: each decomposed subtask could carry an explicit terminal
condition (e.g. "leave the state compatible with the next subtask's own precondition"), so
this class of bug is checked for systematically at decomposition time rather than discovered
one skill-pair at a time. **Not implemented** -- recorded here as a future-work direction, per
explicit instruction to just document it for now.

**3. A framing worth keeping for the future MPC-cost-term work** (see "Implementation status"
above -- the user's stated eventual goal of feeding absolute time bounds into the MPC as a
cost term, e.g. "pick up the bowl within 5 seconds"): this paper treats STL satisfaction as a
*hard constraint* (via MILP encoding) and keeps cost (robustness margin, control energy)
strictly separate. That suggests a hard deadline ("must finish by T") is more naturally a
feasibility constraint enforced at each subtask boundary than a cost penalty, while "try to be
fast without a hard failure" is the right shape for an actual cost term -- worth not
conflating the two mechanisms when that work is eventually designed.

**What was deliberately not ported:** the MILP encoding of STL robustness and the
Lipschitz-constant disturbance-tube backward recursion (their Theorem 3, computing robust
feasible terminal sets under adversarial disturbance growth) assume a classical
robust-control-theoretic setup that has no counterpart in STeP's current stack. No attempt was
made to adapt that machinery.

## Implemented: soft, log-only timing-overrun monitoring (2026-09-12)

Follow-up to a user question ("why is transport's F[0,180] bigger than place's F[0,120]?")
that led to a full audit of every existing temporal-constraint mechanism in the codebase,
then a literature pass, then this concrete, deliberately small feature.

**What the audit found:** both `vlm-stl/system1/nominal_mpc.py` (legacy Cartesian MPC) and its
ported successor `step/system_one/motion/mpc.py` (`JointSpaceMPC`) already have a complete,
differentiable timing-cost mechanism (`TimingWindow`, `_timing_cost_and_grad`, an
urgency-ramp toward an event target as a deadline approaches). It is **100% dead code**:
`TimingWindow(...)` is never constructed anywhere, `MPCContext.timing_windows` is never
populated by any real call site (every `MPCContext(...)` construction passes only
`reference=`), and its input data (`sub["applied_constraints"]` with `type: "timing_window"`)
is never emitted by any real generated plan and isn't even a field the JSON schema allows.
The legacy (non-`ompl_mpc`) execution path has the same gap one level down:
`executor._timing_windows_for_subtask` reads that same never-populated `applied_constraints`
field. So "the baseline of not having temporal-constraint monitoring" is not hypothetical --
it is exactly what runs today, on both backends.

**Literature checked, and why it doesn't directly fix the gap:** Pant/Abbas/Mangharam,
"Smooth Operator: Control Using the Smooth Robustness of Temporal Logic" (CCTA 2017) gives the
standard closed-form smooth max/min for STL robustness (`max̃_k(a) = (1/k)ln(Σe^(ka_i))`, with a
provable one-sided error bound `0 ≤ max̃_k(a) − max(a) ≤ ln(m)/k`) and frames the MPC problem as
maximizing that smooth robustness directly. It assumes the smooth-max is taken over the
*entire* relevant window in one optimization. `JointSpaceMPC`'s receding horizon is only
14-16 steps; a decomposed subtask's absolute window (from the SLICE work above) is typically
90-270+ steps. Porting the formula into `_timing_cost_and_grad` as-is doesn't fix anything --
that mechanism structurally can't see a deadline until it's already inside the tiny lookahead,
producing last-second-panic behavior no matter how principled the per-tick math is. Also
checked: putting a timing cost into the RRTConnect motion planner itself -- not viable, since
`ompl_planner.py` uses plain `og.RRTConnect` (a first-solution planner with no
`OptimizationObjective`, unlike RRT*/BIT*) and RRTConnect operates in pure configuration
space with no notion of time at all; real timing only gets decided later, in
`time_parameterize.py`'s velocity-limited pacing.

**What was actually built, per explicit user direction** ("the high-level monitoring is what's
important... if it fails, it would most likely be over the time limit, so just log it"): a
soft, log-only, end-to-end path from the SLICE-decomposed formula's own absolute deadlines to
the next System-2 recall, with **no** changes to `max_subtask_steps` (an unrelated, separate
per-skill hard cap) and no attempt to gate, abort, or repair timing at the System 1 level:

1. `step/system_two/stl_plan.py:decompose()` now attaches `subtask["stl_deadline"] =
   {"start_step", "end_step"}` from the goal's own `goal_absolute_window()` -- already
   computed for the sequential-timing validation, just exposed on the output now too.
2. `step/system_one/executor.py:_annotate_timing_violation()` (new): after each subtask
   finishes (success or failure), if the cumulative absolute step count exceeds
   `stl_deadline["end_step"]`, stamps a `timing_violation` fact onto that
   `SkillExecutionRecord.info` -- skill, object, declared deadline, actual step, overrun size,
   whether it succeeded anyway. Purely observational: never changes `record.done`, never
   aborts anything. A no-op for `--stl-mode skill_graph` plans (no `stl_deadline` to read).
   Distinct from the pre-existing, still-dead `_timing_window_violation_for_subtask`
   (different, unpopulated `applied_constraints` mechanism, left untouched).
3. `step/libero_runner.py:_annotate_timing_overruns()` (new): on a System-2 recall (the
   existing retry loop in `main()`, not the separate and still-disabled
   `_run_subtasks_on_env_with_system2_recall`/`recall_controller.py` subsystem), scans the
   just-finished attempt's `summary["records"]` for any `timing_violation` and stamps a
   `[Precomputed: ...]` note onto the relevant entity in the fresh scene_context -- mirroring
   the existing `_annotate_already_satisfied_objects` pattern -- describing exactly how much
   the previous attempt overran and by how much, so the recalled plan can decide (a larger
   budget, a different approach, or leaving it as-is) with real numbers instead of guessing.

**Tests:** `tests/test_timing_overrun.py` (9 deterministic tests, no live env) covers
`decompose()` attaching the right deadline, the annotation no-op on legacy plans and
within-budget runs, flagging both a failed and a late-but-successful overrun without
clobbering existing `record.info`, and `_annotate_timing_overruns` stamping/creating entities
correctly and no-op'ing on a clean run or an exception-fallback summary with no `records` key.
Full suite: 119 passed (same one pre-existing, unrelated flaky OMPL test as before). Also
re-ran all 8 real-scene cases end-to-end -- `stl_deadline` now appears correctly on every
subtask (confirmed cumulative, e.g. the 7-goal blocking scenario's last subtask reads
`{"start_step": 780, "end_step": 900}`), and skill sequences still match the legacy planner
exactly.

**Not done, by design:** no attempt to enforce, gate, or auto-repair timing at the System 1
level; no change to any MPC cost function; no change to `time_parameterize.py`'s pacing. This
is deliberately the smallest change that turns "zero temporal-constraint monitoring" into
"the next planning attempt is told exactly what went over budget and by how much" -- matching
"doesn't have to be perfect, just better than the baseline of not having it."

## Implemented: a "wait" skill for not-before constraints (2026-09-12)

Follow-up to the timing-overrun monitoring above: monitoring tells you a deadline was missed
*after the fact*, but says nothing about a **minimum-delay** constraint ("place it at least 5
seconds after picking it up") -- if the robot gets there early, it needs to actually wait, not
just get flagged for being early. Scoped to pick, transport, and place only, per explicit
instruction.

**Design:** `decompose()` already computes each goal's own absolute window
(`goal_absolute_window()`); its `start_step` **is** the not-before bound. All that was missing
was (a) a prompt path for the VLM to actually place a gap there when the instruction calls for
one, and (b) something in System 1 that holds position when it arrives at a subtask before its
own declared `start_step`.

1. **`step/system_one/motion/servo.py`: `wait_steps(adapter, n_steps, gripper_cmd,
   step_callback=None)`** (new, module-level). Holds the current end-effector pose completely
   still for `n_steps` ticks -- a zero Cartesian-delta `_step_env` call each tick, with
   `gripper_cmd` held constant (open pre-grasp, closed while carrying). Works with any adapter
   exposing `_step_env(cart_pos, cart_ori, gripper_cmd)` -- every `_ServoMixin` subclass and
   `LiberoPlaceAdapter` (which defines its own identically-shaped `_step_env` instead of
   inheriting the mixin) both qualify, so one function serves all three skills without
   duplicating the loop.
2. **`step/system_one/executor.py`: `_wait_for_stl_start(adapter, sub, *, task_steps_so_far,
   gripper_cmd, env)`** (new). Reads `sub["stl_deadline"]["start_step"]` (set by `decompose()`);
   if the real cumulative absolute step count hasn't reached it yet, calls `wait_steps` for the
   remaining gap and returns the steps consumed (0 in the common case -- no declared bound, or
   already caught up). Wired into `_run_pick_with_motion_pipeline` (gripper held **open** --
   nothing grasped yet), `_run_transport_with_motion_pipeline`, and
   `_run_place_with_motion_pipeline` (gripper held **closed** -- still carrying the object),
   called once before each adapter's own `.run(...)`, with the wait steps added into the
   returned step count (so `_annotate_timing_violation`'s overrun accounting, video capture,
   and logs all stay consistent). `open`/`close`/`turn_on` deliberately untouched, per scope.
3. **`build_global_developer_prompt` rule 2b** (new): tells the VLM that a stated minimum
   delay between two events becomes an explicit gap in the absolute timeline -- shift the
   *later* goal's `start_step` out (never shrink a duration to force a fit), and only when the
   natural cumulative chaining wouldn't already satisfy it.

**Verified against the real VLM** (`outputs/moka_task_verify`'s scene): asked for "place it on
the stove... at least 8 seconds (160 steps) after you pick it up" -- the model correctly
determined transport's own natural 180-step span (90->270) already exceeds a 160-step
requirement from pick's own end (90), so it emitted the plain zero-gap chain, no gap needed
(pick[0,90] -> transport[90,270] -> place[270,390]) -- exactly the right call, not a missed
case. Then asked for "at least 30 seconds (600 steps)" instead: it correctly opened a genuine
gap, place[690,810] instead of place[270,390] -- a 420-step wait between transport ending at
270 and place's own window starting at 690. Both decomposed cleanly with `decompose_and_validate`
and no `_validate_sequential_timing` violations (a gap is valid; `_validate_sequential_timing`
only rejects `start < previous_end`, never `start > previous_end`).

**Tests:** `tests/test_wait_skill.py` (7 tests, stub adapter, no live env) covers `wait_steps`
holding an exact zero-delta count with the given gripper command, no-op on zero/negative steps,
`step_callback` firing per tick, `_wait_for_stl_start` no-op'ing with no `stl_deadline` or when
already at/past `start_step`, holding the exact computed gap, and using the right gripper
command for a pre-grasp (pick) vs. carrying (transport/place) wait. Full suite: 126 passed
(same one pre-existing, unrelated flaky OMPL test).

**Verified with a real live rollout (2026-09-12), LIBERO Spatial task 0** ("pick up the black
bowl between the plate and the ramekin and place it on the plate"), custom prompt requesting a
minimum delay, `--stl-mode stl_spec` (the default), `--render --save-video`, full System
1 execution (not `--dry-run`):

- **First run, "at least 10 seconds (200 steps)":** decomposed to pick[0,90] ->
  transport[90,270] -> place[290,410] (a genuine 20-step gap). Task succeeded end-to-end, but
  the wait never actually engaged (`wait_steps: 0` for all three subtasks) -- pick took 345 real
  steps against its own 90-step *planned* budget (32 grasp candidates evaluated, most
  infeasible -- genuine OMPL/IK search cost, not a bug), so by the time transport finished
  (real cumulative step 429) the plan's own *static* schedule already considered place's
  not-before bound (290) satisfied. **Real finding, not a bug in the check itself:** the wait
  gate compares real elapsed steps against the *plan's* declared `start_step` (computed once,
  offline, from nominal per-goal durations) -- not against when the referenced event *actually*
  finished in this rollout. Here place started only 84 real steps (~4.2s) after pick's *real*
  completion, not the requested 10s -- satisfied on paper, not in the sense a person would mean
  it, once pick ran far behind its own planned duration. Worth having found this from a real
  rollout rather than assuming the plan-time check was sufcient.
- **Second run, "at least 30 seconds (600 steps)":** decomposed to pick[0,90] ->
  transport[90,270] -> place[690,810] (a large, deliberately-overkill 420-step planned gap,
  chosen precisely so it would survive a repeat of the first run's pick overrun). Pick again
  took 347 real steps, transport 83 more (real cumulative 430) -- still 260 steps short of
  place's 690 not-before bound, so this time the wait *did* engage:
  `wait_steps: 260` exactly matches `690 - 430`. Place's total steps (333) = 260 (wait) + 73
  (actual place motion, consistent with run 1's ~75-step place). Place finished at real
  absolute step 763, under its 810 deadline -- `timing_violation: None` for place, while pick
  and transport still correctly logged their own (harmless, unrelated) overruns. Overall
  `success: true`.
- **Visually confirmed via the saved video** (`outputs/libero_spatial_task0_wait_test_30s/rollout.mp4`,
  383 frames at `video_every_n_steps=2`): sampled frames at real steps ~430, ~560, and ~690
  (spanning the entire 260-step wait) show the arm in a visually identical pose -- holding
  still, gripper closed around the bowl, exactly as `wait_steps` is supposed to look. The frame
  just after (~step 720) shows the arm visibly moving again into its place-descent motion. The
  final frame shows the bowl resting on the plate with positive STL robustness for
  `Placed(...)`, confirming genuine success, not a monitor false-positive.

**Net result:** the full pipeline -- global-formula VLM plan -> SLICE decomposition with a
genuine not-before gap -> System 1 physically waiting out that gap -> successful completion,
correctly logged -- works, on a real task, not just synthetic unit tests. The one real caveat
(not-before measured against the plan's schedule, not the referenced event's real completion)
is now a known, documented limitation rather than an untested assumption.
