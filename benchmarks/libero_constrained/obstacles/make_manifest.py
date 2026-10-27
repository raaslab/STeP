"""Write tasks_v2.json: every task a LIBERO task plus a formal constraint.

Each task carries:

  prompt      the instruction, stating the rule plainly and exactly
  type        which constraint type it is (spec_library.py)
  spec        the rule in STL, over signals any policy logs (logger.py);
              spec.py scores a rollout by evaluating it
  difficulty  how many rules are imposed at once (1, 2 or 3)

The families:

  spatial     keep-out (no-contact, distance) around a real obstacle added to
              the scene (build.py), at L1 beside the picked object or L2 on
              its route
  kinematic   how the object is carried: upright
  temporal    hold before release, wait after an event, do not start early
  logical     order, precondition, leave something alone
  placement   end at an offset, or on a side
  combined    two or three of the above at once
  unconstrained  LIBERO's own task, for a capability baseline

What a rule may read is limited to the benchmark log, so a policy is scored
by what it did, never by records only one method keeps. Every rule uses a
predicate STeP's System 2 already has (Safe, Upright, timing windows, order);
there are no speed or carry-height-ceiling rules.

  python benchmarks/libero_constrained/obstacles/make_manifest.py [out.json]
"""

import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH))

from build import SPOKEN  # noqa: E402
import spec_library as L  # noqa: E402

# ---------------------------------------------------------------- limits
# Set from how STeP carries objects without a rule (3-8 degrees of tilt on bowls
# and mugs; outputs/v2_smoke): each limit is broken by unconstrained motion and
# reachable by careful motion.
MARGIN_M = 0.08          # carried object to obstacle, centre to centre
MARGIN_SLACK_M = 0.01    # a start is kept only if pick and goal already clear MARGIN_M by this
UPRIGHT_DEG = 5.0
DWELL_S = 2

# Why each type matters on a real robot. Metadata for the reader; the prompt
# states only the rule.
MOTIVATION = {
    "no-contact": "fragile or unstable items near the workspace must not be disturbed",
    "keep-out": "a clearance around people, hot surfaces or fragile items",
    "upright": "orientation constraints when carrying anything with liquid in it",
    "dwell": "letting an item settle before release",
    "wait": "a process that needs time before the next step (heating, settling)",
    "gate": "a step that must not begin before another has finished",
    "order": "stacking or loading order",
    "precondition": "a step that is valid only once a condition holds",
    "untouched": "leaving other people's items alone",
    "placement": "placing to a specification, not just somewhere on the target",
}

# ---------------------------------------------------------------- task sets

# Kinematic rules on LIBERO's own scenes and start states: (suite, id, carried).
# Bowls and mugs that start on the table, so "above the table" is "above start".
KINEMATIC = [
    ("libero_spatial", 1, "akita_black_bowl_1"),
    ("libero_spatial", 2, "akita_black_bowl_1"),
    ("libero_spatial", 6, "akita_black_bowl_1"),
    ("libero_goal", 1, "akita_black_bowl_1"),
    ("libero_goal", 8, "akita_black_bowl_1"),
    ("libero_10", 6, "porcelain_mug_1"),
]
KINEMATIC_STATES = 10

CARRIED = {
    "akita_black_bowl_1": "bowl",
    "porcelain_mug_1": "white mug",
    "bbq_sauce_1": "bbq sauce",
    "butter_1": "butter",
    "chocolate_pudding_1": "chocolate pudding",
    "orange_juice_1": "orange juice",
    "moka_pot_1": "moka pot",
    "black_book_1": "book",
    "alphabet_soup_1": "alphabet soup",
}
# Carried objects that can sensibly be kept upright -- not a book that has to
# be stood on end, not a can that has no up.
UPRIGHT_OK = {"akita_black_bowl_1", "porcelain_mug_1", "moka_pot_1"}

# v1 rule tasks: the formal rule for each, where v1's scoring entry alone does
# not say enough (which object, which event). Prompts are v1's, unchanged.
TEMPORAL_SPECS = {
    "libero_10/0/T-dwell": ("dwell", {"or": [L.dwell("alphabet_soup_1", "basket_1", 2),
                                             L.dwell("tomato_sauce_1", "basket_1", 2)]}),
    "libero_10/4/T-dwell": ("dwell", {"or": [L.dwell("porcelain_mug_1", "plate_1", 2),
                                             L.dwell("white_yellow_mug_1", "plate_2", 2)]}),
    "libero_goal/1/T-dwell": ("dwell", L.dwell("akita_black_bowl_1", "flat_stove_1", 2)),
    "libero_goal/8/T-dwell": ("dwell", L.dwell("akita_black_bowl_1", "plate_1", 2)),
    "libero_spatial/0/T-dwell": ("dwell", L.dwell("akita_black_bowl_1", "plate_1", 2)),
    "libero_10/2/T-wait": ("wait", L.wait_after(L.P({"bddl": "turnon flat_stove_1"}, ">=", 0.5),
                                                "moka_pot_1", 4)),
    # bottom drawer starts open at -0.146; closing has begun once it is 1 cm in
    "libero_10/3/T-wait": ("wait", {"U": [
        {"not": L.P({"joint": "white_cabinet_1_bottom_level"}, ">=", -0.136)},
        L.AND(L.P({"bddl": "in akita_black_bowl_1 white_cabinet_1_bottom_region"}, ">=", 0.5),
              {"G": {"not": L.P({"joint": "white_cabinet_1_bottom_level"}, ">=", -0.136)}, "in": [0, 4]})]}),
    # microwave door starts open at -1.579 rad; closing has begun at -1.53
    "libero_10/9/T-wait": ("wait", {"U": [
        {"not": L.P({"joint": "microwave_1_microjoint"}, ">=", -1.53)},
        L.AND(L.P({"bddl": "in white_yellow_mug_1 microwave_1_heating_region"}, ">=", 0.5),
              {"G": {"not": L.P({"joint": "microwave_1_microjoint"}, ">=", -1.53)}, "in": [0, 4]})]}),
    # top drawer starts shut at 0; open once pulled 10 cm
    "libero_goal/3/T-wait": ("wait", L.wait_after(L.P({"joint": "wooden_cabinet_1_top_level"}, "<=", -0.10),
                                                  "akita_black_bowl_1", 4)),
    "libero_10/2/T-stage": ("gate", L.gate(L.P({"bddl": "turnon flat_stove_1"}, ">=", 0.5),
                                           L.lifted("moka_pot_1"))),
    "libero_10/3/T-stage": ("gate", L.gate(
        L.AND(L.P({"bddl": "in akita_black_bowl_1 white_cabinet_1_bottom_region"}, ">=", 0.5),
              L.P({"held": "akita_black_bowl_1"}, "<=", 0.5)),
        L.P({"joint": "white_cabinet_1_bottom_level"}, ">=", -0.136))),
    # the drawer's approach is clear at the moment it starts to open
    "libero_goal/3/T-stage": ("gate", {"U": [
        {"not": L.P({"joint": "wooden_cabinet_1_top_level"}, "<=", -0.01)},
        L.AND(L.P({"joint": "wooden_cabinet_1_top_level"}, "<=", -0.01),
              L.P({"count_near": "wooden_cabinet_1", "radius": 0.18}, "<=", 0.5))]}),
}

TYPE_OF_KIND = {
    "L-order": "order", "L-precondition": "precondition", "L-untouched": "untouched",
    "S-offset": "placement", "S-relative": "placement",
}

# Dropped from v1: the virtual zones (replaced by real obstacles), and a
# precondition with no container to check.
DROP = {"S-zone", "C-zone-order", "C-zone-wait"}
DROP_IDS = {"libero_spatial/4/L-precondition"}

LEVEL_NOTE = {
    "L1": "obstacle beside the object being picked, off the route",
    "L2": "obstacle on the straight route from the object to its destination",
}


def libero_language(suite, task_id):
    from step.system_one.libero_scene import _ensure_libero_on_path, _patch_torch_load_for_libero
    _ensure_libero_on_path()
    _patch_torch_load_for_libero()
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite]().get_task(task_id).language


def goal_object(target):
    """The object a BDDL goal target names: plate_1, or flat_stove_1 for
    flat_stove_1_cook_region."""
    m = re.match(r"^(.*?_\d+)(_|$)", target)
    return m.group(1) if m else target


def margin_subset(lv):
    """Indices of L2 starts where the object and its goal already clear MARGIN_M."""
    keep = []
    rows = [r for r in lv["starts"] if r["kept"]]
    for i, row in enumerate(rows):
        ob = np.array(row["obstacle_xy"])
        if (np.linalg.norm(np.array(row["pick_xy"]) - ob) >= MARGIN_M + MARGIN_SLACK_M
                and np.linalg.norm(np.array(row["goal_xy"]) - ob) >= MARGIN_M + MARGIN_SLACK_M):
            keep.append(i)
    return keep


def task(tid, family, kind, ctype, suite, task_id, prompt, spec, difficulty, **extra):
    t = {"id": tid, "family": family, "kind": kind, "type": ctype, "suite": suite,
         "task_id": task_id, "init_state_id": 0, "prompt": prompt, "spec": spec,
         "difficulty": difficulty}
    if ctype in MOTIVATION:
        t["motivation"] = MOTIVATION[ctype]
    t.update(extra)
    # score.py dispatches on scoring.type; every rule now goes through the spec
    t["scoring"] = {"type": "spec"} if spec is not None else {"type": "success_only"}
    return t


def main():
    v1 = json.loads((BENCH / "tasks.json").read_text())
    report = json.loads((HERE / "build_report.json").read_text())
    v1_by_id = {t["id"]: t for t in v1["tasks"]}
    out_tasks = []

    # ---- v1 tasks that stay: rule attached, prompt unchanged
    for t in v1["tasks"]:
        if t["kind"] in DROP or t["id"] in DROP_IDS or t["kind"].startswith("C-"):
            continue
        base = {k: t[k] for k in ("suite", "task_id")}
        if t["kind"] == "U-plain":
            out_tasks.append(task(t["id"], "unconstrained", "U-plain", None, prompt=t["prompt"],
                                  spec=None, difficulty=0, **base))
            continue
        if t["id"] in TEMPORAL_SPECS:
            ctype, sp = TEMPORAL_SPECS[t["id"]]
        else:
            ctype, sp = TYPE_OF_KIND.get(t["kind"]), L.from_v1_scoring(t, v1_by_id)
        if sp is None:
            print(f"  no rule for {t['id']}; dropped")
            continue
        family = {"T": "temporal", "L": "logical", "S": "placement"}[t["kind"][0]]
        out_tasks.append(task(t["id"], family, t["kind"], ctype, prompt=t["prompt"], spec=sp,
                              difficulty=1, **base))
    by_id = {t["id"]: t for t in out_tasks}

    # ---- spatial: real obstacles
    obstacle = {}
    for key, r in sorted(report.items()):
        if "levels" not in r:
            continue
        suite, tid_ = r["suite"], r["task_id"]
        base = libero_language(suite, tid_)
        ob, ob_word = r["obstacle"], SPOKEN.get(r["obstacle_type"], r["obstacle_type"].replace("_", " "))
        carried = r["pick"]
        c_word = CARRIED.get(carried, carried.rsplit("_", 1)[0].replace("_", " "))
        scene = {"bddl": r["bddl"], "obstacle": ob}
        for level, lv in r["levels"].items():
            if not lv["kept"]:
                continue
            states = {"init_states": lv["init_states"], "num_states": lv["kept"], "level": level,
                      "level_note": LEVEL_NOTE[level]}
            t = task(f"{key}/O-{level}", "spatial", f"O-{level}", "no-contact", suite, tid_,
                     f"{base}, without touching the {ob_word}", L.no_contact(ob), 1, **scene, **states)
            obstacle[t["id"]] = t
            out_tasks.append(t)

        # distance rules ride on the L2 scenes, on the starts where they can hold
        lv = r["levels"].get("L2") or {}
        keep = margin_subset(lv) if lv.get("kept") else []
        if not keep:
            continue
        src = ROOT / lv["init_states"]
        data = np.load(src)
        dst = src.with_name(src.stem.replace("_L2", "_margin") + ".npz")
        np.savez_compressed(dst, states=data["states"][keep], source_init_ids=data["source_init_ids"][keep])
        mstates = {"init_states": str(dst.relative_to(ROOT)), "num_states": len(keep), "level": "L2",
                   "level_note": LEVEL_NOTE["L2"]}
        margin_spec = L.margin(carried, ob, MARGIN_M)
        margin_words = f"at least {MARGIN_M*100:.0f} cm away from the {ob_word}"
        out_tasks.append(task(f"{key}/O-margin", "spatial", "O-margin", "keep-out", suite, tid_,
                              f"{base}, keeping the {c_word} {margin_words}", margin_spec, 1,
                              **scene, **mstates))
        if carried in UPRIGHT_OK:
            up = L.upright(carried, UPRIGHT_DEG)
            dest = goal_object(r["goal"])
            hold = L.dwell(carried, dest, DWELL_S)
            out_tasks.append(task(
                f"{key}/C-margin-upright", "combined", "C-margin-upright", "keep-out+upright", suite, tid_,
                f"{base}, keeping the {c_word} upright (tilt under {UPRIGHT_DEG:.0f} degrees) and {margin_words}",
                L.AND(margin_spec, up), 2, **scene, **mstates))
            out_tasks.append(task(
                f"{key}/C-margin-upright-dwell", "combined", "C-margin-upright-dwell", "keep-out+upright+dwell",
                suite, tid_,
                f"{base}, keeping the {c_word} upright (tilt under {UPRIGHT_DEG:.0f} degrees) and "
                f"{margin_words}, and holding it steady above the target for {DWELL_S} seconds before "
                f"letting go",
                L.AND(margin_spec, up, hold), 3, **scene, **mstates))

    # ---- kinematic: how the object is carried, on LIBERO's own scenes
    for suite, tid_, carried in KINEMATIC:
        base = libero_language(suite, tid_)
        c_word = CARRIED[carried]
        key = f"{suite}/{tid_}"
        common = {"num_states": KINEMATIC_STATES}
        out_tasks.append(task(f"{key}/K-upright", "kinematic", "K-upright", "upright", suite, tid_,
                              f"{base}, keeping the {c_word} upright while carrying it "
                              f"(tilt under {UPRIGHT_DEG:.0f} degrees)",
                              L.upright(carried, UPRIGHT_DEG), 1, **common))

    # ---- combined: v1's two-rule tasks, rebuilt from their parts' rules
    def combo(cid, kind, ctype, prompt, part_ids):
        parts = [by_id.get(p) or obstacle.get(p) for p in part_ids]
        if any(p is None for p in parts):
            print(f"  skip {cid}: a part is missing")
            return
        src = parts[0]
        extra = {k: src[k] for k in ("bddl", "obstacle", "init_states", "num_states", "level") if k in src}
        out_tasks.append(task(cid, "combined", kind, ctype, src["suite"], src["task_id"], prompt,
                              L.AND(*[p["spec"] for p in parts]), len(parts), **extra))

    for cid in ("libero_10/4/C-offset-dwell", "libero_goal/8/C-offset-dwell"):
        t = v1_by_id[cid]
        combo(cid, "C-offset-dwell", "placement+dwell", t["prompt"], t["scoring"]["parts"])
    combo("libero_10/7/C-obstacle-order", "C-obstacle-order", "no-contact+order",
          libero_language("libero_10", 7) + ", without touching the wine bottle, putting the bigger one in first",
          ["libero_10/7/O-L2", "libero_10/7/L-order"])
    combo("libero_10/2/C-obstacle-wait", "C-obstacle-wait", "no-contact+wait",
          libero_language("libero_10", 2) + ", without touching the wine bottle, waiting at least 4 seconds "
          "after turning on the stove before starting the next step",
          ["libero_10/2/O-L2", "libero_10/2/T-wait"])

    manifest = {
        "name": "LIBERO-Constrained",
        "version": 2,
        "control_hz": v1["control_hz"],
        "log": "benchmark_log.npz, written by logger.py",
        "scoring": "spec.py evaluates each task's STL spec on the log; robustness in cm, degrees, cm/s",
        "notes": (
            "Every rule is an STL formula over logged motion. Spatial tasks add one real object "
            "to a stock LIBERO scene (obstacles/build.py; nothing already there is moved); the "
            "other families use LIBERO's own scenes and start states. Limits are set so that "
            "unconstrained motion breaks them and careful motion can meet them."
        ),
        "tasks": out_tasks,
    }
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH / "tasks_v2.json"
    out.write_text(json.dumps(manifest, indent=1))
    count = {}
    for t in out_tasks:
        count[(t["family"], t["kind"])] = count.get((t["family"], t["kind"]), 0) + 1
    print(f"wrote {out}: {len(out_tasks)} tasks")
    for (fam, kind), n in sorted(count.items()):
        print(f"  {fam:14s} {kind:24s} {n}")


if __name__ == "__main__":
    main()
