# Contact-GraspNet grasp generator (in-process, on the user's GPU machine)

## Context

The current pick pipeline's only `GraspGenerator` is `PrivilegedTopDownGraspGenerator`
(`step/system_one/motion/grasp_generator.py`) — a hand-written heuristic sampler
(center-grasp + rim-edge candidates from an oriented-bounding-box fit). The user judged
this "too rudimentary for more complex objects" and asked to integrate AnyGrasp
(`github.com/graspnet/anygrasp_sdk`) for better grasp selection.

Research findings that changed the plan:

- **This dev machine (macOS, Apple M2, no NVIDIA GPU) cannot run AnyGrasp, or any
  mainstream learned 6-DoF grasp detector, at all** — every one of them (AnyGrasp,
  GraspNet-baseline, Contact-GraspNet) requires CUDA. This is a hardware fact, not an
  integration-difficulty problem.
- **AnyGrasp specifically also gates its detection code behind a per-machine license
  request** (~5 business days), on top of the CUDA requirement — its actual weights/code
  aren't open source.
- Presented this to the user; they confirmed they have access to a separate CUDA machine
  and said: *"Feel free to use the Cap-X grasp net if its easier to implement."* Cap-X
  (`github.com/capgym/cap-x`, MIT licensed) is the same upstream project this codebase's
  `grasp_generator.py` already credits for its original privileged-grasp starting point,
  and it has a complete, working Contact-GraspNet client/server integration already built —
  fully open source, no license wait, and its `hand_from_fingertip_pose()` counterpart is
  already anticipated in this codebase's own code.

**Decision: implement a Contact-GraspNet generator (not AnyGrasp) that calls the model
in-process** (not over a network server) **on the user's CUDA machine**, using the exact
same underlying library Cap-X's own server wraps. The user confirmed they'll run the whole
pick step directly on their GPU machine when learned grasping is enabled, and explicitly
asked to skip the extra server process — so instead of mirroring Cap-X's FastAPI server +
HTTP client split, this generator imports and calls Contact-GraspNet's `GraspEstimator`
directly, the same way Cap-X's own server does internally before it ever serializes
anything over HTTP. STeP does not vendor or reimplement any CUDA code — it only gains a
thin wrapper around a class Cap-X already depends on, plus a new candidate generator that
plugs into the existing `PickSkill` pipeline exactly like the heuristic generator does
today.

**Deployment implication of this choice**: when `contact_graspnet.enabled: true`, the
*entire* `run_libero_step.py` process (sim included, not just the grasp call) must run on
a machine with `contact_graspnet_pytorch` + CUDA + the checkpoint installed — there's no
longer a network boundary letting the sim run on this Mac while only inference happens
remotely. With `enabled: false` (the shipped default), nothing changes: this Mac keeps
developing/testing the heuristic-only path exactly as today.

Confirmed by re-reading `pick_skill.py`, `types.py`, `config.py`, and `scene_query.py`
directly (not assumed): `PickSkill` already treats "which generator produced this
candidate" as irrelevant — its job is scoring/IK/collision feasibility over whatever
candidate list it's handed. This means the new generator is purely additive: it runs
*alongside* the existing heuristic generator, and is off by default, so nothing changes
for anyone without a configured server.

## Cap-X's confirmed in-process call (to reuse directly, not over HTTP)

Read verbatim from `capx/serving/launch_contact_graspnet_server.py` — this is what its
FastAPI endpoint does internally before ever touching HTTP/base64, and it's exactly what
our generator will call directly instead:

```python
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch.checkpoints import CheckpointIO

estimator = GraspEstimator(global_config)          # global_config: dict, from a YAML
checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir, model=estimator.model)
checkpoint_io.load("model.pt")

pred_grasps_cam, scores, contact_pts, _ = estimator.predict_scene_grasps(
    pc_full,                       # (M, 3) full-scene point cloud
    pc_segments={1: pc_segment},   # dict: segmap_id -> (K, 3) target-object points
    local_regions=True,
    filter_grasps=True,
    forward_passes=1,
)
grasps_i = pred_grasps_cam[1]   # (N, 4, 4) SE3 matrices, contact/fingertip frame
scores_i = scores[1]            # (N,)
contact_pts_i = contact_pts[1]  # (N, 3)
```

Cap-X's own client-side code then applies a hardcoded 0.12m Z-offset to go from this output
frame to their hand frame. **We will not copy that constant** — this codebase already has a
*measured* equivalent, `grasp_generator.hand_from_fingertip_pose()` (reads the real
fingertip→hand transform via `SceneQuery.hand_to_tcp_transform`), consistent with this
package's established practice of measuring real transforms instead of hardcoding them.

## Real gaps found while validating this plan against the actual code (not hypothetical)

1. **`PickSkill` reads `self.grasp_generator.gripper_max_opening_m` directly**
   (`pick_skill.py:199`) — this attribute lives on `PrivilegedTopDownGraspGenerator`, not on
   the abstract `GraspGenerator` base. The new composite wrapper must expose it too, or
   `PickSkill` raises `AttributeError` the first time any candidate has a non-`None` width.
2. **`SceneQuery.body_surface_points_local()`** (the only existing "real geometry" sampler,
   used today for OBB-fitting) **is far too sparse for point-cloud-based grasp inference** —
   confirmed by reading `_geom_sample_points_local`: it returns only 8 box corners, ~33
   points for a capsule/cylinder, 7 for a sphere, and **zero points for a plane geom** (so
   the table surface contributes nothing). Real Contact-GraspNet expects hundreds+ of
   points. A new, denser sampler is needed — gridding box faces and adding more
   rings/latitude bands for round primitives — still pure ground-truth-mesh geometry, no
   camera rendering involved.
3. **No existing way to enumerate "other nearby bodies"** for `pc_full` scene context —
   `LiberoPickAdapter` doesn't have `scene_context`/`SceneResolver` wired in at all, only
   `env`+`motion_config`. Adding a `SceneQuery.nearby_body_names(point_world, radius_m)`
   method (raw MuJoCo body iteration, filtered to free-jointed bodies only — this
   automatically excludes the fixed robot/table/world bodies without needing any new
   scene_context plumbing) is simpler than threading `SceneResolver` through.
4. **`contact_graspnet_pytorch` (and torch+CUDA) must never be imported when
   `enabled: false`** (the shipped default), since this Mac's dev/test env has none of
   those. Mirror the existing lazy-import-with-clear-error pattern this codebase already
   uses for `mujoco` in `scene_query.py`: import only inside the method that needs it, and
   only when actually invoked with `enabled: true`.
5. **No existing "call an optional external dependency, degrade gracefully" pattern** in
   this codebase to mirror (the one other external call, `system_two/planner.py`'s OpenAI
   client, is unguarded and fatal-on-failure by design). This generator introduces that
   pattern fresh: any error from the estimator (bad checkpoint, CUDA OOM, malformed
   geometry, missing package) is caught and `generate()` returns `[]`, never raises — a
   misconfigured/unavailable model just means the heuristic generator's candidates carry
   the pick attempt alone, exactly like today.
6. **Testability, addressed by dependency injection.** Since inference is now a direct
   Python call instead of an HTTP round-trip, `ContactGraspNetGenerator` takes an
   `estimator_factory` constructor argument (defaulting to the real lazy loader shown
   above). Tests inject a fake factory returning a stub object whose
   `predict_scene_grasps(...)` returns a fixed canned tuple — this fully exercises pose
   conversion, scoring, and candidate construction on this Mac with zero GPU, zero network,
   and zero `contact_graspnet_pytorch` import.
7. **Known, accepted limitation for this pass**: the point cloud sent has no table/floor
   representation (plane geoms sample to zero points) and is a full, unoccluded view (not a
   realistic single-camera partial view) — this is the simplest buildable v1 given this is
   sim ground truth, not a real camera. If real testing on the user's GPU machine shows
   disappointing grasp quality, the documented next step is switching to a simulated
   depth-camera capture (mirroring Cap-X's own `agentview`/`robot0_eye_in_hand` multi-view
   fusion) instead of ground-truth mesh sampling — out of scope for this pass.

## Implementation

**New file: `step/system_one/motion/contact_graspnet_generator.py`**
- `_load_real_estimator(config: ContactGraspNetConfig)`: the ONLY place
  `contact_graspnet_pytorch`/torch is imported, done lazily inside this function body (not
  at module top-level) so importing this file never requires CUDA/the package to be
  installed. Loads the YAML config from `config.checkpoint_dir`, constructs `GraspEstimator`,
  loads `model.pt` via `CheckpointIO`, moves it to `config.device`, returns the estimator.
  Raises a clear `ImportError`/`FileNotFoundError` with install/setup instructions if the
  package or checkpoint isn't present — mirroring `scene_query.py`'s existing pattern for a
  missing `mujoco` install.
- `class ContactGraspNetGenerator(GraspGenerator)`:
  - `__init__(self, config: ContactGraspNetConfig, *, estimator_factory=None, hand_site="gripper0_right_ft_frame", tcp_site="gripper0_right_grip_site", gripper_max_opening_m: float = 0.08)`
    — `estimator_factory` defaults to `_load_real_estimator`; tests inject a fake. The
    estimator itself is lazy-loaded and cached on first use (`self._estimator = None`
    until `_get_estimator()` is first called), so constructing this generator (e.g. inside
    `CompositeGraspGenerator`) never by itself touches CUDA or loads a checkpoint.
  - `generate(self, scene, object_names) -> list[GraspCandidate]`: returns `[]` immediately
    if `not self.config.enabled`. Otherwise, per object: build `pc_segment` (dense
    world-frame points for that object, via the new dense sampler) and `pc_full`
    (`pc_segment` unioned with dense points from `scene.nearby_body_names(object_center,
    config.point_cloud_radius_m, exclude=[object_name])`); call
    `estimator.predict_scene_grasps(pc_full, pc_segments={1: pc_segment},
    local_regions=config.local_regions, filter_grasps=config.filter_grasps,
    forward_passes=config.forward_passes)`; read back `pred_grasps_cam[1]`/`scores[1]`;
    convert each pose via `hand_from_fingertip_pose(...)`; clip `generator_score` to
    `[0, 1]`; set `gripper_width=None` (v1 — deriving a real width from `contact_pts` needs
    a live model output to validate the semantics against first, not guessed); cap the
    returned list at `config.max_candidates` by score. Any exception anywhere in this path
    (missing package/checkpoint, CUDA OOM, malformed output, empty scene points) is caught
    and logged as a diagnostic — the method returns `[]` for that object, never raises, so
    one object's inference failure doesn't block picks on other objects or fall through to
    a crash.

**Edit: `step/system_one/motion/grasp_generator.py`**
- Add `CompositeGraspGenerator(GraspGenerator)`: wraps a list of generators, `generate()`
  concatenates their results; `gripper_max_opening_m` is read from the wrapped generators at
  construction (`ValueError` if they disagree — it's a property of the physical gripper, not
  of the algorithm, so a mismatch means misconfiguration, not a valid state to silently
  paper over).

**Edit: `step/system_one/motion/scene_query.py`**
- `nearby_body_names(self, point_world, radius_m, exclude=()) -> list[str]`: iterate
  `self.model.nbody`, keep only free-jointed bodies (`mjtJoint.mjJNT_FREE`) within
  `radius_m`, excluding names in `exclude`.
- `body_surface_points_dense_local(self, body_name, ...) -> np.ndarray`: same per-geom-type
  branch structure as `_geom_sample_points_local`, but grids box/mesh faces (not just 8
  corners) and adds more rings/bands for capsule/cylinder/sphere. Used only by the new
  generator — `_object_xy_obb_world`'s existing OBB fit keeps using the sparse version
  (unaffected, no regression risk to today's heuristic generator).

**Edit: `step/system_one/motion/config.py`**
```python
@dataclass
class ContactGraspNetConfig:
    enabled: bool = False
    checkpoint_dir: str = ""   # path to the Contact-GraspNet checkpoint (model.pt + config.yaml)
    device: str = "cuda"
    forward_passes: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    max_candidates: int = 32
    point_cloud_radius_m: float = 0.15
```
Wire into `MotionConfig` (`contact_graspnet: ContactGraspNetConfig = field(default_factory=...)`)
and `from_scene_context` (`ContactGraspNetConfig(**raw.get("contact_graspnet", {}))`) —
identical pattern to the existing `DrawerConfig`.

**Edit: `configs/libero_template.json`** — add a `"contact_graspnet": {"enabled": false,
"checkpoint_dir": "", "device": "cuda", "forward_passes": 1, "local_regions": true,
"filter_grasps": true, "max_candidates": 32, "point_cloud_radius_m": 0.15}` block next to
the existing `"drawer"` block. Stays disabled by default — zero behavior change for
existing runs/tests. On the GPU machine, the user sets `checkpoint_dir` to wherever they
installed the real Contact-GraspNet weights and flips `enabled: true`.

**Edit: `step/system_one/motion/libero_pick_adapter.py`** (`_build_pick_skill`):
```python
gen: GraspGenerator = PrivilegedTopDownGraspGenerator(self.config.grasp)
if self.config.contact_graspnet.enabled:
    from .contact_graspnet_generator import ContactGraspNetGenerator
    gen = CompositeGraspGenerator([gen, ContactGraspNetGenerator(self.config.contact_graspnet)])
```

## Tests (all runnable in this sandbox, no GPU/network needed)

- `tests/test_scene_query.py`: `nearby_body_names` (radius/exclude/free-joint-only
  filtering) and `body_surface_points_dense_local` (point count actually increases vs. the
  sparse version; still bounds the real geometry) — both `@requires_libero`.
- `tests/test_grasp_generator.py`: `CompositeGraspGenerator` concatenation using plain stub
  `GraspGenerator`s (no libero needed); `gripper_max_opening_m` delegation, and a
  constructor `ValueError` on mismatched wrapped generators.
- New `tests/test_contact_graspnet_generator.py`:
  - `enabled=False` → `generate()` returns `[]`, confirms `estimator_factory` is never
    called (no lazy load attempted) and no import of `contact_graspnet_pytorch` occurs.
  - Point-cloud construction shape/world-frame correctness against the real `libero_scene`
    fixture (`@requires_libero`, no fake/real model needed for this part).
  - Fake-estimator happy path: inject `estimator_factory=lambda cfg: FakeEstimator(...)`
    where `FakeEstimator.predict_scene_grasps(...)` returns a canned
    `(pred_grasps_cam, scores, contact_pts, None)` tuple shaped like the real API; assert
    the resulting `GraspCandidate`s' pose conversion matches a direct
    `hand_from_fingertip_pose()` call on the same input, `generator_score` is clipped to
    `[0,1]`, `gripper_width is None`, and `max_candidates` truncation is respected.
    (`@requires_libero` for the real hand/tcp transform, no CUDA/package needed.)
  - Fake estimator raising (simulating CUDA OOM / malformed output / missing checkpoint) →
    `generate()` returns `[]` without propagating the exception.
  - `_load_real_estimator` itself: a lightweight test confirming it raises a clear,
    actionable error (not an unhandled `ModuleNotFoundError`) when
    `contact_graspnet_pytorch` isn't installed — this always runs on this Mac's env, which
    genuinely lacks the package, so it's a real (not hypothetical) regression guard.
- `tests/test_pick_skill_or.py`: one new test running `PickSkill` with
  `CompositeGraspGenerator([PrivilegedTopDownGraspGenerator(...), stub_generator])` to
  confirm OR-set feasibility scoring treats both generators' candidates identically.
- Full suite: `python -m pytest tests/ -q` (conda env `robosuite_residual_grasp_demo`) must
  stay green — currently 68/68, all new tests added on top.

## Verification

1. All new/updated unit tests above, run locally — this fully exercises point-cloud
   construction, pose conversion, config wiring, and graceful-fallback behavior via the
   injected fake estimator, without needing actual CUDA or the real package.
2. Confirm `enabled: false` (the shipped default) leaves every existing behavior/test
   byte-identical — no regression risk for the current heuristic-only pipeline.
3. **What I cannot verify from this machine**: actual Contact-GraspNet grasp quality, and
   that `_load_real_estimator`'s exact API usage matches the real installed package version
   (it's transcribed from reading Cap-X's server source, not executed against a real
   checkpoint). Once this lands, the user needs to install `contact_graspnet_pytorch` +
   its checkpoint on their CUDA machine (this repo, running there), set `checkpoint_dir`,
   flip `enabled: true`, and run one real `libero_goal`/`libero_object` task end-to-end so
   we can check the resulting `execution_summary.json` together and fix up any real-API
   mismatch `_load_real_estimator` has — consistent with this project's standing rule of
   confirming real runs rather than declaring success from tests alone.
