"""VisionPerception: builds and maintains the perceived world from RGB-D
and proprioception only.

Lifecycle per env:
  attach(raw_env)   after every reset (robosuite rebuilds the sim on reset)
  discover()        first keyframe: VLM categories -> SAM3 -> Set-of-Mark names
  refresh()         later keyframes: re-detect known categories, re-associate
                    by 3D proximity, re-fit geometry
  on_step()         every env step: grasp/release events from the gripper's
                    own joint state, carry held objects with the TCP, couple
                    a grasped drawer to the hand

Keyframes happen at reset, at each subtask start, after every release, on
recall, and optionally every cfg.refresh_every_steps. In between, objects
stay where they were last seen unless the robot is carrying them -- the
same perceive-then-act assumption VoxPoser and OK-Robot make.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..system_one.motion.panda import EE_SITE, FINGER_JOINT_NAMES, GRIPPER_BODIES, ROBOT_BODIES
from ..system_one.motion.scene_query import SceneQuery, quat_to_rotmat_wxyz
from . import labeling
from .config import PerceptionConfig
from .geometry import clean_points, fit_plane_normal, fit_support_plane, heightmap_shape
from .rgbd import RGBDFrame, capture_rgbd
from .segmenter import Detection, Sam3Segmenter
from .twin import PerceivedSceneQuery, TwinSync, build_twin_model
from .world import PerceivedDrawer, PerceivedKnob, PerceivedObject, WorldEstimate, yaw_quat_wxyz

_LEVELS_BY_COUNT = {1: [""], 2: ["top", "bottom"], 3: ["top", "middle", "bottom"]}
_PART_WORDS = ("drawer", "handle", "knob", "door", "compartment", "burner")


class _Candidate:
    """One SAM3 instance lifted to 3D."""

    def __init__(self, det: Detection, frame: RGBDFrame, points: np.ndarray) -> None:
        self.det = det
        self.frame = frame
        self.points = points
        self.centroid = points.mean(axis=0) if len(points) else np.full(3, np.nan)


def _pose_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    T[:3, :3] = mat.reshape(3, 3)
    T[:3, 3] = pos
    return T


def _matrix_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T[:3, :3].reshape(9))
    return T[:3, 3].copy(), quat


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(max(1, np.logical_or(a, b).sum()))


class VisionPerception:
    def __init__(
        self,
        cfg: PerceptionConfig | None = None,
        *,
        instruction: str = "",
        prior: dict[str, Any] | None = None,
        debug_dir: str | Path | None = None,
        vlm_model: str | None = None,
    ) -> None:
        self.cfg = cfg or PerceptionConfig()
        self.instruction = instruction
        self.prior = prior or {}
        self.vlm_model = vlm_model
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.segmenter = Sam3Segmenter.shared()
        self.world = WorldEstimate()
        self.categories: list[str] = []
        self.sim = None
        self._world_xml: str | None = None
        self.twin_model: mujoco.MjModel | None = None
        self.sync: TwinSync | None = None
        self.step_count = 0
        self._gripper_cmd = -1.0
        self._closed_steps = 0
        self._open_steps = 0
        self._grasp_evaluated = False
        # Per grasp: TCP height at attach time, and the lift-and-look verdict
        # (None until the arm has lifted).
        self._grasp_tcp_z: float | None = None
        self._grasp_width_confirmed = False
        self._grasp_visual_ok: bool | None = None
        self._grasp_visual_step = 0
        self._pending_refresh_after: int | None = None
        self.keyframe_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ wiring

    def attach(self, raw_env: Any) -> None:
        """Bind to the robosuite env's current sim. The task XML is used only
        for the robot's own model (everything else is stripped)."""
        self.sim = raw_env.sim
        self._world_xml = raw_env.model.get_xml()
        self.sim._step_perception = self
        self._finger_addrs = None
        self._rebuild_twin()

    def _rebuild_twin(self) -> None:
        self.twin_model = build_twin_model(self._world_xml, self.world)
        self.sync = TwinSync(self.twin_model, self.sim.model._model, self.world)

    def scene_query(self) -> PerceivedSceneQuery:
        data = self.sync.snapshot(self.sim.data)
        return PerceivedSceneQuery(self.twin_model, data, perception=self, live_sim=self.sim)

    # ------------------------------------------------------------- proprioception

    def _finger_opening_and_speed(self) -> tuple[float, float]:
        if self._finger_addrs is None:
            scene = self.scene_query()
            names = []
            for n in FINGER_JOINT_NAMES:
                try:
                    scene.joint_id(n)
                    names.append(mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_JOINT, scene.joint_id(n)))
                except KeyError:
                    pass
            live = self.sim.model._model
            ids = [mujoco.mj_name2id(live, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
            self._finger_addrs = [(int(live.jnt_qposadr[j]), int(live.jnt_dofadr[j])) for j in ids if j >= 0]
        q = [float(self.sim.data.qpos[a]) for a, _ in self._finger_addrs]
        v = [float(self.sim.data.qvel[d]) for _, d in self._finger_addrs]
        if len(q) != 2:
            return 0.0, 0.0
        return abs(q[0] - q[1]), max(abs(v[0]), abs(v[1]))

    def _tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.scene_query().site_pose(EE_SITE)

    def fingers_stalled(self) -> bool:
        if self._gripper_cmd <= 0.0 or self._closed_steps < 3:
            return False
        return self._finger_opening_and_speed()[1] < 0.01

    def held_object(self) -> PerceivedObject | None:
        for obj in self.world.objects.values():
            if obj.tcp_from_object is not None:
                return obj
        return None

    def is_grasped(self) -> bool:
        """Finger width proves a grasp on anything thicker than a few mm, but a
        bowl's 1.4 mm rim closes to the same width as an empty gripper. Those
        stay provisional until the arm has lifted, then the cameras decide:
        did the object come up with the hand, or is it still on the table?"""
        if self._gripper_cmd <= 0.0:
            return False
        if self._articulation_coupled():
            return True
        held = self.held_object()
        if held is None:
            return False
        if self._grasp_width_confirmed:
            return True
        # Look once the hand is clearly up (the pick's test lift is 5 cm; at
        # 2 cm "lifted" and "left on the table" are too close to tell apart),
        # then again every 60 steps of carrying, since width can't show a slip.
        due = self._grasp_visual_ok is None or self.step_count - self._grasp_visual_step >= 60
        if due and self._tcp_pose()[0][2] - self._grasp_tcp_z > 0.04:
            self._grasp_visual_ok = self._visually_held(held)
            self._grasp_visual_step = self.step_count
            if not self._grasp_visual_ok:
                held.tcp_from_object = None
                self._pending_refresh_after = self.step_count + 5
                return False
        return True

    def _visually_held(self, obj: PerceivedObject) -> bool:
        frames = self._frames()
        scene = self.scene_query()
        predicted = obj.centroid()
        dists = [float(np.linalg.norm(c.centroid - predicted)) for cs in self._candidates(frames, [obj.category], scene).values() for c in cs]
        # Held = seen where the hand would be carrying it. Points next to the
        # fingers are filtered as robot, so allow a few cm of offset. Not
        # comparing against where it rested: the approach often nudges it.
        # Not seen at all (occluded by the hand): trust the contact.
        held = not dists or min(dists) < 0.06
        self._log("grasp_visual_check", object=obj.name, held=held, dist_to_hand=round(min(dists), 4) if dists else None, detections=len(dists))
        return held

    def on_step(self, action: np.ndarray) -> None:
        self.step_count += 1
        cmd = float(np.asarray(action).reshape(-1)[-1]) if np.asarray(action).size else self._gripper_cmd
        closing = cmd > 0.0
        if closing and self._gripper_cmd <= 0.0:
            self._closed_steps, self._grasp_evaluated = 0, False
        if not closing and self._gripper_cmd > 0.0:
            self._release()
        self._gripper_cmd = cmd
        if closing:
            self._closed_steps += 1
            coupled = self.held_object() is not None or self._articulation_coupled()
            if self._closed_steps >= 8 and not coupled and not self._grasp_evaluated and self.fingers_stalled():
                self._attach_nearest()
                self._grasp_evaluated = True
        else:
            self._open_steps += 1

        tcp_pos, tcp_quat = self._tcp_pose()
        held = self.held_object()
        if held is not None:
            if self._grasp_width_confirmed and self._finger_opening_and_speed()[0] < self.cfg.grasp_slip_width_m:
                # Fingers that were held apart have closed: the object slipped out.
                held.tcp_from_object = None
                self._log("grasp_slipped", object=held.name)
                self._pending_refresh_after = self.step_count + 10
            else:
                held.pos, held.quat_wxyz = _matrix_pose(_pose_matrix(tcp_pos, tcp_quat) @ held.tcp_from_object)
        self._update_drawers(tcp_pos)
        self._update_knobs(tcp_quat)

        if self._pending_refresh_after is not None and self.step_count >= self._pending_refresh_after:
            self._pending_refresh_after = None
            self.refresh(reason="post_release")
        elif self.cfg.refresh_every_steps and self.step_count % int(self.cfg.refresh_every_steps) == 0:
            self.refresh(reason="periodic")

    def _attach_nearest(self) -> None:
        tcp_pos, tcp_quat = self._tcp_pose()
        best_drawer, best_drawer_dist = None, np.inf
        for d in self.world.drawers:
            dist = float(np.linalg.norm(tcp_pos - (d.handle_center + d.pull_axis * d.qpos)))
            if dist < best_drawer_dist:
                best_drawer, best_drawer_dist = d, dist
        scene = self.scene_query()
        best_obj, best_dist = None, np.inf
        for obj in self.world.objects.values():
            if obj.is_static_fixture or obj.shape is None:
                continue
            ext = scene.body_collision_extent_world(obj.name)
            if ext is None:
                continue
            centre = scene.body_pose(obj.name)[0] + ext.center_offset
            # Distance from the TCP to the object's bounding box.
            dist = float(np.linalg.norm(np.maximum(np.abs(tcp_pos - centre) - ext.half_extents, 0.0)))
            if dist < best_dist:
                best_obj, best_dist = obj, dist
        best_knob, best_knob_dist = None, np.inf
        for k in self.world.knobs:
            top = k.center + k.axis * (2.0 * k.half[2])
            dist = max(0.0, float(np.linalg.norm(tcp_pos - top)) - float(k.half[0]))
            if dist < best_knob_dist:
                best_knob, best_knob_dist = k, dist
        if best_knob is not None and best_knob_dist < min(best_dist, best_drawer_dist, 0.03):
            best_knob.grasp_tcp_rot = quat_to_rotmat_wxyz(tcp_quat)
            best_knob.grasp_qpos = best_knob.qpos
            self._log("grasp_knob", knob=best_knob.body_name, dist=round(best_knob_dist, 4))
            return
        if best_drawer is not None and best_drawer_dist < min(best_dist, 0.05):
            best_drawer.grasp_tcp_offset = float(tcp_pos @ best_drawer.pull_axis) - best_drawer.qpos
            self._log("grasp_drawer", drawer=best_drawer.body_name, dist=round(best_drawer_dist, 4))
            return
        if best_obj is not None and best_dist <= self.cfg.attach_max_dist_m:
            T_tcp = _pose_matrix(tcp_pos, tcp_quat)
            best_obj.tcp_from_object = np.linalg.inv(T_tcp) @ _pose_matrix(best_obj.pos, best_obj.quat_wxyz)
            opening = self._finger_opening_and_speed()[0]
            self._grasp_tcp_z = float(tcp_pos[2])
            self._grasp_width_confirmed = opening > self.cfg.grasp_min_width_m
            self._grasp_visual_ok = None
            self._log("grasp_object", object=best_obj.name, dist=round(best_dist, 4), opening_mm=round(opening * 1000, 2))
        else:
            self._log("grasp_unassigned", nearest=getattr(best_obj, "name", None), dist=round(best_dist, 4))

    def _release(self) -> None:
        released = False
        for obj in self.world.objects.values():
            if obj.tcp_from_object is not None:
                obj.tcp_from_object = None
                released = True
        for d in self.world.drawers:
            if d.grasp_tcp_offset is not None:
                d.grasp_tcp_offset = None
                released = True
        for k in self.world.knobs:
            if k.grasp_tcp_rot is not None:
                k.grasp_tcp_rot = None
                released = True
        self._open_steps = 0
        self._grasp_width_confirmed, self._grasp_visual_ok = False, None
        if released:
            # Let the object settle before looking again.
            self._pending_refresh_after = self.step_count + 15

    def _articulation_coupled(self) -> bool:
        return any(d.grasp_tcp_offset is not None for d in self.world.drawers) or any(
            k.grasp_tcp_rot is not None for k in self.world.knobs
        )

    def _update_knobs(self, tcp_quat: np.ndarray) -> None:
        """A grasped knob turns with the hand: its angle is the hand's
        rotation about the knob axis since the grasp."""
        from scipy.spatial.transform import Rotation

        for k in self.world.knobs:
            if k.grasp_tcp_rot is None:
                continue
            delta = quat_to_rotmat_wxyz(tcp_quat) @ k.grasp_tcp_rot.T
            angle = float(Rotation.from_matrix(delta).as_rotvec() @ k.axis)
            k.qpos = float(np.clip(k.grasp_qpos + angle, *k.range_rad))

    def _update_drawers(self, tcp_pos: np.ndarray) -> None:
        for d in self.world.drawers:
            along = float(tcp_pos @ d.pull_axis)
            if d.grasp_tcp_offset is not None:
                d.qpos = float(np.clip(along - d.grasp_tcp_offset, 0.0, d.travel_m))
                continue
            # A hand pressed into the front panel has pushed the drawer in.
            front_closed = float(d.panel_center @ d.pull_axis) + d.panel_half[0]
            lateral = np.cross(np.array([0.0, 0.0, 1.0]), d.pull_axis)
            rel = tcp_pos - d.panel_center
            if abs(float(rel @ lateral)) < d.panel_half[1] and abs(float(rel[2])) < d.panel_half[2]:
                pushed_to = along - front_closed
                if -0.03 < pushed_to < d.qpos:
                    d.qpos = float(max(0.0, pushed_to))

    # ----------------------------------------------------------------- sensing

    def _frames(self) -> list[RGBDFrame]:
        frames = []
        for cam in self.cfg.camera_names:
            try:
                frames.append(capture_rgbd(self.sim, cam, self.cfg.render_height, self.cfg.render_width))
            except Exception:
                continue
        return frames

    def _robot_keep_mask(self, points: np.ndarray, scene: SceneQuery) -> np.ndarray:
        keep = np.ones(len(points), dtype=bool)
        m = self.cfg.robot_filter_margin_m
        for body in ROBOT_BODIES + GRIPPER_BODIES:
            try:
                gids = scene.body_geom_ids(body, include_children=False)
            except KeyError:
                continue
            for g in gids:
                lo, hi = scene.geom_world_aabb(g)
                inside = np.all((points >= lo - m) & (points <= hi + m), axis=1)
                keep &= ~inside
        return keep

    def _lift(self, det: Detection, frame: RGBDFrame, pts_world: np.ndarray, scene: SceneQuery, erode_px: int) -> np.ndarray:
        import cv2

        mask = det.mask
        if erode_px > 0:
            k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), k).astype(bool)
        valid = mask & (frame.depth > 0) & (frame.depth < self.cfg.depth_max_m)
        pts = pts_world[valid]
        if len(pts) == 0:
            return pts
        pts = pts[self._robot_keep_mask(pts, scene)]
        if self.world.support_z is not None:
            pts = pts[pts[:, 2] > self.world.support_z + self.cfg.table_clearance_m]
        return pts

    def _candidates(
        self, frames: list[RGBDFrame], prompts: list[str], scene: SceneQuery, erode_px: int | None = None
    ) -> dict[str, list[_Candidate]]:
        """Per camera: every prompt's instances, NMS'd across prompts (the
        higher-scoring prompt keeps an instance two prompts both found)."""
        erode = self.cfg.mask_erode_px if erode_px is None else erode_px
        out: dict[str, list[_Candidate]] = {}
        for frame in frames:
            h, w = frame.depth.shape
            dets: list[Detection] = []
            for p in prompts:
                for d in self.segmenter.segment(frame.rgb, p, self.cfg.sam3_score_threshold):
                    area = d.mask.sum() / float(h * w)
                    if self.cfg.min_mask_area_frac <= area <= self.cfg.max_mask_area_frac:
                        dets.append(d)
            if frame.camera_name != self.cfg.camera_names[0]:
                # The wrist camera sees many objects cut off by its frame edge;
                # the main camera has a full view of those, so drop the partial ones.
                dets = [d for d in dets if not (d.mask[0].any() or d.mask[-1].any() or d.mask[:, 0].any() or d.mask[:, -1].any())]
            dets.sort(key=lambda d: -d.score)
            kept: list[Detection] = []
            for d in dets:
                if all(_mask_iou(d.mask, k.mask) < self.cfg.mask_nms_iou for k in kept):
                    kept.append(d)
            pts_world = frame.points_world()
            cands = []
            for d in kept:
                pts = clean_points(self._lift(d, frame, pts_world, scene, erode), self.cfg)
                if len(pts) >= self.cfg.cluster_min_points:
                    cands.append(_Candidate(d, frame, pts))
            out[frame.camera_name] = cands
        return out

    def _fit_support(self, frames: list[RGBDFrame], scene: SceneQuery) -> None:
        chunks = []
        for frame in frames:
            pts = frame.points_world()[(frame.depth > 0) & (frame.depth < self.cfg.depth_max_m)]
            chunks.append(pts[self._robot_keep_mask(pts, scene)])
        pts = np.concatenate(chunks)
        z = fit_support_plane(pts, self.cfg)
        if z is None:
            return
        self.world.support_z = z
        on_plane = pts[np.abs(pts[:, 2] - z) < 0.01]
        if len(on_plane):
            self.world.support_xy_bounds = (np.percentile(on_plane[:, :2], 1, axis=0), np.percentile(on_plane[:, :2], 99, axis=0))

    # ---------------------------------------------------------------- keyframes

    def discover(self) -> WorldEstimate:
        t0 = time.time()
        frames = self._frames()
        scene = self.scene_query()
        self._fit_support(frames, scene)

        prior_objects = self.prior.get("objects") or []
        if prior_objects:
            return self._discover_from_prior(frames, scene, prior_objects, t0)

        agent = next((f for f in frames if f.camera_name == self.cfg.camera_names[0]), frames[0])
        if labeling.vlm_available():
            # Parts come from fixture discovery (handles, knobs), not from
            # segmenting them as objects of their own.
            self.categories = [c for c in labeling.propose_categories(agent.rgb, self.instruction, self.vlm_model) if not any(p in c for p in _PART_WORDS)]
        else:
            self.categories = ["bowl", "plate", "box", "cup", "bottle", "can", "basket", "cabinet", "stove", "book", "mug"]
        cands = self._candidates(frames, self.categories, scene)
        marks = self._group_marks(cands, agent.camera_name)

        if labeling.vlm_available() and marks:
            names, missing, grounded = labeling.name_marks(
                self._mark_images(frames, marks, "discovery"),
                [g[0].det.prompt for g in marks],
                self.instruction,
                self.vlm_model,
                self._support_facts(marks),
                self._mark_crops(marks),
            )
            # SAM3 handles plain head nouns ("box") far better than phrases
            # ("cookie box"), so try both.
            expanded: list[str] = []
            for m in missing:
                # Robot and scenery aren't objects; parts aren't either (above).
                if any(w in m for w in ("robot", "gripper", "table", "floor", "wall") + _PART_WORDS):
                    continue
                expanded += [m, m.split()[-1]]
            missing = [m for m in dict.fromkeys(expanded) if m not in self.categories]
            if missing:
                # One more round for whatever the VLM says the segmenter missed.
                self.categories += missing
                cands = self._candidates(frames, self.categories, scene)
                marks = self._group_marks(cands, agent.camera_name)
                names, _, grounded = labeling.name_marks(
                    self._mark_images(frames, marks, "discovery_round2"),
                    [g[0].det.prompt for g in marks],
                    self.instruction,
                    self.vlm_model,
                    self._support_facts(marks),
                    self._mark_crops(marks),
                )
            self._log(
                "naming", marks=[{"mark": k + 1, **n} for k, n in enumerate(names)], facts=self._support_facts(marks), grounded=grounded, missing=missing
            )
        else:
            names = [
                {"keep": True, "name": f"{g[0].det.prompt}_{i + 1}", "role": "manipulated_object", "description": ""}
                for i, g in enumerate(marks)
            ]

        facts = self._support_facts(marks)
        used: set[str] = set()
        mark_names: dict[int, str] = {}
        for k, (group, label) in enumerate(zip(marks, names), start=1):
            if not label.get("keep", True):
                continue
            # A drawer or handle seen as an object of its own: those come from
            # fixture discovery.
            if (str(label.get("what_it_is", "")).lower().split() or [""])[-1] in _PART_WORDS:
                continue
            name = labeling.snake(label.get("name") or group[0].det.prompt)
            base, k = name, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name)
            role = label.get("role", "manipulated_object")
            obj = PerceivedObject(
                name=name,
                category=group[0].det.prompt,
                role="support_object" if role == "fixture" else role,
                description=label.get("description", ""),
                score=max(c.det.score for c in group),
                is_static_fixture=role == "fixture",
            )
            self._set_geometry(obj, [c.points for c in group])
            if obj.shape is not None:
                self.world.objects[name] = obj
                mark_names[k] = name
        # Depth-derived stacking, for System 2: "rests on mark 4" -> "rests on ramekin".
        for k, name in mark_names.items():
            fact = facts[k - 1]
            if fact.startswith("rests on mark "):
                under = mark_names.get(int(fact.split()[3]))
                if under:
                    self.world.objects[name].relation = f"rests on {under}"

        self._discover_drawers(frames, scene)
        self._discover_knobs(frames, scene)
        self._rebuild_twin()
        self._log("discover", seconds=round(time.time() - t0, 2), categories=self.categories, objects=sorted(self.world.objects))
        self._save_world()
        return self.world

    def _group_marks(self, cands: dict[str, list[_Candidate]], primary: str) -> list[list[_Candidate]]:
        """One group per physical object, holding its instance in each camera
        that saw it: main-camera instances first, then wrist-camera ones the
        main camera missed (occluded, or out of its view)."""
        groups = [[c] for c in cands.get(primary, [])]
        for cam, cs in cands.items():
            if cam == primary:
                continue
            for c in cs:
                options = [g for g in groups if all(m.frame.camera_name != cam for m in g) and self._same_object(c, g[0])]
                if options:
                    min(options, key=lambda g: np.linalg.norm(g[0].centroid - c.centroid)).append(c)
                else:
                    groups.append([c])
        return groups

    @staticmethod
    def _same_object(a: _Candidate, b: _Candidate) -> bool:
        """Two views of one object see different faces of it, so a large
        object's centroids can sit far apart: same category with largely
        overlapping extents also counts. Containment alone would merge a can
        with the basket it sits in."""
        if np.linalg.norm(a.centroid - b.centroid) < 0.04:
            return True
        if a.det.prompt != b.det.prompt:
            return False
        lo_a, hi_a = a.points.min(axis=0) - 0.005, a.points.max(axis=0) + 0.005
        lo_b, hi_b = b.points.min(axis=0) - 0.005, b.points.max(axis=0) + 0.005
        inter = np.prod(np.clip(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0, None))
        smaller = min(np.prod(hi_a - lo_a), np.prod(hi_b - lo_b))
        return bool(inter > 0.3 * smaller)

    def _support_facts(self, marks: list[list[_Candidate]]) -> list[str]:
        """What each mark rests on, from depth: stacking is often ambiguous
        in a single image ("the bowl on the ramekin" vs one behind it)."""
        if self.world.support_z is None:
            return [""] * len(marks)
        z0 = self.world.support_z
        stats = []
        for g in marks:
            pts = np.concatenate([c.points for c in g])
            stats.append((float(np.percentile(pts[:, 2], 2)), float(np.percentile(pts[:, 2], 98)), pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)))
        facts = []
        for i, (bottom, _, lo, hi) in enumerate(stats):
            # The camera looks down at an angle, so even objects on the table
            # usually show their lowest point 1-2 cm up.
            if bottom - z0 < 0.025:
                facts.append("rests on the table")
                continue
            # Under = footprints overlap and the other's top meets this bottom.
            # The supporter is often mostly hidden beneath, so its visible
            # footprint can be small; any overlap counts.
            under = [
                j + 1
                for j, (_, top_j, lo_j, hi_j) in enumerate(stats)
                if j != i and -0.025 < bottom - top_j < 0.015 and np.all(np.minimum(hi, hi_j) > np.maximum(lo, lo_j))
            ]
            height = f"{(bottom - z0) * 100:.0f} cm above the table"
            facts.append(
                f"rests on mark {under[0]} ({height})" if under else f"lowest visible point {height}: resting on something unmarked, or its base is hidden"
            )
        return facts

    def _mark_crops(self, marks: list[list[_Candidate]], max_crops: int = 14) -> list[tuple[int, np.ndarray]]:
        """Close-ups of each small mark from a 2x render of the camera that
        sees it best (1024 px, still below a real RealSense's 1280x720)."""
        import cv2
        from PIL import Image

        hi_res: dict[str, np.ndarray] = {}
        crops = []
        for k, group in enumerate(marks, start=1):
            best = max(group, key=lambda c: c.det.mask.sum())
            h, w = best.det.mask.shape
            rows, cols = np.nonzero(best.det.mask)
            if len(rows) == 0 or (np.ptp(rows) + 1) * (np.ptp(cols) + 1) > 0.15 * h * w:
                continue  # big things (cabinets, baskets) are legible already
            cam = best.frame.camera_name
            if cam not in hi_res:
                hi_res[cam] = capture_rgbd(self.sim, cam, 2 * h, 2 * w).rgb
            img = hi_res[cam]
            margin = max(8, int(0.15 * max(np.ptp(rows), np.ptp(cols))))
            r0, r1 = max(0, 2 * (rows.min() - margin)), min(2 * h, 2 * (rows.max() + margin))
            c0, c1 = max(0, 2 * (cols.min() - margin)), min(2 * w, 2 * (cols.max() + margin))
            crop = Image.fromarray(img[r0:r1, c0:c1])
            crop.thumbnail((256, 256))
            crop = np.array(crop)  # writable copy for the label
            cv2.rectangle(crop, (0, 0), (24, 18), (0, 0, 0), -1)
            cv2.putText(crop, str(k), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            crops.append((k, crop))
            if len(crops) >= max_crops:
                break
        if crops:
            self._save_image("discovery_crops.png", np.concatenate([np.pad(c, ((0, 256 - c.shape[0]), (0, 256 - c.shape[1]), (0, 0))) for _, c in crops], axis=1))
        return crops

    def _mark_images(self, frames: list[RGBDFrame], marks: list[list[_Candidate]], tag: str) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs = []
        for frame in frames:
            members = [(k, c) for k, g in enumerate(marks, start=1) for c in g if c.frame is frame]
            if not members:
                continue
            marked = labeling.draw_marks(frame.rgb, [c.det.mask for _, c in members], [k for k, _ in members])
            self._save_image(f"{tag}_marks_{frame.camera_name}.png", marked)
            pairs.append((frame.rgb, marked))
        return pairs

    def _discover_from_prior(self, frames, scene, prior_objects, t0) -> WorldEstimate:
        """Re-identify the objects a previous process already named (same
        initial state, a fresh env), by category and 3D position."""
        self.categories = sorted({p["category"] for p in prior_objects})
        cands = self._candidates(frames, self.categories, scene)
        agent_name = self.cfg.camera_names[0]
        for p in prior_objects:
            obj = PerceivedObject(
                name=p["name"],
                category=p["category"],
                role=p.get("role", "manipulated_object"),
                description=p.get("description", ""),
                is_static_fixture=bool(p.get("is_static_fixture")),
                relation=p.get("relation", ""),
            )
            obj.pos = np.asarray(p["pos"], dtype=np.float64)
            obj._assoc_point = np.asarray(p.get("centroid", p["pos"]), dtype=np.float64)
            obj._assoc_radius = float(p.get("radius", 0.0))
            self.world.objects[obj.name] = obj
        self._associate_and_update(cands, agent_name, require_shape=True)
        # Anything not found again keeps no geometry and is dropped.
        self.world.objects = {n: o for n, o in self.world.objects.items() if o.shape is not None}
        self._discover_drawers(frames, scene)
        self._discover_knobs(frames, scene)
        self._rebuild_twin()
        self._log("discover_from_prior", seconds=round(time.time() - t0, 2), objects=sorted(self.world.objects))
        self._save_world()
        return self.world

    def refresh(self, reason: str = "") -> None:
        if not self.world.objects:
            return
        t0 = time.time()
        frames = self._frames()
        scene = self.scene_query()
        cands = self._candidates(frames, self.categories or sorted({o.category for o in self.world.objects.values()}), scene)
        self._associate_and_update(cands, self.cfg.camera_names[0], require_shape=False)
        self._refresh_drawers(frames, scene)
        self._rebuild_twin()
        self._log("refresh", reason=reason, seconds=round(time.time() - t0, 2))

    def _associate_and_update(self, cands: dict[str, list[_Candidate]], primary: str, require_shape: bool) -> None:
        """Hungarian assignment of detections to tracks per camera and
        category, gated on 3D distance; matched tracks are re-fitted from
        every view that saw them."""
        from scipy.optimize import linear_sum_assignment

        tracks = [o for o in self.world.objects.values() if o.tcp_from_object is None]
        evidence: dict[str, list[np.ndarray]] = {o.name: [] for o in tracks}
        for cam, cs in cands.items():
            for category in {o.category for o in tracks}:
                ts = [o for o in tracks if o.category == category]
                ds = [c for c in cs if c.det.prompt == category]
                if not ts or not ds:
                    continue
                points = [t.centroid() if t.shape is not None else t._assoc_point for t in ts]
                cost = np.array([[np.linalg.norm(p - d.centroid) for d in ds] for p in points])
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    # A cabinet's centroid moves more between views than a cup's.
                    radius = ts[r].footprint_radius() if ts[r].shape is not None else getattr(ts[r], "_assoc_radius", 0.0)
                    gate = max(self.cfg.track_gate_m, 0.5 * radius) * (2.0 if require_shape else 1.0)
                    if cost[r, c] < gate:
                        evidence[ts[r].name].append(ds[c].points)
                        ts[r].score = max(ts[r].score, ds[c].det.score)
        for obj in tracks:
            if evidence[obj.name]:
                self._set_geometry(obj, evidence[obj.name])
                obj.last_seen_step = self.step_count

    def _set_geometry(self, obj: PerceivedObject, point_sets: list[np.ndarray]) -> None:
        pts = clean_points(np.concatenate(point_sets), self.cfg) if len(point_sets) > 1 else point_sets[0]
        support = self.world.support_z if self.world.support_z is not None else float(pts[:, 2].min())
        shape = heightmap_shape(pts, support, self.cfg)
        if shape is None:
            return
        obj.shape = shape
        obj.pos = shape.origin_world.copy()
        obj.quat_wxyz = yaw_quat_wxyz(shape.yaw)
        obj._points = pts  # kept for drawer fitting

    # ------------------------------------------------------------------ drawers

    def _discover_drawers(self, frames: list[RGBDFrame], scene: SceneQuery) -> None:
        self.world.drawers = []
        if not self.cfg.part_prompts:
            return
        handles = self._handle_candidates(frames, scene)
        if not handles:
            return
        cabinets = [o for o in self.world.objects.values() if o.is_static_fixture and getattr(o, "_points", None) is not None]
        orphans = [h for h in handles if not cabinets or min(self._xy_distance_to_shape(h.mean(axis=0), c) for c in cabinets) >= 0.06]
        if orphans:
            # Handles with no cabinet around them: the VLM split the cabinet
            # into panels or dropped it. Segment it directly instead.
            cabinets += self._cabinets_around(frames, scene, orphans)
        if not cabinets:
            return
        by_cabinet: dict[str, list[np.ndarray]] = {}
        for h in handles:
            owner = min(cabinets, key=lambda c: self._xy_distance_to_shape(h.mean(axis=0), c))
            if self._xy_distance_to_shape(h.mean(axis=0), owner) < 0.06:
                by_cabinet.setdefault(owner.name, []).append(h)
        for cab_name, hs in by_cabinet.items():
            self.world.drawers.extend(self._fit_cabinet_drawers(self.world.objects[cab_name], hs))
        # The carcass must not include open drawers or the handles: refit it
        # from points behind the closed front plane.
        for cab_name in by_cabinet:
            cab = self.world.objects[cab_name]
            ds = [d for d in self.world.drawers if d.cabinet == cab_name]
            front = float(ds[0].panel_center @ ds[0].pull_axis) + ds[0].panel_half[0]
            pts = cab._points[cab._points @ ds[0].pull_axis <= front - ds[0].panel_half[0] * 2]
            shape = heightmap_shape(pts, self.world.support_z, self.cfg)
            if shape is not None:
                cab.shape, cab.pos, cab.quat_wxyz = shape, shape.origin_world.copy(), yaw_quat_wxyz(shape.yaw)

    def _cabinets_around(self, frames, scene, handles: list[np.ndarray]) -> list[PerceivedObject]:
        """A cabinet for each segmented "cabinet" instance that has a handle
        on it. Matched per handle: stray handles on background furniture
        must not drag the match away from the real one."""
        groups = self._group_marks(self._candidates(frames, ["cabinet"], scene), self.cfg.camera_names[0])
        clouds = [np.concatenate([c.points for c in g]) for g in groups]
        hits: set[int] = set()
        for h in handles:
            c = h.mean(axis=0)[:2]
            inside = [i for i, pts in enumerate(clouds) if np.all(c >= pts[:, :2].min(axis=0) - 0.06) and np.all(c <= pts[:, :2].max(axis=0) + 0.06)]
            if inside:
                hits.add(max(inside, key=lambda i: len(clouds[i])))
        out = []
        for i in sorted(hits):
            name, k = "cabinet", 2
            while name in self.world.objects:
                name, k = f"cabinet_{k}", k + 1
            cab = PerceivedObject(name=name, category="cabinet", role="support_object", description="cabinet with drawers", is_static_fixture=True)
            self._set_geometry(cab, [clouds[i]])
            if cab.shape is not None:
                self.world.objects[name] = cab
                out.append(cab)
                self._log("cabinet_from_handles", name=name)
        return out

    def _handle_candidates(self, frames, scene) -> list[np.ndarray]:
        return self._part_candidates(frames, scene, self.cfg.part_prompts)

    def _part_candidates(self, frames, scene, prompts: list[str]) -> list[np.ndarray]:
        """Part point clouds merged across views (thin parts: 1 px erosion)."""
        cands = self._candidates(frames, prompts, scene, erode_px=1)
        merged: list[np.ndarray] = []
        for cs in cands.values():
            for c in cs:
                for i, m in enumerate(merged):
                    if np.linalg.norm(m.mean(axis=0) - c.centroid) < 0.03:
                        merged[i] = np.concatenate([m, c.points])
                        break
                else:
                    merged.append(c.points)
        return merged

    def _discover_knobs(self, frames, scene) -> None:
        """One knob per static fixture (the twin names it <fixture>_knob).
        Its axis is the outward normal of the surface it is mounted on."""
        self.world.knobs = []
        fixtures = [o for o in self.world.objects.values() if o.is_static_fixture and getattr(o, "_points", None) is not None]
        if not fixtures or not self.cfg.knob_prompts:
            return
        handles = [d.handle_center + d.pull_axis * d.qpos for d in self.world.drawers]
        best: dict[str, tuple[int, PerceivedKnob]] = {}
        for pts in self._part_candidates(frames, scene, self.cfg.knob_prompts):
            c = pts.mean(axis=0)
            if any(np.linalg.norm(c - h) < 0.04 for h in handles):
                continue
            radius = float(np.max(np.linalg.norm(pts[:, :2] - c[:2], axis=1)))
            # The VLM sometimes keeps the knob itself as a fixture ("black clamp
            # stand"); it can't own itself, and its solid boxes would fill the
            # space the fingers turn in.
            owners = [f for f in fixtures if not self._is_knob_itself(f, c, radius)]
            if not owners:
                continue
            owner = min(owners, key=lambda f: self._xy_distance_to_shape(c, f))
            # Knobs sit at a fixture's edge or just behind it (LIBERO's stove
            # knob is ~6 cm off the burner's base), not necessarily on top.
            if self._xy_distance_to_shape(c, owner) > 0.08:
                continue
            fp = owner._points
            dist = np.linalg.norm(fp - c, axis=1)
            ring = fp[(dist > radius + 0.01) & (dist < radius + 0.06)]
            fit = fit_plane_normal(ring, self.cfg.drawer_plane_threshold_m)
            if fit is None:
                n, d = np.array([0.0, 0.0, 1.0]), -float(np.median(ring[:, 2])) if len(ring) else -float(pts[:, 2].min())
            else:
                n, d = fit
            if (c - (ring.mean(axis=0) if len(ring) else c - n)) @ n < 0:
                n, d = -n, -d
            # A mounting surface within 10 deg of level is level: the plane fit
            # on a small ring of points is otherwise tilted by noise.
            if n[2] > np.cos(np.radians(10.0)):
                n, d = np.array([0.0, 0.0, 1.0]), -float(np.median(ring[:, 2])) if len(ring) else -float(pts[:, 2].min())
            height = max(float(np.max(pts @ n) + d), 0.005)
            perp = (pts - c) - np.outer((pts - c) @ n, n)
            radius = max(float(np.max(np.linalg.norm(perp, axis=1))), 0.01)
            knob = PerceivedKnob(
                fixture=owner.name,
                center=c - n * (float(c @ n) + d),
                axis=n,
                half=np.array([radius, radius, 0.5 * height]),
                range_rad=tuple(self.cfg.knob_range_rad),
            )
            self._fit_knob_handle(knob, pts, height)
            if owner.name not in best or len(pts) > best[owner.name][0]:
                best[owner.name] = (len(pts), knob)
        # Largest first; a smaller detection near a kept knob is the same knob
        # seen partially.
        knobs: list[PerceivedKnob] = []
        for _, k in sorted(best.values(), key=lambda nk: -nk[0]):
            if all(np.linalg.norm(k.center[:2] - kept.center[:2]) > 0.05 for kept in knobs):
                knobs.append(k)
        # The VLM may already have named the knob as an object of its own.
        for k in knobs:
            for name, obj in list(self.world.objects.items()):
                if name != k.fixture and self._is_knob_itself(obj, k.center, k.half[0]):
                    del self.world.objects[name]
                    self._log("object_is_knob", object=name, knob=k.body_name)
        self.world.knobs = [k for k in knobs if k.fixture in self.world.objects]

    @staticmethod
    def _fit_knob_handle(knob: PerceivedKnob, pts: np.ndarray, height: float) -> None:
        """The handle is the knob's upper half: its midpoint and long
        direction come from those points' extent in the mounting plane."""
        h = (pts - knob.center) @ knob.axis
        top = pts[h > 0.5 * height]
        if len(top) < 10:
            return
        rel = (top - knob.center) - np.outer((top - knob.center) @ knob.axis, knob.axis)
        evals, evecs = np.linalg.eigh(np.cov((rel - rel.mean(axis=0)).T))
        if evals[-1] < 1.3**2 * evals[-2]:
            return  # round: no handle direction to grasp across
        long_dir = evecs[:, -1] - knob.axis * (evecs[:, -1] @ knob.axis)
        long_dir /= np.linalg.norm(long_dir)
        thin_dir = np.cross(knob.axis, long_dir)
        along = np.percentile(rel @ long_dir, [3, 97])
        across = np.percentile(rel @ thin_dir, [3, 97])
        knob.grip_offset = 0.5 * along.sum() * long_dir + 0.5 * across.sum() * thin_dir
        knob.grip_dir = long_dir
        knob.grip_half = np.array([max(0.5 * np.ptp(along), 0.005), max(0.5 * np.ptp(across), 0.004), 0.5 * height])

    @staticmethod
    def _is_knob_itself(obj: PerceivedObject, knob_center: np.ndarray, knob_radius: float) -> bool:
        """Knob-sized and centred on the knob."""
        return bool(
            np.linalg.norm(obj.centroid()[:2] - knob_center[:2]) < knob_radius + 0.02
            and obj.footprint_radius() < knob_radius + 0.03
        )

    @staticmethod
    def _xy_distance_to_shape(p: np.ndarray, obj: PerceivedObject) -> float:
        pts = obj._points[:, :2]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return float(np.linalg.norm(np.maximum(np.maximum(lo - p[:2], p[:2] - hi), 0.0)))

    def _fit_cabinet_drawers(self, cab: PerceivedObject, handle_sets: list[np.ndarray]) -> list[PerceivedDrawer]:
        from scipy.spatial import cKDTree

        carcass = cab._points
        handle_all = np.concatenate(handle_sets)
        near_handle = cKDTree(handle_all).query(carcass, distance_upper_bound=0.015)[0] < np.inf
        body_pts = carcass[~near_handle]
        ez = np.array([0.0, 0.0, 1.0])

        axis = self._front_face_axis(body_pts, [h.mean(axis=0) for h in handle_sets])
        lateral = np.cross(ez, axis)
        fronts = []
        for h in handle_sets:
            c = h.mean(axis=0)
            local = body_pts[np.linalg.norm(body_pts - c, axis=1) < 0.08]
            # A drawer's own front is the most forward surface around its handle
            # (an open drawer's sides reach back from it).
            fronts.append(float(np.percentile(local @ axis, 90)) if len(local) else float(c @ axis) - 0.02)

        # Closed front: the top surface's front edge, which never moves.
        top_z = cab.shape.top_z
        top_pts = body_pts[body_pts[:, 2] > top_z - 0.01]
        f_closed = float(np.percentile(top_pts @ axis, 97)) if len(top_pts) > 20 else min(fronts)
        # An open top drawer's own upper edge reaches into top_pts and drags
        # the edge out with it; closed drawers sit flush, so the least proud
        # front is the closed position.
        f_closed = min(f_closed, min(fronts))
        proj = body_pts @ axis
        # Depth of the carcass itself: an open drawer sticking out would
        # inflate it, and with it the travel every opening is a fraction of.
        depth = float(f_closed - proj[proj <= f_closed + 0.005].min())
        travel = self.cfg.drawer_travel_frac_of_depth * depth

        front_band = body_pts[proj > f_closed - 0.03]
        lat = front_band @ lateral if len(front_band) else body_pts @ lateral
        lat_lo, lat_hi = float(np.percentile(lat, 2)), float(np.percentile(lat, 98))
        bottom_z = cab.shape.bottom_z

        order = sorted(range(len(handle_sets)), key=lambda i: -handle_sets[i].mean(axis=0)[2])
        levels = _LEVELS_BY_COUNT.get(len(order), [f"level{i}" for i in range(len(order))])
        if len(order) > 1:
            # By height within the cabinet rather than by count, so a handle
            # SAM3 missed (hidden behind an open drawer) doesn't shift the rest.
            span = max(top_z - cab.shape.bottom_z, 1e-6)
            by_height = []
            for i in order:
                frac = (handle_sets[i].mean(axis=0)[2] - cab.shape.bottom_z) / span
                by_height.append("top" if frac > 2 / 3 else ("bottom" if frac < 1 / 3 else "middle"))
            if len(set(by_height)) == len(by_height):
                levels = by_height
        # The carcass's back is often out of view, so its depth undershoots;
        # a drawer seen open proves the travel is at least that far.
        openings = [fronts[i] - f_closed for i in order]
        travel = max(travel, 1.05 * max(openings))
        zs = [handle_sets[i].mean(axis=0)[2] for i in order]
        bounds = [top_z - 0.01] + [0.5 * (zs[k] + zs[k + 1]) for k in range(len(zs) - 1)] + [bottom_z + 0.01]

        drawers = []
        for k, i in enumerate(order):
            h = handle_sets[i]
            c = h.mean(axis=0)
            # Opening = how far this drawer's front sits proud of the top's edge.
            q = fronts[i] - f_closed
            q = q if q > 0.01 else 0.0
            z_hi, z_lo = bounds[k], bounds[k + 1]
            panel_center = axis * (f_closed - 0.01) + lateral * (0.5 * (lat_lo + lat_hi)) + ez * (0.5 * (z_hi + z_lo))
            hp = np.stack([h @ axis, h @ lateral, h[:, 2]], axis=1)
            handle_half = np.maximum(0.5 * (hp.max(axis=0) - hp.min(axis=0)), 0.005)
            drawers.append(
                PerceivedDrawer(
                    cabinet=cab.name,
                    level=levels[k],
                    handle_center=c - axis * q,
                    handle_half=handle_half,
                    panel_center=panel_center,
                    panel_half=np.array([0.01, max(0.5 * (lat_hi - lat_lo) - 0.005, 0.02), max(0.5 * (z_hi - z_lo) - 0.004, 0.01)]),
                    pull_axis=axis,
                    travel_m=float(max(travel, 0.05)),
                    qpos=min(q, travel),
                )
            )
        return drawers

    def _front_face_axis(self, pts: np.ndarray, handle_centers: list[np.ndarray]) -> np.ndarray:
        """Horizontal outward normal of the face the handles stick out of.

        Picked among the carcass's vertical planes as the one with the most
        handles just in front of it. A plane fit around each handle on its
        own fails when that drawer is open: its sides dominate the fit.
        """
        import open3d as o3d

        ez = np.array([0.0, 0.0, 1.0])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(0.005)
        best: tuple[int, int, np.ndarray] | None = None
        for _ in range(6):
            if len(pcd.points) < 100:
                break
            (a, b, c, d), inliers = pcd.segment_plane(self.cfg.drawer_plane_threshold_m, 3, 500)
            n = np.array([a, b, c])
            d /= np.linalg.norm(n)
            n /= np.linalg.norm(n)
            if abs(n[2]) < 0.15 and len(inliers) >= 100:
                dists = np.array([h @ n + d for h in handle_centers])
                if np.median(dists) < 0:
                    n, dists = -n, -dists
                # Closed drawers' handles sit 0.5-6 cm proud of the front face;
                # a side face is a cabinet half-width away from all of them.
                near = int(np.sum((dists > 0.003) & (dists < 0.06)))
                if near and (best is None or (near, len(inliers)) > best[:2]):
                    best = (near, len(inliers), n)
            pcd = pcd.select_by_index(inliers, invert=True)
        if best is None:
            n = np.mean(handle_centers, axis=0) - pts.mean(axis=0)
        else:
            n = best[2]
        n = n - ez * n[2]
        return n / np.linalg.norm(n)

    def _refresh_drawers(self, frames, scene) -> None:
        """Re-measure each drawer's opening from where its handle is now."""
        free = [d for d in self.world.drawers if d.grasp_tcp_offset is None]
        if not free:
            return
        handles = [h.mean(axis=0) for h in self._handle_candidates(frames, scene)]
        for d in free:
            if not handles:
                break
            # Handles only move along the pull axis: match on the other two.
            def off_axis(p):
                r = p - d.handle_center
                return float(np.linalg.norm(r - d.pull_axis * (r @ d.pull_axis)))

            best = min(handles, key=off_axis)
            if off_axis(best) < 0.03:
                d.qpos = float(np.clip((best - d.handle_center) @ d.pull_axis, 0.0, d.travel_m))
                if d.qpos < 0.01:
                    d.qpos = 0.0

    # ----------------------------------------------- segmentation for grasping

    def render_depth_and_body_ids(self, scene: SceneQuery, camera_name: str, height: int, width: int):
        """Real depth plus a body-id image built from perception: each
        perceived object's category re-segmented and matched to it by 3D
        position. What Contact-GraspNet sees in place of the simulator's
        ground-truth segmentation."""
        frame = capture_rgbd(self.sim, camera_name, height, width)
        body_id = np.full(frame.depth.shape, -1, dtype=np.int64)
        pts_world = frame.points_world()
        objs = [o for o in self.world.objects.values() if o.shape is not None]
        for category in sorted({o.category for o in objs}):
            for det in self.segmenter.segment(frame.rgb, category, self.cfg.sam3_score_threshold):
                pts = self._lift(det, frame, pts_world, scene, self.cfg.mask_erode_px)
                if len(pts) < self.cfg.cluster_min_points:
                    continue
                c = np.median(pts, axis=0)
                cat_objs = [o for o in objs if o.category == category]
                best = min(cat_objs, key=lambda o: np.linalg.norm(o.centroid() - c))
                if np.linalg.norm(best.centroid() - c) < self.cfg.track_gate_m:
                    body_id[det.mask & (body_id < 0)] = scene.body_id(best.name)
        return frame.depth, body_id

    # ------------------------------------------------------------- reporting

    def object_positions(self) -> dict[str, np.ndarray]:
        return {n: o.pos.astype(np.float32).copy() for n, o in self.world.objects.items() if not o.is_static_fixture}

    def _log(self, event: str, **fields) -> None:
        entry = {"event": event, "step": self.step_count, **fields}
        self.keyframe_log.append(entry)
        print(f"[perception] {json.dumps(entry)}")
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            with open(self.debug_dir / "perception_log.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _save_image(self, name: str, rgb: np.ndarray) -> None:
        if self.debug_dir is None:
            return
        from PIL import Image

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(self.debug_dir / name)

    def _save_world(self) -> None:
        if self.debug_dir is None:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        (self.debug_dir / "perceived_world.json").write_text(json.dumps(self.world.to_json(), indent=1))
