"""VLM-driven scene discovery for the real-robot backend.

At startup the user typically has no idea what is on the table, only that the
robot can be told to "pick up the mug" or "stack the blocks". This module:

1. Captures one RGB-D frame from the primary RealSense.
2. Asks an LLM (default ``gpt-4o``) to enumerate every visible workspace
   entity, classify its role (manipulated_object / obstacle / support_object /
   goal_region), and return a 2D pixel hit for each.
3. Optionally refines each pixel hit with SAM to recover an accurate 3D
   centroid + radius.
4. Builds and returns a fully-formed ``scene_context`` dict that can be
   handed straight to System 2 and System 1.

The resulting scene_context inherits the *platform* / coordinate / placement
metadata from a "base" scene_context the user already has on disk, but its
``entities`` and ``geometry`` sections are completely replaced by what the
VLM saw.
"""

from __future__ import annotations

import base64
import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .projector import Projector
from .sam_segmenter import SAMSegmenter, estimate_geometry_from_mask


SCENE_DISCOVERY_PROMPT = """You are a perception assistant for a robot manipulation system. You see ONE image from a fixed RealSense camera looking at a tabletop workspace.

Your job is to enumerate EVERY workspace-relevant entity you can see and classify its role for a pick-and-place robot. Output ONLY a JSON object of the form:

{
  "entities": [
    {
      "name": "<lowercase_snake_case_canonical_name>",
      "label": "<short human-readable description, e.g. 'red wooden block'>",
      "object_class": "<single noun like cube, mug, box, tray, knife, region>",
      "role": "<one of: manipulated_object, obstacle, support_object, goal_region, container>",
      "x": <int 0-1000>,
      "y": <int 0-1000>,
      "aliases": ["<phrase a user might say to refer to this entity>", ...]
    },
    ...
  ]
}

Role rules:
- manipulated_object: small enough to be grasped by a parallel-jaw gripper (e.g. cube, mug, block, can, bottle).
- obstacle: any rigid item the robot must avoid (e.g. cardboard box, fixture, tool laid on its side).
- support_object: a flat object the robot might place onto (e.g. a tray, a knife laid flat as a "shelf").
- goal_region: a marked area on the table (e.g. a coloured tape square, a printed target circle, a placemat).
- container: an open vessel the robot might place items INTO (e.g. a bowl, a basket, a box with an open top).

Coordinate rules:
- The x and y coordinates are NORMALIZED to 0-1000 range, regardless of the image resolution.
- Pick a stable, central pixel for each entity (the body of a mug, the geometric centre of a box).
- Avoid thin edges, shadows, and reflective highlights.

Naming rules:
- ``name`` must be lowercase_snake_case and unique within the scene. Disambiguate similar items by colour or position (e.g. red_block vs blue_block, left_mug vs right_mug).
- The robot itself, the table, walls, the camera, and out-of-workspace items must be IGNORED.
- Aliases must be short natural-language phrases a user might say (e.g. "the pink mug", "left cube"). Include the colour and noun for visual items.

Output ONLY the JSON object. No prose, no markdown, no commentary.
"""


@dataclass
class DiscoveredEntity:
    name: str
    label: str
    object_class: str
    role: str
    pixel: tuple[int, int]
    aliases: List[str]
    xyz_robot: Optional[np.ndarray] = None
    radius: Optional[float] = None
    surface_points: Optional[np.ndarray] = None
    visible: bool = True
    notes: str = ""


@dataclass
class SceneDiscovery:
    """VLM-only scene-discovery client.

    Pair with :class:`SAMSegmenter` for high-quality 3D geometry; if SAM is
    unavailable the discovery still works but each entity gets only a
    single-pixel deprojection.
    """

    projector: Projector
    workspace_bounds: Optional[Dict[str, List[float]]] = None
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    timeout_sec: int = 90
    sam: Optional[SAMSegmenter] = None
    table_z: float = 0.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None

    # -------------------------------------------------------- main entry

    def discover(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        *,
        pcd: Any = None,
    ) -> List[DiscoveredEntity]:
        """Run the VLM enumeration and (optionally) refine with SAM."""
        if self.api_key is None:
            print("[scene_discovery] OPENAI_API_KEY missing; returning empty scene.")
            return []
        try:
            raw = self._ask_vlm_for_entities(color_image)
        except Exception as exc:
            print(f"[scene_discovery] VLM call failed: {exc}")
            return []

        h, w = color_image.shape[:2]
        results: List[DiscoveredEntity] = []
        seen_names: set[str] = set()
        for ent in raw:
            name = str(ent.get("name", "")).strip().lower().replace(" ", "_")
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            try:
                px = int(round(float(ent.get("x", 500)) * w / 1000.0))
                py = int(round(float(ent.get("y", 500)) * h / 1000.0))
            except Exception:
                continue
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            role = str(ent.get("role", "manipulated_object"))
            object_class = str(ent.get("object_class", "object"))
            label = str(ent.get("label", name))
            aliases = [
                str(a).strip()
                for a in ent.get("aliases", [])
                if isinstance(a, str) and a.strip()
            ]
            if label and label.lower() not in {a.lower() for a in aliases}:
                aliases.append(label)
            entity = DiscoveredEntity(
                name=name,
                label=label,
                object_class=object_class,
                role=role,
                pixel=(px, py),
                aliases=aliases,
            )
            est = None
            try:
                est = estimate_geometry_from_mask(
                    image=color_image,
                    depth_image=depth_image,
                    px=px,
                    py=py,
                    sam=self.sam,
                    projector=self.projector,
                    pcd=pcd,
                    label=name,
                    table_z=float(self.table_z),
                )
            except Exception as exc:
                print(f"[scene_discovery] SAM geometry failed for {name!r}: {exc}")
            if est is None:
                try:
                    xyz = self.projector.pixel_to_robot_frame(px, py, depth_image)
                except Exception as exc:
                    entity.visible = False
                    entity.notes = f"deprojection failed: {exc}"
                    results.append(entity)
                    continue
                entity.xyz_robot = np.asarray(xyz, dtype=np.float32)
                entity.radius = 0.05
                entity.notes = "single-pixel deprojection"
            else:
                entity.xyz_robot = np.asarray(est.top_xyz, dtype=np.float32)
                entity.radius = float(est.radius)
                entity.surface_points = np.asarray(est.surface_points, dtype=np.float32)
                entity.notes = "VLM+SAM" if est.num_points > 1 else "VLM"
            if (
                self.workspace_bounds
                and entity.xyz_robot is not None
                and not self.projector.is_in_workspace(entity.xyz_robot, self.workspace_bounds)
            ):
                entity.notes += "; OUTSIDE workspace bounds"
            results.append(entity)
        return results

    # ------------------------------------------- scene_context construction

    def build_scene_context(
        self,
        base_scene_context: dict,
        discovered: List[DiscoveredEntity],
    ) -> dict:
        """Merge discovered entities into a base platform scene_context."""
        ctx = copy.deepcopy(base_scene_context)
        ctx["entities"] = []
        ctx["geometry"] = {}
        for ent in discovered:
            if ent.xyz_robot is None:
                continue
            entity_block: Dict[str, Any] = {
                "name": ent.name,
                "role": ent.role,
                "object_class": ent.object_class,
                "aliases": _dedup(ent.aliases + [ent.name.replace("_", " ")]),
                "perception": {"vlm_label": ent.label},
            }
            if ent.radius is not None:
                entity_block["perception"]["expected_radius"] = float(ent.radius)
            ctx["entities"].append(entity_block)

            geom: Dict[str, Any] = {
                "xy": [float(ent.xyz_robot[0]), float(ent.xyz_robot[1])],
                "z": float(ent.xyz_robot[2]),
            }
            if ent.radius is not None:
                geom["radius"] = float(ent.radius)
                if ent.role == "goal_region":
                    geom["descend_radius"] = max(0.05, float(ent.radius) + 0.02)
            ctx["geometry"][ent.name] = geom

        # Make sure the platform block is present and the backend is "real".
        ctx.setdefault("platform", {})["backend"] = "real"
        # Sensible defaults if the base config didn't provide them.
        ctx.setdefault(
            "execution_proxy",
            {
                "default_grasp_mode": "top",
                "num_blocks": 0,
                "post_success_steps": 40,
                "max_subtask_steps": 400,
                "env_horizon": 800,
            },
        )
        ctx.setdefault(
            "placement_defaults",
            {
                "table_surface_z": float(self.table_z),
                "table_goal_z": float(self.table_z) + 0.02,
                "stack_extra_z": 0.05,
                "support_object_goal_z": float(self.table_z) + 0.04,
            },
        )
        ctx.setdefault(
            "coordinate_convention",
            {
                "frame": "robot_base",
                "left": "-x",
                "right": "+x",
                "forward": "-y",
                "backward": "+y",
                "offset_units": "meters",
                "notes": "Auto-generated; verify against your camera+robot mounting.",
            },
        )
        ctx.setdefault(
            "pass_through_defaults",
            {
                "default_radius": 0.05,
                "default_offset_from_reference": 0.10,
                "radius": 0.10,
                "offset_from_reference": 0.10,
                "interpretation": "broad corridor / half-plane proxy",
            },
        )
        ctx.setdefault("scene_name", "real_auto_discovered")
        ctx.setdefault(
            "description",
            "Scene context auto-discovered from a RealSense capture using VLM (+ optional SAM).",
        )
        return ctx

    # ----------------------------------------------------------- internals

    def _ask_vlm_for_entities(self, color_image: np.ndarray) -> List[dict]:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required for SceneDiscovery") from exc
        client = OpenAI(api_key=self.api_key)
        try:
            import cv2  # type: ignore

            ok, buffer = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("cv2 failed to encode image")
            img_bytes = buffer.tobytes()
        except Exception:
            from io import BytesIO

            from PIL import Image  # type: ignore

            buf = BytesIO()
            Image.fromarray(color_image).save(buf, format="JPEG", quality=90)
            img_bytes = buf.getvalue()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SCENE_DISCOVERY_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.1,
            timeout=self.timeout_sec,
        )
        text = resp.choices[0].message.content or ""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return []
        return list(obj.get("entities", []))


def _dedup(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        key = v.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(v.strip())
    return out
