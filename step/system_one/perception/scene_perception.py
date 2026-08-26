"""VLM-driven 3D entity localization for the real-robot backend.

Given an RGB+Depth frame and a scene_context.json describing the entities the
robot should be aware of, this module asks an LLM (OpenAI's GPT-4o family by
default) to point at each entity in the image. The 2D pixel hits are then
deprojected into the robot base frame via :class:`Projector`, producing a
``Dict[str, np.ndarray]`` mapping each canonical entity name to its 3D XYZ.

The localizer is intentionally minimal:

* It does NOT call System 2 — it only locates known scene entities so the
  STL/MPC stage machine has live positions to plan against.
* It uses prompt engineering similar to ``interactive_planner.py`` from
  ``temp_real_robot_files/``: normalized 1000x1000 pixel coordinates and a
  permissive JSON-only output format.
* It can fall back to a "manual" mode that simply reads the geometry section
  of scene_context.json (useful for dry-run testing without a VLM call).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .projector import Projector
from .sam_segmenter import (
    GeometryEstimate,
    SAMSegmenter,
    estimate_geometry_from_mask,
)


SCENE_LOCALIZE_PROMPT = """You are a perception assistant for a robot manipulation system. You see one image from a fixed overhead RealSense camera looking at the robot's workspace.

You are given a JSON list of named entities that the robot should track. Your job is to identify the pixel center of each entity in the image and return its normalized (0-1000) image coordinates.

Output ONLY a JSON object of the form:
{
  "entities": [
    {"name": "<canonical entity name>", "x": <int 0-1000>, "y": <int 0-1000>, "visible": true},
    {"name": "<canonical entity name>", "visible": false}
  ]
}

Rules:
- The x and y coordinates are NORMALIZED to 0-1000 range, regardless of the actual image resolution.
- Use the canonical entity names exactly as given. Do not invent new entity names.
- If an entity is not visible in the image, set visible=false and omit x/y.
- Choose a stable, central pixel for each entity (e.g., the body of a mug, the geometric center of a cardboard box).
- Avoid thin edges, shadows, and reflective highlights.
- Output ONLY the JSON object. No prose, no markdown.

The entities are:
{entities_json}
"""


@dataclass
class EntityLocalization:
    name: str
    role: str
    visible: bool
    pixel: Optional[tuple[int, int]] = None
    xyz_robot: Optional[np.ndarray] = None
    radius: Optional[float] = None
    surface_points: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    notes: str = ""


@dataclass
class ScenePerception:
    """Localizes scene_context entities in 3D using a VLM (+ optional SAM)."""

    projector: Projector
    workspace_bounds: dict[str, list[float]] = field(default_factory=dict)
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    timeout_sec: int = 60
    sam: Optional[SAMSegmenter] = None
    table_z: float = 0.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None

    # ------------------------------------------------------------ entry points

    def localize_from_scene_context(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        scene_context: dict,
        *,
        manual_only: bool = False,
        pcd: Any = None,
    ) -> Dict[str, EntityLocalization]:
        """Find every entity declared in scene_context and return its 3D pose.

        Strategy, in order:
          1. If ``manual_only`` is True or the entity already has a
             geometry.xy in scene_context, use that XY directly (lift to z=0
             plane unless geometry includes a z).
          2. Otherwise, call the VLM with the image and ask for a pixel hit.
          3. Deproject the pixel through the projector + depth image into the
             robot base frame.

        Entities that resolve to neither a geometry nor a VLM hit are returned
        with ``visible=False`` and no xyz.
        """
        entities: List[Dict[str, Any]] = list(scene_context.get("entities", []))
        geometry = scene_context.get("geometry", {})
        out: Dict[str, EntityLocalization] = {}

        # Pass 1: anything that has explicit geometry.xy in scene_context.
        deferred: List[Dict[str, Any]] = []
        for ent in entities:
            name = ent["name"]
            geom = geometry.get(name, {})
            if "xy" in geom and not manual_only:
                # Honor explicit geometry but still try VLM if no explicit z is set.
                deferred.append(ent)
                continue
            if "xy" in geom:
                xyz = self._geometry_to_xyz(geom, scene_context)
                out[name] = EntityLocalization(
                    name=name,
                    role=ent.get("role", "unknown"),
                    visible=True,
                    xyz_robot=xyz,
                    notes="from scene_context.geometry",
                )
            else:
                deferred.append(ent)

        if not deferred:
            return out

        if manual_only or self.api_key is None:
            for ent in deferred:
                name = ent["name"]
                out[name] = EntityLocalization(
                    name=name,
                    role=ent.get("role", "unknown"),
                    visible=False,
                    notes="manual mode and no geometry available",
                )
            return out

        # Pass 2: ask the VLM to point at every deferred entity at once.
        try:
            vlm_hits = self._ask_vlm_for_pixels(color_image, deferred)
        except Exception as exc:
            print(f"[scene_perception] VLM call failed ({exc}). Falling back to geometry only.")
            vlm_hits = {}

        # Pass 3: deproject each pixel hit into the robot frame, refining with
        # SAM when available so the position represents the real object centre
        # rather than whichever single pixel the VLM happened to pick.
        for ent in deferred:
            name = ent["name"]
            geom = geometry.get(name, {})
            hit = vlm_hits.get(name)
            if hit is None or not hit.get("visible", False):
                if "xy" in geom:
                    xyz = self._geometry_to_xyz(geom, scene_context)
                    out[name] = EntityLocalization(
                        name=name,
                        role=ent.get("role", "unknown"),
                        visible=True,
                        xyz_robot=xyz,
                        notes="VLM did not see it; used scene_context.geometry",
                    )
                else:
                    out[name] = EntityLocalization(
                        name=name,
                        role=ent.get("role", "unknown"),
                        visible=False,
                        notes="VLM did not see it and no geometry fallback",
                    )
                continue
            px, py = int(hit["x"]), int(hit["y"])
            est = self._estimate_geometry(color_image, depth_image, px, py, label=name, pcd=pcd)
            if est is None:
                if "xy" in geom:
                    xyz = self._geometry_to_xyz(geom, scene_context)
                    out[name] = EntityLocalization(
                        name=name,
                        role=ent.get("role", "unknown"),
                        visible=True,
                        pixel=(px, py),
                        xyz_robot=xyz,
                        notes="VLM hit but deprojection failed; used scene_context.geometry",
                    )
                else:
                    out[name] = EntityLocalization(
                        name=name,
                        role=ent.get("role", "unknown"),
                        visible=False,
                        pixel=(px, py),
                        notes="VLM hit but deprojection failed and no geometry fallback",
                    )
                continue
            xyz = est.top_xyz
            in_ws = (
                not self.workspace_bounds
                or self.projector.is_in_workspace(xyz, self.workspace_bounds)
            )
            notes = "VLM+SAM-localized" if est.num_points > 1 else "VLM-localized"
            if not in_ws:
                notes += "; OUTSIDE workspace bounds"
            out[name] = EntityLocalization(
                name=name,
                role=ent.get("role", "unknown"),
                visible=True,
                pixel=(px, py),
                xyz_robot=np.asarray(xyz, dtype=np.float32),
                radius=float(est.radius),
                surface_points=np.asarray(est.surface_points, dtype=np.float32),
                mask=est.mask,
                notes=notes,
            )
        return out

    def _estimate_geometry(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        px: int,
        py: int,
        *,
        label: str,
        pcd: Any = None,
    ) -> Optional[GeometryEstimate]:
        """Return SAM-refined geometry, falling back to a single-pixel hit."""
        try:
            return estimate_geometry_from_mask(
                image=color_image,
                depth_image=depth_image,
                px=px,
                py=py,
                sam=self.sam,
                projector=self.projector,
                pcd=pcd,
                label=label,
                table_z=float(self.table_z),
            )
        except Exception as exc:
            print(f"[scene_perception] SAM geometry failed for {label!r}: {exc}; "
                  f"falling back to single-pixel deprojection.")
        try:
            xyz = self.projector.pixel_to_robot_frame(int(px), int(py), depth_image)
        except Exception as exc:
            print(f"[scene_perception] single-pixel deprojection failed for {label!r}: {exc}")
            return None
        return GeometryEstimate(
            top_xyz=np.asarray(xyz, dtype=np.float32),
            center_xy=np.asarray(xyz[:2], dtype=np.float32),
            radius=0.05,
            mask=np.zeros(color_image.shape[:2], dtype=bool),
            surface_points=np.asarray([xyz[:2]], dtype=np.float32),
            num_points=1,
        )

    # ----------------------------------------------------------- internals

    def _geometry_to_xyz(self, geom: dict, scene_context: dict) -> np.ndarray:
        xy = geom.get("xy", [0.0, 0.0])
        z = geom.get("z")
        if z is None:
            z = float(
                scene_context.get("placement_defaults", {}).get(
                    "table_surface_z", 0.0
                )
            )
        return np.asarray([float(xy[0]), float(xy[1]), float(z)], dtype=np.float32)

    def _ask_vlm_for_pixels(self, color_image: np.ndarray, entities: List[dict]) -> Dict[str, dict]:
        if self.api_key is None:
            return {}
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required for ScenePerception") from exc

        client = OpenAI(api_key=self.api_key)

        h, w = color_image.shape[:2]
        try:
            import cv2  # type: ignore

            ok, buffer = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("cv2 failed to encode image")
            img_bytes = buffer.tobytes()
        except Exception:
            # Fallback: use PIL if cv2 is unavailable.
            from io import BytesIO

            from PIL import Image  # type: ignore

            buf = BytesIO()
            Image.fromarray(color_image).save(buf, format="JPEG", quality=90)
            img_bytes = buf.getvalue()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        compact_entities = [
            {
                "name": e["name"],
                "role": e.get("role", "unknown"),
                "object_class": e.get("object_class", ""),
                "vlm_label": e.get("perception", {}).get("vlm_label", e["name"]),
                "aliases": e.get("aliases", []),
            }
            for e in entities
        ]
        prompt = SCENE_LOCALIZE_PROMPT.replace(
            "{entities_json}", json.dumps(compact_entities, indent=2)
        )

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.1,
            timeout=self.timeout_sec,
        )
        text = resp.choices[0].message.content or ""
        # Be lenient with surrounding text.
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {}

        out: Dict[str, dict] = {}
        for hit in obj.get("entities", []):
            name = hit.get("name")
            if name is None:
                continue
            visible = bool(hit.get("visible", True))
            entry: dict[str, Any] = {"visible": visible}
            if visible and "x" in hit and "y" in hit:
                entry["x"] = int(round(float(hit["x"]) * w / 1000.0))
                entry["y"] = int(round(float(hit["y"]) * h / 1000.0))
                entry["x"] = max(0, min(entry["x"], w - 1))
                entry["y"] = max(0, min(entry["y"], h - 1))
            out[name] = entry
        return out


def save_color_image(color_image: np.ndarray, save_path: str | Path) -> str:
    """Save a BGR OpenCV image to a PNG/JPEG path.

    Internal convention for the real camera pipeline is BGR because RealSense is
    opened with rs.format.bgr8 and the legacy OpenCV/cloud_grasp code expects
    BGR. cv2.imwrite expects BGR, so that path is correct. If cv2 is unavailable
    and we fall back to PIL, convert BGR -> RGB first.
    """
    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore

        cv2.imwrite(save_path, color_image)
    except Exception:
        from PIL import Image  # type: ignore

        arr = color_image
        if getattr(arr, "ndim", 0) == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        Image.fromarray(arr).save(save_path)
    return save_path
