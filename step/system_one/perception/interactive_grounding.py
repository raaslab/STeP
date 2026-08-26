"""SoM-first real-world perception grounding for VLM-STL.

This module adapts the perception flow from the working legacy interactive
planner into the VLM-STL real backend.

Core idea:
1. Capture RGB-D elsewhere.
2. If use_som=True, run SAM over the entire image first and overlay numbered
   marks using cloud_grasp.prepare_som_image(...).
3. Send the marked image to the VLM and ask it for mark numbers instead of
   raw coordinates.
4. Resolve marks -> pixels.
5. Run cloud_grasp.get_object_geometry_with_sam(...) at those pixels to obtain
   robot-frame geometry.
6. Write geometry + live_scene_state into scene_context.

This intentionally mirrors the reference interactive planner's successful
pattern: SAM/SoM before VLM when --som is used, then SAM geometry refinement
after VLM selection.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


LOCALIZE_KNOWN_PROMPT = """You are a robot perception assistant.

You see ONE image from a fixed RealSense camera looking at a tabletop robot
workspace. You are given a fixed list of canonical entity names. Your ONLY job
is to find where each entity is in the image. Do NOT invent new names.

CRITICAL RULES:
- Your response must contain EXACTLY the same names as the list below — no more, no less.
- Copy each name character-for-character. Do not paraphrase, rename, or add extras.
- If you cannot find an entity, set visible=false with mark=null and no coordinates.

Output ONLY a valid JSON object:
{
  "entities": [
    {
      "name": "<EXACT canonical name from the list>",
      "visible": true,
      "x": <int 0-1000>,
      "y": <int 0-1000>,
      "mark": <int or null>
    },
    {
      "name": "<EXACT canonical name from the list>",
      "visible": false,
      "mark": null
    }
  ]
}

Entities to localize (use these names EXACTLY):
{entities_json}

Coordinate rules:
- Coordinates are normalized 0-1000.
- x=0 is left image edge, x=1000 is right image edge.
- y=0 is top image edge, y=1000 is bottom image edge.
- For cubes/blocks/graspable objects, select the center of the visible top surface.
- For paper/regions, select the center of the visible region.
- Avoid shadows, edges, rims, walls, and highlights.

Return ONLY JSON. No extra keys, no renamed entities.
"""

DISCOVER_PROMPT = """You are a robot perception assistant.

You see ONE image from a fixed RealSense camera looking at a tabletop robot
workspace. Enumerate every visible workspace-relevant entity.

Output ONLY a valid JSON object:
{
  "entities": [
    {
      "name": "<lowercase_snake_case_unique_name>",
      "label": "<short display label>",
      "object_class": "<cube|block|paper|chest|tray|mug|box|region|object>",
      "role": "<manipulated_object|goal_region|container|support_object|obstacle>",
      "x": <int 0-1000>,
      "y": <int 0-1000>,
      "mark": <int or null>,
      "aliases": ["<alias>", "..."]
    }
  ]
}

Role rules:
- manipulated_object: small graspable object, e.g. cube/block/mug.
- goal_region: flat target area, e.g. white paper.
- container: open chest/basket/box/bowl that can hold objects.
- support_object: object that can support placement on top.
- obstacle: object to avoid.

Coordinate rules:
- Coordinates are normalized 0-1000.
- For manipulated objects, select the center of the visible top surface.
- For paper/regions, select the center of the visible region.
- Avoid shadows, edges, rims, walls, and highlights.

Return ONLY JSON.
"""

SOM_ADDITION = """
SET-OF-MARKS IS ACTIVE:
The image has colored numbered labels over SAM-generated segments. Prefer using
the "mark" field when a numbered mark badge (black box with white number) is
directly overlaid on the target entity.
- If a numbered badge is directly ON the correct cube/object/paper, output that
  mark number.
- CRITICAL: If you cannot see a badge directly on the target object, set
  mark=null and estimate x/y from the object's visible color and shape instead.
  Do NOT pick a badge from a different object just because it is nearby.
- Never pick a mark from the wrong object. A green badge outline is just a
  boundary — only the black numbered badge counts as a mark.
- For cubes/blocks: the mark badge will be centered on the cube's top surface.
- For white paper: the mark badge will be on the paper, not on a shadow.
"""

SOM_DISCOVER_PROMPT = """You are a robot perception assistant.

The image shows a tabletop robot workspace with SET-OF-MARKS overlaid.
Each black-box numbered badge (1, 2, 3 ...) identifies a distinct SAM-segmented
region. Your PRIMARY task: produce one JSON entity for EVERY mark number you can
see in the image. Do not skip any mark.

For each mark decide:
- Is it the robot arm / table background / wall? → skip it (only valid exception).
- Is it a flat colored region (paper, mat, tray)? → role = "goal_region".
- Is it a small 3-D graspable object (cube, block, cylinder, mug)? → role = "manipulated_object".
- Is it a container (chest, basket, box that holds things)? → role = "container".
- Is it a large immovable obstacle? → role = "obstacle".

A small object sitting ON TOP of a paper still gets its own entity with
role="manipulated_object". Do not merge it with the paper beneath it.

Output ONLY a valid JSON object:
{
  "entities": [
    {
      "name": "<lowercase_snake_case_unique_name>",
      "label": "<short display label>",
      "object_class": "<cube|block|cylinder|paper|tray|chest|mug|box|region|object>",
      "role": "<manipulated_object|goal_region|container|support_object|obstacle>",
      "x": <int 0-1000>,
      "y": <int 0-1000>,
      "mark": <int>,
      "aliases": ["<alias>", "..."]
    }
  ]
}

Coordinate rules:
- Coordinates are normalized 0-1000.
- For manipulated objects, select the center of the visible top surface.
- For paper/regions, select the center of the visible region.

Return ONLY JSON.
"""


def _load_env_file(env_file: str | None = ".env") -> None:
    if not env_file:
        return
    p = Path(env_file).expanduser()
    candidates = [p] if p.is_absolute() else [
        Path.cwd() / p,
        Path(__file__).resolve().parents[2] / p,
    ]
    env_path = next((c for c in candidates if c.is_file()), None)
    if env_path is None:
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _add_legacy_path(legacy_root: str | Path | None) -> Path:
    if legacy_root is None:
        legacy_root = "temp_real_robot_files"
    root = Path(legacy_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _import_cloud_grasp(legacy_root: str | Path | None):
    root = _add_legacy_path(legacy_root)
    try:
        import cloud_grasp  # type: ignore
        return cloud_grasp
    except Exception as exc:
        raise RuntimeError(
            f"Could not import cloud_grasp from {root}. "
            "Set platform.perception.legacy_perception_root or use "
            "--legacy-perception-root."
        ) from exc


def _image_to_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buffer).decode("utf-8")


def draw_coordinate_grid(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    tick_color = (220, 220, 220)
    label_color = (255, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for v in range(0, 1001, 100):
        px = int(v / 1000 * (w - 1))
        py = int(v / 1000 * (h - 1))
        cv2.line(img, (px, 0), (px, 6), tick_color, 1)
        cv2.putText(img, str(v), (max(px - 12, 0), 16), font, 0.32, label_color, 1, cv2.LINE_AA)
        cv2.line(img, (px, h - 1), (px, h - 7), tick_color, 1)
        cv2.line(img, (0, py), (6, py), tick_color, 1)
        cv2.putText(img, str(v), (2, max(py + 4, 8)), font, 0.32, label_color, 1, cv2.LINE_AA)
        cv2.line(img, (w - 1, py), (w - 7, py), tick_color, 1)
    return img


def _compact_entities(scene_context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": e["name"],
            "role": e.get("role", "unknown"),
            "object_class": e.get("object_class", ""),
            "vlm_label": e.get("perception", {}).get("vlm_label", e["name"]),
            "aliases": e.get("aliases", []),
        }
        for e in scene_context.get("entities", [])
        if e.get("name")
    ]


def _query_vlm_entities(
    *,
    image: np.ndarray,
    scene_context: dict[str, Any],
    mode: str,
    model: str,
    som_marks: dict[int, Any] | None,
) -> tuple[list[dict[str, Any]], np.ndarray, str, str]:
    h, w = image.shape[:2]
    if mode == "discover":
        # With SoM active, use the mark-first prompt so every visible mark gets
        # its own entity. Without SoM, fall back to the generic discover prompt.
        prompt = SOM_DISCOVER_PROMPT if som_marks else DISCOVER_PROMPT
    else:
        prompt = LOCALIZE_KNOWN_PROMPT.replace(
            "{entities_json}", json.dumps(_compact_entities(scene_context), indent=2)
        )

    if som_marks and mode != "discover":
        # SOM_ADDITION is only needed for localize-known; discover already has
        # the mark-first instructions baked into SOM_DISCOVER_PROMPT.
        prompt += "\n" + SOM_ADDITION
        vlm_image = image
    else:
        vlm_image = draw_coordinate_grid(image)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env or export it.")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(vlm_image)}"}},
            ],
        }],
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    print(f"[interactive_grounding] querying OpenAI ({model}), som={bool(som_marks)}")
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    raw_vlm_text = resp.json()["choices"][0]["message"]["content"] or ""
    text = raw_vlm_text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    obj = json.loads(text)
    entities = list(obj.get("entities", []))

    for ent in entities:
        mark = ent.get("mark")
        if som_marks and mark is not None:
            try:
                mid = int(mark)
                if mid in som_marks:
                    cx, cy, _ = som_marks[mid]
                    ent["x_px"] = int(cx)
                    ent["y_px"] = int(cy)
                    ent["_from_som"] = True
                    ent["_mark"] = mid
            except Exception:
                pass
        if "x_px" not in ent and "x" in ent and "y" in ent:
            ent["x_px"] = max(0, min(int(float(ent["x"]) * w / 1000.0), w - 1))
            ent["y_px"] = max(0, min(int(float(ent["y"]) * h / 1000.0), h - 1))
    return entities, vlm_image, raw_vlm_text, prompt


def _role_for_known(scene_context: dict[str, Any], name: str | None) -> str:
    for e in scene_context.get("entities", []):
        if e.get("name") == name:
            return e.get("role", "unknown")
    return "unknown"


def _class_for_known(scene_context: dict[str, Any], name: str | None) -> str:
    for e in scene_context.get("entities", []):
        if e.get("name") == name:
            return e.get("object_class", "object")
    return "object"


def _aliases_for_entity(entity: dict[str, Any]) -> list[str]:
    aliases = [a for a in entity.get("aliases", []) if isinstance(a, str)]
    label = entity.get("label")
    name = entity.get("name")
    if label and label not in aliases:
        aliases.append(str(label))
    if name:
        natural = str(name).replace("_", " ")
        if natural not in aliases:
            aliases.append(natural)
    return aliases


def _normalize_sam_result(res):
    if res is None:
        return None, None, None, None, None
    vals = list(res)
    while len(vals) < 5:
        vals.append(None)
    return vals[:5]


def _refine_entity_geometry(
    *,
    entity: dict[str, Any],
    color_image: np.ndarray,
    depth_image: np.ndarray,
    projector: Any,
    cloud_grasp: Any,
    sam_segmenter: Any,
    pcd: Any,
    outdir: Path | None,
) -> dict[str, Any] | None:
    name = str(entity.get("name", "")).strip().lower().replace(" ", "_")
    if not name or entity.get("visible") is False:
        return None
    px = entity.get("x_px")
    py = entity.get("y_px")
    if px is None or py is None:
        return None
    px, py = int(px), int(py)
    label = str(entity.get("label") or entity.get("vlm_label") or name)
    notes: list[str] = []
    mask = None

    try:
        res = cloud_grasp.get_object_geometry_with_sam(
            color_image, depth_image, px, py, sam_segmenter, projector, label=label, pcd=pcd
        )
        top_pt, center_xy, radius, mask, surface_points = _normalize_sam_result(res)
    except Exception as exc:
        top_pt, center_xy, radius, mask, surface_points = None, None, None, None, None
        notes.append(f"SAM call failed: {exc}")

    if top_pt is not None:
        # Keep px, py as the VLM's chosen point (SoM mark centroid).
        # SAM geometry gives us the 3D position and radius but we don't
        # override the 2D pixel the VLM deliberately selected.
        xyz = [float(top_pt[0]), float(top_pt[1]), float(top_pt[2])]
        xy = xyz[:2] if center_xy is None else [float(center_xy[0]), float(center_xy[1])]
        radius_f = None if radius is None else float(radius)
        notes.append("SoM-first pipeline: VLM mark/pixel + SAM mask geometry")
    else:
        try:
            xyz_np = projector.pixel_to_robot_frame(px, py, depth_image)
            xyz = [float(xyz_np[0]), float(xyz_np[1]), float(xyz_np[2])]
            xy = xyz[:2]
            radius_f = float(entity.get("expected_radius", 0.05) or 0.05)
            notes.append("SoM-first pipeline: fallback single-pixel deprojection")
        except Exception as exc:
            return {
                "name": name,
                "visible": False,
                "pixel": [px, py],
                "notes": f"deprojection failed: {exc}",
            }

    if outdir is not None and mask is not None:
        try:
            cv2.imwrite(str(outdir / f"mask_{name}.png"), (mask.astype(np.uint8) * 255))
        except Exception:
            pass

    return {
        "name": name,
        "label": label,
        "role": str(entity.get("role") or "unknown"),
        "object_class": str(entity.get("object_class") or "object"),
        "visible": True,
        "pixel": [px, py],
        "mark": entity.get("_mark"),
        "from_som": bool(entity.get("_from_som", False)),
        "xyz_robot": xyz,
        "xy_robot": xy,
        "z_robot": xyz[2],
        "radius": radius_f,
        "aliases": _aliases_for_entity(entity),
        "notes": "; ".join(notes),
    }


def _build_scene_context(base: dict[str, Any], rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    ctx = copy.deepcopy(base)
    if mode == "discover":
        ctx["entities"] = []
        ctx["geometry"] = {}
        for row in rows:
            if not row.get("visible") or row.get("xyz_robot") is None:
                continue
            ent = {
                "name": row["name"],
                "role": row.get("role", "unknown"),
                "object_class": row.get("object_class", "object"),
                "aliases": row.get("aliases") or [row["name"].replace("_", " ")],
                "perception": {"vlm_label": row.get("label", row["name"])},
            }
            if row.get("radius") is not None:
                ent["perception"]["expected_radius"] = float(row["radius"])
            ctx["entities"].append(ent)

    geom = ctx.setdefault("geometry", {})
    live = {}
    for row in rows:
        if not row.get("visible") or row.get("xyz_robot") is None:
            continue
        name = row["name"]
        xyz = [float(v) for v in row["xyz_robot"]]
        xy = [float(v) for v in row["xy_robot"]]
        geom.setdefault(name, {})
        geom[name]["xy"] = xy
        geom[name]["z"] = float(xyz[2])
        geom[name]["xyz"] = xyz
        if row.get("radius") is not None:
            geom[name]["radius"] = float(row["radius"])
            if row.get("role") == "goal_region":
                geom[name]["descend_radius"] = max(0.05, float(row["radius"]) + 0.02)
        live[name] = {
            "xyz": xyz,
            "xy": xy,
            "z": float(xyz[2]),
            "role": row.get("role"),
            "object_class": row.get("object_class"),
            "radius": row.get("radius"),
            "pixel": row.get("pixel"),
            "mark": row.get("mark"),
            "from_som": bool(row.get("from_som")),
            "source": "som_first_interactive_grounding",
            "notes": row.get("notes", ""),
        }

    ctx.setdefault("platform", {})["backend"] = "real"
    ctx["live_scene_state"] = {
        "objects": live,
        "notes": "Generated by SoM-first grounding: SAM marks before VLM, then SAM geometry refinement.",
    }
    return ctx


def draw_debug_overlay(image: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    disp = image.copy()
    for row in rows:
        if not row.get("visible") or row.get("pixel") is None:
            continue
        px, py = row["pixel"]
        color = (0, 255, 0) if row.get("role") == "manipulated_object" else (0, 200, 255)
        cv2.circle(disp, (int(px), int(py)), 12, color, 2)
        label = row["name"]
        if row.get("from_som"):
            label += f" m{row.get('mark')}"
        cv2.putText(disp, label, (int(px) + 14, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if row.get("radius") is not None:
            cv2.putText(disp, f"R={float(row['radius']):.2f}", (int(px) + 14, int(py) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return disp


def _draw_vlm_response_overlay(
    vlm_image: np.ndarray,
    entities: list[dict[str, Any]],
    path: Path,
) -> None:
    """Draw VLM-chosen coordinates as circles on the image sent to the VLM.

    Mirrors interactive_planner.py's visualize_plan overlay style: colored
    circle + label at each entity's raw VLM pixel, before SAM refinement.
    """
    img = vlm_image.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    role_color = {
        "manipulated_object": (0, 255, 0),
        "goal_region":        (0, 200, 255),
        "container":          (255, 100, 0),
        "obstacle":           (0, 0, 255),
    }

    for i, ent in enumerate(entities):
        if ent.get("visible") is False:
            continue
        px = ent.get("x_px")
        py = ent.get("y_px")
        if px is None or py is None:
            continue
        px, py = int(px), int(py)
        if not (0 <= px < w and 0 <= py < h):
            continue

        role = str(ent.get("role", ""))
        color = role_color.get(role, (200, 200, 0))
        name = str(ent.get("name") or ent.get("label") or f"ent{i+1}")
        mark = ent.get("mark") or ent.get("_mark")

        # Circle + cross like interactive_planner
        cv2.circle(img, (px, py), 14, color, 2)
        cv2.drawMarker(img, (px, py), color, cv2.MARKER_CROSS, 12, 1)

        label = f"#{mark} {name}" if mark is not None else name
        # Shadow then bright text
        cv2.putText(img, label, (px + 16, py - 6), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (px + 16, py - 6), font, 0.45, color, 1, cv2.LINE_AA)

        coord_str = f"({ent.get('x', '?')},{ent.get('y', '?')})"
        cv2.putText(img, coord_str, (px + 16, py + 10), font, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, coord_str, (px + 16, py + 10), font, 0.35, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(path), img)


def ground_scene_context_interactive(
    *,
    scene_context: dict[str, Any],
    color_image: np.ndarray,
    depth_image: np.ndarray,
    projector: Any,
    mode: str,
    model: str | None = None,
    legacy_perception_root: str | Path | None = "temp_real_robot_files",
    use_som: bool = False,
    use_sam: bool = True,
    pcd: Any = None,
    outdir: str | Path | None = None,
    env_file: str | None = ".env",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ground a scene_context using the reference interactive planner's flow."""
    _load_env_file(env_file)
    cloud_grasp = _import_cloud_grasp(legacy_perception_root)

    out_path = None if outdir is None else Path(outdir)
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    # Load and apply external mask if available
    original_color_image = color_image.copy()
    mask_path = Path("mask.png")
    if mask_path.exists():
        try:
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if mask_img.shape[:2] != color_image.shape[:2]:
                    mask_img = cv2.resize(mask_img, (color_image.shape[1], color_image.shape[0]))
                
                # Apply mask to color image
                color_image = cv2.bitwise_and(color_image, color_image, mask=mask_img)
                
                # Apply mask to depth image
                depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask_img)
                print(f"[interactive_grounding] Applied external mask from {mask_path}")
        except Exception as e:
            print(f"[interactive_grounding] WARNING: Failed to apply mask.png: {e}")

    sam_segmenter = None
    if use_sam:
        try:
            sam_segmenter = cloud_grasp.SAMSegmenter()
        except Exception as exc:
            print(f"[interactive_grounding] SAM unavailable: {exc}")
            sam_segmenter = None

    # CRITICAL: this is the requested behavior. With --som, run SAM over the
    # full image first, draw numbered marks, and send that marked image to VLM.
    vlm_image = original_color_image
    som_marks = None
    if use_som and sam_segmenter is not None and hasattr(cloud_grasp, "prepare_som_image"):
        print("[interactive_grounding] SoM enabled: running SAM before VLM...")
        try:
            # We pass the MASKED image for mask generation, but the ORIGINAL image for labeling
            vlm_image, som_marks = cloud_grasp.prepare_som_image(original_color_image, sam_segmenter, masked_image=color_image)
            if not som_marks:
                print("[interactive_grounding] SoM produced no marks; falling back to coordinate grid.")
                vlm_image = color_image
                som_marks = None
            else:
                print(f"[interactive_grounding] SoM generated {len(som_marks)} marks.")
                if out_path is not None:
                    cv2.imwrite(str(out_path / "som_labels.png"), vlm_image)
        except Exception as exc:
            print(f"[interactive_grounding] SoM failed: {exc}; falling back to coordinate grid.")
            vlm_image = color_image
            som_marks = None

    model = model or os.environ.get("OPENAI_MODEL") or scene_context.get("platform", {}).get("perception", {}).get("model", "gpt-4o")
    entities, sent_to_vlm, _raw_vlm_text, _vlm_prompt = _query_vlm_entities(
        image=vlm_image,
        scene_context=scene_context,
        mode=mode,
        model=model,
        som_marks=som_marks,
    )

    if out_path is not None:
        cv2.imwrite(str(out_path / "capture.png"), color_image)
        cv2.imwrite(str(out_path / "vlm_input.png"), sent_to_vlm)
        print(f"[interactive_grounding] VLM input image saved to {out_path / 'vlm_input.png'}")
        try:
            np.save(out_path / "depth.npy", depth_image)
        except Exception:
            pass
        # Draw VLM-chosen coords as circles on the image that was sent to the VLM
        try:
            _draw_vlm_response_overlay(sent_to_vlm, entities, out_path / "vlm_response.png")
            print(f"[interactive_grounding] VLM response image saved to {out_path / 'vlm_response.png'}")
        except Exception as exc:
            print(f"[interactive_grounding] Warning: could not save vlm_response.png: {exc}")

    known_by_name = {e.get("name"): e for e in scene_context.get("entities", [])}
    rows: list[dict[str, Any]] = []
    for ent in entities:
        if ent.get("visible") is False:
            rows.append({
                "name": ent.get("name"),
                "role": _role_for_known(scene_context, ent.get("name")),
                "object_class": _class_for_known(scene_context, ent.get("name")),
                "visible": False,
                "notes": "VLM marked not visible",
            })
            continue
        if mode == "localize-known":
            known = known_by_name.get(ent.get("name"), {})
            ent.setdefault("role", known.get("role", "unknown"))
            ent.setdefault("object_class", known.get("object_class", "object"))
            ent.setdefault("label", known.get("perception", {}).get("vlm_label", ent.get("name")))
            ent.setdefault("aliases", known.get("aliases", []))
        row = _refine_entity_geometry(
            entity=ent,
            color_image=color_image,
            depth_image=depth_image,
            projector=projector,
            cloud_grasp=cloud_grasp,
            sam_segmenter=sam_segmenter,
            pcd=pcd,
            outdir=out_path,
        )
        if row is not None:
            rows.append(row)

    new_ctx = _build_scene_context(scene_context, rows, mode)
    if out_path is not None:
        cv2.imwrite(str(out_path / "perception_overlay.png"), draw_debug_overlay(color_image, rows))
        report = {
            "mode": mode,
            "pipeline": "som_first_interactive_grounding",
            "model": model,
            "som": bool(som_marks),
            "sam": bool(sam_segmenter is not None),
            "pcd_fusion": bool(pcd is not None),
            "entities": rows,
        }
        (out_path / "perception_report.json").write_text(json.dumps(report, indent=2))
        (out_path / "scene_context_perception.json").write_text(json.dumps(new_ctx, indent=2))

    return new_ctx, rows
