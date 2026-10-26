"""VLM grounding: which object categories to segment, and what to call
each segmented instance.

Same model and client as System 2. First the VLM proposes the generic nouns
SAM3 should segment (SAM3 finds "bowl" reliably but misses asset-style names
like "ramekin"). Then every candidate mask is drawn as a numbered mark
(Set-of-Mark, as in CoPa), the VLM identifies each mark, and a separate call
grounds the instruction's objects to marks, renaming those in the
instruction's own words.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

import numpy as np

_CATEGORY_PROMPT = (
    "You see a robot manipulation scene. List every distinct KIND of physical object or "
    "piece of furniture on or around the work surface that the robot might interact with or "
    "collide with: containers, food items, kitchenware, books, cabinets, stoves, racks, "
    "baskets, etc. Use short generic nouns a segmentation model knows (e.g. \"bowl\", \"box\", "
    "\"bottle\", \"mug\", \"plate\", \"cabinet\", \"stove\", \"basket\", \"book\", \"can\"), one entry "
    "per kind, not per instance. Prefer the plain category over brand or material words. "
    "Never list the robot, the table, the floor or the walls."
)

_IMAGES_PREAMBLE = (
    "The images come in pairs from the robot's cameras: a clean view, then the same view where "
    "each numbered mark outlines one candidate object found by a segmentation model. The first "
    "pair is the main camera; a second pair, if present, is the wrist camera. A number appearing "
    "in both views is the same object seen from both cameras; judge it from whichever view shows "
    "it better (an object hidden behind another in one view can be plain in the other). "
)

_IDENTIFY_PROMPT = _IMAGES_PREAMBLE + (
    "The segmenter's category for a mark is only a rough guess and is often wrong (a cup or "
    "ramekin found as \"bowl\", a box as \"book\"), so identify every object yourself from its "
    "appearance in the clean views. For EVERY mark:\n"
    "- what_it_is: what the object actually is, judged by shape, size, label text and appearance\n"
    "- keep: false if it is not a real separate object (robot part, table, shadow, a piece of "
    "another mark's object, background), else true\n"
    "- name: a short unique snake_case name from what it is, with appearance or position words to "
    "tell identical-looking objects apart (e.g. black_bowl_left, black_bowl_right)\n"
    "- role: \"manipulated_object\" for small movable things, \"support_object\" for things "
    "objects are placed on or into (plates, baskets, trays, racks), \"fixture\" for large "
    "immovable furniture (cabinets, stoves, microwaves, shelves)\n"
    "- description: a few words on its appearance\n"
    "Image left/right is the robot's left/right.\n"
    "Finally, in unmarked_objects list a generic noun for every clearly visible object on or "
    "around the work surface that has NO mark (empty list if none)."
)

_GROUND_PROMPT = _IMAGES_PREAMBLE + (
    "Each mark has already been identified (list below, with depth-measured support facts, "
    "which are reliable). For EVERY object the task instruction mentions, find the mark it "
    "refers to:\n"
    "- phrase: the instruction's words for the object\n"
    "- evidence: check the mark's identity and every relation the instruction states (on, in, "
    "between, next to, left of) against the marks; use the support facts for on/in\n"
    "- mark: the mark number, or 0 if no mark is that object. The kind of object must match "
    "(a bowl is not a cup, a box is not a can) -- never pick a different kind just because it is "
    "the closest match, a wrong object is worse than none. Colour, material and brand words may "
    "not match what you see exactly (rendered colours and labels differ), so do not reject a "
    "mark of the right kind for those alone; when several marks are the right kind, decide by "
    "the relations, then by those words. The instruction may also name an object differently "
    "from its identification above (a ramekin identified as a small cup, ketchup as a sauce "
    "bottle); judge by what the object physically is, not by matching words.\n"
    "- name: a short snake_case name made from the instruction's words (e.g. "
    "black_bowl_between_plate_and_ramekin)\n"
    "- noun: one generic noun for what the object is (e.g. \"bowl\", \"can\", \"box\")\n"
    "Parts of an object (a drawer, door, knob, handle, compartment, burner) are handled "
    "separately: give them mark 0, never the mark of the whole object they belong to."
)

def _client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())


def vlm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _image_url(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ask_json(model: str, text: str, images: list[np.ndarray], schema: dict[str, Any], name: str) -> dict[str, Any]:
    content = [{"type": "input_text", "text": text}]
    content += [{"type": "input_image", "image_url": _image_url(im)} for im in images]
    resp = _client().responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
    )
    return json.loads(resp.output_text)


def default_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-5.4")


def propose_categories(rgb: np.ndarray, instruction: str, model: str | None = None) -> list[str]:
    schema = {
        "type": "object",
        "properties": {"categories": {"type": "array", "items": {"type": "string"}}},
        "required": ["categories"],
        "additionalProperties": False,
    }
    out = _ask_json(model or default_model(), f"{_CATEGORY_PROMPT}\n\nTask instruction: {instruction}", [rgb], schema, "categories")
    seen: list[str] = []
    for c in out.get("categories", []):
        c = str(c).strip().lower()
        if c and c not in seen:
            seen.append(c)
    return seen


def draw_marks(rgb: np.ndarray, masks: list[np.ndarray], labels: list[int]) -> np.ndarray:
    """CoPa-style marks: light fill, outline, and the number at the mask's
    most interior point so it never sits on a neighbouring object."""
    import cv2

    out = rgb.astype(np.float32).copy()
    colors = [np.random.default_rng(k).integers(60, 255, size=3) for k in labels]
    for mask, color in zip(masks, colors):
        out[mask] = 0.8 * out[mask] + 0.2 * color
    out = out.astype(np.uint8)
    for k, mask, color in zip(labels, masks, colors):
        m8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, [int(c) for c in color], 1)
        # Pad so the image border counts as outside the mask; otherwise a mask
        # cut off by the frame puts its number on the edge.
        dist = cv2.distanceTransform(np.pad(m8, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        row, col = np.unravel_index(int(np.argmax(dist)), dist.shape)
        label = str(k)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x0, y0 = int(col - tw / 2) - 2, int(row - th / 2) - 2
        cv2.rectangle(out, (x0, y0), (x0 + tw + 4, y0 + th + 5), (0, 0, 0), -1)
        cv2.putText(out, label, (x0 + 2, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def name_marks(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    categories: list[str],
    instruction: str,
    model: str | None = None,
    facts: list[str] | None = None,
    crops: list[tuple[int, np.ndarray]] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """One record per mark (1-based order) with keep/name/role/description,
    nouns for objects the segmenter missed, and the raw grounding answers.

    Two calls, as in CoPa: identify every mark without the instruction, then
    ground each object the instruction mentions to a mark or to none. Asking
    for both at once put the instruction's nouns on whatever mark looked
    closest, even marks the same answer identified as something else.
    """
    model = model or default_model()
    images = [im for pair in image_pairs for im in pair]
    facts = facts or [""] * len(categories)
    crop_note = ""
    if crops:
        # Close-ups make label text legible (ketchup vs bbq sauce, butter vs pudding).
        images += [c for _, c in crops]
        crop_note = (
            "\n\nAfter the camera images come higher-resolution close-up crops of these marks, in "
            f"this order: {', '.join(str(k) for k, _ in crops)}. Read labels and details from them."
        )

    identify_schema = {
        "type": "object",
        "properties": {
            "marks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mark": {"type": "integer"},
                        "what_it_is": {"type": "string"},
                        "keep": {"type": "boolean"},
                        "name": {"type": "string"},
                        "role": {"type": "string", "enum": ["manipulated_object", "support_object", "fixture"]},
                        "description": {"type": "string"},
                    },
                    "required": ["mark", "what_it_is", "keep", "name", "role", "description"],
                    "additionalProperties": False,
                },
            },
            "unmarked_objects": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["marks", "unmarked_objects"],
        "additionalProperties": False,
    }
    hint = "\n".join(f"mark {k}: segmenter guessed \"{c}\"" for k, c in enumerate(categories, start=1))
    out = _ask_json(model, f"{_IDENTIFY_PROMPT}{crop_note}\n\n{hint}", images, identify_schema, "mark_identities")
    by_mark = {int(m["mark"]): m for m in out.get("marks", [])}
    records = [
        dict(by_mark.get(k, {"mark": k, "what_it_is": "", "keep": True, "name": "", "role": "manipulated_object", "description": ""}))
        for k in range(1, len(categories) + 1)
    ]
    missing = [str(c).strip().lower() for c in out.get("unmarked_objects", []) if str(c).strip()]

    ground_schema = {
        "type": "object",
        "properties": {
            "referents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "phrase": {"type": "string"},
                        "evidence": {"type": "string"},
                        "mark": {"type": "integer"},
                        "name": {"type": "string"},
                        "noun": {"type": "string"},
                    },
                    "required": ["phrase", "evidence", "mark", "name", "noun"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["referents"],
        "additionalProperties": False,
    }
    listing = "\n".join(
        f"mark {r['mark']}: {r['what_it_is']} ({r['description']}){'; ' + f if f else ''}"
        for r, f in zip(records, facts)
        if r.get("keep", True)
    )
    text = f"{_GROUND_PROMPT}{crop_note}\n\nTask instruction: {instruction}\n\nMarks:\n{listing}"
    grounded = _ask_json(model, text, images, ground_schema, "referents").get("referents", [])
    for ref in grounded:
        k = int(ref.get("mark", 0))
        if 1 <= k <= len(records) and records[k - 1].get("keep", True) and ref.get("name"):
            records[k - 1]["name"] = ref["name"]
            records[k - 1]["referent"] = ref.get("phrase", "")
            records[k - 1]["evidence"] = ref.get("evidence", "")
        elif k == 0 and (ref.get("noun") or ref.get("phrase")):
            missing.append(str(ref.get("noun") or ref["phrase"]).strip().lower())
    return records, missing, grounded


def snake(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return s or "object"
