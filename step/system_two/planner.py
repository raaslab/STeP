from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .prompts import build_developer_prompt, build_global_developer_prompt, build_user_payload


class DirectSTLPlanner:
    def __init__(
        self,
        *,
        schema: dict[str, Any],
        scene_context: dict[str, Any] | None = None,
        model: str | None = None,
        developer_prompt_fn: Callable[[dict[str, Any] | None], str] | None = None,
    ) -> None:
        """`developer_prompt_fn` is an EXPERIMENTAL hook (see
        STL_DECOMPOSITION_PLAN.md): pass `build_global_developer_prompt` (or
        use `mode="stl_spec"`, see `for_mode`) to get the single-global-formula
        prompt variant instead of the default per-subtask one. Defaults to
        today's `build_developer_prompt`, so existing callers are unaffected.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.4")
        self.schema = schema
        self.scene_context = scene_context
        self.developer_prompt_fn = developer_prompt_fn or build_developer_prompt

    @classmethod
    def for_mode(
        cls,
        mode: str,
        *,
        schema: dict[str, Any],
        scene_context: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> "DirectSTLPlanner":
        """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md) convenience constructor.
        `mode="skill_graph"` (default pipeline) or `mode="stl_spec"` (single
        global formula, decomposed after the fact by `stl_plan.decompose`)."""
        if mode == "stl_spec":
            prompt_fn = build_global_developer_prompt
        elif mode == "skill_graph":
            prompt_fn = build_developer_prompt
        else:
            raise ValueError(f"unknown DirectSTLPlanner mode {mode!r}")
        return cls(schema=schema, scene_context=scene_context, model=model, developer_prompt_fn=prompt_fn)

    @staticmethod
    def _image_to_data_url(path: str | Path) -> str:
        p = Path(path)
        suffix = p.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        return f"data:image/{mime};base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"

    def plan(self, *, instruction: str, image_path: str | Path) -> tuple[dict[str, Any], str]:
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self.developer_prompt_fn(self.scene_context),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_user_payload(
                                instruction=instruction,
                                scene_context=self.scene_context,
                                schema=self.schema,
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": self._image_to_data_url(image_path),
                        },
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "direct_stl_plan",
                    "schema": self.schema,
                    "strict": False,
                }
            },
        )
        raw_text = resp.output_text
        return json.loads(raw_text), raw_text
