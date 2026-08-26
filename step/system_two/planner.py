from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from .prompts import build_developer_prompt, build_user_payload


class DirectSTLPlanner:
    def __init__(
        self,
        *,
        schema: dict[str, Any],
        scene_context: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.schema = schema
        self.scene_context = scene_context

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
                            "text": build_developer_prompt(self.scene_context),
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
