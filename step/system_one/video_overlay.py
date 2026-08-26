"""Extracted mixin for System1Executor.

This file is intentionally method-only: it preserves behavior from the previous
monolithic executor.py while making the main executor easier to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import time

import numpy as np

from .base_env import BaseManipulationEnv
from .execution_types import SkillExecutionRecord
from .scene_resolver import SceneResolver



class VideoOverlayMixin:
    @staticmethod
    def _wrap_text_for_width(draw, text: str, font, max_width: int, max_lines: int = 4) -> list[str]:
        words = str(text).split()
        if not words:
            return [""]
        lines: list[str] = []
        cur = ""
        for word in words:
            candidate = word if not cur else f"{cur} {word}"
            try:
                bbox = draw.textbbox((0, 0), candidate, font=font)
                candidate_width = bbox[2] - bbox[0]
            except Exception:
                candidate_width = 8 * len(candidate)
            if candidate_width <= max_width or not cur:
                cur = candidate
            else:
                lines.append(cur)
                cur = word
                if len(lines) >= max_lines:
                    break
        if cur and len(lines) < max_lines:
            lines.append(cur)
        if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
            last = lines[-1]
            while len(last) > 3:
                candidate = last.rstrip() + "..."
                try:
                    bbox = draw.textbbox((0, 0), candidate, font=font)
                    if bbox[2] - bbox[0] <= max_width:
                        lines[-1] = candidate
                        break
                except Exception:
                    lines[-1] = candidate
                    break
                last = last[:-1]
        return lines

    @classmethod
    def _add_prompt_caption(
        cls,
        frame: np.ndarray,
        prompt: str,
        *,
        precondition: str = "",
        postcondition: str = "",
        reflection_summary: str = "",
        subtask_description: str = "",
        stl_formula: str = "",
        robustness: str = "",
        show_state: bool = True,
    ) -> np.ndarray:
        """Append fixed-size caption bands below the frame.

        All frames in an imageio/ffmpeg movie must have identical dimensions.
        Therefore the overlay reserves a constant number of lines, regardless
        of how long the current precondition/postcondition/reflection text is.
        """
        if not prompt and not show_state:
            return np.asarray(frame, dtype=np.uint8)
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            raise RuntimeError(
                "Saving captioned videos requires Pillow. Install it with: pip install pillow"
            ) from exc

        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        pad_x = 20
        pad_y = 12
        font_size = max(18, min(30, img.width // 24))
        state_font_size = max(14, int(font_size * 0.72))
        font = None
        state_font = None
        font_candidates = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    state_font = ImageFont.truetype(font_path, state_font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        if state_font is None:
            state_font = ImageFont.load_default()

        dummy = Image.new("RGB", (img.width, 10), "black")
        draw = ImageDraw.Draw(dummy)

        prompt_max_lines = 3 if prompt else 0
        subtask_max_lines = 2
        stl_max_lines = 2
        robust_max_lines = 1
        pre_max_lines = 2
        post_max_lines = 2
        refl_max_lines = 3
        state_max_lines = (
            subtask_max_lines
            + stl_max_lines
            + robust_max_lines
            + pre_max_lines
            + post_max_lines
            + refl_max_lines
        ) if show_state else 0

        prompt_lines: list[str] = []
        if prompt_max_lines:
            prompt_lines = cls._wrap_text_for_width(
                draw,
                f"Prompt: {prompt}",
                font,
                max_width=img.width - 2 * pad_x,
                max_lines=prompt_max_lines,
            )
        while len(prompt_lines) < prompt_max_lines:
            prompt_lines.append("")

        state_lines: list[str] = []
        if show_state:
            for label, value, budget in [
                ("Subtask", subtask_description, subtask_max_lines),
                ("STL", stl_formula, stl_max_lines),
                ("Robustness", robustness, robust_max_lines),
                ("Precondition", precondition, pre_max_lines),
                ("Postcondition", postcondition, post_max_lines),
                ("Recall", reflection_summary, refl_max_lines),
            ]:
                text = str(value or "").strip() or "—"
                lines = cls._wrap_text_for_width(
                    draw,
                    f"{label}: {text}",
                    state_font,
                    max_width=img.width - 2 * pad_x,
                    max_lines=budget,
                )
                while len(lines) < budget:
                    lines.append("")
                state_lines.extend(lines[:budget])

        try:
            bbox = draw.textbbox((0, 0), "Ag", font=font)
            line_h = max(font_size + 6, bbox[3] - bbox[1] + 8)
        except Exception:
            line_h = font_size + 8
        try:
            sbbox = draw.textbbox((0, 0), "Ag", font=state_font)
            state_line_h = max(state_font_size + 5, sbbox[3] - sbbox[1] + 7)
        except Exception:
            state_line_h = state_font_size + 7

        prompt_h = 2 * pad_y + line_h * prompt_max_lines if prompt_max_lines else 0
        state_h = 2 * pad_y + state_line_h * state_max_lines if show_state else 0

        out = Image.new("RGB", (img.width, img.height + prompt_h + state_h), "black")
        out.paste(img, (0, 0))
        draw = ImageDraw.Draw(out)

        y = img.height
        if prompt_max_lines:
            draw.rectangle((0, y, img.width, y + prompt_h), fill=(0, 0, 0))
            yy = y + pad_y
            for line in prompt_lines[:prompt_max_lines]:
                if line:
                    draw.text((pad_x, yy), line, fill=(255, 255, 255), font=font)
                yy += line_h
            y += prompt_h

        if show_state:
            draw.rectangle((0, y, img.width, y + state_h), fill=(70, 0, 0))
            yy = y + pad_y
            for line in state_lines[:state_max_lines]:
                if line:
                    draw.text((pad_x, yy), line, fill=(255, 235, 235), font=state_font)
                yy += state_line_h

        return np.asarray(out, dtype=np.uint8)

    def _normalize_video_frame_size(self, frame: np.ndarray) -> np.ndarray:
        """Pad/crop to first written video frame size as a safety guard."""
        arr = np.asarray(frame, dtype=np.uint8)
        if self._video_frame_shape is None:
            self._video_frame_shape = tuple(arr.shape)
            return arr
        target_h, target_w = self._video_frame_shape[:2]
        h, w = arr.shape[:2]
        if h == target_h and w == target_w:
            return arr
        fixed = np.zeros((target_h, target_w, arr.shape[2]), dtype=np.uint8)
        copy_h = min(h, target_h)
        copy_w = min(w, target_w)
        fixed[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
        return fixed

    def _open_video_writer(self) -> None:
        if self.video_path is None or self._video_writer is not None:
            return
        try:
            import imageio.v2 as imageio
        except Exception as exc:
            raise RuntimeError(
                "Saving videos requires imageio and imageio-ffmpeg. Install with: pip install imageio imageio-ffmpeg pillow"
            ) from exc
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self._video_writer = imageio.get_writer(
            str(self.video_path), fps=max(1, self.video_fps)
        )

    def _capture_video_frame(self, env: BaseManipulationEnv, *, force: bool = False) -> None:
        if self.video_path is None:
            return
        self._video_step_count += 1
        if not force and (self._video_step_count % self.video_every_n_steps != 0):
            return
        self._open_video_writer()
        frame = env.render_rgb_frame(
            camera_name=self.video_camera,
            width=self.video_width,
            height=self.video_height,
        )
        if self.video_flip_vertical:
            frame = np.flipud(frame)
        if self.video_flip_horizontal:
            frame = np.fliplr(frame)
        # Keep the robustness line live during the current subtask.
        self._video_current_robustness = self._format_video_robustness(env)
        frame = self._add_prompt_caption(
            frame,
            self.video_prompt,
            precondition=self._video_current_precondition,
            postcondition=self._video_current_postcondition,
            reflection_summary=self._video_current_reflection_summary,
            subtask_description=self._video_current_subtask_description,
            stl_formula=self._video_current_stl_formula,
            robustness=self._video_current_robustness,
            show_state=self.video_state_overlay,
        )
        frame = self._normalize_video_frame_size(frame)
        self._video_writer.append_data(frame)
        self._video_frame_count += 1

    def _close_video_writer(self) -> None:
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            finally:
                self._video_writer = None

    def _set_video_subtask_overlay(self, subtask: dict[str, Any] | None) -> None:
        if not isinstance(subtask, dict):
            self._video_current_precondition = ""
            self._video_current_postcondition = ""
            self._video_current_subtask_description = ""
            self._video_current_stl_formula = ""
            self._video_current_robustness = ""
            self._video_current_subtask = None
            return
        self._video_current_subtask = subtask
        self._video_current_subtask_description = self._subtask_description(subtask)
        self._video_current_precondition = str(subtask.get("precondition") or "")
        self._video_current_postcondition = str(subtask.get("postcondition") or "")
        formula = self._formula_string(subtask)
        self._video_current_stl_formula = str(formula or "—")
        self._video_current_robustness = "—"

    def _set_video_reflection_overlay(self, result: dict[str, Any] | None) -> None:
        if not isinstance(result, dict):
            return
        # This field is now used for recall/replanning summaries. Keep the
        # method name for backward compatibility with old reflection code.
        summary = str(result.get("recall_summary") or result.get("reflection_summary") or "").strip()
        if result.get("trigger") and not summary:
            summary = f"Recall trigger={result.get('trigger')}"
        action = str(result.get("repair_action") or "").strip()
        failure = str(result.get("failure_type") or "").strip()
        prefix = ""
        if not result.get("recall_summary") and (action or failure):
            prefix = f"[{action or 'reflection'}"
            if failure:
                prefix += f" / {failure}"
            prefix += "] "
        self._video_current_reflection_summary = (prefix + summary).strip()

    def _set_video_final_overlay(self, result: dict[str, Any] | None = None) -> None:
        self._video_current_subtask = None
        self._video_current_stl_formula = "Final task verification"
        self._video_current_robustness = "—"
        self._video_current_precondition = "All planned high-level subtasks have executed."
        self._video_current_postcondition = "Final verifier confirms the original user task is complete."
        if result is not None:
            if isinstance(result, dict):
                if bool(result.get("repair_required", False)):
                    self._video_current_robustness = "-1.000 (task incomplete)"
                else:
                    self._video_current_robustness = "+1.000 (task complete)"
            self._set_video_reflection_overlay(result)


