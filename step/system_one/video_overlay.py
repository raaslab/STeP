"""Extracted mixin for System1Executor.

This file is intentionally method-only: it preserves behavior from the previous
monolithic executor.py while making the main executor easier to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import os
import time

import numpy as np

from .base_env import BaseManipulationEnv
from .execution_types import SkillExecutionRecord
from .scene_resolver import SceneResolver



class VideoOverlayMixin:
    # Sized independently of the 256px agentview render so widening the
    # caption panel never shrinks the simulation image.
    _PANEL_MIN_WIDTH = 620

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
        """Compose the sim frame beside a fixed-size caption panel.

        The caption goes to the RIGHT of the simulation frame, not below it.
        The sim render is only 256px wide, so the previous below-the-frame bands
        had to wrap every line at 256px and ellipsized most of the prompt and
        essentially every STL formula. A side panel gets its own generous width
        (_PANEL_MIN_WIDTH) without touching the sim image, which is pasted at
        its native resolution.

        All frames in an imageio/ffmpeg movie must have identical dimensions.
        Therefore the overlay reserves a constant number of lines, regardless
        of how long the current precondition/postcondition/reflection text is,
        and rounds the canvas up to a multiple of 16 (ffmpeg's macro block size,
        otherwise the writer silently rescales every frame).
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
        pad_x = 16
        pad_y = 12
        # Panel width is independent of the sim frame's own (small) width --
        # that's the whole point of putting the caption beside it.
        panel_w = max(cls._PANEL_MIN_WIDTH, img.width)
        text_w = panel_w - 2 * pad_x
        font_size = 17
        state_font_size = 15
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

        # Generous budgets: a full STL formula line can be long.
        prompt_max_lines = 3 if prompt else 0
        subtask_max_lines = 2
        stl_max_lines = 4
        robust_max_lines = 2
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
                max_width=text_w,
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
                    max_width=text_w,
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
        panel_h = prompt_h + state_h

        def _round16(v: int) -> int:
            return int(((int(v) + 15) // 16) * 16)

        canvas_w = _round16(img.width + panel_w)
        canvas_h = _round16(max(img.height, panel_h))

        out = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        # Sim frame: pasted at native resolution, never scaled.
        out.paste(img, (0, 0))
        draw = ImageDraw.Draw(out)

        panel_x = img.width
        y = 0
        if prompt_max_lines:
            draw.rectangle((panel_x, y, canvas_w, y + prompt_h), fill=(18, 18, 18))
            yy = y + pad_y
            for line in prompt_lines[:prompt_max_lines]:
                if line:
                    draw.text((panel_x + pad_x, yy), line, fill=(255, 255, 255), font=font)
                yy += line_h
            y += prompt_h

        if show_state:
            draw.rectangle((panel_x, y, canvas_w, y + state_h), fill=(52, 10, 10))
            yy = y + pad_y
            for line in state_lines[:state_max_lines]:
                if line:
                    draw.text((panel_x + pad_x, yy), line, fill=(255, 235, 235), font=state_font)
                yy += state_line_h
            y += state_h

        # Fill any remainder of the panel column (when the sim frame is taller
        # than the text) so the panel reads as one block, not a floating band.
        if y < canvas_h:
            draw.rectangle((panel_x, y, canvas_w, canvas_h), fill=(18, 18, 18))

        return np.asarray(out, dtype=np.uint8)

    def _on_skill_step(self, env: BaseManipulationEnv) -> None:
        """Per-step hook for the ompl_mpc skill adapters.

        Those adapters drive the env directly instead of going through
        step_nominal(), so neither the video capture nor the LIBERO-
        Constrained margin sampler would otherwise see their steps. Both hang
        off the single step_callback they already expose.
        """
        self._capture_video_frame(env)
        self._sample_constraint_margins(env)

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
        # Per-episode drawing caches: the table height and the camera projection
        # are both fixed for a rollout but must not carry into the next one.
        self._keepout_table_z = None
        self._keepout_proj_cache = None
        self._video_writer = imageio.get_writer(
            str(self.video_path), fps=max(1, self.video_fps)
        )

    def _keepout_projector(self, env: BaseManipulationEnv, width: int, height: int):
        """World xyz -> frame pixel, for the camera the video is rendered from.

        MuJoCo cameras look down -Z with +Y up, so this is the ordinary pinhole
        projection with a sign flip on z. Cached per episode: the camera does not
        move, and rebuilding it every frame would re-read the model each step.
        """
        cached = getattr(self, "_keepout_proj_cache", None)
        if cached is not None and cached[0] == (self.video_camera, width, height):
            return cached[1]
        sim = getattr(getattr(env, "env", None), "sim", None) or getattr(env, "sim", None)
        if sim is None:
            return None
        try:
            cid = sim.model.camera_name2id(self.video_camera)
            fovy = float(sim.model.cam_fovy[cid])
            cam_pos = np.asarray(sim.data.cam_xpos[cid], dtype=float)
            cam_mat = np.asarray(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)
        except Exception:
            return None
        focal = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)

        def project(p):
            q = cam_mat.T @ (np.asarray(p, dtype=float) - cam_pos)
            if q[2] >= -1e-6:
                return None
            u = focal * (q[0] / -q[2]) + width / 2.0
            v = focal * (q[1] / -q[2]) + height / 2.0
            return (u, height - v)

        self._keepout_proj_cache = ((self.video_camera, width, height), project)
        return project

    def _draw_keepout_zones(self, env: BaseManipulationEnv, frame: np.ndarray) -> np.ndarray:
        """Paint every active keep-out onto the frame as a filled region on the table."""
        from .geometry_sdf import CircleSDF

        zones = [
            c for c in (getattr(env, "sdf_constraints", []) or [])
            if str(getattr(c, "role", "")) in {"obstacle", "keepout"}
            and isinstance(getattr(c, "primitive", None), CircleSDF)
            and c.primitive.convention == "outside_positive"
        ]
        if not zones:
            return frame
        height, width = frame.shape[0], frame.shape[1]
        project = self._keepout_projector(env, width, height)
        if project is None:
            return frame

        # Measured once and held fixed: recomputing per frame as a median would
        # shift the zone whenever a carried object is lifted. Uses the minimum, not the median, since the lowest object is the one actually resting on the table.
        table_z = getattr(self, "_keepout_table_z", None)
        if table_z is None:
            try:
                zs = [
                    float(np.asarray(p).reshape(-1)[2])
                    for p in (env.scene_objects or {}).values()
                    if p is not None and np.isfinite(np.asarray(p).reshape(-1)[2])
                ]
                table_z = (min(zs) - 0.02) if zs else 0.90
            except Exception:
                table_z = 0.90
            self._keepout_table_z = table_z

        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return frame

        img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        drew = False
        for zone in zones:
            centre = np.asarray(zone.primitive.center_xy, dtype=float)
            radius = float(zone.primitive.radius) + float(getattr(zone, "margin", 0.0))
            pts = []
            for i in range(64):
                a = 2.0 * math.pi * i / 64.0
                p = project([centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a), table_z])
                if p is None:
                    pts = []
                    break
                pts.append(p)
            if len(pts) < 3:
                continue
            draw.polygon(pts, fill=(200, 45, 25, 80))
            draw.line(pts + [pts[0]], fill=(200, 45, 25, 230), width=3)
            drew = True
        if not drew:
            return frame
        return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"), dtype=np.uint8)

    def _draw_attached_collision_spheres(self, env: BaseManipulationEnv, frame: np.ndarray) -> np.ndarray:
        """Paints the carried object's sphere-fit collision approximation
        (what the attached-object margin check actually sees, not the object's real mesh)."""
        attached = getattr(self, "_motion_attached_state", None)
        if attached is None or not attached.collision_spheres_object_frame:
            return frame
        height, width = frame.shape[0], frame.shape[1]
        project = self._keepout_projector(env, width, height)
        if project is None:
            return frame
        sim = getattr(getattr(env, "env", None), "sim", None) or getattr(env, "sim", None)
        if sim is None:
            return frame
        try:
            from .motion.collision import compose_world_pose
            from .motion.panda import EE_SITE
            from .motion.scene_query import SceneQuery, quat_to_rotmat_wxyz
            from PIL import Image, ImageDraw
        except ImportError:
            return frame

        scene = SceneQuery.from_robosuite_sim(sim)
        hand_pos, hand_quat = scene.site_pose(EE_SITE)
        obj_pos, obj_quat = compose_world_pose(hand_pos, hand_quat, attached.grasp_transform_hand_from_object)
        R_obj = quat_to_rotmat_wxyz(obj_quat)
        cid = sim.model.camera_name2id(self.video_camera)
        cam_right = np.asarray(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 0]

        img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        drew = False
        for local_offset, radius in attached.collision_spheres_object_frame:
            center_world = obj_pos + R_obj @ local_offset
            center_px = project(center_world)
            edge_px = project(center_world + radius * cam_right)
            if center_px is None or edge_px is None:
                continue
            # Perspective-correct screen radius -- a fixed pixel size would be
            # wrong as the carried object moves nearer/farther from the camera.
            r_px = math.hypot(edge_px[0] - center_px[0], edge_px[1] - center_px[1])
            if r_px < 1.0:
                continue
            bbox = [center_px[0] - r_px, center_px[1] - r_px, center_px[0] + r_px, center_px[1] + r_px]
            draw.ellipse(bbox, fill=(30, 144, 255, 90), outline=(30, 144, 255, 200), width=2)
            drew = True
        if not drew:
            return frame
        return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"), dtype=np.uint8)

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
        # The keep-out is a region, not an object, so it's invisible in the
        # simulator; drawn here or the video can't be used to judge a violation.
        frame = self._draw_keepout_zones(env, frame)
        frame = self._draw_attached_collision_spheres(env, frame)
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


