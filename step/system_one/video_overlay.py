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

    # ---------------------------------------------------------------- design
    # One dark ground, cards for each piece of state, a single amber accent and
    # one monospace face for anything the system computed. The old panel was a
    # dark-red block of flush-left text rows, three of which read "—" on almost
    # every frame; it was legible but it read as a debug dump, not something to
    # put in a talk.

    _BG = (18, 19, 23)
    _CARD = (28, 30, 36)
    _CARD_HI = (36, 39, 46)
    _INK = (238, 240, 244)
    _DIM = (142, 150, 164)
    _FAINT = (92, 99, 112)
    _ACCENT = (240, 174, 76)
    _OK = (104, 206, 150)
    _BAD = (232, 110, 92)

    _FONTS: dict = {}

    @classmethod
    def _font(cls, kind: str, size: int):
        key = (kind, size)
        if key in cls._FONTS:
            return cls._FONTS[key]
        from PIL import ImageFont

        paths = {
            "sans": ["/System/Library/Fonts/HelveticaNeue.ttc",
                     "/System/Library/Fonts/Supplemental/Arial.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
            "bold": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                     "/System/Library/Fonts/HelveticaNeue.ttc",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
            "mono": ["/System/Library/Fonts/Menlo.ttc",
                     "/System/Library/Fonts/SFNSMono.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
        }[kind]
        f = None
        for path in paths:
            if os.path.exists(path):
                try:
                    f = ImageFont.truetype(path, size)
                    break
                except Exception:
                    pass
        if f is None:
            f = ImageFont.load_default()
        cls._FONTS[key] = f
        return f

    @staticmethod
    def _tracked(draw, xy, text, font, fill, spacing=1.6):
        """Letter-spaced small caps, for the eyebrow labels."""
        x, y = xy
        for ch in text:
            draw.text((x, y), ch, font=font, fill=fill)
            x += draw.textlength(ch, font=font) + spacing
        return x

    @classmethod
    def _wrap(cls, draw, text, font, max_w, max_lines):
        words, lines, cur = str(text).split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if draw.textlength(t, font=font) <= max_w or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
                if len(lines) == max_lines:
                    break
        if cur and len(lines) < max_lines:
            lines.append(cur)
        if lines and draw.textlength(lines[-1], font=font) > max_w:
            while lines[-1] and draw.textlength(lines[-1] + "…", font=font) > max_w:
                lines[-1] = lines[-1][:-1]
            lines[-1] += "…"
        return lines

    def _card(self, draw, x, y, w, h, fill=None):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=fill or self._CARD)
        return y + h

    def _pill(self, draw, x, y, text, font, fg, bg):
        tw = draw.textlength(text, font=font)
        draw.rounded_rectangle((x, y, x + tw + 18, y + 21), radius=10, fill=bg)
        draw.text((x + 9, y + 3), text, font=font, fill=fg)
        return x + tw + 18

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

    def _add_prompt_caption(
        self,
        frame: np.ndarray,
        prompt: str | None,
        *,
        precondition: str = "",
        postcondition: str = "",
        reflection_summary: str = "",
        subtask_description: str = "",
        stl_formula: str = "",
        robustness: str = "",
        show_state: bool = True,
    ) -> np.ndarray:
        """Compose the sim render and the state panel into one presentable frame.

        The sim image is pasted at native resolution and never scaled; the panel
        is drawn beside it, and the band left under the render carries the
        timeline and the legend for the in-scene overlays.
        """
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            raise RuntimeError(
                "Saving captioned videos requires Pillow. Install it with: pip install pillow"
            ) from exc

        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        PANEL = max(self._PANEL_MIN_WIDTH, img.width)
        PAD = 22
        STRIP = 92                                   # timeline + legend band
        canvas_w = (img.width + PANEL + 15) // 16 * 16
        canvas_h = (max(img.height + STRIP, 560) + 15) // 16 * 16

        out = Image.new("RGB", (canvas_w, canvas_h), self._BG)
        out.paste(img, (0, 0))
        d = ImageDraw.Draw(out)

        f_eyebrow = self._font("bold", 11)
        f_title = self._font("bold", 19)
        f_body = self._font("sans", 15)
        f_small = self._font("sans", 13)
        f_mono = self._font("mono", 12)
        f_mono_s = self._font("mono", 11)
        f_big = self._font("bold", 34)

        px = img.width + PAD
        pw = canvas_w - px - PAD
        y = PAD

        # --- the task ---------------------------------------------------------
        self._tracked(d, (px, y), "LIBERO-CONSTRAINED", f_eyebrow, self._ACCENT)
        y += 20
        for line in self._wrap(d, prompt or "", f_title, pw, 2):
            d.text((px, y), line, font=f_title, fill=self._INK)
            y += 25
        y += 10

        # --- what System 1 is doing, and where in the plan --------------------
        if show_state and subtask_description:
            marks = getattr(self, "_hud_subtask_marks", []) or []
            total = max(1, getattr(self, "_hud_subtask_total", 0) or len(marks) or 1)
            cur = (getattr(self, "_hud_subtask_index", 0) or 0) + 1
            h = 92 if stl_formula else 54
            self._card(d, px, y, pw, h)
            self._tracked(d, (px + 16, y + 13), "SUBTASK", f_eyebrow, self._FAINT)
            pill = f"{min(cur, total)} / {total}"
            tw = d.textlength(pill, font=f_mono_s)
            d.rounded_rectangle((px + pw - 20 - tw - 16, y + 10, px + pw - 16, y + 29),
                                radius=9, fill=self._CARD_HI)
            d.text((px + pw - 28 - tw, y + 13), pill, font=f_mono_s, fill=self._DIM)
            d.text((px + 16, y + 29), self._wrap(d, subtask_description, f_body, pw - 32, 1)[0],
                   font=f_body, fill=self._INK)
            if stl_formula:
                yy = y + 54
                for line in self._wrap(d, stl_formula, f_mono, pw - 32, 2):
                    d.text((px + 16, yy), line, font=f_mono, fill=self._DIM)
                    yy += 15
            y += h + 12

        # --- the constraint, as a shape over time -----------------------------
        trace = getattr(self, "_hud_robustness", []) or []
        if show_state:
            ch = 128
            self._card(d, px, y, pw, ch)
            ex = self._tracked(d, (px + 16, y + 13), "ROBUSTNESS", f_eyebrow, self._FAINT)
            d.text((ex + 14, y + 12), "below zero = constraint violated",
                   font=f_small, fill=self._FAINT)
            if trace:
                v = trace[-1]
                col = self._OK if v >= 0 else self._BAD
                d.text((px + 16, y + 30), f"{v:+.3f}", font=f_big, fill=col)
                label = str(robustness or "")
                if "(" in label:
                    d.text((px + 150, y + 44),
                           label.split("(", 1)[1].rstrip(")").split(";")[0],
                           font=f_small, fill=self._DIM)
                gx, gy, gw, gh = px + 16, y + 74, pw - 32, 40
                d.rounded_rectangle((gx, gy, gx + gw, gy + gh), radius=5, fill=(22, 24, 29))
                lo, hi = min(trace), max(trace)
                span = max(hi - lo, 0.08)
                lo, hi = lo - 0.12 * span, hi + 0.12 * span
                zy = gy + gh * (hi - 0.0) / (hi - lo)
                zy = min(max(zy, gy), gy + gh)
                d.line([(gx, zy), (gx + gw, zy)], fill=(74, 80, 92), width=1)
                n = min(len(trace), gw)
                stride = len(trace) / n
                pts = []
                for i in range(n):
                    val = trace[min(len(trace) - 1, int(i * stride))]
                    pts.append((gx + i * (gw / n),
                                min(max(gy + gh * (hi - val) / (hi - lo), gy), gy + gh)))
                for i in range(1, len(pts)):
                    d.line([pts[i - 1], pts[i]],
                           fill=self._BAD if pts[i][1] > zy else self._OK, width=2)
                if pts:
                    d.ellipse([pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3],
                              fill=self._INK)
            else:
                d.text((px + 16, y + 40), "—", font=f_big, fill=self._FAINT)
            y += ch + 12

        # --- only shown when there is something to show -----------------------
        for label, text in (("RECALL", reflection_summary),
                            ("PRECONDITION", precondition),
                            ("POSTCONDITION", postcondition)):
            t = str(text or "").strip()
            if not t or t == "—" or not show_state:
                continue
            lines = self._wrap(d, t, f_small, pw - 32, 2)
            h = 26 + 16 * len(lines)
            self._card(d, px, y, pw, h)
            self._tracked(d, (px + 16, y + 9), label, f_eyebrow, self._FAINT)
            yy = y + 24
            for line in lines:
                d.text((px + 16, yy), line, font=f_small,
                       fill=self._ACCENT if label == "RECALL" else self._DIM)
                yy += 16
            y += h + 10

        # --- band under the render: timeline, then legend ---------------------
        by = img.height
        d.rectangle((0, by, img.width, canvas_h), fill=self._BG)
        marks = getattr(self, "_hud_subtask_marks", []) or []
        total = max(1, getattr(self, "_hud_subtask_total", 0) or len(marks) or 1)
        cur = getattr(self, "_hud_subtask_index", 0) or 0
        tx, tw_ = PAD, img.width - 2 * PAD
        seg = tw_ / total
        for i in range(total):
            sx = tx + i * seg
            col = self._OK if i < cur else (self._ACCENT if i == cur else (48, 52, 60))
            d.rounded_rectangle((sx, by + 16, sx + seg - 3, by + 22), radius=3, fill=col)
            name = marks[i][1] if i < len(marks) else ""
            if name and seg > 52:
                d.text((sx + 1, by + 27), name[:12], font=f_mono_s,
                       fill=self._INK if i == cur else self._FAINT)
        lx = PAD
        for sw, label in (((80, 190, 255), "end-effector path"),
                          ((200, 45, 25), "keep-out zone"),
                          ((30, 144, 255), "carried object")):
            d.ellipse((lx, by + 54, lx + 8, by + 62), fill=sw)
            d.text((lx + 14, by + 51), label, font=f_small, fill=self._FAINT)
            lx += 30 + int(d.textlength(label, font=f_small))

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
        # HUD state, per episode: where the end effector has been, how the
        # robustness bottleneck has moved, and which events punctuated the run.
        self._hud_trail: list[tuple[int, int]] = []
        self._hud_robustness: list[float] = []
        self._hud_events: list[tuple[int, str]] = []
        self._hud_subtask_marks: list[tuple[int, str]] = []
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

    # ------------------------------------------------------------------ HUD

    def _hud_record(self, env) -> None:
        """Per-frame HUD samples: the end effector's pixel, and the robustness
        bottleneck. Both are cheap reads; drawing happens once per frame below.
        """
        import numpy as _np

        # end-effector trail, projected with the same camera the zones use
        try:
            proj = self._keepout_projector(env, self.video_width, self.video_height)
            # Read the site live from the sim, the way the collision overlay
            # does. env.prev_ee_pos only updates inside step_nominal(), and the
            # ompl_mpc skill adapters drive the env directly -- so during a
            # skill it goes stale for long stretches and then jumps, which the
            # gap filter below then discards as a teleport. The result was a
            # trail of one dot.
            from .motion.panda import EE_SITE
            from .motion.scene_query import SceneQuery

            sim = getattr(getattr(env, "env", None), "sim", None) or getattr(env, "sim", None)
            ee = SceneQuery.from_robosuite_sim(sim).site_pose(EE_SITE)[0] if sim is not None else None
            if proj is not None and ee is not None:
                px = proj(_np.asarray(ee, dtype=float)[:3])
                if px is not None:
                    x, y = int(px[0]), int(px[1])
                    if not self._hud_trail or (x, y) != self._hud_trail[-1]:
                        self._hud_trail.append((x, y))
                        if len(self._hud_trail) > 400:
                            del self._hud_trail[0]
        except Exception:
            pass

        # robustness bottleneck, as a number rather than a formatted string, so
        # it can be drawn as a trace. A single printed value says nothing about
        # whether a constraint is being approached or recovered from.
        try:
            sub = self._video_current_subtask
            if isinstance(sub, dict):
                margins = self._solver_robustness_margins(env, sub)
                if margins:
                    self._hud_robustness.append(float(min(margins.values())))
                    if len(self._hud_robustness) > 600:
                        del self._hud_robustness[0]
        except Exception:
            pass

    def _draw_hud_trail(self, frame):
        """The path the end effector has taken, fading toward the oldest point.

        A still frame shows where the arm is; the trail shows how it got there,
        which is what makes a detour around a keep-out zone or a replan visible
        at all in a recording.
        """
        if len(getattr(self, "_hud_trail", ())) < 2:
            return frame
        try:
            from PIL import Image, ImageDraw
            import numpy as _np
        except Exception:
            return frame
        img = Image.fromarray(_np.asarray(frame, dtype=_np.uint8)).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        pts = self._hud_trail
        n = len(pts)
        for i in range(1, n):
            p0, p1 = pts[i - 1], pts[i]
            # A gap between consecutive samples means the arm was repositioned
            # (a replan, a homing move, a new subtask), not that it travelled
            # that line. Drawing the chord would invent a path it never took.
            if abs(p1[0] - p0[0]) + abs(p1[1] - p0[1]) > 38:
                continue
            a = int(30 + 190 * (i / n))           # older points fade out
            w = 2 if i < n - 40 else 3            # the recent tail reads heavier
            d.line([p0, p1], fill=(80, 190, 255, a), width=w)
        d.ellipse([pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4],
                  fill=(120, 215, 255, 235))
        return _np.asarray(Image.alpha_composite(img, layer).convert("RGB"))

    def _draw_hud_panel(self, draw, x0, y0, w, h, font, small):
        """Timeline, robustness trace and legend, drawn into the caption panel.

        Replaces three rows that read "Precondition: — / Postcondition: — /
        Recall: —" on almost every frame. What a viewer actually needs is where
        the run is, and whether the constraint is holding.
        """
        ink, dim = (236, 240, 245), (150, 160, 172)
        y = y0

        # --- subtask timeline -------------------------------------------------
        marks = getattr(self, "_hud_subtask_marks", []) or []
        total = max(1, getattr(self, "_hud_subtask_total", 0) or len(marks) or 1)
        cur = getattr(self, "_hud_subtask_index", 0) or 0
        draw.text((x0, y), "PROGRESS", fill=dim, font=small)
        y += 18
        seg_w = w / total
        for i in range(total):
            sx = x0 + i * seg_w
            done, active = i < cur, i == cur
            col = (60, 170, 120) if done else ((70, 150, 220) if active else (58, 62, 70))
            draw.rectangle((sx + 1, y, sx + seg_w - 2, y + 11), fill=col)
            label = marks[i][1] if i < len(marks) else ""
            if label and seg_w > 46:
                draw.text((sx + 4, y + 14), label[:11], fill=ink if active else dim, font=small)
        y += 42

        # --- robustness trace -------------------------------------------------
        trace = getattr(self, "_hud_robustness", []) or []
        draw.text((x0, y), "ROBUSTNESS   (below the line = constraint violated)", fill=dim, font=small)
        y += 18
        ph = 58
        draw.rectangle((x0, y, x0 + w, y + ph), fill=(26, 29, 34))
        if trace:
            lo, hi = min(trace), max(trace)
            span = max(hi - lo, 0.08)
            lo, hi = lo - 0.1 * span, hi + 0.1 * span
            zero_y = y + ph * (hi - 0.0) / (hi - lo)
            zero_y = min(max(zero_y, y), y + ph)
            draw.line([(x0, zero_y), (x0 + w, zero_y)], fill=(120, 90, 90), width=1)
            step = max(1.0, len(trace) / max(1.0, w))
            pts = []
            for i in range(int(len(trace) / step)):
                v = trace[min(len(trace) - 1, int(i * step))]
                px = x0 + i * (w / max(1, int(len(trace) / step)))
                py = y + ph * (hi - v) / (hi - lo)
                pts.append((px, min(max(py, y), y + ph)))
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    neg = pts[i][1] > zero_y
                    draw.line([pts[i - 1], pts[i]],
                              fill=(225, 95, 75) if neg else (90, 200, 150), width=2)
                draw.ellipse([pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3],
                             fill=(255, 255, 255))
            draw.rectangle((x0 + w - 92, y + 1, x0 + w - 2, y + 17), fill=(26, 29, 34))
            draw.text((x0 + w - 88, y + 2), f"{trace[-1]:+.3f}", fill=ink, font=small)
        y += ph + 10

        # --- events -----------------------------------------------------------
        ev = (getattr(self, "_hud_events", []) or [])[-3:]
        if ev:
            draw.text((x0, y), "EVENTS", fill=dim, font=small)
            y += 18
            for _, label in ev:
                draw.text((x0 + 4, y), f"• {label[:68]}", fill=(235, 200, 120), font=small)
                y += 17
        return y

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
        self._hud_record(env)
        frame = self._draw_hud_trail(frame)
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
        # Position in the plan, for the HUD's progress bar.
        try:
            idx = int(subtask.get("subtask_id", len(getattr(self, "_hud_subtask_marks", []))))
        except Exception:
            idx = len(getattr(self, "_hud_subtask_marks", []))
        self._hud_subtask_index = idx
        try:
            self._hud_subtask_total = max(int(getattr(self, "_hud_subtask_total", 0) or 0), idx + 1)
        except Exception:
            pass
        skill = str(subtask.get("skill") or "")
        marks = getattr(self, "_hud_subtask_marks", None)
        if marks is None:
            marks = self._hud_subtask_marks = []
        if not any(m[0] == idx for m in marks):
            marks.append((idx, skill))
            marks.sort()
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
        if self._video_current_reflection_summary:
            ev = getattr(self, "_hud_events", None)
            if ev is None:
                ev = self._hud_events = []
            ev.append((getattr(self, "_video_frame_count", 0),
                       "replan: " + self._video_current_reflection_summary))

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


