"""SAM3 text-prompted instance segmentation.

Runs in-process by default (one ~3.5 GB bf16 model per process, loaded on
first use). Set STEP_SAM3_URL to use cap-x's launch_sam3_server instead,
which lets several rollout processes share one model.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    mask: np.ndarray  # (H, W) bool
    box_xyxy: np.ndarray  # (4,) pixels
    score: float
    prompt: str


class Sam3Segmenter:
    _shared: "Sam3Segmenter | None" = None

    def __init__(self, device: str = "cuda", server_url: str | None = None) -> None:
        self.device = device
        self.server_url = server_url
        self._processor = None
        # The image embedding is reused across prompts on the same frame.
        self._cached_key: int | None = None
        self._cached_state = None

    @classmethod
    def shared(cls) -> "Sam3Segmenter":
        if cls._shared is None:
            cls._shared = cls(server_url=os.environ.get("STEP_SAM3_URL") or None)
        return cls._shared

    def _ensure_model(self) -> None:
        if self._processor is not None:
            return
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(device=self.device)
        # Thresholding is done by the caller so one encode serves every threshold.
        self._processor = Sam3Processor(model, device=self.device, confidence_threshold=0.05)

    def segment(self, rgb: np.ndarray, prompt: str, score_threshold: float) -> list[Detection]:
        if self.server_url:
            dets = self._segment_via_server(rgb, prompt)
        else:
            dets = self._segment_local(rgb, prompt)
        return [d for d in dets if d.score >= score_threshold]

    def _segment_local(self, rgb: np.ndarray, prompt: str) -> list[Detection]:
        import torch
        from PIL import Image

        self._ensure_model()
        key = hash(np.ascontiguousarray(rgb).tobytes())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if key != self._cached_key:
                self._cached_state = self._processor.set_image(Image.fromarray(np.ascontiguousarray(rgb)))
                self._cached_key = key
            else:
                self._processor.reset_all_prompts(self._cached_state)
            out = self._processor.set_text_prompt(state=self._cached_state, prompt=prompt)
        if out.get("masks") is None or len(out["scores"]) == 0:
            return []
        masks = out["masks"].squeeze(1).cpu().numpy().astype(bool)
        boxes = out["boxes"].float().cpu().numpy()
        scores = out["scores"].float().cpu().numpy()
        return [Detection(m, b, float(s), prompt) for m, b, s in zip(masks, boxes, scores)]

    def _segment_via_server(self, rgb: np.ndarray, prompt: str) -> list[Detection]:
        import requests
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
        payload = {"image_base64": base64.b64encode(buf.getvalue()).decode(), "text_prompt": prompt}
        resp = requests.post(f"{self.server_url}/segment", json=payload, timeout=60)
        resp.raise_for_status()
        out = []
        for item in resp.json().get("results", []):
            mask = np.frombuffer(base64.b64decode(item["mask_base64"]), dtype=np.uint8).reshape(item["shape"]).astype(bool)
            out.append(Detection(mask, np.asarray(item["box"], dtype=np.float64), float(item["score"]), prompt))
        return out
