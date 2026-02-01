from __future__ import annotations

import base64
import io
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

def bgr_to_data_url_png(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode frame to PNG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode frame to PNG")
    return buf.tobytes()

@dataclass
class FrameRecord:
    timestamp: str
    reason: str
    classes: List[str]
    avg_confidence: float
    image_data: str  # data URL (PNG)
    # extra
    detections: List[Dict[str, Any]]
    is_compliant: bool

class FrameStore:
    """In-memory store for the dashboard + report generation."""

    def __init__(self, max_items: int = 200):
        self.max_items = max_items
        self._lock = threading.Lock()
        self.compliant: List[FrameRecord] = []
        self.non_compliant: List[FrameRecord] = []

    def clear(self) -> None:
        with self._lock:
            self.compliant.clear()
            self.non_compliant.clear()

    def add(self, rec: FrameRecord) -> None:
        with self._lock:
            target = self.compliant if rec.is_compliant else self.non_compliant
            target.append(rec)

            # enforce max size across both lists (drop oldest)
            total = len(self.compliant) + len(self.non_compliant)
            while total > self.max_items:
                # drop from the older bucket first
                if self.non_compliant:
                    self.non_compliant.pop(0)
                elif self.compliant:
                    self.compliant.pop(0)
                total = len(self.compliant) + len(self.non_compliant)

    def to_payload(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total": len(self.compliant) + len(self.non_compliant),
                "compliant": [asdict(x) for x in self.compliant],
                "non_compliant": [asdict(x) for x in self.non_compliant],
            }

@dataclass
class PipelineStats:
    total_frames_processed: int = 0
    frames_with_person: int = 0
    frames_without_person: int = 0
    compliant_frames: int = 0
    non_compliant_frames: int = 0

    @property
    def person_detection_rate(self) -> str:
        if self.total_frames_processed == 0:
            return "0.0%"
        return f"{(self.frames_with_person / self.total_frames_processed) * 100:.1f}%"

    @property
    def compliance_rate(self) -> str:
        denom = self.compliant_frames + self.non_compliant_frames
        if denom == 0:
            return "0.0%"
        return f"{(self.compliant_frames / denom) * 100:.1f}%"
