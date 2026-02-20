from __future__ import annotations

import base64
import heapq
import threading
from dataclasses import asdict, dataclass
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
    """
    Stored frame for dashboard/reporting.

    - avg_confidence is the ranking score your pipeline decides (can be pure conf or a composite score).
    - track_id groups frames that belong to the same person/head across frames.
    """
    timestamp: str
    reason: str
    classes: List[str]
    avg_confidence: float
    image_data: str
    detections: List[Dict[str, Any]]
    is_compliant: Optional[bool]  # True / False / None (NO_PERSON)
    track_id: str = "unknown"


class TrackFrameStore:
    """
    In-memory store — keeps top-K frames per (bucket, track_id).

    Bucket:
      - True  => compliant
      - False => non-compliant

    For each track_id inside a bucket, we keep a min-heap of size top_k.
    """
    def __init__(self, top_k: int = 2):
        self.top_k = max(1, int(top_k))
        self._lock = threading.Lock()

        # tiebreaker counter so heap never compares FrameRecord directly
        self._counter = 0

        # {True: {track_id: heap}, False: {track_id: heap}}
        # heap items are (score, counter, FrameRecord)
        self._tracks: Dict[bool, Dict[str, List[Tuple[float, int, FrameRecord]]]] = {
            True: {},
            False: {},
        }

    def clear(self) -> None:
        with self._lock:
            self._tracks = {True: {}, False: {}}
            self._counter = 0

    def add(self, rec: FrameRecord) -> None:
        """
        Store COMPLIANT / NON-COMPLIANT frames grouped by track_id.
        If rec.is_compliant is None (NO_PERSON), ignore it.
        """
        if rec.is_compliant is None:
            return

        bucket = True if rec.is_compliant is True else False
        track_id = (getattr(rec, "track_id", None) or "unknown").strip() or "unknown"

        with self._lock:
            heap = self._tracks[bucket].setdefault(track_id, [])
            self._counter += 1
            item = (float(rec.avg_confidence), self._counter, rec)

            if len(heap) < self.top_k:
                heapq.heappush(heap, item)
                return

            # replace only if stronger than weakest
            if item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    def _sorted_track_records(self, bucket: bool, track_id: str) -> List[FrameRecord]:
        heap = self._tracks[bucket].get(track_id, [])
        return [rec for _, _, rec in sorted(heap, key=lambda x: -x[0])]

    def _flatten_bucket(self, bucket: bool) -> List[FrameRecord]:
        items: List[Tuple[float, int, FrameRecord]] = []
        for heap in self._tracks[bucket].values():
            items.extend(heap)
        return [rec for _, _, rec in sorted(items, key=lambda x: -x[0])]

    @property
    def compliant(self) -> List[FrameRecord]:
        """Legacy flat list (best-score-first) across all compliant tracks."""
        with self._lock:
            return self._flatten_bucket(True)

    @property
    def non_compliant(self) -> List[FrameRecord]:
        """Legacy flat list (best-score-first) across all non-compliant tracks."""
        with self._lock:
            return self._flatten_bucket(False)

    def to_payload(self) -> Dict[str, Any]:
        """
        Returns BOTH grouped and legacy-flat payloads.

        New keys:
          - compliant_tracks: {track_id: [FrameRecord...]}
          - non_compliant_tracks: {track_id: [FrameRecord...]}

        Legacy keys kept to avoid breaking existing UI/report code:
          - compliant: [FrameRecord...]
          - non_compliant: [FrameRecord...]
        """
        with self._lock:
            compliant_tracks = {
                tid: [asdict(r) for r in self._sorted_track_records(True, tid)]
                for tid in sorted(self._tracks[True].keys())
            }
            non_compliant_tracks = {
                tid: [asdict(r) for r in self._sorted_track_records(False, tid)]
                for tid in sorted(self._tracks[False].keys())
            }

            compliant_flat = [asdict(r) for r in self._flatten_bucket(True)]
            non_compliant_flat = [asdict(r) for r in self._flatten_bucket(False)]

            return {
                "total": len(compliant_flat) + len(non_compliant_flat),
                "total_tracks": len(compliant_tracks) + len(non_compliant_tracks),
                "compliant": compliant_flat,
                "non_compliant": non_compliant_flat,
                "compliant_tracks": compliant_tracks,
                "non_compliant_tracks": non_compliant_tracks,
            }


# Keep the old name used everywhere else in your codebase:
FrameStore = TrackFrameStore


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
