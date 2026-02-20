from __future__ import annotations

import threading
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

from camera.streamer import CameraReader
from detectors.person_detector import PersonDetector
from detectors.compliance_detector import ComplianceDetector, ComplianceResult
from core.state import FrameStore, FrameRecord, PipelineStats, bgr_to_data_url_png
from alerts.beeper import Beeper

logger = logging.getLogger(__name__)

# Target inference rate — 8 FPS is a good balance between responsiveness and CPU load
INFERENCE_FPS = 8
INFERENCE_INTERVAL = 1.0 / float(max(1, INFERENCE_FPS))


def _score_frame(detections: List[Dict[str, Any]], frame_w: int, frame_h: int) -> float:
    """
    Combines confidence + centrality so partial/edge-of-frame detections rank lower.

    Score = 0.7 * avg_confidence + 0.3 * avg_centrality
      - avg_confidence: mean detection confidence
      - avg_centrality: 1.0 = centered, 0.0 = at edge
    """
    if not detections:
        return 0.0

    avg_conf = float(np.mean([float(d["conf"]) for d in detections]))

    centrality_scores: List[float] = []
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0

        # normalized distance from center: 0=center, 1=edge/corner
        dx = abs(cx - frame_w / 2.0) / (frame_w / 2.0 + 1e-6)
        dy = abs(cy - frame_h / 2.0) / (frame_h / 2.0 + 1e-6)

        centrality_scores.append(max(0.0, 1.0 - ((dx + dy) / 2.0)))

    avg_centrality = float(np.mean(centrality_scores)) if centrality_scores else 0.0
    return 0.7 * avg_conf + 0.3 * avg_centrality


def _xyxy_iou(a: List[float], b: List[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area + 1e-6
    return float(inter_area / union)


def _pick_track_bbox(detections: List[Dict[str, Any]]) -> Optional[List[float]]:
    """
    Pick a representative bbox to track the person/head.

    For PPE models where detections are around the head area
    (head_net/beard_net/exposed_hair/etc), the highest-confidence bbox
    is usually a good proxy for the "head track".
    """
    if not detections:
        return None
    best = max(detections, key=lambda d: float(d.get("conf", 0.0) or 0.0))
    xyxy = best.get("xyxy")
    if not (isinstance(xyxy, (list, tuple)) and len(xyxy) == 4):
        return None
    return [float(x) for x in xyxy]


class _SimpleIOUTracker:
    """
    Tiny IOU-based tracker to assign a stable track_id across frames.

    Intentionally simple (no Kalman/SORT). Works well for fixed CCTV angles
    where the head moves smoothly and detections are consistent.
    """

    def __init__(self, iou_thres: float = 0.40, ttl_sec: float = 2.0):
        self.iou_thres = float(iou_thres)
        self.ttl_sec = float(ttl_sec)
        self._next_id = 1
        # track_id -> {"bbox": [x1,y1,x2,y2], "last_seen": unix_ts}
        self._tracks: Dict[str, Dict[str, Any]] = {}

    def _expire(self, now: float) -> None:
        dead = [
            tid
            for tid, t in self._tracks.items()
            if (now - float(t["last_seen"])) > self.ttl_sec
        ]
        for tid in dead:
            self._tracks.pop(tid, None)

    def assign(self, bbox: Optional[List[float]], now: float) -> str:
        """Return an existing track_id if bbox matches, else create a new one."""
        if bbox is None:
            return "unknown"

        self._expire(now)

        best_tid: Optional[str] = None
        best_iou = 0.0
        for tid, t in self._tracks.items():
            iou = _xyxy_iou(bbox, t["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        if best_tid is not None and best_iou >= self.iou_thres:
            self._tracks[best_tid]["bbox"] = bbox
            self._tracks[best_tid]["last_seen"] = now
            return best_tid

        tid = f"T{self._next_id}"
        self._next_id += 1
        self._tracks[tid] = {"bbox": bbox, "last_seen": now}
        return tid


def annotate_frame(frame_bgr: np.ndarray, result: ComplianceResult) -> np.ndarray:
    img = frame_bgr.copy()

    status = result.status
    if result.is_compliant is True:
        color = (94, 197, 34)  # green BGR
    elif result.is_compliant is False:
        color = (68, 68, 239)  # red BGR
    else:
        color = (180, 180, 180)  # grey for NO_PERSON / NO_HEAD

    # Banner
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), color, thickness=-1)
    cv2.putText(
        img,
        f"{status}: {result.reason}",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Detection boxes
    for det in result.detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        cls = str(det["class_name"])
        conf = float(det["conf"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{cls} {conf:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    # Timestamp
    cv2.putText(
        img,
        result.timestamp,
        (10, img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return img


class Pipeline:
    """Background pipeline:
    - Read frames from CameraReader
    - If person/head present => run compliance detector
    - Update latest result + in-memory store
    """

    def __init__(
        self,
        camera: CameraReader,
        person_detector: Optional[PersonDetector],
        compliance_detector: ComplianceDetector,
        store: FrameStore,
        beeper: Beeper,
    ):
        self.camera = camera
        self.person_detector = person_detector
        self.compliance_detector = compliance_detector
        self.store = store
        self.beeper = beeper

        self.stats = PipelineStats()
        self._lock = threading.Lock()
        self._latest_result: Optional[ComplianceResult] = None
        self._latest_annotated: Optional[np.ndarray] = None

        # Simple IOU tracker to group same head/person across frames
        self._tracker = _SimpleIOUTracker(iou_thres=0.40, ttl_sec=2.0)

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.camera.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.camera.stop()

    def latest_result(self) -> Optional[ComplianceResult]:
        with self._lock:
            return self._latest_result

    def latest_annotated(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_annotated is None else self._latest_annotated.copy()

    def _set_no_person_result(self, reason: str = "no_person_detected") -> None:
        """
        Critical: overwrite latest result + clear annotated frame
        so previous COMPLIANT/NON-COMPLIANT never persists.
        """
        ts = datetime.now().isoformat(timespec="seconds")
        no_person = ComplianceResult(
            is_compliant=None,
            status="NO_PERSON",
            reason=reason,
            detected_classes=[],
            confidence_scores={},
            violations=[],
            detections=[],
            timestamp=ts,
        )
        with self._lock:
            self._latest_result = no_person
            self._latest_annotated = None

    def _loop(self) -> None:
        logger.info("Pipeline started")
        last_infer = 0.0

        while self._running:
            pkt = self.camera.latest()
            if pkt is None:
                time.sleep(0.05)
                continue

            # Rate-limit inference
            now = time.time()
            if now - last_infer < INFERENCE_INTERVAL:
                time.sleep(0.005)
                continue
            last_infer = now

            frame = pkt.frame_bgr
            self.stats.total_frames_processed += 1

            # Person/head gating
            has_person = True
            if self.person_detector is not None:
                try:
                    has_person = bool(self.person_detector.has_person(frame))
                except Exception:
                    logger.exception("Person detector failed; treating as no person")
                    has_person = False

            if not has_person:
                self.stats.frames_without_person += 1
                self._set_no_person_result("no_person_detected")
                continue

            self.stats.frames_with_person += 1

            # Compliance
            result = self.compliance_detector.detect(frame)
            annotated = annotate_frame(frame, result)

            # Always update UI output
            with self._lock:
                self._latest_result = result
                self._latest_annotated = annotated

            # If compliance returns NO_PERSON, do not store/alert
            if result.status == "NO_PERSON":
                continue

            # Stats
            if result.is_compliant is True:
                self.stats.compliant_frames += 1
            elif result.is_compliant is False:
                self.stats.non_compliant_frames += 1

            # Alerts
            if result.is_compliant is False:
                if "exposed_hair" in result.violations or "exposed_beard" in result.violations:
                    self.beeper.beep_critical()
                else:
                    self.beeper.beep_warning()

            # Store in dashboard (Top-K per track) using a score
            h, w = frame.shape[:2]
            score = _score_frame(result.detections, w, h)

            # Track identity (same head/person across frames)
            track_bbox = _pick_track_bbox(result.detections)
            track_id = self._tracker.assign(track_bbox, now)

            rec = FrameRecord(
                timestamp=result.timestamp,
                reason=result.reason,
                classes=result.detected_classes,
                avg_confidence=score,
                image_data=bgr_to_data_url_png(annotated),
                detections=result.detections,
                is_compliant=result.is_compliant,
                track_id=track_id,
            )
            self.store.add(rec)

    def run_on_external_frame(self, frame_bgr: np.ndarray):
        """
        Upload helper.
        Returns: (result_dict, annotated_frame_bgr | None)
        """
        has_person = True
        if self.person_detector is not None:
            has_person = bool(self.person_detector.has_person(frame_bgr))

        if not has_person:
            ts = datetime.now().isoformat(timespec="seconds")
            res = ComplianceResult(
                is_compliant=None,
                status="NO_PERSON",
                reason="no_person_detected",
                detected_classes=[],
                confidence_scores={},
                violations=[],
                detections=[],
                timestamp=ts,
            )
            return res.to_dict(), None

        result = self.compliance_detector.detect(frame_bgr)
        if result.status == "NO_PERSON":
            return result.to_dict(), None

        annotated = annotate_frame(frame_bgr, result)
        return result.to_dict(), annotated
