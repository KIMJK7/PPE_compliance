from __future__ import annotations

import threading
import time
import logging
from dataclasses import asdict
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

from camera.streamer import CameraReader
from detectors.person_detector import PersonDetector
from detectors.compliance_detector import ComplianceDetector, ComplianceResult
from core.state import FrameStore, FrameRecord, PipelineStats, bgr_to_data_url_png
from alerts.beeper import Beeper

logger = logging.getLogger(__name__)

def annotate_frame(frame_bgr: np.ndarray, result: ComplianceResult) -> np.ndarray:
    img = frame_bgr.copy()
    # banner
    status = result.status
    color = (34, 197, 94) if result.is_compliant else (239, 68, 68)  # BGR-ish? actually OpenCV uses BGR
    # OpenCV uses BGR; we used RGB values; convert:
    color = (color[2], color[1], color[0])
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), color, thickness=-1)
    cv2.putText(img, f"{status}: {result.reason}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    for det in result.detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        cls = det["class_name"]
        conf = det["conf"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # timestamp
    cv2.putText(img, result.timestamp, (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

class Pipeline:
    """Background pipeline:
    - Read frames from CameraReader
    - If person present => run compliance detector
    - Update latest results + in-memory store
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

    def _loop(self) -> None:
        logger.info("Pipeline started")
        while self._running:
            pkt = self.camera.latest()
            if pkt is None:
                time.sleep(0.05)
                continue

            frame = pkt.frame_bgr
            self.stats.total_frames_processed += 1

            # person gating
            has_person = True
            if self.person_detector is not None:
                has_person = self.person_detector.has_person(frame)

            if has_person:
                self.stats.frames_with_person += 1
                result = self.compliance_detector.detect(frame)
                annotated = annotate_frame(frame, result)

                # alert logic
                if not result.is_compliant:
                    # critical if exposed_hair or exposed_beard, else warning
                    if "exposed_hair" in result.violations or "exposed_beard" in result.violations:
                        self.beeper.beep_critical()
                    else:
                        self.beeper.beep_warning()

                # update stats
                if result.is_compliant:
                    self.stats.compliant_frames += 1
                else:
                    self.stats.non_compliant_frames += 1

                # create record for dashboard
                avg_conf = float(np.mean([d["conf"] for d in result.detections])) if result.detections else 0.0
                rec = FrameRecord(
                    timestamp=result.timestamp,
                    reason=result.reason,
                    classes=result.detected_classes,
                    avg_confidence=avg_conf,
                    image_data=bgr_to_data_url_png(annotated),
                    detections=result.detections,
                    is_compliant=result.is_compliant,
                )
                self.store.add(rec)

                with self._lock:
                    self._latest_result = result
                    self._latest_annotated = annotated
            else:
                self.stats.frames_without_person += 1

            time.sleep(0.001)
    
    def run_on_external_frame(self, frame_bgr):
        """
        Runs the same detection logic on an externally provided frame (upload image/video frame).
        Returns: (result_dict, annotated_frame_bgr)
        """
        # Person gating (if person detector exists)
        has_person = True
        person_boxes = []
        if self.person_detector is not None:
            person_boxes = self.person_detector.detect(frame_bgr)
            has_person = len(person_boxes) > 0

        detections = []
        classes = []
        avg_conf = 0.0
        status = "NON-COMPLIANT"
        reason = "required PPE not detected"

        annotated = frame_bgr.copy()

        if has_person:
            detections = self.compliance_detector.detect(frame_bgr)
            classes = sorted(list({d["class"] for d in detections}))
            if len(detections) > 0:
                avg_conf = float(sum([d["conf"] for d in detections]) / len(detections))

            # Use your same compliance decision (reuse existing logic if you already have it)
            # If you already have a decision function in pipeline, call that instead.
            status, reason = self._decide_compliance(classes)

            # Draw detections using existing helper if you have one
            annotated = self.compliance_detector.draw(frame_bgr, detections, status=status)

        result = {
            "status": status,
            "reason": reason,
            "classes": classes,
            "detections": detections,
            "avg_confidence": avg_conf,
            "has_person": has_person,
        }
        return result, annotated

